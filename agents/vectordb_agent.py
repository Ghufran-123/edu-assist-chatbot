from __future__ import annotations

import os
import re
import json
import logging
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from rapidfuzz import fuzz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger("eduassist.vectordb_agent")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "intfloat/e5-base-v2"
    use_e5_prefix: bool = True
    device: str = "cpu"
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 160
    separators: Tuple[str, ...] = ("\n\n", "\n", ".", " ", "")


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 15                   # FAISS candidates to fetch
    rerank_k: int = 8             # top candidates to rerank
    top_chunks: int = 5           # ✅ raised from 2 → 5 for richer LLM context
    min_chars: int = 180
    max_distance: float = 0.90    # ✅ relaxed from 0.85 — catches more relevant chunks
    min_keyword_score: int = 25   # ✅ lowered from 45 — short queries like "code jung" now pass
    tie_window: float = 0.05


class VectorDBAgent:
    def __init__(
        self,
        vector_path: str,
        *,
        embedding_cfg: EmbeddingConfig = EmbeddingConfig(),
        chunking_cfg: ChunkingConfig = ChunkingConfig(),
        retrieval_cfg: RetrievalConfig = RetrievalConfig(),
    ):
        self.vector_path = vector_path
        os.makedirs(self.vector_path, exist_ok=True)

        self.embedding_cfg = embedding_cfg
        self.chunking_cfg = chunking_cfg
        self.retrieval_cfg = retrieval_cfg

        self._state_path = os.path.join(self.vector_path, "_index_state.json")

        self.embeddings = self._init_embeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_cfg.chunk_size,
            chunk_overlap=self.chunking_cfg.chunk_overlap,
            separators=list(self.chunking_cfg.separators),
        )

        self.vectorstore: Optional[FAISS] = self._load_vectorstore()

    # ------------------ Public ------------------

    def add_documents_from_folder(self, folder_path: str) -> None:
        if not os.path.isdir(folder_path):
            logger.warning("Document folder not found: %s", folder_path)
            return

        files: List[str] = []
        for root, _, names in os.walk(folder_path):
            for name in names:
                ext = os.path.splitext(name)[1].lower()
                if ext in (".pdf", ".txt"):
                    files.append(os.path.join(root, name))

        if not files:
            logger.info("No documents found in %s", folder_path)
            return

        state = self._load_state()
        changed = [fp for fp in files if state.get(fp) != self._file_sig(fp)]
        if not changed:
            logger.info("All documents unchanged, skipping indexing.")
            return

        logger.info("Indexing %d changed file(s)...", len(changed))

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for fp in changed:
            extracted = self._extract_file(fp)
            for raw_text, meta in extracted:
                for chunk in self._chunk(raw_text):
                    clean = self._clean_text(chunk)
                    if self._is_low_quality(clean):
                        continue

                    prepared = self._format_passage(clean)
                    h = self._hash(prepared)

                    meta2 = dict(meta)
                    meta2.update({
                        "file_path": fp,
                        "file_name": os.path.basename(fp),
                        "chunk_hash": h,
                    })

                    texts.append(prepared)
                    metas.append(meta2)

        texts, metas = self._dedup(texts, metas)
        if not texts:
            logger.info("No new chunks to index after dedup.")
            return

        logger.info("Adding %d chunks to FAISS...", len(texts))

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metas)
        else:
            self.vectorstore.add_texts(texts, metadatas=metas)

        self.vectorstore.save_local(self.vector_path)

        for fp in changed:
            state[fp] = self._file_sig(fp)
        self._save_state(state)

        logger.info("✅ Indexing complete.")

    def search(self, query: str) -> Optional[Union[str, List[str]]]:
        """
        Search the vectorstore.
        Returns top-N chunks as a list (for LLM multi-chunk context),
        or None if nothing found.
        """
        if not self.vectorstore:
            return None

        raw_query = (query or "").strip()
        if not raw_query:
            return None

        embed_query = self._format_query(raw_query)

        try:
            results = self.vectorstore.similarity_search_with_score(
                embed_query, k=self.retrieval_cfg.k
            )
        except Exception as e:
            logger.error("FAISS search error: %s", e)
            return None

        if not results:
            return None

        candidates: List[Dict[str, Any]] = []
        for doc, score in results:
            distance = float(score)
            if distance > self.retrieval_cfg.max_distance:
                continue

            content = (doc.page_content or "").strip()
            if not content:
                continue

            text = self._strip_passage_prefix(content)
            if self._is_low_quality(text):
                continue

            kw = int(fuzz.token_set_ratio(raw_query.lower(), text.lower()))
            if kw < self.retrieval_cfg.min_keyword_score:
                continue

            candidates.append({"distance": distance, "kw": kw, "text": text})

        if not candidates:
            return None

        # ✅ Sort: primary by distance (lower=better), secondary by kw score (higher=better)
        candidates.sort(key=lambda x: (x["distance"], -x["kw"]))

        # ✅ Apply tie-breaking on top candidates
        candidates = self._tie_break_top(candidates)

        # ✅ Return top-N chunks as a list for the LLM to use as merged context
        top_chunks = [c["text"] for c in candidates[: self.retrieval_cfg.top_chunks]]

        return top_chunks if top_chunks else None

    # ------------------ Tie-breaking ------------------

    def _tie_break_top(self, candidates: List[Dict]) -> List[Dict]:
        """Swap top-2 if they're within tie_window and second has better kw score."""
        if len(candidates) < 2:
            return candidates
        best, second = candidates[0], candidates[1]
        if (
            abs(best["distance"] - second["distance"]) <= self.retrieval_cfg.tie_window
            and second["kw"] > best["kw"]
        ):
            candidates[0], candidates[1] = second, best
        return candidates

    # ------------------ Embeddings ------------------

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        model_kwargs = {"device": self.embedding_cfg.device}
        encode_kwargs = {"normalize_embeddings": self.embedding_cfg.normalize_embeddings}
        return HuggingFaceEmbeddings(
            model_name=self.embedding_cfg.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def _format_query(self, raw: str) -> str:
        return f"query: {raw}" if self.embedding_cfg.use_e5_prefix else raw

    def _format_passage(self, text: str) -> str:
        return f"passage: {text}" if self.embedding_cfg.use_e5_prefix else text

    # ------------------ Load/save ------------------

    def _load_vectorstore(self) -> Optional[FAISS]:
        try:
            if os.path.exists(os.path.join(self.vector_path, "index.faiss")):
                logger.info("Loading existing FAISS index from %s", self.vector_path)
                return FAISS.load_local(
                    self.vector_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
        except Exception as e:
            logger.warning("Failed to load FAISS index: %s", e)
        return None

    # ------------------ Extraction ------------------

    def _extract_file(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        if ext == ".txt":
            return self._extract_txt(file_path)
        return []

    def _extract_pdf(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        ✅ Uses pymupdf (fitz) instead of PyPDF2.
        pymupdf handles complex layouts, tables, and unicode far better.
        Install: pip install pymupdf
        """
        out: List[Tuple[str, Dict[str, Any]]] = []
        try:
            import fitz  # pymupdf
            doc = fitz.open(file_path)
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    out.append((text, {"source_type": "pdf", "page": i}))
            doc.close()
        except ImportError:
            logger.warning(
                "pymupdf not installed. Falling back to PyPDF2. "
                "Run: pip install pymupdf"
            )
            out = self._extract_pdf_pypdf2(file_path)
        except Exception as e:
            logger.error("PDF extraction failed for %s: %s", file_path, e)
        return out

    def _extract_pdf_pypdf2(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Fallback PDF extractor using PyPDF2."""
        out: List[Tuple[str, Dict[str, Any]]] = []
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        out.append((text, {"source_type": "pdf", "page": i}))
        except Exception as e:
            logger.error("PyPDF2 fallback failed for %s: %s", file_path, e)
        return out

    def _extract_txt(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            return [(text, {"source_type": "txt"})] if text else []
        except Exception as e:
            logger.error("TXT extraction failed for %s: %s", file_path, e)
            return []

    # ------------------ Chunking / cleaning ------------------

    def _chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text or "")

    _space_re = re.compile(r"\s+", flags=re.UNICODE)

    def _clean_text(self, text: str) -> str:
        text = (text or "").strip()
        text = self._space_re.sub(" ", text).strip()
        return text

    def _strip_passage_prefix(self, text: str) -> str:
        t = text.strip()
        if t.lower().startswith("passage:"):
            return t.split(":", 1)[1].strip()
        return t

    def _is_low_quality(self, text: str) -> bool:
        if not text:
            return True
        if len(text) < self.retrieval_cfg.min_chars:
            return True
        if len(text.split()) < 20:
            return True
        return False

    # ------------------ Dedup / state ------------------

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def _dedup(
        self, texts: List[str], metas: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        seen = set()
        nt, nm = [], []
        for t, m in zip(texts, metas):
            h = m.get("chunk_hash") or self._hash(t)
            if h in seen:
                continue
            seen.add(h)
            nt.append(t)
            nm.append(m)
        return nt, nm

    def _file_sig(self, fp: str) -> str:
        try:
            st = os.stat(fp)
            return f"{st.st_mtime_ns}-{st.st_size}"
        except Exception:
            return "missing"

    def _load_state(self) -> Dict[str, str]:
        if not os.path.exists(self._state_path):
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _save_state(self, state: Dict[str, str]) -> None:
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save index state: %s", e)