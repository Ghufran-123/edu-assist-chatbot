from __future__ import annotations

import hashlib
import json
import shelve
import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from agents.faq_agent import FAQAgent
from agents.vectordb_agent import VectorDBAgent
from agents.webscraper_agent import WebScraperAgent
from agents.llm_agent import LLMAgent
from agents.consistency_check_agent import ConsistencyCheckAgent
from agents.summarizing_agent import SummarizingAgent

logger = logging.getLogger("eduassist.orchestrator")


class OrchestratorAgent:
    """
    Full 8-step RAG pipeline for EDU Assist — Iteration 4.

      [1] FAQ Agent       — MongoDB fuzzy + core-topic match
      [2] VectorDB Agent  — FAISS semantic search (top-3 chunks)
      [3] WebScraper      — Live KIET website crawl (top-3 chunks)
      [4] LLM Agent       — Generates answer from merged chunks + conversation history
      [5] Consistency     — Hallucination detection + retry
      [6] Summarizer      — Student-friendly bullet-point output
      [7] Link Appender   — Appends relevant KIET links
      [8] FAQ Storage     — Stores verified Q&A back to MongoDB

    Iteration 4 additions:
      - Answer caching     — Same question returns instantly from cache ✅
      - Confidence scores  — Every answer tagged high / medium / low ✅
      - History-aware LLM  — Conversation history passed to LLM ✅

    Smart routing:
      - Time-sensitive / admission date queries  → WEB first
      - Program / event / admission queries      → WEB first
      - All others                               → VECTOR first
    """

    # ✅ Iteration 4: Cache settings
    _CACHE_TTL_SECONDS = 3600       # 1 hour — cached answers expire after this
    _CACHE_MAX_SIZE    = 500        # max entries before LRU eviction

    LINK_KEYWORDS = {
        "admission", "apply", "application", "form",
        "last date", "closing date", "deadline", "eligibility",
        "entry test", "merit", "apply now",
        "program", "programs", "degree", "bachelor", "bs", "ms",
        "courses", "department", "faculty",
        "code jung", "event", "hackathon", "seminar", "workshop", "fest",
        "fee", "fees", "scholarship", "financial",
        "contact", "address", "location", "campus",
        "website", "portal", "link", "page", "online",
    }

    FALLBACK_LINKS = {
        # Admissions
        "admission":    ("🎓 Admissions",            "https://admissions.kiet.edu.pk/"),
        "apply":        ("🎓 Apply Online",           "https://kiet.edu.pk/apply"),
        "eligibility":  ("📋 Admission Process",      "https://admissions.kiet.edu.pk/admission-process/"),
        "requirement":  ("📋 Admission Process",      "https://admissions.kiet.edu.pk/admission-process/"),
        "entry test":   ("📝 Aptitude Test Samples",  "https://admissions.kiet.edu.pk/sample-test-paper/"),
        "aptitude":     ("📝 Aptitude Test Samples",  "https://admissions.kiet.edu.pk/sample-test-paper/"),
        "last date":    ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "closing date": ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "deadline":     ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "schedule":     ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "merit":        ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        # Fee & Scholarship
        "fee":          ("💰 Fee Structure",          "https://kiet.edu.pk/fee-structure/"),
        "scholarship":  ("🏆 Scholarship & Discount", "https://admissions.kiet.edu.pk/scholarship-fee-discount/"),
        "discount":     ("🏆 Scholarship & Discount", "https://admissions.kiet.edu.pk/scholarship-fee-discount/"),
        # Programs
        "program":      ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "courses":      ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "bachelor":     ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "degree":       ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "software":     ("💻 COCIS Faculty",          "https://cocis.kiet.edu.pk/"),
        "computer":     ("💻 COCIS Faculty",          "https://cocis.kiet.edu.pk/"),
        "ai":           ("💻 COCIS Faculty",          "https://cocis.kiet.edu.pk/"),
        "management":   ("💼 COMS Faculty",           "https://coms.kiet.edu.pk/"),
        "mba":          ("💼 COMS Faculty",           "https://coms.kiet.edu.pk/"),
        "bba":          ("💼 COMS Faculty",           "https://coms.kiet.edu.pk/"),
        # Events
        "event":        ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "code jung":    ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "hackathon":    ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "seminar":      ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "workshop":     ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        # Student
        "portal":       ("🖥 Student LMS Portal",     "https://lms.kiet.edu.pk/kietlms/my/Student_Portal.php"),
        "lms":          ("🖥 Student LMS Portal",     "https://lms.kiet.edu.pk/kietlms/my/Student_Portal.php"),
        "transport":    ("🚌 Transport Services",      "https://kiet.edu.pk/services/students-transport-services/"),
        "sports":       ("⚽ Sports Activities",       "https://kiet.edu.pk/sports-activities/"),
        "alumni":       ("🤝 Alumni",                  "https://kiet.edu.pk/alumni/"),
        "job":          ("💼 Jobs at KIET",            "https://kiet.edu.pk/jobs/"),
        # Contact & Info
        "contact":      ("📞 Departments Contact",    "https://kiet.edu.pk/departments-contact/"),
        "address":      ("🗺 Route Map",               "https://kiet.edu.pk/route-map/"),
        "location":     ("🗺 Route Map",               "https://kiet.edu.pk/route-map/"),
        "calendar":     ("📆 Academic Calendar",       "https://kiet.edu.pk/academics/academic-calendar/"),
        "faq":          ("❓ Admissions FAQs",         "https://kiet.edu.pk/faq/"),
        "about":        ("🏫 About KIET",              "https://kiet.edu.pk/about/"),
    }

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        vector_path: str,
        urls=None,
        base_domain: Optional[str] = None,
        *,
        faq_min_score: int = 82,
        enable_smart_routing: bool = True,
        enable_timing_logs: bool = False,
        consistency_max_retries: int = 1,
        debug: bool = False,
    ):
        self.faq_agent = FAQAgent(mongo_uri, db_name, collection_name)
        self.vector_agent = VectorDBAgent(vector_path)
        self.webscraper_agent = WebScraperAgent(
            urls=urls,
            base_domain=base_domain or "kiet.edu.pk",
        )
        self.llm_agent = LLMAgent()
        self.consistency_agent = ConsistencyCheckAgent(
            self.llm_agent,
            max_retries=consistency_max_retries,
            enable_llm_verify=False,
            debug=debug,
        )
        self.summarizing_agent = SummarizingAgent(self.llm_agent, debug=debug)

        self.faq_min_score    = int(faq_min_score)
        self.enable_smart_routing  = bool(enable_smart_routing)
        self.enable_timing_logs    = bool(enable_timing_logs)
        self.debug = debug

        # Answer cache — in-memory with disk persistence via shelve
        # Survives server restarts; thread-safe via lock
        self._answer_cache: Dict[str, tuple] = {}
        self._cache_lock = threading.Lock()
        self._cache_path = str(Path(vector_path) / "answer_cache")
        self._load_persistent_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        conversation_history: str = "",     # ✅ Iteration 4 — full history string
    ) -> Dict[str, str]:
        question = (question or "").strip()
        if not question:
            return self._fallback()

        # ✅ Iteration 4: Check answer cache first (bypass entire pipeline)
        cached = self._get_cached_answer(question)
        if cached is not None:
            logger.info("⚡ Cache hit for: '%s'", question[:50])
            cached["from_cache"] = "true"
            return cached

        t0 = time.perf_counter()
        timings: Dict[str, float] = {}

        # Step 1: FAQ (fast path)
        faq_result = self._timed("FAQ", timings, lambda: self._run_faq(question))
        if faq_result is not None:
            faq_result["answer"] = self._append_links(
                question, faq_result["answer"], [], "FAQ"
            )
            # ✅ Iteration 4: Confidence — FAQ = high (verified, fuzzy matched)
            faq_result["confidence"]       = "high"
            faq_result["confidence_label"] = "✅ Verified Answer"
            faq_result["from_cache"]       = "false"

            self._set_cached_answer(question, faq_result)

            if self.enable_timing_logs:
                self._log(["FAQ"], timings, time.perf_counter() - t0, "FAQ")
            return faq_result

        # Step 2: Retrieve chunks
        order = self._decide_order(question)
        chunk: Optional[List[str]] = None
        chunk_source: Optional[str] = None

        for name in order:
            if name == "FAQ":
                continue
            result = self._timed(name, timings, lambda n=name: self._run_retriever(n, question))
            if result:
                chunk = result
                chunk_source = name
                logger.info("Retrieval success via %s (%d chunks)", name, len(chunk))
                break

        if not chunk:
            logger.info("No chunks found for: '%s'", question[:60])
            if self.enable_timing_logs:
                self._log(order, timings, time.perf_counter() - t0, "Fallback")
            return self._fallback()

        # Step 3: Extract URLs + strip Source: lines before LLM
        source_urls  = self._extract_source_urls(chunk)
        clean_chunk  = self._strip_source_lines(chunk)

        # Step 4: LLM generation — ✅ Iteration 4: pass conversation_history
        t_llm = time.perf_counter()
        raw_answer = self.llm_agent.generate(
            question=question,
            context=clean_chunk,
            conversation_history=conversation_history,   # ✅ NEW
        )
        timings["LLM"] = time.perf_counter() - t_llm

        if not raw_answer:
            logger.warning("LLM returned empty answer.")
            return self._fallback()

        # Step 5: Consistency check
        chunk_text = "\n\n".join(clean_chunk) if isinstance(clean_chunk, list) else clean_chunk
        t_cc = time.perf_counter()
        final_answer, passed, attempts = self.consistency_agent.check_and_fix(
            question=question, answer=raw_answer, chunk=chunk_text,
        )
        timings["ConsistencyCheck"] = time.perf_counter() - t_cc
        logger.info("Consistency check: passed=%s attempts=%d", passed, attempts)

        # Step 6: Summarize
        t_sum = time.perf_counter()
        summarized = self.summarizing_agent.summarize(final_answer)
        timings["Summarize"] = time.perf_counter() - t_sum

        # Step 7: Append links
        summarized = self._append_links(question, summarized, source_urls, chunk_source)

        # Step 8: Store to FAQ
        self._store_to_faq(question, summarized)

        # ✅ Iteration 4: Confidence — medium if consistency passed, low if not
        if passed:
            confidence       = "medium"
            confidence_label = "🔍 AI Generated"
        else:
            confidence       = "low"
            confidence_label = "⚠️ Low Confidence"

        result = {
            "source":                f"{chunk_source} Agent -> LLM",
            "answer":                summarized,
            "consistency_passed":    str(passed),
            "consistency_attempts":  str(attempts),
            "confidence":            confidence,
            "confidence_label":      confidence_label,
            "from_cache":            "false",
        }

        # ✅ Iteration 4: Store in cache — only medium/high confidence answers
        if passed:
            self._set_cached_answer(question, result)

        if self.enable_timing_logs:
            self._log(
                order + ["LLM", "ConsistencyCheck", "Summarize"],
                timings, time.perf_counter() - t0, chunk_source or "Unknown"
            )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Iteration 4: Answer Cache
    # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # Answer Cache — persistent (shelve) + in-memory, thread-safe
    # ──────────────────────────────────────────────────────────────────────────

    def _load_persistent_cache(self) -> None:
        """Load cache from disk into memory on startup."""
        try:
            with shelve.open(self._cache_path) as db:
                now = time.time()
                loaded = 0
                for key, (result, ts) in db.items():
                    if (now - ts) <= self._CACHE_TTL_SECONDS:
                        self._answer_cache[key] = (result, ts)
                        loaded += 1
            logger.info("Persistent cache loaded: %d valid entries", loaded)
        except Exception as e:
            logger.warning("Could not load persistent cache: %s", e)

    def _persist_cache_entry(self, key: str, result: dict, ts: float) -> None:
        """Write a single entry to disk (non-blocking best-effort)."""
        try:
            with self._cache_lock:
                with shelve.open(self._cache_path) as db:
                    db[key] = (result, ts)
        except Exception as e:
            logger.debug("Persistent cache write error (non-fatal): %s", e)

    def _cache_key(self, question: str) -> str:
        """SHA-256 hash of lowercased stripped question."""
        return hashlib.sha256(question.lower().strip().encode()).hexdigest()

    def _get_cached_answer(self, question: str) -> Optional[Dict[str, str]]:
        """Return cached answer if it exists and hasn't expired."""
        key = self._cache_key(question)
        with self._cache_lock:
            entry = self._answer_cache.get(key)
        if entry is None:
            return None
        result, ts = entry
        if (time.time() - ts) > self._CACHE_TTL_SECONDS:
            with self._cache_lock:
                self._answer_cache.pop(key, None)
            logger.debug("Cache expired for: '%s'", question[:40])
            return None
        return dict(result)

    def _set_cached_answer(self, question: str, result: Dict[str, str]) -> None:
        """Store answer in memory and persist to disk. Evicts oldest if full."""
        key = self._cache_key(question)
        ts = time.time()
        with self._cache_lock:
            self._answer_cache[key] = (dict(result), ts)
            # LRU eviction
            if len(self._answer_cache) > self._CACHE_MAX_SIZE:
                oldest = min(self._answer_cache, key=lambda k: self._answer_cache[k][1])
                self._answer_cache.pop(oldest, None)
                logger.debug("Cache evicted oldest entry (size limit reached)")
        # Persist to disk in background thread — never blocks the request
        t = threading.Thread(
            target=self._persist_cache_entry,
            args=(key, dict(result), ts),
            daemon=True,
        )
        t.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval Runners
    # ──────────────────────────────────────────────────────────────────────────

    def _run_faq(self, question: str) -> Optional[Dict[str, str]]:
        try:
            dbg = self.faq_agent.answer_with_debug(question)
            if dbg and int(dbg.get("score", 0) or 0) >= self.faq_min_score:
                ans = str(dbg.get("answer", "")).strip()
                if ans:
                    return {"source": "FAQ Agent", "answer": ans}
        except Exception as e:
            logger.warning("FAQ agent error: %s", e)
        return None

    def _run_retriever(self, name: str, question: str) -> Optional[List[str]]:
        try:
            if name == "VECTOR":
                result = self.vector_agent.search(question)
            elif name == "WEB":
                result = self.webscraper_agent.scrape(question)
            else:
                return None

            if isinstance(result, list):
                chunks = [c for c in result if isinstance(c, str) and c.strip()]
                return chunks if chunks else None
            if isinstance(result, str) and result.strip():
                return [result.strip()]
            return None

        except Exception as e:
            logger.warning("%s retriever error: %s", name, e)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # URL Extraction & Link Appending
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_source_urls(self, chunks: List[str]) -> List[str]:
        urls: List[str] = []
        seen: set = set()
        for chunk in chunks:
            for line in chunk.splitlines():
                line = line.strip()
                if line.lower().startswith("source:"):
                    url = line.split(":", 1)[1].strip()
                    if url.startswith("//"):
                        url = "https:" + url
                    if url not in seen and url.startswith("http"):
                        seen.add(url)
                        urls.append(url)
        return urls

    def _strip_source_lines(self, chunks: List[str]) -> List[str]:
        cleaned: List[str] = []
        for chunk in chunks:
            kept = [
                line for line in chunk.splitlines()
                if not line.strip().lower().startswith("source:")
            ]
            cleaned.append("\n".join(kept).strip())
        return cleaned

    def _is_link_worthy(self, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in self.LINK_KEYWORDS)

    def _get_fallback_links(self, question: str) -> List[tuple]:
        q = question.lower()
        seen_urls: set = set()
        links: List[tuple] = []
        for kw, (label, url) in self.FALLBACK_LINKS.items():
            if kw in q and url not in seen_urls:
                seen_urls.add(url)
                links.append((label, url))
        return links[:2]

    def _append_links(
        self,
        question: str,
        answer: str,
        source_urls: List[str],
        chunk_source: Optional[str],
    ) -> str:
        if not self._is_link_worthy(question):
            return answer
        if "contact the University directly" in answer:
            return answer
        if "not available in the current data" in answer:
            return answer

        link_lines: List[str] = []

        if source_urls and chunk_source == "WEB":
            link_lines.append("\n\U0001f517 *Relevant Links:*")
            for url in source_urls[:2]:
                link_lines.append("   \u2022 " + url)
        else:
            fallbacks = self._get_fallback_links(question)
            if fallbacks:
                link_lines.append("\n\U0001f517 *Useful Links:*")
                for label, url in fallbacks:
                    link_lines.append("   \u2022 " + label + ": " + url)

        if link_lines:
            return answer + "\n" + "\n".join(link_lines)
        return answer

    # ──────────────────────────────────────────────────────────────────────────
    # FAQ Storage
    # ──────────────────────────────────────────────────────────────────────────

    def _store_to_faq(self, question: str, answer: str) -> None:
        if not question or not answer:
            return
        if answer.strip() == self._fallback()["answer"].strip():
            logger.debug("FAQ store skipped: fallback answer")
            return
        if "not available in the current data" in answer.lower():
            logger.debug("FAQ store skipped: not available phrase")
            return
        if len(answer.strip()) < 15:
            logger.debug("FAQ store skipped: answer too short (%d chars)", len(answer.strip()))
            return

        try:
            existing = self.faq_agent.collection.find_one(
                {"question": question.strip()}, {"_id": 1}
            )
            if existing:
                return
            self.faq_agent.collection.insert_one(
                {"question": question.strip(), "answer": answer.strip()}
            )
            self.faq_agent.invalidate_cache()
            logger.info("Stored new FAQ entry: '%s'", question[:60])
        except Exception as e:
            logger.warning("FAQ store error: %s", e)

    # ──────────────────────────────────────────────────────────────────────────
    # Smart Routing
    # ──────────────────────────────────────────────────────────────────────────

    def _decide_order(self, question: str) -> List[str]:
        base = ["FAQ", "VECTOR", "WEB"]
        if not self.enable_smart_routing:
            return base

        q = question.lower()

        time_sensitive = any(k in q for k in [
            "latest", "recent", "updated", "notice", "announcement",
            "deadline", "timetable", "schedule", "merit list",
            "result", "today", "this week", "new",
            "last date", "closing date", "due date", "last day",
            "when is", "when are", "when does", "when will",
            "open now", "currently", "this year",
            str(datetime.now().year), str(datetime.now().year + 1),
            "admission date", "apply by", "submission date",
        ])
        if time_sensitive:
            logger.info("Smart routing: time-sensitive -> WEB first")
            return ["FAQ", "WEB", "VECTOR"]

        web_first = any(k in q for k in [
            "event", "competition", "fest", "code jung", "hackathon",
            "seminar", "workshop", "expo", "tech fest", "gaming",
            "how many", "list of", "all programs", "all courses",
            "programs offering", "programs offered", "kiet offering",
            "bachelor", "bachelors", "bs programs", "ms programs",
            "degree", "courses offered", "what programs",
            "admission", "apply", "eligibility", "requirement",
            "merit", "entry test", "form", "apply online",
        ])
        if web_first:
            logger.info("Smart routing: web-first query -> WEB before VECTOR")
            return ["FAQ", "WEB", "VECTOR"]

        return base

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _timed(self, name: str, timings: dict, fn):
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        return result

    def _fallback(self) -> Dict[str, str]:
        return {
            "source":            "None",
            "confidence":        "none",
            "confidence_label":  "❓ No Answer Found",
            "from_cache":        "false",
            "answer": (
                "I'm sorry, I couldn't find information on that.\n"
                "Please contact the University directly:\n"
                "\u2022 \U0001f4de Phone: 02136628381 / 02136679314\n"
                "\u2022 \U0001f4e7 Email: admissions@kiet.edu.pk"
            ),
        }

    def _log(self, order, timings, total, chosen):
        parts = " | ".join([f"{k}={timings.get(k, 0):.3f}s" for k in order])
        logger.info("\u23f1 chosen=%s total=%.3fs | %s", chosen, total, parts)

    def close(self) -> None:
        try:
            self.faq_agent.close()
        except Exception:
            pass