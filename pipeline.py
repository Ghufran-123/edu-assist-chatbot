"""
pipeline.py

Responsibilities:
- Validate configuration at startup (fail fast)
- Load seed URLs from urls.txt
- Build OrchestratorAgent (thread-safe singleton)
- Trigger VectorDB indexing on document changes
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from agents.orchestrator_agent import OrchestratorAgent
from config import settings
from config.settings import validate as validate_settings

# =====================================
# Logging Setup
# =====================================
logger = logging.getLogger("eduassist.pipeline")

# =====================================
# Startup Validation — fail fast
# =====================================
try:
    validate_settings()
except EnvironmentError as _e:
    logger.critical("Startup validation failed: %s", _e)
    raise

# =====================================
# Flags
# =====================================
BUILD_INDEX_ON_START = os.getenv("BUILD_INDEX_ON_START", "1").lower() in {"1", "true", "yes"}
FORCE_REINDEX        = os.getenv("FORCE_REINDEX", "0").lower() in {"1", "true", "yes"}
DEBUG_MODE           = os.getenv("DEBUG_MODE", "0").lower() in {"1", "true", "yes"}

META_FILE = Path(settings.VECTOR_DB_FOLDER) / ".index_meta.json"

# =====================================
# Thread-safe Singleton
# =====================================
_bot: Optional[OrchestratorAgent] = None
_index_ready: bool = False
_bot_lock = threading.Lock()


# =====================================
# URL Loader
# =====================================
def load_urls(file_path: str, base_domain: Optional[str] = None) -> List[str]:
    path = Path(file_path)
    if not path.exists():
        logger.warning("URLs file not found: %s", file_path)
        return []

    urls: List[str] = []
    seen = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):   # ✅ Skip blank lines and comments
            continue

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            logger.warning("Invalid URL skipped: %s", url)
            continue

        if base_domain and base_domain not in parsed.netloc:
            logger.debug("Non-base-domain URL skipped: %s", url)
            continue

        if url not in seen:
            seen.add(url)
            urls.append(url)

    logger.info("Loaded %d URLs from %s", len(urls), file_path)
    return urls


# =====================================
# Document Change Detection
# =====================================
def _iter_doc_files(folder: str) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return sorted([
        f for f in folder_path.rglob("*")
        if f.is_file() and f.suffix.lower() in {".pdf", ".txt"}
    ])


def _docs_signature(folder: str) -> dict:
    folder_path = Path(folder)
    items = []
    for f in _iter_doc_files(folder):
        try:
            stat = f.stat()
            items.append({
                "path": str(f.relative_to(folder_path)),
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
            })
        except FileNotFoundError:
            continue
    return {"folder": str(folder_path.resolve()), "count": len(items), "items": items}


def _load_meta() -> Optional[dict]:
    try:
        if META_FILE.exists():
            return json.loads(META_FILE.read_text())
    except Exception as e:
        logger.warning("Meta file read error: %s", e)
    return None


def _save_meta(meta: dict) -> None:
    try:
        META_FILE.parent.mkdir(parents=True, exist_ok=True)
        META_FILE.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        logger.warning("Meta file write error: %s", e)


def docs_changed(folder: str) -> bool:
    current = _docs_signature(folder)
    prev = _load_meta()
    if prev is None:
        return True
    return prev.get("items") != current.get("items")


# =====================================
# VectorDB Indexing
# =====================================
def ensure_index(bot: OrchestratorAgent) -> None:
    global _index_ready

    if _index_ready:
        return

    doc_folder = settings.DOCUMENT_FOLDER

    if not Path(doc_folder).exists():
        logger.info("Document folder missing — skipping indexing.")
        _index_ready = True
        return

    if not bot.vector_agent:
        logger.warning("Vector agent not available — skipping indexing.")
        _index_ready = True
        return

    if not (FORCE_REINDEX or docs_changed(doc_folder)):
        logger.info("Documents unchanged — skipping re-index.")
        _index_ready = True
        return

    logger.info("📚 Indexing documents into FAISS...")
    start = time.time()
    bot.vector_agent.add_documents_from_folder(doc_folder)
    elapsed = time.time() - start
    logger.info("✅ VectorDB indexed in %.2fs", elapsed)

    _save_meta(_docs_signature(doc_folder))
    _index_ready = True


# =====================================
# Bot Builder
# =====================================
def _build_bot() -> OrchestratorAgent:
    urls = load_urls(
        settings.URLS_FILE,
        base_domain=settings.BASE_DOMAIN,
    )

    return OrchestratorAgent(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.DB_NAME,
        collection_name=settings.FAQ_COLLECTION,
        vector_path=settings.VECTOR_DB_FOLDER,
        urls=urls,
        base_domain=settings.BASE_DOMAIN,
        consistency_max_retries=1,
        enable_timing_logs=DEBUG_MODE,
        debug=DEBUG_MODE,
    )


# =====================================
# Public Getter — Thread-safe Singleton
# =====================================
def get_bot() -> OrchestratorAgent:
    global _bot

    # ✅ Fast path — no lock needed once initialized
    if _bot is not None:
        return _bot

    # ✅ Slow path — lock prevents race condition on first load
    with _bot_lock:
        if _bot is None:
            logger.info("🚀 Initializing EDU Assist pipeline...")
            _bot = _build_bot()
            logger.info("✅ Pipeline ready.")

            if BUILD_INDEX_ON_START:
                ensure_index(_bot)
            else:
                logger.info("Lazy mode — skipping index build on startup.")

    return _bot