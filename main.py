"""
main.py — EDU Assist FastAPI Server (Iteration 7)

Project structure:
  main.py, pipeline.py         → project root
  agents/                      → all agent modules
  config/settings.py           → configuration
  config/urls.txt              → crawl URL list
  static/                      → HTML pages
  utils/ui_helpers.py          → input helpers
  admin/logger.py              → query analytics

Endpoints:
  GET    /                     → static/index.html
  GET    /chat                 → static/chat.html
  GET    /admin                → static/admin.html  (Admin Dashboard)
  POST   /api/ask              → SSE stream
  GET    /api/health           → bot readiness

  Admin (require X-Admin-Key header):
  GET    /api/admin/stats      → dashboard KPIs
  GET    /api/admin/logs       → query logs
  GET    /api/admin/faqs       → list FAQs
  POST   /api/admin/faqs       → add FAQ
  DELETE /api/admin/faqs/{id}  → delete FAQ
  GET    /api/admin/urls       → list URLs
  POST   /api/admin/urls       → add URL
  DELETE /api/admin/urls       → remove URL
  GET    /api/admin/vectordb   → indexed files
  POST   /api/admin/reindex    → trigger reindex

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import get_bot
from utils.ui_helpers import (
    enrich_with_context,
    get_followup_suggestions,
    get_smart_fallback,
    validate_input,
)
from admin.logger import query_logger

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eduassist.main")

# ── Bot singleton ────────────────────────────────────────────
_bot = None

def _get_bot():
    global _bot
    if _bot is None:
        logger.info("🚀 Loading EDU Assist pipeline...")
        _bot = get_bot()
        logger.info("✅ Pipeline ready.")
    return _bot


# ── Rate limiter (simple in-memory) ──────────────────────────
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
_rate_store: dict[str, list[float]] = defaultdict(list)

def _is_rate_limited(session_id: str) -> bool:
    """Returns True if this session has exceeded RATE_LIMIT_PER_MINUTE requests."""
    now = time.time()
    window = [t for t in _rate_store[session_id] if now - t < 60]
    _rate_store[session_id] = window
    if len(window) >= _RATE_LIMIT:
        return True
    _rate_store[session_id].append(now)
    return False


# ── Admin auth ───────────────────────────────────────────────
_ADMIN_KEY      = os.getenv("ADMIN_API_KEY", "")
_ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1234")

# ── Session store (in-memory, token → expiry timestamp) ──────
# Sessions expire after 8 hours. A restart clears all sessions.
_SESSION_TTL    = int(os.getenv("ADMIN_SESSION_TTL", str(8 * 3600)))  # seconds
_sessions: dict[str, float] = {}   # token → expiry


def _create_session() -> str:
    """Generate a new session token and store it."""
    token = secrets.token_hex(32)
    _sessions[token] = time.time() + _SESSION_TTL
    return token


def _is_valid_session(token: str | None) -> bool:
    """Return True if the token exists and has not expired."""
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None:
        return False
    if time.time() > expiry:
        _sessions.pop(token, None)
        return False
    return True


# ── Lifespan (replaces deprecated @app.on_event) ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 EDU Assist starting up...")
    yield

    logger.info("🛑 EDU Assist shutting down...")
    try:
        bot = _get_bot()
        bot.close()
    except Exception:
        pass

    try:
        query_logger.close()
    except Exception:
        pass

    logger.info("🛑 Shutdown complete.")

# ── App ─────────────────────────────────────────────────────
app = FastAPI(
    title="EDU Assist",
    description="KIET University AI Chatbot — Iteration 6",
    version="6.0.0",
    lifespan=lifespan,
)

# ── CORS — configurable via .env ─────────────────────────────
# Local dev:  CORS_ORIGINS=*
# Production: CORS_ORIGINS=https://kiet.edu.pk,https://chat.kiet.edu.pk
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
CORS_ORIGINS = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)
logger.info("CORS origins: %s", CORS_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

# ── Static files ────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
ASSETS_DIR = STATIC_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


# ════════════════════════════════════════════════════════════
# Page routes — serve HTML files
# ════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/chat", include_in_schema=False)
async def chat_page():
    return FileResponse(STATIC_DIR / "chat.html")

@app.get("/admin/login", include_in_schema=False)
async def admin_login_page():
    """Serve the admin login page."""
    return FileResponse(STATIC_DIR / "admin_login.html")


@app.get("/admin", include_in_schema=False)
async def admin_page(admin_session: str | None = Cookie(default=None)):
    """
    Serve the admin dashboard.
    Redirects to /admin/login if the session cookie is missing or expired.
    """
    if not _is_valid_session(admin_session):
        return RedirectResponse(url="/admin/login", status_code=302)
    return FileResponse(STATIC_DIR / "admin.html")


# ════════════════════════════════════════════════════════════
# Admin — Login / Logout
# ════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/admin/login")
async def admin_login(body: LoginRequest):
    """
    Validate admin credentials and issue a session cookie.
    Credentials are set via ADMIN_USERNAME and ADMIN_PASSWORD in .env.
    Default: admin / admin1234  — change in production!
    """
    if body.username != _ADMIN_USERNAME or body.password != _ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = _create_session()
    response = JSONResponse({"success": True, "token": token})
    response.set_cookie(
        key="admin_session",
        value=token,
        httponly=True,          # not accessible via JS — XSS safe
        samesite="lax",         # CSRF protection
        max_age=_SESSION_TTL,
        path="/",
    )
    logger.info("Admin login successful — session issued.")
    return response


@app.post("/api/admin/logout")
async def admin_logout(admin_session: str | None = Cookie(default=None)):
    """Invalidate the current session and clear the cookie."""
    if admin_session:
        _sessions.pop(admin_session, None)
    response = JSONResponse({"success": True})
    response.delete_cookie(key="admin_session", path="/")
    logger.info("Admin logged out.")
    return response


@app.get("/api/admin/session")
async def admin_session_check(admin_session: str | None = Cookie(default=None)):
    """
    Called by admin.html on load to verify session is still valid.
    Returns 200 if valid, 401 if not.
    """
    if not _is_valid_session(admin_session):
        raise HTTPException(status_code=401, detail="Session expired or not found.")
    return JSONResponse({"valid": True})


@app.get("/api/admin/stats")
async def admin_stats(x_admin_key: str = Header(default="")):
    """Protected admin analytics endpoint."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized — invalid ADMIN_API_KEY")
    return JSONResponse({
        "total_questions":   query_logger.get_total_questions(),
        "questions_today":   query_logger.get_questions_today(),
        "avg_response_ms":   query_logger.get_avg_response_time_ms(),
        "source_distribution": query_logger.get_source_distribution(),
        "confidence_distribution": query_logger.get_confidence_distribution(),
        "top_questions":     query_logger.get_top_questions(10),
        "unanswered":        query_logger.get_unanswered_questions(10),
        "daily_counts":      query_logger.get_daily_counts(14),
    })


# ════════════════════════════════════════════════════════════
# Admin — Query Logs
# ════════════════════════════════════════════════════════════

@app.get("/api/admin/logs")
async def admin_logs(
    limit: int = 100,
    source: str = "",
    x_admin_key: str = Header(default=""),
):
    """Recent query logs from MongoDB, newest first. Optional ?source= filter."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not query_logger._db_ready or not query_logger._client:
        return JSONResponse([])

    try:
        col = query_logger._client["EDU_ANALYTICS"]["query_logs"]
        filt = {}
        if source:
            filt["source"] = {"$regex": source, "$options": "i"}
        docs = list(
            col.find(filt, {"_id": 0})
               .sort("timestamp", -1)
               .limit(max(1, min(limit, 500)))
        )
        for d in docs:
            if "timestamp" in d:
                d["timestamp"] = d["timestamp"].strftime("%H:%M:%S")
        return JSONResponse(docs)
    except Exception as e:
        logger.warning("admin_logs error: %s", e)
        return JSONResponse([])


# ════════════════════════════════════════════════════════════
# Admin — FAQ Manager
# ════════════════════════════════════════════════════════════

class FAQCreateRequest(BaseModel):
    question: str
    answer:   str
    category: str = "General"


@app.get("/api/admin/faqs")
async def admin_faqs_list(x_admin_key: str = Header(default="")):
    """Return all FAQ entries from MongoDB."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    bot = _get_bot()
    try:
        docs = list(bot.faq_agent.collection.find(
            {}, {"_id": 1, "question": 1, "answer": 1, "category": 1}
        ))
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return JSONResponse(docs)
    except Exception as e:
        logger.warning("admin_faqs_list error: %s", e)
        return JSONResponse([])


@app.post("/api/admin/faqs")
async def admin_faqs_create(body: FAQCreateRequest, x_admin_key: str = Header(default="")):
    """Add a new FAQ entry to MongoDB and invalidate the in-memory cache."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    q = (body.question or "").strip()
    a = (body.answer   or "").strip()
    if not q or not a:
        raise HTTPException(status_code=422, detail="question and answer are required")

    bot = _get_bot()
    try:
        existing = bot.faq_agent.collection.find_one({"question": q}, {"_id": 1})
        if existing:
            raise HTTPException(status_code=409, detail="FAQ with this question already exists")
        result = bot.faq_agent.collection.insert_one({
            "question": q,
            "answer":   a,
            "category": body.category or "General",
        })
        bot.faq_agent.invalidate_cache()
        return JSONResponse({"id": str(result.inserted_id), "question": q, "answer": a})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_faqs_create error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/faqs/{faq_id}")
async def admin_faqs_delete(faq_id: str, x_admin_key: str = Header(default="")):
    """Delete a FAQ entry by MongoDB _id."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    bot = _get_bot()
    try:
        from bson import ObjectId   # deferred — avoids crash if pymongo<4
        result = bot.faq_agent.collection.delete_one({"_id": ObjectId(faq_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="FAQ entry not found")
        bot.faq_agent.invalidate_cache()
        return JSONResponse({"deleted": faq_id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_faqs_delete error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════
# Admin — URL Manager
# ════════════════════════════════════════════════════════════

# Resolve urls.txt path from config (matches config/urls.txt in project structure)
from config import settings as _cfg

_URLS_FILE = Path(__file__).parent / _cfg.URLS_FILE
if not _URLS_FILE.exists():
    _URLS_FILE = Path(_cfg.URLS_FILE)   # fallback: relative to cwd


def _read_urls() -> list:
    if not _URLS_FILE.exists():
        return []
    return [
        ln.strip() for ln in _URLS_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#") and ln.strip().startswith("http")
    ]


def _write_urls(urls: list) -> None:
    _URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    out = []
    for u in urls:
        u = u.strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    _URLS_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")


class URLRequest(BaseModel):
    url: str


@app.get("/api/admin/urls")
async def admin_urls_list(x_admin_key: str = Header(default="")):
    """Return all URLs from urls.txt with domain info."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    from urllib.parse import urlparse
    return JSONResponse([
        {"url": u, "domain": urlparse(u).netloc, "status": "ok"}
        for u in _read_urls()
    ])


@app.post("/api/admin/urls")
async def admin_urls_add(body: URLRequest, x_admin_key: str = Header(default="")):
    """Append a URL to urls.txt."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    url = (body.url or "").strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=422, detail="URL must start with http/https")
    urls = _read_urls()
    if url in urls:
        raise HTTPException(status_code=409, detail="URL already exists")
    urls.append(url)
    _write_urls(urls)
    return JSONResponse({"added": url, "total": len(urls)})


@app.delete("/api/admin/urls")
async def admin_urls_delete(body: URLRequest, x_admin_key: str = Header(default="")):
    """Remove a URL from urls.txt."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    url = (body.url or "").strip()
    urls = _read_urls()
    if url not in urls:
        raise HTTPException(status_code=404, detail="URL not found")
    _write_urls([u for u in urls if u != url])
    return JSONResponse({"deleted": url, "total": len(urls) - 1})


# ════════════════════════════════════════════════════════════
# Admin — VectorDB file list
# ════════════════════════════════════════════════════════════

@app.get("/api/admin/vectordb")
async def admin_vectordb(x_admin_key: str = Header(default="")):
    """List all document files and their indexing status."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    import json as _json
    from config import settings as _s

    doc_folder    = Path(_s.DOCUMENT_FOLDER)
    vector_folder = Path(_s.VECTOR_DB_FOLDER)
    state_file    = vector_folder / "_index_state.json"

    indexed_state: dict = {}
    try:
        if state_file.exists():
            indexed_state = _json.loads(state_file.read_text())
    except Exception:
        pass

    files = []
    if doc_folder.exists():
        for f in sorted(doc_folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".docx", ".html", ".md"}:
                try:
                    st  = f.stat()
                    sig = f"{st.st_mtime_ns}-{st.st_size}"
                    files.append({
                        "name":   f.name,
                        "path":   str(f.relative_to(doc_folder)),
                        "type":   f.suffix.lstrip(".").lower(),
                        "size":   f"{st.st_size / 1024 / 1024:.1f} MB",
                        "date":   time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
                        "status": "indexed" if indexed_state.get(str(f)) == sig else "pending",
                    })
                except Exception:
                    continue

    return JSONResponse({
        "files":   files,
        "total":   len(files),
        "indexed": sum(1 for f in files if f["status"] == "indexed"),
        "pending": sum(1 for f in files if f["status"] == "pending"),
    })


# ════════════════════════════════════════════════════════════
# Admin — Trigger Reindex
# ════════════════════════════════════════════════════════════

class ReindexRequest(BaseModel):
    type: str = "incremental"   # "incremental" | "full"


@app.post("/api/admin/reindex")
async def admin_reindex(body: ReindexRequest, x_admin_key: str = Header(default="")):
    """Trigger a background reindex (incremental or full)."""
    if _ADMIN_KEY and x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    import threading
    from config import settings as _s
    from pipeline import ensure_index

    bot           = _get_bot()
    vector_folder = _s.VECTOR_DB_FOLDER

    def _run():
        try:
            if body.type == "full":
                state_file = Path(vector_folder) / "_index_state.json"
                if state_file.exists():
                    state_file.unlink()
                if bot.vector_agent:
                    bot.vector_agent.vectorstore = None
            ensure_index(bot)
            logger.info("Admin reindex (%s) complete.", body.type)
        except Exception as e:
            logger.error("Admin reindex error: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "type": body.type})


# ════════════════════════════════════════════════════════════
# Health check
# ════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """
    Returns bot readiness status.
    Called by chat.html on page load to show/hide the loading overlay.
    """
    try:
        bot = _get_bot()
        return JSONResponse({
            "status": "ready",
            "faq_agent":    bot.faq_agent    is not None,
            "vector_agent": bot.vector_agent is not None,
            "llm_agent":    bot.llm_agent    is not None,
        })
    except Exception as e:
        return JSONResponse({"status": "loading", "error": str(e)}, status_code=503)


# ════════════════════════════════════════════════════════════
# /api/ask — Main SSE streaming endpoint
# ════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question:   str
    history:    list  = []   # [{"role": "user"|"assistant", "content": str}]
    session_id: str   = ""


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _determine_confidence(source: str, consistency_passed: str) -> str:
    """Map pipeline output to confidence level string."""
    if "FAQ" in source:
        return "high"
    if consistency_passed == "True":
        return "medium"
    return "low"


async def _stream_answer(request: AskRequest) -> AsyncIterator[str]:
    """
    Core SSE generator.

    SSE event sequence:
      1. pipeline_step  — FAQ check
      2. pipeline_step  — Retrieval (VectorDB or Web)
      3. pipeline_step  — LLM generating  (only on LLM path)
      4. token          — one per LLM token  (only on LLM path)
      5. pipeline_step  — Consistency check  (only on LLM path)
      6. done           — final metadata (source, confidence, links, suggestions)

    For FAQ / Cache hits: steps 1 + done only (instant response).
    """
    bot        = _get_bot()
    question   = (request.question or "").strip()
    session_id = request.session_id or str(uuid.uuid4())
    t_start    = time.perf_counter()

    # ── Rate limiting ─────────────────────────────────────
    if _is_rate_limited(session_id):
        yield _sse("error", {"message": "Too many requests. Please wait a moment before asking again."})
        return

    # ── Input validation ──────────────────────────────────
    recent_qs = [
        m["content"] for m in request.history if m.get("role") == "user"
    ]
    error = validate_input(question, recent_qs)
    if error:
        yield _sse("error", {"message": error})
        return

    # ── Smart fallback check ──────────────────────────────
    # Catches: hostel, timetable, results, complaints, jobs, lost & found
    # Saves 15-25s pipeline run for these known categories
    fallback = get_smart_fallback(question)
    if fallback:
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        yield _sse("done", {
            "answer":      fallback["message"],
            "source":      "Smart Fallback",
            "confidence":  "medium",
            "from_cache":  False,
            "links":       [{"label": fallback["link_label"], "url": fallback["link_url"]}],
            "suggestions": [],
        })
        query_logger.log(
            question=question, enriched_question=question,
            source="Smart Fallback", confidence="medium",
            from_cache=False, response_time_ms=elapsed_ms,
            session_id=session_id,
        )
        return

    # ── Context enrichment ────────────────────────────────
    enriched = enrich_with_context(question, request.history)
    if enriched != question:
        logger.info("Context enriched: '%s' → '%s'", question[:40], enriched[:60])

    # ── Step 1: FAQ ───────────────────────────────────────
    yield _sse("pipeline_step", {
        "step": 1, "label": "Checking FAQ database", "status": "active"
    })

    faq_result = None
    try:
        dbg = bot.faq_agent.answer_with_debug(enriched)
        if dbg and int(dbg.get("score", 0) or 0) >= bot.faq_min_score:
            ans = str(dbg.get("answer", "")).strip()
            if ans:
                faq_result = {
                    "answer": ans,
                    "source": "FAQ Agent",
                    "consistency_passed": "True",
                }
    except Exception as e:
        logger.warning("FAQ check error: %s", e)

    if faq_result:
        yield _sse("pipeline_step", {
            "step": 1, "label": "FAQ match found ✓", "status": "done", "time": 0.1
        })
        answer   = bot._append_links(enriched, faq_result["answer"], [], "FAQ")
        suggs    = get_followup_suggestions(question, answer)
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        yield _sse("done", {
            "answer":      answer,
            "source":      "FAQ Agent",
            "confidence":  "high",
            "from_cache":  False,
            "links":       [],
            "suggestions": suggs,
        })
        query_logger.log(
            question=question, enriched_question=enriched,
            source="FAQ Agent", confidence="high",
            from_cache=False, response_time_ms=elapsed_ms,
            session_id=session_id,
        )
        return

    yield _sse("pipeline_step", {
        "step": 1, "label": "FAQ — no match", "status": "done"
    })

    # ── Step 2: Retrieval ─────────────────────────────────
    order = bot._decide_order(enriched)
    retrieval_label = "Searching KIET website" if order[1] == "WEB" else "Searching knowledge base"

    yield _sse("pipeline_step", {
        "step": 2, "label": retrieval_label, "status": "active"
    })

    chunk = None
    chunk_source = None
    for name in order:
        if name == "FAQ":
            continue
        result = bot._run_retriever(name, enriched)
        if result:
            chunk = result
            chunk_source = name
            break

    if not chunk:
        yield _sse("pipeline_step", {
            "step": 2, "label": "No relevant content found", "status": "done"
        })
        fallback_answer = bot._fallback()["answer"]
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        yield _sse("done", {
            "answer":      fallback_answer,
            "source":      "Fallback",
            "confidence":  "fallback",
            "from_cache":  False,
            "links":       [],
            "suggestions": [],
        })
        query_logger.log(
            question=question, enriched_question=enriched,
            source="Fallback", confidence="fallback",
            from_cache=False, response_time_ms=elapsed_ms,
            session_id=session_id,
        )
        return

    yield _sse("pipeline_step", {
        "step": 2, "label": f"{retrieval_label} ✓", "status": "done"
    })

    # ── Step 3: LLM Generation — real token streaming ────
    yield _sse("pipeline_step", {
        "step": 3, "label": "Generating answer", "status": "active"
    })

    source_urls = bot._extract_source_urls(chunk)
    clean_chunk = bot._strip_source_lines(chunk)
    tokens_so_far: list[str] = []

    try:
        loop = asyncio.get_event_loop()
        gen = bot.llm_agent.generate_stream(question=enriched, context=clean_chunk)
        # Run blocking LLM iterator in a thread so the event loop isn't blocked
        while True:
            token_text = await loop.run_in_executor(None, next, gen, None)
            if token_text is None:
                break
            if token_text:
                tokens_so_far.append(token_text)
                yield _sse("token", {"token": token_text})
    except StopIteration:
        pass
    except Exception as e:
        logger.error("LLM streaming error: %s", e)

    full_answer = "".join(tokens_so_far).strip()

    if not full_answer:
        full_answer = bot._fallback()["answer"]
        elapsed_ms  = int((time.perf_counter() - t_start) * 1000)
        yield _sse("done", {
            "answer":      full_answer,
            "source":      "Fallback",
            "confidence":  "fallback",
            "from_cache":  False,
            "links":       [],
            "suggestions": [],
        })
        return

    yield _sse("pipeline_step", {
        "step": 3, "label": "Answer generated ✓", "status": "done"
    })

    # ── Step 4: Consistency check ─────────────────────────
    yield _sse("pipeline_step", {
        "step": 4, "label": "Consistency check", "status": "active"
    })

    chunk_text = "\n\n".join(clean_chunk) if isinstance(clean_chunk, list) else clean_chunk
    final_answer, passed, _ = bot.consistency_agent.check_and_fix(
        question=enriched, answer=full_answer, chunk=chunk_text,
    )

    yield _sse("pipeline_step", {
        "step": 4,
        "label": "Consistency check ✓" if passed else "Consistency — regenerated",
        "status": "done",
    })

    # ── Step 5: Summarize + links ─────────────────────────
    summarized = bot.summarizing_agent.summarize(final_answer)
    summarized = bot._append_links(enriched, summarized, source_urls, chunk_source)
    bot._store_to_faq(enriched, summarized)

    confidence = _determine_confidence(
        f"{chunk_source} Agent", str(passed)
    )
    suggs      = get_followup_suggestions(question, summarized)
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    # ── Build links list for frontend ────────────────────
    links = []
    for url in source_urls[:2]:
        links.append({"label": url, "url": url})
    if not links:
        for label, url in bot._get_fallback_links(enriched):
            links.append({"label": label, "url": url})

    yield _sse("done", {
        "answer":      summarized,
        "source":      f"{chunk_source} → LLM",
        "confidence":  confidence,
        "from_cache":  False,
        "links":       links,
        "suggestions": suggs,
    })

    query_logger.log(
        question=question, enriched_question=enriched,
        source=f"{chunk_source} → LLM", confidence=confidence,
        from_cache=False, response_time_ms=elapsed_ms,
        session_id=session_id,
    )


@app.post("/api/ask")
async def ask(request: AskRequest):
    """
    Main chat endpoint. Returns a Server-Sent Events stream.

    The browser connects with EventSource / fetch + ReadableStream,
    receives pipeline_step, token, and done events in real time.
    """
    return StreamingResponse(
        _stream_answer(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering
        },
    )