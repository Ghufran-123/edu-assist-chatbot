"""
config/settings.py

All configuration loaded from environment variables via .env file.
Secrets (HuggingFace token, MongoDB URI) NEVER hardcoded here.

Setup:
  1. Copy .env.example to .env
  2. Fill in your real values
  3. .env is gitignored — never committed
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

# ============================================
# Load .env from project root — multiple fallback paths
# ============================================
def _find_and_load_env() -> None:
    """
    Try multiple locations for .env file.
    Handles both structures:
      - project/.env  (settings.py is in config/)
      - project/.env  (settings.py is in project root)
    """
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",  # config/settings.py → ../
        Path(__file__).resolve().parent / ".env",          # settings.py in root
        Path.cwd() / ".env",                               # current working directory
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(dotenv_path=path, override=True)
            print(f"[settings] ✅ Loaded .env from: {path}")
            return
    print("[settings] ⚠️  No .env file found. Checked:")
    for p in candidates:
        print(f"   - {p}")

_find_and_load_env()

logger = logging.getLogger("eduassist.settings")


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ================================
# MongoDB
# ================================
MONGO_URI       = _optional("MONGO_URI",      "mongodb://localhost:27017")
DB_NAME         = _optional("DB_NAME",        "FAQ_AGENT")
FAQ_COLLECTION  = _optional("FAQ_COLLECTION", "FAQs")

# ================================
# Paths
# ================================
DOCUMENT_FOLDER  = _optional("DOCUMENT_FOLDER",  "data/documents")
VECTOR_DB_FOLDER = _optional("VECTOR_DB_FOLDER", "data/vectorstore")
URLS_FILE        = _optional("URLS_FILE",         "config/urls.txt")
BASE_DOMAIN      = _optional("BASE_DOMAIN",       "kiet.edu.pk")

# ================================
# HuggingFace / LLM
# ================================
HUGGINGFACE_TOKEN = _optional("HUGGINGFACE_TOKEN", "")

if not HUGGINGFACE_TOKEN:
    logger.warning(
        "[settings] HUGGINGFACE_TOKEN not set — gated models (e.g. Llama) will fail to download."
    )

# ✅ Default to 3B model — best balance of speed and quality on CPU
# ✅ Optimized for AMD Ryzen 5 7430U (16GB RAM, CPU-only)
# Llama 3.1 1B — best speed/quality balance for this hardware
LLM_MODEL       = _optional("LLM_MODEL",       "meta-llama/Llama-3.2-1B-Instruct")
LLM_MAX_TOKENS  = int(_optional("LLM_MAX_TOKENS",  "256"))
LLM_TEMPERATURE = float(_optional("LLM_TEMPERATURE", "0.0"))  # greedy = fastest on CPU


# ================================
# Startup Validation
# ================================
def validate() -> None:
    """
    Called once at pipeline startup.
    Catches misconfigurations early — fail fast with clear error messages.
    """
    errors = []
    warnings = []

    if not Path(DOCUMENT_FOLDER).exists():
        warnings.append(f"DOCUMENT_FOLDER '{DOCUMENT_FOLDER}' does not exist yet — indexing will be skipped.")

    if not Path(URLS_FILE).exists():
        errors.append(f"URLS_FILE '{URLS_FILE}' not found.")

    if LLM_MAX_TOKENS < 64:
        errors.append(f"LLM_MAX_TOKENS={LLM_MAX_TOKENS} is too low (minimum 64).")

    if not (0.0 <= LLM_TEMPERATURE <= 2.0):
        errors.append(f"LLM_TEMPERATURE={LLM_TEMPERATURE} is out of range (0.0–2.0).")

    if not HUGGINGFACE_TOKEN:
        warnings.append("HUGGINGFACE_TOKEN not set — only public models will work.")

    for w in warnings:
        logger.warning("[settings] %s", w)

    if errors:
        for e in errors:
            logger.error("[settings] %s", e)
        raise EnvironmentError(
            "[settings] Startup validation failed:\n" + "\n".join(errors)
        )

    logger.info("[settings] ✅ Configuration validated — model=%s tokens=%d",
                LLM_MODEL, LLM_MAX_TOKENS)