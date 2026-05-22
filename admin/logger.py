"""
admin/logger.py

Logs every chatbot query to EDU_ANALYTICS MongoDB database.
Called at the end of every /api/ask request in main.py.

Database:    EDU_ANALYTICS   (separate from FAQ_AGENT — zero risk to existing data)
Collection:  query_logs

Document structure:
{
    "question":          str,    # original question from user
    "enriched_question": str,    # after context enrichment (may differ)
    "source":            str,    # "FAQ Agent" / "Web Agent → LLM" / "Cache" etc
    "confidence":        str,    # "high" / "medium" / "low" / "fallback"
    "from_cache":        bool,
    "response_time_ms":  int,
    "timestamp":         datetime (UTC),
    "session_id":        str,    # browser session identifier
}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from config.settings import MONGO_URI

logger = logging.getLogger("eduassist.analytics.logger")

# ── Settings ──────────────────────────────────────────────
# These can be overridden via .env
import os
ANALYTICS_DB         = os.getenv("ANALYTICS_DB_NAME",          "EDU_ANALYTICS")
LOGS_COLLECTION      = os.getenv("ANALYTICS_LOGS_COLLECTION",   "query_logs")
STATS_COLLECTION     = os.getenv("ANALYTICS_STATS_COLLECTION",  "daily_stats")


class QueryLogger:
    """
    Thread-safe query logger.
    Uses a single MongoClient connection shared across all requests.
    Failures are logged but NEVER raise — chatbot must never crash due to logging.
    """

    _client:     Optional[MongoClient] = None
    _db_ready:   bool                  = False

    def __init__(self):
        if not QueryLogger._client:
            try:
                QueryLogger._client  = MongoClient(
                    MONGO_URI, serverSelectionTimeoutMS=3000
                )
                db = QueryLogger._client[ANALYTICS_DB]

                # ✅ Ensure indexes exist for fast dashboard queries
                logs = db[LOGS_COLLECTION]
                logs.create_index([("timestamp", DESCENDING)])
                logs.create_index([("session_id", ASCENDING)])
                logs.create_index([("source",    ASCENDING)])
                logs.create_index([("confidence",ASCENDING)])

                QueryLogger._db_ready = True
                logger.info("✅ QueryLogger connected to %s.%s", ANALYTICS_DB, LOGS_COLLECTION)
            except PyMongoError as e:
                logger.warning("⚠️ QueryLogger failed to connect: %s", e)
                QueryLogger._db_ready = False

        self._client = QueryLogger._client

    # ─────────────────────────────────────────────
    # Public — log a query
    # ─────────────────────────────────────────────

    def log(
        self,
        *,
        question:          str,
        enriched_question: str,
        source:            str,
        confidence:        str,
        from_cache:        bool,
        response_time_ms:  int,
        session_id:        str  = "anonymous",
    ) -> None:
        """
        Insert one query log document.
        Silent on failure — chatbot response is never blocked by this.
        """
        if not QueryLogger._db_ready or not self._client:
            return
        try:
            doc = {
                "question":          question.strip(),
                "enriched_question": enriched_question.strip(),
                "source":            source,
                "confidence":        confidence,
                "from_cache":        from_cache,
                "response_time_ms":  int(response_time_ms),
                "timestamp":         datetime.now(timezone.utc),
                "session_id":        session_id,
            }
            self._client[ANALYTICS_DB][LOGS_COLLECTION].insert_one(doc)
        except Exception as e:
            logger.debug("QueryLogger.log error (non-fatal): %s", e)

    # ─────────────────────────────────────────────
    # Dashboard data helpers (used by Iteration 7)
    # ─────────────────────────────────────────────

    def get_total_questions(self) -> int:
        """Total questions ever asked."""
        if not QueryLogger._db_ready:
            return 0
        try:
            return self._client[ANALYTICS_DB][LOGS_COLLECTION].count_documents({})
        except Exception:
            return 0

    def get_source_distribution(self) -> dict:
        """Returns {source_name: count} for all sources."""
        if not QueryLogger._db_ready:
            return {}
        try:
            pipeline = [
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": DESCENDING}},
            ]
            results = list(
                self._client[ANALYTICS_DB][LOGS_COLLECTION].aggregate(pipeline)
            )
            return {r["_id"]: r["count"] for r in results if r["_id"]}
        except Exception as e:
            logger.debug("get_source_distribution error: %s", e)
            return {}

    def get_confidence_distribution(self) -> dict:
        """Returns {confidence_level: count}."""
        if not QueryLogger._db_ready:
            return {}
        try:
            pipeline = [
                {"$group": {"_id": "$confidence", "count": {"$sum": 1}}},
            ]
            results = list(
                self._client[ANALYTICS_DB][LOGS_COLLECTION].aggregate(pipeline)
            )
            return {r["_id"]: r["count"] for r in results if r["_id"]}
        except Exception as e:
            logger.debug("get_confidence_distribution error: %s", e)
            return {}

    def get_avg_response_time_ms(self) -> float:
        """Average response time across all queries in ms."""
        if not QueryLogger._db_ready:
            return 0.0
        try:
            pipeline = [
                {"$group": {"_id": None, "avg": {"$avg": "$response_time_ms"}}}
            ]
            result = list(
                self._client[ANALYTICS_DB][LOGS_COLLECTION].aggregate(pipeline)
            )
            return round(result[0]["avg"], 1) if result else 0.0
        except Exception as e:
            logger.debug("get_avg_response_time_ms error: %s", e)
            return 0.0

    def get_questions_today(self) -> int:
        """Questions asked since midnight UTC today."""
        if not QueryLogger._db_ready:
            return 0
        try:
            today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return self._client[ANALYTICS_DB][LOGS_COLLECTION].count_documents(
                {"timestamp": {"$gte": today}}
            )
        except Exception:
            return 0

    def get_daily_counts(self, days: int = 14) -> list:
        """
        Returns last N days of daily question counts.
        Format: [{"date": "2026-03-01", "count": 42}, ...]
        """
        if not QueryLogger._db_ready:
            return []
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$timestamp"
                            }
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": ASCENDING}},
                {"$limit": days},
            ]
            results = list(
                self._client[ANALYTICS_DB][LOGS_COLLECTION].aggregate(pipeline)
            )
            return [{"date": r["_id"], "count": r["count"]} for r in results]
        except Exception as e:
            logger.debug("get_daily_counts error: %s", e)
            return []

    def get_top_questions(self, limit: int = 10) -> list:
        """
        Most frequently asked questions.
        Format: [{"question": str, "count": int, "source": str}, ...]
        """
        if not QueryLogger._db_ready:
            return []
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": "$question",
                        "count": {"$sum": 1},
                        "source": {"$first": "$source"},
                        "avg_ms": {"$avg": "$response_time_ms"},
                    }
                },
                {"$sort": {"count": DESCENDING}},
                {"$limit": limit},
            ]
            results = list(
                self._client[ANALYTICS_DB][LOGS_COLLECTION].aggregate(pipeline)
            )
            return [
                {
                    "question": r["_id"],
                    "count":    r["count"],
                    "source":   r.get("source", "Unknown"),
                    "avg_ms":   round(r.get("avg_ms", 0)),
                }
                for r in results
            ]
        except Exception as e:
            logger.debug("get_top_questions error: %s", e)
            return []

    def get_unanswered_questions(self, limit: int = 20) -> list:
        """
        Questions that hit the fallback — useful for adding to FAQ.
        Format: [{"question": str, "count": int}, ...]
        """
        if not QueryLogger._db_ready:
            return []
        try:
            pipeline = [
                {"$match": {"confidence": "fallback"}},
                {
                    "$group": {
                        "_id": "$question",
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"count": DESCENDING}},
                {"$limit": limit},
            ]
            results = list(
                self._client[ANALYTICS_DB][LOGS_COLLECTION].aggregate(pipeline)
            )
            return [{"question": r["_id"], "count": r["count"]} for r in results]
        except Exception as e:
            logger.debug("get_unanswered_questions error: %s", e)
            return []

    def close(self) -> None:
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass


# ── Module-level singleton ─────────────────────────────────
# Import and use this in main.py:
#   from admin.logger import query_logger
#   query_logger.log(question=..., source=..., ...)
query_logger = QueryLogger()