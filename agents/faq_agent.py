from __future__ import annotations

import re
import time
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient
from rapidfuzz import fuzz

logger = logging.getLogger("eduassist.faq_agent")


@dataclass(frozen=True)
class FAQItem:
    question: str
    answer: str
    norm_question: str
    tokens: Tuple[str, ...]


# ✅ Expanded smalltalk set — covers more common student greetings
SMALLTALK_EXACT = {
    "hi", "hello", "hey", "thanks", "thank you", "thankyou",
    "ok", "okay", "k", "bye", "goodbye", "good morning",
    "good afternoon", "good evening", "good night",
    "sup", "yo", "wassup", "whatsup", "what's up",
    "how are you", "how r u", "who are you", "what are you",
    "nice", "cool", "great", "awesome", "perfect",
    "hola", "salam", "assalam", "assalamualaikum",
}

SMALLTALK_RESPONSES = [
    "👋 Hello! I'm EDU Assist, your KIET university chatbot. Ask me anything about admissions, fees, programs, or events!",
    "👋 Hi there! I'm EDU Assist. How can I help you with KIET university today?",
    "👋 Hello! Happy to help. Ask me about KIET programs, fees, admissions, or anything else!",
]


# ✅ Question/filler words stripped before topic matching
# Allows 'tell me about X' to match 'what is X' by comparing only the core topic
QUESTION_WORDS = {
    'tell', 'me', 'about', 'what', 'is', 'are', 'was', 'were',
    'who', 'when', 'where', 'why', 'how', 'which', 'the', 'a', 'an',
    'give', 'show', 'explain', 'describe', 'list', 'find', 'get',
    'can', 'you', 'please', 'do', 'does', 'did', 'i', 'want', 'know',
    'information', 'info', 'details', 'detail', 'regarding', 'related',
    'some', 'any', 'all', 'more', 'please', 'kindly', 'kiet',
}


class FAQAgent:
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        *,
        threshold: int = 70,
        top_k: int = 5,
        cache_ttl_seconds: int = 300,
    ):
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self.collection = self.client[db_name][collection_name]

        self.threshold = int(threshold)
        self.top_k = int(top_k)
        self.cache_ttl_seconds = int(cache_ttl_seconds)

        self._cache: List[FAQItem] = []
        self._cache_loaded_at: float = 0.0
        self._cache_initialized: bool = False   # ✅ Sentinel flag — distinct from empty cache

        # ✅ Thread-safety lock — prevents double-load in Streamlit's threaded env
        self._cache_lock = threading.Lock()

        # ✅ Rotating smalltalk response index
        self._smalltalk_idx: int = 0

    # ------------------ Public ------------------

    def answer(self, user_question: str) -> Optional[str]:
        dbg = self.answer_with_debug(user_question)
        return dbg["answer"] if dbg else None

    def answer_with_debug(self, user_question: str) -> Optional[Dict[str, Any]]:
        user_question = (user_question or "").strip()
        if not user_question:
            return None

        # ✅ Smalltalk → return friendly greeting immediately
        # Avoids wasteful fallthrough through all 6 pipeline agents
        if self._looks_like_smalltalk(user_question):
            response = SMALLTALK_RESPONSES[self._smalltalk_idx % len(SMALLTALK_RESPONSES)]
            self._smalltalk_idx += 1
            return {
                "answer": response,
                "score": 100,
                "matched_question": "__smalltalk__",
            }

        faqs = self._get_cached_faqs()
        if not faqs:
            return None

        norm_q = self._normalize(user_question)
        q_tokens = set(self._tokenize(norm_q))
        core_q = self._core_topic(norm_q)   # ✅ Strip question words for topic matching

        scored: List[Tuple[int, FAQItem]] = []
        for item in faqs:
            # ✅ Two-pass scoring:
            # Pass 1: full token_set_ratio (catches exact/near-exact matches)
            full_score = int(fuzz.token_set_ratio(norm_q, item.norm_question))

            # Pass 2: core topic ratio (catches paraphrase matches)
            # "tell me about code jung" → core: "code jung"
            # "what is code jung event" → core: "code jung event"
            # overlap/shorter = 2/2 = 100 ✅
            # Uses SHORTER set in denominator so extra words don't penalize score
            core_faq = self._core_topic(item.norm_question)
            if core_q and core_faq:
                core_tokens_q = set(core_q.split())
                core_tokens_f = set(core_faq.split())
                intersection = core_tokens_q & core_tokens_f
                shorter = min(len(core_tokens_q), len(core_tokens_f))
                core_score = int(len(intersection) / shorter * 100) if shorter else 0
            else:
                core_score = 0

            # ✅ Use the better of the two scores
            best = max(full_score, core_score)
            scored.append((best, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return None

        candidates = scored[: max(1, self.top_k)]
        best_score, best_item = self._tie_break(candidates, q_tokens)

        if best_score >= self.threshold:
            # ✅ Topic mismatch guard — prevents false positives
            # Even high-scoring matches are rejected if topic words don't overlap
            if not self._topic_words_overlap(norm_q, best_item.norm_question):
                logger.info(
                    "FAQ rejected (topic mismatch): score=%d query='%s' matched='%s'",
                    best_score, norm_q[:50], best_item.norm_question[:50],
                )
                return None

            logger.info(
                "FAQ hit: score=%d matched='%s'",
                best_score, best_item.question[:60],
            )
            return {
                "answer": best_item.answer,
                "score": best_score,
                "matched_question": best_item.question,
            }

        logger.debug("FAQ miss: best_score=%d < threshold=%d", best_score, self.threshold)
        return None

    def invalidate_cache(self) -> None:
        """Reset the in-memory FAQ cache so it reloads from MongoDB on next query."""
        with self._cache_lock:
            self._cache = []
            self._cache_loaded_at = 0.0
            self._cache_initialized = False

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    # ------------------ Cache / Load ------------------

    def _get_cached_faqs(self) -> List[FAQItem]:
        now = time.time()

        # ✅ Fast path — cache loaded and still fresh
        if self._cache_initialized and (now - self._cache_loaded_at) < self.cache_ttl_seconds:
            return self._cache

        # ✅ Slow path — acquire lock before reloading
        with self._cache_lock:
            # Double-check inside lock using sentinel flag (not list truthiness)
            # This correctly handles the case where MongoDB returns 0 docs
            if self._cache_initialized and (now - self._cache_loaded_at) < self.cache_ttl_seconds:
                return self._cache

            self._cache = self._load_from_mongo()
            self._cache_loaded_at = time.time()
            self._cache_initialized = True   # ✅ Mark as loaded regardless of result
            logger.info("FAQ cache refreshed: %d entries loaded.", len(self._cache))

        return self._cache

    def _load_from_mongo(self) -> List[FAQItem]:
        items: List[FAQItem] = []
        try:
            docs = list(self.collection.find({}, {"_id": 0}))
        except Exception as e:
            logger.error("MongoDB load error: %s", e)
            return items

        raw: List[Dict[str, Any]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            if isinstance(doc.get("faqs"), list):
                raw.extend([x for x in doc["faqs"] if isinstance(x, dict)])
            elif "question" in doc and "answer" in doc:
                raw.append(doc)

        for f in raw:
            q = f.get("question")
            a = f.get("answer")
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            q = q.strip()
            a = a.strip()
            if not q or not a:
                continue
            nq = self._normalize(q)
            items.append(
                FAQItem(
                    question=q,
                    answer=a,
                    norm_question=nq,
                    tokens=self._tokenize(nq),
                )
            )

        return items

    # ------------------ Ranking ------------------

    def _tie_break(
        self, candidates: List[Tuple[int, FAQItem]], q_tokens: set
    ) -> Tuple[int, FAQItem]:
        best_score, _ = candidates[0]
        close = [c for c in candidates if (best_score - c[0]) <= 3]
        if len(close) <= 1:
            return candidates[0]

        def overlap(item: FAQItem) -> int:
            return len(q_tokens.intersection(item.tokens))

        close.sort(
            key=lambda x: (x[0], overlap(x[1]), len(x[1].tokens), len(x[1].question)),
            reverse=True,
        )
        return close[0][0], close[0][1]

    # ------------------ Text utils ------------------

    _punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)
    _space_re = re.compile(r"\s+", flags=re.UNICODE)

    def _normalize(self, text: str) -> str:
        text = (text or "").lower().strip()
        text = self._punct_re.sub(" ", text)
        text = self._space_re.sub(" ", text).strip()
        return text

    def _tokenize(self, norm_text: str) -> Tuple[str, ...]:
        return tuple(t for t in norm_text.split() if t)

    def _core_topic(self, norm_text: str) -> str:
        """
        Strip question/filler words leaving only the core topic.
        'tell me about code jung' → 'code jung'
        'what is the last date of admission' → 'last date admission'
        """
        tokens = [t for t in norm_text.split() if t not in QUESTION_WORDS]
        return ' '.join(tokens)

    def _topic_words_overlap(self, query_norm: str, faq_norm: str) -> bool:
        """
        ✅ Topic overlap guard — prevents false FAQ matches.
        Rejects a FAQ match if the main topic words of the query
        don't appear in the matched FAQ question at all.

        Example:
            query = "tell me about code jung"
            faq   = "tell me about dr abdullah qualifications"
            → topic words of query: {"code", "jung"}
            → none found in faq → REJECT

        Only applies when query has meaningful topic words (nouns/names).
        Skips common words like "tell", "about", "what", "is", "the".
        """
        # Words to ignore (stop words)
        stop_words = {
            "tell", "me", "about", "what", "is", "are", "the", "a", "an",
            "how", "many", "does", "do", "kiet", "university", "please",
            "can", "you", "i", "want", "know", "give", "provide", "show",
            "get", "find", "explain", "describe", "list", "all", "any",
            "have", "has", "had", "will", "would", "could", "should",
            "its", "their", "there", "which", "who", "when", "where",
        }

        query_tokens = set(self._tokenize(query_norm)) - stop_words
        faq_tokens = set(self._tokenize(faq_norm)) - stop_words

        # If query has no meaningful topic words, skip the check
        if not query_tokens:
            return True

        # At least 1 topic word must overlap
        overlap = query_tokens & faq_tokens
        has_overlap = len(overlap) >= 1

        if not has_overlap:
            logger.debug(
                "Topic mismatch: query_topics=%s faq_topics=%s",
                query_tokens, faq_tokens
            )

        return has_overlap

    def _looks_like_smalltalk(self, text: str) -> bool:
        norm = self._normalize(text)

        # Exact match
        if norm in SMALLTALK_EXACT:
            return True

        # Very short input (1-2 chars)
        if len(norm) <= 2:
            return True

        # ✅ Fuzzy match against smalltalk set for typos like "helo", "thankss"
        for phrase in SMALLTALK_EXACT:
            if len(phrase) > 3 and fuzz.ratio(norm, phrase) >= 88:
                return True

        return False