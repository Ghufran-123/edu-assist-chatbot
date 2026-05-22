from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from rapidfuzz import fuzz

logger = logging.getLogger("eduassist.consistency_check")


# =========================
# Prompts — Plain text (no chat tags)
# ✅ Consistent with llm_agent.py prompt style
# =========================

SELF_VERIFY_PROMPT = """You are a strict fact-checker for a university chatbot.

You are given a source chunk (the ONLY allowed source of truth) and a generated answer.

Your job:
- Check if the answer contains ANY information NOT found in the source chunk.
- Check if the answer contradicts the source chunk.

Respond ONLY with one word:
  PASS   — answer is fully supported by the source chunk
  FAIL   — answer contains hallucinated or contradictory information

Source Chunk:
{chunk}

Generated Answer:
{answer}

Verdict (PASS or FAIL):"""


REGENERATE_PROMPT = """You are an AI assistant for a university chatbot.

The previous answer failed a hallucination check.
Generate a NEW answer using ONLY the provided context below. Do NOT add anything outside the context.

Rules:
- Read the context carefully and extract relevant information.
- Answer directly and clearly using ONLY what is in the context.
- If the context contains the answer, always provide it.
- Only say "The information is not available in the current data." if the context truly has NO relevant information.
- Be concise and factual.

Context:
{context}

Student Question:
{question}

Answer:"""


class ConsistencyCheckAgent:
    """
    Checks generated answers for hallucinations using:
    1. Keyword overlap check (fast, no LLM call)
    2. LLM self-verify prompt (optional deep check)

    Retries up to max_retries times if hallucination detected.

    ✅ Improvements:
    - Prompts aligned with plain-text style (no chat tags)
    - NOT_AVAILABLE answer auto-passes (no point retrying it)
    - Overlap threshold lowered to 20 — less false positives
    - Better logging with overlap scores visible
    """

    NOT_AVAILABLE_PHRASE = "the information is not available in the current data"

    def __init__(
        self,
        llm_agent,
        *,
        max_retries: int = 1,
        keyword_overlap_threshold: float = 0.20,   # ✅ lowered from 0.25 — fewer false fails
        enable_llm_verify: bool = False,            # ✅ off by default — saves LLM calls
        debug: bool = False,
    ):
        self.llm = llm_agent
        self.max_retries = max_retries
        self.keyword_overlap_threshold = keyword_overlap_threshold
        self.enable_llm_verify = enable_llm_verify
        self.debug = debug

    # =========================
    # Public Entry Point
    # =========================

    def check_and_fix(
        self,
        question: str,
        answer: str,
        chunk: str,
    ) -> Tuple[str, bool, int]:
        """
        Returns:
            (final_answer, passed, attempts_used)
        """
        answer = (answer or "").strip()
        chunk = (chunk or "").strip()

        # ✅ If answer is "not available", pass it directly — no point retrying
        if self.NOT_AVAILABLE_PHRASE in answer.lower():
            if self.debug:
                logger.debug("Answer is 'not available' — skipping consistency check.")
            return answer, True, 0

        attempt = 0
        while attempt <= self.max_retries:
            passed, reason = self._check(answer, chunk)

            if self.debug:
                logger.debug("attempt=%d passed=%s reason=%s", attempt, passed, reason)

            if passed:
                return answer, True, attempt

            attempt += 1
            if attempt > self.max_retries:
                break

            if self.debug:
                logger.debug("Regenerating attempt %d (reason=%s)...", attempt, reason)

            new_answer = self._regenerate(question, chunk)
            if not new_answer:
                break
            answer = new_answer

        if self.debug:
            logger.debug("All retries exhausted — returning last answer.")

        return answer, False, attempt

    # =========================
    # Check Logic
    # =========================

    def _check(self, answer: str, chunk: str) -> Tuple[bool, str]:
        # Step 1: Fast keyword overlap
        overlap_ok, overlap_reason = self._keyword_overlap_check(answer, chunk)
        if not overlap_ok:
            return False, overlap_reason

        # Step 2: LLM self-verify (optional)
        if self.enable_llm_verify:
            llm_ok, llm_reason = self._llm_self_verify(answer, chunk)
            if not llm_ok:
                return False, llm_reason

        return True, "ok"

    def _keyword_overlap_check(self, answer: str, chunk: str) -> Tuple[bool, str]:
        score = fuzz.token_set_ratio(
            self._normalize(answer),
            self._normalize(chunk),
        )
        threshold = int(self.keyword_overlap_threshold * 100)

        if self.debug:
            logger.debug("keyword_overlap score=%d threshold=%d", score, threshold)

        if score < threshold:
            return False, f"keyword_overlap_low (score={score}, threshold={threshold})"

        return True, f"keyword_overlap_ok (score={score})"

    def _llm_self_verify(self, answer: str, chunk: str) -> Tuple[bool, str]:
        prompt = SELF_VERIFY_PROMPT.format(
            chunk=chunk[:1500],
            answer=answer[:800],
        )
        try:
            raw = self.llm._generate(prompt)
            verdict = self._extract_verdict(raw)

            if self.debug:
                logger.debug("LLM verdict raw='%s' parsed='%s'", raw[:80], verdict)

            if verdict == "PASS":
                return True, "llm_verified"
            else:
                return False, f"llm_verdict={verdict}"

        except Exception as e:
            if self.debug:
                logger.debug("LLM self-verify error: %s", e)
            return True, "llm_verify_skipped"

    def _extract_verdict(self, raw: str) -> str:
        upper = (raw or "").upper()
        if "PASS" in upper:
            return "PASS"
        if "FAIL" in upper:
            return "FAIL"
        return "UNKNOWN"

    # =========================
    # Regeneration
    # =========================

    def _regenerate(self, question: str, chunk: str) -> Optional[str]:
        prompt = REGENERATE_PROMPT.format(
            question=question,
            context=chunk[:2000],   # ✅ increased from 1500
        )
        try:
            return self.llm._generate(prompt)
        except Exception as e:
            if self.debug:
                print(f"[ConsistencyCheck] Regeneration error: {e}")
            return None

    # =========================
    # Utils
    # =========================

    _space_re = re.compile(r"\s+")

    def _normalize(self, text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        return self._space_re.sub(" ", text).strip()