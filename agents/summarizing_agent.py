from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("eduassist.summarizing_agent")

# =========================
# Prompt Template
# ✅ Plain text prompt — consistent with llm_agent.py style
# ✅ No chat tags
# =========================

SUMMARIZE_PROMPT = """You are a helpful university assistant summarizing information for students.

Convert the following answer into a clear, student-friendly bullet-point summary.

Rules:
- Use simple, easy-to-understand language.
- Present information as short bullet points starting with "•" symbol.
- Each bullet point should cover ONE key fact.
- Do NOT add any information not present in the original answer.
- Do NOT use headers or bold text.
- Maximum 6 bullet points.

Original Answer:
{answer}

Bullet-Point Summary:
•"""


class SummarizingAgent:
    """
    Summarizes a verified LLM answer into student-friendly bullet points.

    ✅ Improvements:
    - Prompt starts with "•" to guide model output format
    - Plain text prompt (no chat tags) — consistent with llm_agent
    - Smarter fallback: sentence-split for medium answers
    - Handles "not available" answers cleanly without LLM call
    - Better bullet extraction handles more edge cases
    """

    NOT_AVAILABLE_PHRASE = "the information is not available in the current data"

    def __init__(
        self,
        llm_agent,
        *,
        max_bullets: int = 6,
        fallback_to_original: bool = True,
        debug: bool = False,
    ):
        self.llm = llm_agent
        self.max_bullets = max_bullets
        self.fallback_to_original = fallback_to_original
        self.debug = debug

    # =========================
    # Public Entry Point
    # =========================

    def summarize(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return answer

        # ✅ "Not available" answers don't need summarization
        if self.NOT_AVAILABLE_PHRASE in answer.lower():
            return f"• {answer}"

        # ✅ Already bullet-pointed — just clean and return
        if answer.startswith("•"):
            return self._clean_existing_bullets(answer)

        # Short answer — format directly without LLM call (faster)
        if len(answer.split()) <= 40:
            return self._format_short(answer)

        # Long answer — use LLM to summarize
        try:
            summary = self._llm_summarize(answer)
            if summary and len(self._extract_bullets(summary)) >= 1:
                return summary
        except Exception as e:
            if self.debug:
                logger.debug("LLM summarize error: %s", e)

        # Fallback — sentence split
        if self.fallback_to_original:
            return self._format_short(answer)

        return answer

    # =========================
    # LLM Summarization
    # =========================

    def _llm_summarize(self, answer: str) -> Optional[str]:
        # ✅ Prepend "•" so model continues in bullet format
        prompt = SUMMARIZE_PROMPT.format(answer=answer[:2000])

        raw = self.llm._generate(prompt)
        if not raw:
            return None

        if self.debug:
            logger.debug("raw output:\n%s", raw[:300])

        # ✅ Prepend the "•" that was part of the prompt ending
        if not raw.strip().startswith("•"):
            raw = "• " + raw.strip()

        bullets = self._extract_bullets(raw)
        if not bullets:
            return None

        bullets = bullets[: self.max_bullets]
        return "\n".join(bullets)

    # =========================
    # Bullet Extraction
    # =========================

    def _extract_bullets(self, text: str) -> list[str]:
        """
        Extracts bullet lines from LLM output.
        Handles •, -, *, numbered lists, and inline bullets.
        """
        # ✅ Handle inline bullets separated by "•" on same line
        if "\n" not in text and "•" in text:
            parts = text.split("•")
            bullets = []
            for p in parts:
                p = p.strip()
                if len(p) > 8:
                    bullets.append(f"• {p}")
            return bullets

        lines = text.splitlines()
        bullets = []

        for line in lines:
            line = line.strip()
            if not line or len(line) < 8:
                continue

            if re.match(r"^[•\-\*\u2022\u2023\u25e6]", line):
                clean = re.sub(r"^[•\-\*\u2022\u2023\u25e6]\s*", "• ", line)
                bullets.append(clean)
            elif re.match(r"^\d+[\.\ )]\s+", line):
                clean = re.sub(r"^\d+[\.\ )]\s+", "• ", line)
                bullets.append(clean)

        return bullets

    # =========================
    # Clean Existing Bullets
    # =========================

    def _clean_existing_bullets(self, answer: str) -> str:
        """Clean up bullet formatting if answer already has bullets."""
        lines = answer.splitlines()
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("•"):
                line = f"• {line}"
            cleaned.append(line)
        return "\n".join(cleaned[: self.max_bullets])

    # =========================
    # Short Answer Formatter
    # =========================

    def _format_short(self, answer: str) -> str:
        """For short/medium answers, split into sentence-level bullets."""
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        bullets = []

        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                bullets.append(f"• {s}")

        if not bullets:
            return f"• {answer}"

        return "\n".join(bullets[: self.max_bullets])