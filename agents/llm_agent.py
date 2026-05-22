from __future__ import annotations

import logging
import time
import threading
import torch
from typing import Iterator, List, Optional, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from huggingface_hub import login

from config.settings import (
    HUGGINGFACE_TOKEN,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger("eduassist.llm_agent")

RAG_PROMPT_TEMPLATE = """You are an AI assistant for KIET university chatbot.

Answer the student's question using ONLY the provided context below.

Rules:
- Read the context carefully and extract relevant information.
- Answer directly and clearly using ONLY what is in the context.
- If the context contains the answer, always provide it.
- Only say "The information is not available in the current data." if the context truly has NO relevant information.
- Be concise, factual, and helpful for students.
- Do not repeat the question.

Context:
{context}

Student Question:
{question}

Answer:
"""


class LLMAgent:
    """
    LLM Agent — Iteration 6.

    Changes vs Iteration 5:
    - generate_stream() added using TextIteratorStreamer  ✅
    - generate() and _generate() completely untouched     ✅
    - Streaming runs model.generate() in background thread✅
    - CPU thread comment updated (no longer Streamlit)    ✅
    """

    _model     = None
    _tokenizer = None
    _login_done = False

    def __init__(self):

        # ── HuggingFace Login (once) ──
        if not LLMAgent._login_done:
            if HUGGINGFACE_TOKEN:
                try:
                    login(token=HUGGINGFACE_TOKEN)
                    logger.info("✅ HuggingFace login successful")
                except Exception as e:
                    logger.warning("⚠️ HuggingFace login failed: %s", e)
            else:
                logger.warning("⚠️ No HUGGINGFACE_TOKEN — gated models may fail.")
            LLMAgent._login_done = True

        # ── Device ──
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("🖥️ LLM device: %s", self.device)

        # ✅ CPU thread optimization — AMD Ryzen 5 7430U (6C/12T)
        # Leaves 2 threads for OS / FastAPI workers
        if self.device == "cpu":
            num_threads = max(1, torch.get_num_threads() - 2)
            torch.set_num_threads(num_threads)
            logger.info("🧵 CPU threads: %d", num_threads)

        # ── Load model once ──
        if LLMAgent._model is None:
            logger.info("🧠 Loading LLM: %s", LLM_MODEL)

            LLMAgent._tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL, use_fast=True, token=HUGGINGFACE_TOKEN or None,
            )
            if LLMAgent._tokenizer.pad_token is None:
                LLMAgent._tokenizer.pad_token    = LLMAgent._tokenizer.eos_token
                LLMAgent._tokenizer.pad_token_id = LLMAgent._tokenizer.eos_token_id

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            LLMAgent._model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL, torch_dtype=dtype,
                low_cpu_mem_usage=True, token=HUGGINGFACE_TOKEN or None,
            )
            LLMAgent._model.to(self.device)
            LLMAgent._model.eval()

            try:
                if self.device == "cuda":
                    LLMAgent._model = torch.compile(
                        LLMAgent._model, mode="reduce-overhead", backend="inductor"
                    )
            except Exception:
                pass

            logger.info("✅ LLM loaded on %s", self.device)

        self.tokenizer  = LLMAgent._tokenizer
        self.model      = LLMAgent._model
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE

    # ─────────────────────────────────────────────
    # Core generation — UNCHANGED from Iteration 5
    # Used by: ConsistencyCheckAgent, SummarizingAgent
    # ─────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        is_cpu = self.device == "cpu"
        max_input_length = 1024 if is_cpu else 2048
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_input_length, padding=False,
        ).to(self.device)
        input_length = inputs["input_ids"].shape[1]
        max_new = min(self.max_tokens, 256) if is_cpu else self.max_tokens

        logger.info("⚙️ Generating (device=%s max_new=%d input_tokens=%d)...",
                    self.device, max_new, input_length)
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False if is_cpu else (self.temperature > 0),
                temperature=self.temperature if (not is_cpu and self.temperature > 0) else 1.0,
                top_p=0.9 if not is_cpu else None,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.perf_counter() - t0
        new_tokens = outputs[0][input_length:]
        logger.info("✅ Generated %d tokens in %.1fs (%.1f tok/s)",
                    len(new_tokens), elapsed, len(new_tokens) / max(elapsed, 0.01))
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ─────────────────────────────────────────────
    # Shared context preparation
    # ─────────────────────────────────────────────

    def _prepare_context(self, context: Union[str, List[str]]) -> Optional[str]:
        """Merge and truncate context for LLM input. Returns None if empty."""
        if not context:
            return None
        if isinstance(context, list):
            merged = "\n\n---\n\n".join(c.strip() for c in context if c and c.strip())
        else:
            merged = context.strip()
        if not merged:
            return None
        is_cpu = self.device == "cpu"
        return merged[:1500] if is_cpu else merged[:4000]

    def generate(self, question: str, context: Union[str, List[str]], **_kwargs) -> Optional[str]:
        """Complete answer generation."""
        merged = self._prepare_context(context)
        if not merged:
            return None
        prompt = RAG_PROMPT_TEMPLATE.format(question=question.strip(), context=merged)
        try:
            answer = self._generate(prompt)
            return answer if answer else None
        except Exception as e:
            logger.error("LLM generation error: %s", e)
            return None

    # ─────────────────────────────────────────────
    # ✅ NEW — Streaming generation (Iteration 6)
    # Used by: FastAPI /api/ask SSE endpoint
    # ─────────────────────────────────────────────

    def generate_stream(
        self,
        question: str,
        context: Union[str, List[str]],
    ) -> Iterator[str]:
        """
        Stream answer tokens one by one via TextIteratorStreamer.

        How it works:
          1. TextIteratorStreamer is a queue that receives decoded tokens
          2. model.generate() runs in a BACKGROUND THREAD writing into it
          3. This method iterates the queue, yielding tokens to FastAPI
          4. FastAPI sends each token as an SSE event to the browser
          5. Browser appends each token to the chat bubble in real time

        Fallback: if streaming raises any exception, yields full
        answer from generate() as a single chunk so chat never breaks.
        """
        merged = self._prepare_context(context)
        if not merged:
            return

        is_cpu = self.device == "cpu"
        max_new = min(self.max_tokens, 256) if is_cpu else self.max_tokens

        prompt = RAG_PROMPT_TEMPLATE.format(
            question=question.strip(), context=merged
        )

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=1024 if is_cpu else 2048, padding=False,
            ).to(self.device)

            # skip_prompt=True  → only yields tokens AFTER the prompt
            # skip_special_tokens=True → strips <eos>, <pad> etc
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new,
                do_sample=False if is_cpu else (self.temperature > 0),
                temperature=self.temperature if (not is_cpu and self.temperature > 0) else 1.0,
                top_p=0.9 if not is_cpu else None,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # ✅ model.generate() blocks until done — MUST run in separate thread
            t = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
                daemon=True,
            )
            logger.info("⚡ Streaming start (device=%s max_new=%d)", self.device, max_new)
            t0 = time.perf_counter()
            t.start()

            token_count = 0
            for token_text in streamer:
                if token_text:
                    yield token_text
                    token_count += 1

            t.join()
            logger.info("✅ Streamed %d tokens in %.1fs",
                        token_count, time.perf_counter() - t0)

        except Exception as e:
            logger.error("Streaming error (%s) — falling back to generate()", e)
            fallback = self.generate(question, context)
            if fallback:
                yield fallback