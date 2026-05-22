from __future__ import annotations

import asyncio
import httpx
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

logger = logging.getLogger("eduassist.webscraper_agent")


# ============================================================
# Query → URL keyword mapping
# ============================================================
QUERY_URL_HINTS: Dict[str, List[str]] = {
    "admission":    ["admission", "admissions"],
    "fee":          ["fee", "admissions"],
    "scholarship":  ["scholarship", "admissions"],
    "program":      ["program", "academics", "bs-", "bba", "diploma"],
    "faculty":      ["faculty", "kiet-faculty", "cocis", "coms"],
    "software":     ["software", "cocis"],
    "cyber":        ["cyber", "cocis"],
    "data science": ["data-science", "cocis"],
    "business":     ["business", "coms", "bba"],
    "accounting":   ["accounting", "coms"],
    "marketing":    ["marketing", "coms"],
    "department":   ["department", "cocis", "coms", "coe"],
    "contact":      ["contact", "department"],
    "news":         ["news", "events", "stories"],
    "notice":       ["notice", "news", "events"],
    "transfer":     ["transfer"],
    "policy":       ["policy", "unfair"],
    "grading":      ["grading", "examination"],
    "alumni":       ["alumni"],
    "research":     ["oric", "research"],
    "entrepreneur": ["tibic", "tencoms", "entrepreneur"],
    "diploma":      ["diploma"],
    "calendar":     ["calendar", "timetable"],
}


@dataclass(frozen=True)
class WebScraperConfig:
    base_domain: str = "kiet.edu.pk"

    allowed_domains: Tuple[str, ...] = (
        "kiet.edu.pk",
        "coe.kiet.edu.pk",
        "cocis.kiet.edu.pk",
        "coms.kiet.edu.pk",
        "admissions.kiet.edu.pk",
        "tencoms.kiet.edu.pk",
    )

    max_pages_per_query: int = 6   # reduced from 12 — faster crawl
    max_depth: int = 1             # reduced from 2 — less deep crawling
    concurrency: int = 4           # increased from 3 — fetch more pages in parallel

    min_text_chars: int = 150
    min_page_score: int = 35
    max_snippet_lines: int = 20
    early_exit_score: int = 70          # reduced from 80 — exit sooner
    top_chunks: int = 2                 # reduced from 3 — less tokens for LLM

    cache_ttl_seconds: int = 60 * 60
    cache_max_size: int = 200

    total_timeout_s: int = 12      # reduced from 20 — fail faster if site is slow
    request_timeout_s: int = 8     # reduced from 15
    max_retries: int = 0           # reduced from 1 — no retry, saves time

    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    debug: bool = False


class WebScraperAgent:
    def __init__(
        self,
        urls: Optional[List[str]] = None,
        base_domain: str = "kiet.edu.pk",
        config: Optional[WebScraperConfig] = None,
    ):
        self.seed_urls = urls or []
        self.config = config or WebScraperConfig(base_domain=base_domain)
        self.base_domain = self.config.base_domain

        self._allowed_domains_set: Set[str] = {
            d.lower().strip() for d in self.config.allowed_domains
        }

        self._cache: OrderedDict[str, Tuple[float, str]] = OrderedDict()

        # Lazy Playwright check — don't launch browser at startup
        self._use_playwright: Optional[bool] = None  # None = not checked yet

        # httpx timeout — connect: 5s, read: request_timeout_s from config
        self._httpx_timeout = httpx.Timeout(
            connect=5.0,
            read=float(self.config.request_timeout_s),
            write=5.0,
            pool=5.0,
        )
        self._httpx_headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

    # ============================================================
    # Playwright check — lazy (runs on first scrape, not at init)
    # ============================================================

    def _check_playwright(self) -> bool:
        if self._use_playwright is not None:
            return self._use_playwright
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            logger.info("Playwright available - dynamic scraping enabled")
            self._use_playwright = True
        except Exception as e:
            logger.warning(
                "Playwright not available (%s) - using static scraping. "
                "Run: pip install playwright && playwright install chromium",
                type(e).__name__,
            )
            self._use_playwright = False
        return self._use_playwright

    # ============================================================
    # Public Entry
    # ============================================================

    def scrape(self, query: str) -> Optional[List[str]]:
        query = (query or "").strip()
        if not query:
            return None

        # Lazy Playwright check on first scrape
        self._check_playwright()

        start = time.perf_counter()
        out = self._run_async(lambda: self._scrape_async(query))

        if self.config.debug:
            dt = time.perf_counter() - start
            mode = "playwright" if self._use_playwright else "static"
            logger.debug("WebScraper done in %.2fs | mode=%s | chunks=%d",
                         dt, mode, len(out) if out else 0)

        return out if out else None

    async def _scrape_async(self, query: str) -> Optional[List[str]]:
        seeds = self._smart_seeds(query)
        if not seeds:
            return None

        pages = await self._crawl(query, seeds)
        if not pages:
            return None

        return self._best_chunks(query, pages)

    # ============================================================
    # Smart Seed Selection
    # ============================================================

    def _smart_seeds(self, query: str) -> List[str]:
        q = query.lower()
        relevant: List[str] = []
        fallback: List[str] = []

        matched_hints: List[str] = []
        for keyword, hints in QUERY_URL_HINTS.items():
            if keyword in q:
                matched_hints.extend(hints)

        all_seeds = self._build_seed_list()

        for url in all_seeds:
            u = url.lower()
            if matched_hints and any(h in u for h in matched_hints):
                relevant.append(url)
            else:
                fallback.append(url)

        if relevant:
            combined = relevant + fallback
            logger.debug("Smart seeds: %d relevant + %d fallback",
                         len(relevant), len(fallback))
            return combined[:self.config.max_pages_per_query]

        return all_seeds

    def _build_seed_list(self) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()

        for u in self.seed_urls:
            fixed = self._fix_url(u)
            if fixed and fixed not in seen:
                seen.add(fixed)
                out.append(fixed)

        if not out:
            home = self._fix_url(f"https://{self.base_domain}/")
            if home:
                out.append(home)

        return out

    def _fix_url(self, u: str) -> Optional[str]:
        u = (u or "").strip()
        if not u:
            return None
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        u = u.split("#", 1)[0]
        return u if self._allowed(u) else None

    # ============================================================
    # Crawl
    # ============================================================

    async def _crawl(self, query: str, seeds: List[str]) -> List[Tuple[str, str]]:
        visited: Set[str] = set()
        frontier: List[Tuple[str, int, int]] = [(u, 0, 0) for u in seeds]
        collected: List[Tuple[str, str]] = []
        sem = asyncio.Semaphore(self.config.concurrency)
        t0 = time.monotonic()

        if self._use_playwright:
            try:
                browser_config = BrowserConfig(headless=True)
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    await self._crawl_loop(query, crawler, sem, frontier, visited, collected, t0)
                return collected
            except NotImplementedError:
                self._use_playwright = False
                logger.info("Playwright not supported -> fallback to static")
            except Exception as e:
                self._use_playwright = False
                logger.warning("Playwright failed (%s) -> fallback to static", type(e).__name__)

        frontier = [(u, 0, 0) for u in seeds]
        visited.clear()
        collected.clear()
        await self._crawl_loop(query, None, sem, frontier, visited, collected, t0)
        return collected

    async def _crawl_loop(self, query, crawler, sem, frontier, visited, collected, t0):
        while frontier and len(collected) < self.config.max_pages_per_query:
            if (time.monotonic() - t0) > self.config.total_timeout_s:
                logger.debug("Hard timeout %.0fs reached", self.config.total_timeout_s)
                break

            # Early exit if highly relevant page found
            if collected:
                best_so_far = max(
                    int(fuzz.token_set_ratio(query.lower(), text.lower()))
                    for _, text in collected
                )
                if best_so_far >= self.config.early_exit_score:
                    logger.debug("Early exit - high confidence page found (score=%d)", best_so_far)
                    break

            frontier.sort(key=lambda x: x[2])
            batch: List[Tuple[str, int]] = []

            while frontier and len(batch) < self.config.concurrency:
                url, depth, _ = frontier.pop(0)
                url = url.split("#", 1)[0]
                if url in visited or depth > self.config.max_depth:
                    continue
                if not self._allowed(url):
                    continue
                visited.add(url)
                batch.append((url, depth))

            if not batch:
                break

            results = await asyncio.gather(
                *[self._fetch(crawler, sem, url) for url, _d in batch],
                return_exceptions=True,
            )

            for idx, res in enumerate(results):
                if isinstance(res, Exception) or res is None:
                    continue
                url, text, links = res
                if text and len(text) >= self.config.min_text_chars:
                    score = int(fuzz.token_set_ratio(query.lower(), text.lower()))
                    if score >= self.config.min_page_score:
                        collected.append((url, text))

                depth = batch[idx][1]
                next_depth = depth + 1
                if next_depth <= self.config.max_depth:
                    for link, anchor in links:
                        link = link.split("#", 1)[0]
                        if link in visited or not self._allowed(link):
                            continue
                        prio = self._priority(query, link, anchor)
                        frontier.append((link, next_depth, prio))

    # ============================================================
    # Fetch with retry
    # ============================================================

    async def _fetch(self, crawler, sem: asyncio.Semaphore, url: str):
        async with sem:
            cached = self._cache_get(url)
            if cached is not None:
                return (url, cached, [])

            for attempt in range(self.config.max_retries + 1):
                try:
                    result = await self._fetch_once(crawler, url)
                    if result is not None:
                        return result
                except Exception as e:
                    logger.debug("Fetch attempt %d failed for %s: %s", attempt + 1, url, e)
                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.5)

            return None

    async def _fetch_once(self, crawler, url: str):
        if crawler is not None:
            try:
                result = await crawler.arun(url=url, config=CrawlerRunConfig())
                if not result.success or not result.html:
                    return None
                soup = BeautifulSoup(result.html, "html.parser")
                links = self._extract_links(soup, url)
                self._remove_boilerplate(soup)
                main = self._get_main_content(soup)
                text = self._clean_text((main or soup).get_text(separator="\n"))
                if not text:
                    return None
                self._cache_set(url, text)
                return (url, text, links)
            except NotImplementedError:
                self._use_playwright = False
                return await self._fetch_static(url)
            except Exception:
                return await self._fetch_static(url)

        return await self._fetch_static(url)

    async def _fetch_static(self, url: str):
        cached = self._cache_get(url)
        if cached is not None:
            return (url, cached, [])
        try:
            async with httpx.AsyncClient(
                headers=self._httpx_headers,
                timeout=self._httpx_timeout,
                follow_redirects=True,
            ) as client:
                r = await client.get(url)
            if self.config.debug:
                logger.debug("GET %d %s", r.status_code, url)
            if r.status_code != 200 or not r.text:
                return None
            soup = BeautifulSoup(r.text, "html.parser")
            links = self._extract_links(soup, url)
            self._remove_boilerplate(soup)
            main = self._get_main_content(soup)
            text = self._clean_text((main or soup).get_text(separator="\n"))
            if not text:
                return None
            self._cache_set(url, text)
            return (url, text, links)
        except httpx.TimeoutException as e:
            logger.debug("Static fetch timed out for %s: %s", url, e)
            return None
        except Exception as e:
            logger.debug("Static fetch failed for %s: %s", url, e)
            return None

    # ============================================================
    # Best Chunks (returns List[str] for LLM context)
    # ============================================================

    def _best_chunks(self, query: str, pages: List[Tuple[str, str]]) -> Optional[List[str]]:
        q = query.lower()

        scored_pages = []
        for url, text in pages:
            score = int(fuzz.token_set_ratio(q, text.lower()))
            if score >= self.config.min_page_score:
                scored_pages.append((score, url, text))

        if not scored_pages:
            return None

        scored_pages.sort(key=lambda x: x[0], reverse=True)

        all_blocks: List[Tuple[int, str, str]] = []

        for _, url, text in scored_pages[:3]:
            blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
            if not blocks:
                blocks = [text]
            for block in blocks:
                block_score = int(fuzz.token_set_ratio(q, block.lower()))
                if block_score >= self.config.min_page_score:
                    all_blocks.append((block_score, url, block))

        if not all_blocks:
            return None

        all_blocks.sort(key=lambda x: x[0], reverse=True)

        chunks: List[str] = []
        seen_blocks: Set[str] = set()

        for score, url, block in all_blocks[: self.config.top_chunks * 2]:
            block_key = block[:100]
            if block_key in seen_blocks:
                continue
            seen_blocks.add(block_key)
            lines = block.splitlines()[: self.config.max_snippet_lines]
            snippet = "\n".join(lines).strip()
            chunks.append(f"{snippet}\n\nSource: {url}")
            if len(chunks) >= self.config.top_chunks:
                break

        return chunks if chunks else None

    # ============================================================
    # HTML Parsing Helpers
    # ============================================================

    def _get_main_content(self, soup: BeautifulSoup):
        for node in [
            soup.find("main"),
            soup.find("article"),
            soup.find("div", class_="entry-content"),
            soup.find("div", class_="post-content"),
            soup.find("div", class_="page-content"),
            soup.find("div", id="content"),
            soup.find("div", id="main"),
            soup.find("div", id="primary"),
        ]:
            if node and len(node.get_text(strip=True)) > 80:
                return node
        return None

    def _remove_boilerplate(self, soup: BeautifulSoup) -> None:
        for tag_name in [
            "nav", "header", "footer", "aside",
            "script", "style", "noscript", "form", "button",
            "iframe", "svg", "figure",
        ]:
            for t in soup.find_all(tag_name):
                t.decompose()

    _space_re = re.compile(r"\s+", flags=re.UNICODE)
    _noise_patterns = re.compile(
        r"(all rights reserved|powered by wordpress|privacy policy"
        r"|terms of use|subscribe to|follow us|share this"
        r"|back to top|skip to content|toggle navigation"
        r"|copyright \d{4})",
        flags=re.IGNORECASE,
    )

    def _clean_text(self, text: str) -> str:
        lines: List[str] = []
        seen_lines: Set[str] = set()

        for ln in (text or "").splitlines():
            ln = self._space_re.sub(" ", ln).strip()
            if len(ln) <= 2:
                continue
            if "cookie" in ln.lower() and len(ln) < 140:
                continue
            if self._noise_patterns.search(ln):
                continue
            ln_key = ln.lower()
            if ln_key in seen_lines:
                continue
            seen_lines.add(ln_key)
            lines.append(ln)

        return "\n".join(lines).strip()

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            full = urljoin(base_url, href).split("#", 1)[0]
            if not self._allowed(full):
                continue
            low = full.lower()
            if any(x in low for x in [
                "facebook", "twitter", "instagram",
                "linkedin", "youtube", "whatsapp",
                "mailto:", "tel:",
            ]):
                continue
            if full in seen:
                continue
            seen.add(full)
            anchor = (a.get_text(" ", strip=True) or "").strip()
            out.append((full, anchor))

        return out

    def _allowed(self, url: str) -> bool:
        try:
            if not url.startswith(("http://", "https://")):
                return False
            host = (urlparse(url).netloc or "").lower()
            if not host:
                return False
            for d in self._allowed_domains_set:
                if host == d or host.endswith("." + d):
                    return True
            return False
        except Exception:
            return False

    def _priority(self, query: str, url: str, anchor: str) -> int:
        q = query.lower()
        u = url.lower()
        a = (anchor or "").lower()
        hot = [
            "admission", "fee", "scholar", "policy", "calendar",
            "notice", "deadline", "timetable", "program", "department",
        ]
        prio = 50
        if any(k in u for k in hot):
            prio -= 15
        if a and fuzz.partial_ratio(q, a) >= 60:
            prio -= 15
        if fuzz.partial_ratio(q, u) >= 60:
            prio -= 10
        if any(x in u for x in ["/tag/", "/author/", "/page/", "/category/"]):
            prio += 10
        return max(0, prio)

    # ============================================================
    # Cache (LRU with size cap)
    # ============================================================

    def _cache_get(self, url: str) -> Optional[str]:
        item = self._cache.get(url)
        if not item:
            return None
        ts, text = item
        if (time.time() - ts) > self.config.cache_ttl_seconds:
            self._cache.pop(url, None)
            return None
        self._cache.move_to_end(url)
        return text

    def _cache_set(self, url: str, text: str) -> None:
        if url in self._cache:
            self._cache.move_to_end(url)
        self._cache[url] = (time.time(), text)
        while len(self._cache) > self.config.cache_max_size:
            self._cache.popitem(last=False)

    def _run_async(self, coro_factory):
        try:
            return asyncio.run(coro_factory())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro_factory())
            finally:
                loop.close()