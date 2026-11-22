"""Claude client used by agents for reasoning tasks.

* ``structured()`` returns a validated Pydantic object through structured outputs.
* ``web_search()`` runs Claude's server-side web search and returns the answer text plus the
  *actual* result URLs the search returned, so callers can verify citations.
* If no credentials are configured, ``available`` is False and agents use their rule-based fallbacks.

Requests use server-side refusal fallbacks (``fallbacks="default"``). If the primary model declines a
request, the API retries it on a fallback model within the same call.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from ..config import Settings, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}


@dataclass
class WebResult:
    url: str
    title: str
    page_age: str | None = None


@dataclass
class WebSearchOutput:
    text: str
    results: list[WebResult] = field(default_factory=list)
    cited_urls: list[str] = field(default_factory=list)


@dataclass
class LLMCallRecord:
    purpose: str
    model: str
    ok: bool
    seconds: float
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class LLMClient:
    def __init__(self, settings: Settings | None = None, enabled: bool = True):
        self.settings = settings or get_settings()
        self.model = self.settings.model
        self._client = None
        self.calls: list[LLMCallRecord] = []
        self._lock = threading.Lock()
        if enabled and self._has_credentials():
            try:
                import anthropic

                # The SDK resolves credentials itself: API key, auth token, or an `ant auth login` profile.
                self._client = anthropic.Anthropic(max_retries=3, timeout=300)
            except Exception as e:  # pragma: no cover
                log.info("Anthropic client unavailable: %s", e)
                self._client = None

    def _has_credentials(self) -> bool:
        import os

        if self.settings.anthropic_api_key or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_PROFILE"):
            return True
        return os.path.isdir(os.path.expanduser("~/.config/anthropic"))  # `ant auth login` profile store

    @property
    def available(self) -> bool:
        return self._client is not None

    def _record(self, rec: LLMCallRecord) -> None:
        with self._lock:
            self.calls.append(rec)

    # ------------------------------------------------------------------ structured output
    def structured(self, *, purpose: str, system: str, user: str, schema: type[T], effort: str = "medium",
                   max_tokens: int = 16000) -> T | None:
        if not self.available:
            return None
        import anthropic

        t0 = time.time()
        try:
            resp = self._client.beta.messages.parse(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema, output_config={"effort": effort},
                betas=[FALLBACK_BETA], fallbacks="default",
            )
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"model declined ({getattr(resp.stop_details, 'category', None)})")
            if resp.stop_reason == "max_tokens":
                raise RuntimeError("response truncated at max_tokens")
            out = resp.parsed_output
            self._record(LLMCallRecord(purpose, resp.model, out is not None, time.time() - t0, resp._request_id,
                                       resp.usage.input_tokens, resp.usage.output_tokens))
            return out
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError, ValueError) as e:
            log.warning("LLM structured call %s failed: %s", purpose, e)
            self._record(LLMCallRecord(purpose, self.model, False, time.time() - t0, error=str(e)[:300]))
            return None

    # ------------------------------------------------------------------ web search
    def web_search(self, *, purpose: str, system: str, prompt: str, max_uses: int = 5,
                   effort: str = "low", allowed_domains: list[str] | None = None) -> WebSearchOutput | None:
        if not (self.available and self.settings.web_search):
            return None
        import anthropic

        tool = dict(WEB_SEARCH_TOOL, max_uses=max_uses)
        if allowed_domains:
            tool["allowed_domains"] = allowed_domains
        messages: list[dict] = [{"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            resp = None
            for _ in range(4):  # resume server-tool turns that pause
                resp = self._client.beta.messages.create(
                    model=self.model, max_tokens=16000, system=system, messages=messages, tools=[tool],
                    output_config={"effort": effort}, betas=[FALLBACK_BETA], fallbacks="default",
                )
                if resp.stop_reason != "pause_turn":
                    break
                messages = [messages[0], {"role": "assistant", "content": resp.content}]
            if resp is None or resp.stop_reason == "refusal":
                raise RuntimeError("web search declined or empty")
            results: dict[str, WebResult] = {}
            cited: list[str] = []
            texts: list[str] = []
            for block in resp.content:
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    for r in block.content:
                        if getattr(r, "type", "") == "web_search_result":
                            results[r.url] = WebResult(r.url, r.title, getattr(r, "page_age", None))
                elif block.type == "text":
                    texts.append(block.text)
                    for c in getattr(block, "citations", None) or []:
                        u = getattr(c, "url", None)
                        if u:
                            cited.append(u)
            self._record(LLMCallRecord(purpose, resp.model, True, time.time() - t0, resp._request_id,
                                       resp.usage.input_tokens, resp.usage.output_tokens))
            return WebSearchOutput("\n".join(texts).strip(), list(results.values()), cited)
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError) as e:
            log.warning("LLM web search %s failed: %s", purpose, e)
            self._record(LLMCallRecord(purpose, self.model, False, time.time() - t0, error=str(e)[:300]))
            return None
