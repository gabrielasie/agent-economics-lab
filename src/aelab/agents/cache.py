"""Content-addressed cache around the Anthropic client.

This is the one place nondeterminism (the LLM, the network) is allowed to live, so this
module may import anthropic. A run is reproducible from its seed plus this cache:
identical (model, system, user) inputs hash to the same key and replay the stored
completion without calling the API. A miss calls the client once and persists the result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import anthropic

DEFAULT_MAX_TOKENS = 1024


class CompletionClient(Protocol):
    """The single call the cache depends on: a prompt in, a text completion out."""

    def complete(self, model: str, system: str, user: str) -> str: ...


def cache_key(model: str, system: str, user: str) -> str:
    """Stable sha256 over the inputs. Identical inputs give an identical key."""
    canonical = json.dumps(
        {"model": model, "system": system, "user": user},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_text(response: Any) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    return ""


class AnthropicClient:
    """Real completion client. Wraps anthropic.Anthropic, built lazily on first use.

    The lazy construction lets the rest of the lab import this module without an API key;
    the key is only needed when a cache miss actually calls the API. The constant system
    prompt is marked for prompt caching.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._client: anthropic.Anthropic | None = None
        self._max_tokens = max_tokens

    def _ensure_client(self) -> anthropic.Anthropic:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, model: str, system: str, user: str) -> str:
        response = self._ensure_client().messages.create(
            model=model,
            max_tokens=self._max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return _extract_text(response)


class ResponseCache:
    """A content-addressed file cache wrapping a CompletionClient.

    On a hit the stored completion is returned and the wrapped client is not called.
    On a miss the client is called once and the result is written to cache_dir.
    """

    def __init__(self, client: CompletionClient, cache_dir: Path = Path(".cache")) -> None:
        self._client = client
        self._cache_dir = cache_dir

    def complete(self, model: str, system: str, user: str) -> str:
        path = self._cache_dir / f"{cache_key(model, system, user)}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = self._client.complete(model, system, user)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text
