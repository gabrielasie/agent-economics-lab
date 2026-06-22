"""Content-addressed cache around the Anthropic client.

This is the one place nondeterminism (the LLM, the network) is allowed to live, so this
module may import anthropic. A run is reproducible from its seed plus this cache:
identical (model, system, user) inputs hash to the same key and replay the stored
completion without calling the API. A miss calls the client once and persists the result.
The batch client submits one Message Batch and returns completions keyed by custom_id.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import anthropic

DEFAULT_MAX_TOKENS = 1024


class CompletionClient(Protocol):
    """The single call the cache depends on: a prompt in, a text completion out."""

    def complete(self, model: str, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class BatchRequest:
    """One entry in a Message Batch, addressed by custom_id."""

    custom_id: str
    model: str
    system: str
    user: str


class BatchClient(Protocol):
    """Runs a batch of requests and returns completion text keyed by custom_id."""

    def run(self, requests: Sequence[BatchRequest]) -> dict[str, str]: ...


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
    the key is only needed when a cache miss actually calls the API.
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


class AnthropicBatchClient:
    """Real batch client. Submits one Message Batch, polls it to completion, and returns
    completion text keyed by custom_id. Built lazily like AnthropicClient. Failed entries
    are simply absent from the result; the caller decides how to fall back.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, poll_seconds: float = 15.0) -> None:
        self._client: anthropic.Anthropic | None = None
        self._max_tokens = max_tokens
        self._poll_seconds = poll_seconds

    def _ensure_client(self) -> anthropic.Anthropic:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def run(self, requests: Sequence[BatchRequest]) -> dict[str, str]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        client = self._ensure_client()
        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=r.custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=r.model,
                        max_tokens=self._max_tokens,
                        system=[{"type": "text", "text": r.system, "cache_control": {"type": "ephemeral"}}],
                        messages=[{"role": "user", "content": r.user}],
                    ),
                )
                for r in requests
            ]
        )
        while client.messages.batches.retrieve(batch.id).processing_status != "ended":
            time.sleep(self._poll_seconds)
        texts: dict[str, str] = {}
        for result in client.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                texts[result.custom_id] = _extract_text(result.result.message)
        return texts


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
