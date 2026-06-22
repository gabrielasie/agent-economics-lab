"""Tests for the content-addressed response cache.

No network: a fake completion client stands in for the Anthropic client and records
how often it is called, so we can prove hits skip the call and misses call exactly once.
"""

from pathlib import Path

from aelab.agents.cache import ResponseCache, cache_key


class FakeClient:
    """Records calls and returns canned text instead of hitting the network."""

    def __init__(self, text: str = "bid") -> None:
        self.text = text
        self.calls = 0

    def complete(self, model: str, system: str, user: str) -> str:
        self.calls += 1
        return self.text


def test_miss_calls_once_and_persists(tmp_path: Path) -> None:
    fake = FakeClient("hello")
    result = ResponseCache(fake, cache_dir=tmp_path).complete("haiku", "sys", "user")
    assert result == "hello"
    assert fake.calls == 1
    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == "hello"


def test_hit_avoids_the_call(tmp_path: Path) -> None:
    fake = FakeClient("hello")
    cache = ResponseCache(fake, cache_dir=tmp_path)
    cache.complete("haiku", "sys", "user")  # miss
    assert cache.complete("haiku", "sys", "user") == "hello"  # hit
    assert fake.calls == 1


def test_persisted_cache_is_read_by_a_new_instance(tmp_path: Path) -> None:
    ResponseCache(FakeClient("stored"), cache_dir=tmp_path).complete("m", "s", "u")
    fresh = FakeClient("SHOULD-NOT-BE-USED")
    assert ResponseCache(fresh, cache_dir=tmp_path).complete("m", "s", "u") == "stored"
    assert fresh.calls == 0  # served entirely from disk


def test_different_inputs_miss_separately(tmp_path: Path) -> None:
    fake = FakeClient()
    cache = ResponseCache(fake, cache_dir=tmp_path)
    cache.complete("m", "s", "user-a")
    cache.complete("m", "s", "user-b")
    assert fake.calls == 2
    assert len(list(tmp_path.glob("*.txt"))) == 2


def test_identical_inputs_give_identical_keys() -> None:
    assert cache_key("m", "s", "u") == cache_key("m", "s", "u")


def test_different_inputs_give_different_keys() -> None:
    base = cache_key("m", "s", "u")
    assert cache_key("M", "s", "u") != base
    assert cache_key("m", "S", "u") != base
    assert cache_key("m", "s", "U") != base
