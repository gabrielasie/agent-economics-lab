"""Tests for the LLM bidding agent and the batch path. Fully mocked: no network.

A fake completion client and a fake batch client stand in for the cache, so we can check
prompt construction, the malformed-JSON fallback to a truthful bid, and that batch results
join back to the right (funder, invoice) pair by custom_id. The fake batch client enforces
the real Anthropic custom_id constraint, so an id-scheme regression fails here.
"""

import re
from collections.abc import Sequence

from aelab.agents.base import AuctionContext, AuctionRules
from aelab.agents.cache import BatchRequest
from aelab.agents.llm import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    LLMAgent,
    ValidatedLLMAgent,
    batch_bids,
    build_batch_requests,
    decode_custom_id,
    encode_custom_id,
    parse_bid_apr,
)
from aelab.models import Bid, Funder, Invoice, PricingPolicy

CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class FakeCompletionClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.last: tuple[str, str, str] | None = None

    def complete(self, model: str, system: str, user: str) -> str:
        self.last = (model, system, user)
        return self.text


class FakeBatchClient:
    """Stands in for the batch client and enforces the real custom_id constraint."""

    def __init__(self, by_custom_id: dict[str, str]) -> None:
        self.by_custom_id = by_custom_id
        self.seen: list[BatchRequest] = []

    def run(self, requests: Sequence[BatchRequest]) -> dict[str, str]:
        self.seen = list(requests)
        for r in requests:
            assert CUSTOM_ID_PATTERN.match(r.custom_id), f"bad custom_id: {r.custom_id!r}"
        return {
            r.custom_id: self.by_custom_id[r.custom_id]
            for r in requests
            if r.custom_id in self.by_custom_id
        }


def _ctx(memo: str = "") -> AuctionContext:
    invoice = Invoice(invoice_id="INV-1", face_value=100_000.0, days_early=60, memo=memo)
    return AuctionContext(invoice=invoice, rules=AuctionRules(reserve_apr=0.30))


# --- LLMAgent -----------------------------------------------------------------


def test_prompt_construction() -> None:
    fake = FakeCompletionClient('{"apr": 0.11, "rationale": "ok"}')
    bid = LLMAgent(Funder("F1", 0.09), fake).bid(_ctx(memo="net-30"))
    assert bid == Bid("F1", 0.11, rationale="ok")
    assert fake.last is not None
    model, system, user = fake.last
    assert model == DEFAULT_MODEL  # claude-haiku-4-5
    assert system == SYSTEM_PROMPT
    assert "INV-1" in user
    assert "100000" in user
    assert "60" in user
    assert "0.0900" in user  # the funder's true cost
    assert "net-30" in user  # the memo, passed through as data


def test_malformed_json_falls_back_to_truthful_bid() -> None:
    for bad in ["not json", "{}", '{"apr": "abc"}', '{"apr": -0.05}', '{"rationale": "x"}']:
        bid = LLMAgent(Funder("F1", 0.09), FakeCompletionClient(bad)).bid(_ctx())
        assert bid.apr == 0.09  # truthful fallback


def test_valid_json_uses_parsed_apr() -> None:
    agent = LLMAgent(Funder("F1", 0.09), FakeCompletionClient('{"apr": 0.13, "rationale": "y"}'))
    assert agent.bid(_ctx()).apr == 0.13


def test_agent_carries_model_rationale() -> None:
    agent = LLMAgent(Funder("F1", 0.09), FakeCompletionClient('{"apr": 0.13, "rationale": "because"}'))
    assert agent.bid(_ctx()).rationale == "because"


def test_fallback_bid_records_rationale() -> None:
    bid = LLMAgent(Funder("F1", 0.09), FakeCompletionClient("not json")).bid(_ctx())
    assert bid.apr == 0.09  # truthful fallback
    assert "fallback" in bid.rationale  # the bid records why it fell back


def test_parse_bid_apr() -> None:
    assert parse_bid_apr('{"apr": 0.1, "rationale": "x"}') == 0.1
    assert parse_bid_apr('{"apr": 0}') == 0.0
    assert parse_bid_apr("garbage") is None
    assert parse_bid_apr('{"apr": -1}') is None
    assert parse_bid_apr("{}") is None


# A real Haiku return from results/truthfulness_raw.json: valid JSON wrapped in a fence.
REAL_FENCED_RETURN = (
    "```json\n"
    '{"apr": 0.1192, "rationale": "Bidding my true cost of capital of 11.92% APR is '
    "optimal in a sealed-bid second-price reverse auction, as the winner pays the "
    "second-lowest bid rather than their own bid. This eliminates incentive to shade "
    'bids and makes truthful bidding the dominant strategy."}\n'
    "```"
)


def test_parse_bid_apr_tolerates_markdown_fences() -> None:
    assert parse_bid_apr(REAL_FENCED_RETURN) == 0.1192


def test_parse_bid_apr_tolerates_surrounding_prose() -> None:
    assert parse_bid_apr('Here is my bid: {"apr": 0.1, "rationale": "x"} -- thanks!') == 0.1


def test_agent_parses_fenced_json_instead_of_falling_back() -> None:
    # The model fences its JSON; the agent must parse it, not fall back to truthful.
    bid = LLMAgent(Funder("F1", 0.09), FakeCompletionClient(REAL_FENCED_RETURN)).bid(_ctx())
    assert bid.apr == 0.1192


# --- batch path ---------------------------------------------------------------


def test_custom_id_round_trip() -> None:
    assert encode_custom_id(7) == "b7"
    assert decode_custom_id("b7") == 7
    assert decode_custom_id(encode_custom_id(0)) == 0


def test_custom_ids_match_anthropic_pattern() -> None:
    # Funder and invoice ids that would have broken a separator scheme stay clean as indices.
    pairs = [
        (Funder("funder::with::colons", 0.10), Invoice("inv id with spaces!", 1.0, 1)),
        (Funder("F1", 0.20), Invoice("INV-2", 1.0, 1)),
    ]
    requests, _ = build_batch_requests(pairs)
    for request in requests:
        assert CUSTOM_ID_PATTERN.match(request.custom_id)


def test_batch_joins_results_by_custom_id() -> None:
    f0, f1 = Funder("F0", 0.10), Funder("F1", 0.20)
    inv = Invoice("INV-7", 50_000.0, 30)
    responses = {
        encode_custom_id(0): '```json\n{"apr": 0.11, "rationale": "a"}\n```',  # fenced, like reality
        encode_custom_id(1): '{"apr": 0.22, "rationale": "b"}',
    }
    bids = batch_bids([(f0, inv), (f1, inv)], FakeBatchClient(responses))
    assert bids[("F0", "INV-7")] == Bid("F0", 0.11, rationale="a")
    assert bids[("F1", "INV-7")] == Bid("F1", 0.22, rationale="b")


def test_batch_missing_result_falls_back_to_truthful() -> None:
    inv = Invoice("INV-7", 50_000.0, 30)
    bids = batch_bids([(Funder("F0", 0.10), inv)], FakeBatchClient({}))
    assert bids[("F0", "INV-7")].apr == 0.10  # truthful fallback


# --- ValidatedLLMAgent (output-validation defense) ----------------------------


def test_validated_agent_passes_in_policy_bid_through() -> None:
    agent = ValidatedLLMAgent(
        Funder("F", 0.10), FakeCompletionClient('{"apr": 0.12}'), PricingPolicy("v1", 0.05, 0.30)
    )
    assert agent.bid(_ctx()).apr == 0.12  # within bounds: unchanged


def test_validated_agent_clamps_out_of_policy_bid() -> None:
    policy = PricingPolicy("v1", 0.05, 0.30)
    low = ValidatedLLMAgent(Funder("F", 0.10), FakeCompletionClient('{"apr": 0.0001}'), policy)
    assert low.bid(_ctx()).apr == 0.05  # injected near-zero bid rejected; floor used
    high = ValidatedLLMAgent(Funder("F", 0.10), FakeCompletionClient('{"apr": 0.99}'), policy)
    assert high.bid(_ctx()).apr == 0.30  # ceiling
