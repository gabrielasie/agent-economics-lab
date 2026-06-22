"""Tests for the deterministic agents. TruthfulAgent is the honest baseline.

It bids its true cost, conforms to the BiddingAgent seam, and ignores ctx.leaked,
so it never exploits information a leakage attack might expose.
"""

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aelab.agents.base import AuctionContext, AuctionRules, BiddingAgent, LeakedInfo
from aelab.agents.deterministic import PolicyAgent, TruthfulAgent
from aelab.models import Bid, Funder, Invoice, PricingPolicy


def _ctx(
    reserve: float = 0.30, leaked: LeakedInfo | None = None, invoice_id: str = "INV-1"
) -> AuctionContext:
    invoice = Invoice(invoice_id=invoice_id, face_value=100_000.0, days_early=60)
    rules = AuctionRules(reserve_apr=reserve)
    if leaked is None:
        return AuctionContext(invoice=invoice, rules=rules)
    return AuctionContext(invoice=invoice, rules=rules, leaked=leaked)


def test_bids_exactly_true_cost() -> None:
    agent = TruthfulAgent(Funder(party_id="F1", true_cost_apr=0.123))
    bid = agent.bid(_ctx())
    assert isinstance(bid, Bid)
    assert bid.bidder_id == "F1"
    assert bid.apr == 0.123


@given(cost=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_bids_true_cost_property(cost: float) -> None:
    agent = TruthfulAgent(Funder(party_id="F", true_cost_apr=cost))
    assert agent.bid(_ctx()).apr == cost


def test_satisfies_bidding_agent_protocol() -> None:
    agent = TruthfulAgent(Funder(party_id="F1", true_cost_apr=0.10))
    assert isinstance(agent, BiddingAgent)


def test_ignores_leaked_info() -> None:
    agent = TruthfulAgent(Funder(party_id="F1", true_cost_apr=0.10))
    clean = _ctx()
    leaked = _ctx(leaked=LeakedInfo(competitor_bids=(Bid("rival", 0.04), Bid("rival2", 0.06))))
    assert agent.bid(clean) == agent.bid(leaked)


def test_bid_depends_only_on_party() -> None:
    agent = TruthfulAgent(Funder(party_id="F1", true_cost_apr=0.10))
    ctx_a = _ctx(reserve=0.20, invoice_id="A")
    ctx_b = _ctx(reserve=0.50, invoice_id="B")
    assert agent.bid(ctx_a) == agent.bid(ctx_b)


def test_truthful_agent_frozen() -> None:
    agent = TruthfulAgent(Funder(party_id="F1", true_cost_apr=0.10))
    with pytest.raises(dataclasses.FrozenInstanceError):
        agent.party = Funder(party_id="F2", true_cost_apr=0.20)  # type: ignore[misc]


# --- PolicyAgent --------------------------------------------------------------


def _policy() -> PricingPolicy:
    return PricingPolicy(version="v1", min_apr=0.05, max_apr=0.30)


def test_policy_agent_bids_clamped_true_cost() -> None:
    within = PolicyAgent(Funder("H", 0.12), _policy())
    assert within.bid(_ctx()).apr == 0.12  # within bounds: bids true cost
    below = PolicyAgent(Funder("H", 0.01), _policy())
    assert below.bid(_ctx()).apr == 0.05  # below floor -> clamped up
    above = PolicyAgent(Funder("H", 0.90), _policy())
    assert above.bid(_ctx()).apr == 0.30  # above ceiling -> clamped down


def test_policy_agent_satisfies_protocol() -> None:
    assert isinstance(PolicyAgent(Funder("H", 0.12), _policy()), BiddingAgent)


def test_policy_agent_ignores_leaked() -> None:
    agent = PolicyAgent(Funder("H", 0.12), _policy())
    leaked = _ctx(leaked=LeakedInfo(competitor_bids=(Bid("rival", 0.04),)))
    assert agent.bid(_ctx()) == agent.bid(leaked)


def test_policy_agent_frozen() -> None:
    agent = PolicyAgent(Funder("H", 0.12), _policy())
    with pytest.raises(dataclasses.FrozenInstanceError):
        agent.party = Funder("H2", 0.2)  # type: ignore[misc]
