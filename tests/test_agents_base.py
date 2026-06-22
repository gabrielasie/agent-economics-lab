"""Tests for the agent seam: the BiddingAgent protocol and AuctionContext.

The seam is the only way an agent reaches the mechanism. These tests confirm a
trivial agent conforms structurally, the context is frozen, and the sealed-bid
assumption (leaked is empty) holds unless an attack opts in.
"""

import dataclasses

import pytest

from aelab.agents.base import AuctionContext, AuctionRules, BiddingAgent, LeakedInfo
from aelab.models import Bid, Funder, Invoice


def _invoice() -> Invoice:
    return Invoice(invoice_id="INV-1", face_value=100_000.0, days_early=60)


def _ctx(leaked: LeakedInfo | None = None) -> AuctionContext:
    rules = AuctionRules(reserve_apr=0.30)
    if leaked is None:
        return AuctionContext(invoice=_invoice(), rules=rules)
    return AuctionContext(invoice=_invoice(), rules=rules, leaked=leaked)


class FakeAgent:
    """A minimal agent: holds a party and bids its true cost."""

    def __init__(self, party: Funder) -> None:
        self.party = party

    def bid(self, ctx: AuctionContext) -> Bid:
        return Bid(bidder_id=self.party.party_id, apr=self.party.true_cost_apr)


class MissingBid:
    def __init__(self) -> None:
        self.party = Funder(party_id="F1", true_cost_apr=0.10)


class MissingParty:
    def bid(self, ctx: AuctionContext) -> Bid:
        return Bid(bidder_id="x", apr=0.10)


# --- protocol conformance -----------------------------------------------------


def test_fake_agent_satisfies_protocol() -> None:
    agent = FakeAgent(Funder(party_id="F1", true_cost_apr=0.10))
    assert isinstance(agent, BiddingAgent)
    result = agent.bid(_ctx())
    assert isinstance(result, Bid)
    assert result.bidder_id == "F1"
    assert result.apr == 0.10


def test_incomplete_objects_are_not_agents() -> None:
    assert not isinstance(MissingBid(), BiddingAgent)
    assert not isinstance(MissingParty(), BiddingAgent)


# --- AuctionContext -----------------------------------------------------------


def test_auction_context_constructs_and_is_frozen() -> None:
    ctx = _ctx()
    assert ctx.invoice.invoice_id == "INV-1"
    assert ctx.rules.reserve_apr == 0.30
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.rules = AuctionRules(reserve_apr=0.10)  # type: ignore[misc]


def test_leaked_defaults_to_empty() -> None:
    ctx = _ctx()
    assert ctx.leaked.is_empty
    assert ctx.leaked.competitor_bids == ()


def test_leaked_can_be_populated_by_an_attack() -> None:
    leaked = LeakedInfo(competitor_bids=(Bid("rival", 0.09), Bid("rival2", 0.11)))
    ctx = _ctx(leaked)
    assert not ctx.leaked.is_empty
    assert len(ctx.leaked.competitor_bids) == 2
    assert ctx.leaked.competitor_bids[0].bidder_id == "rival"


# --- AuctionRules and LeakedInfo ----------------------------------------------


def test_auction_rules_frozen() -> None:
    rules = AuctionRules(reserve_apr=0.25)
    assert rules.reserve_apr == 0.25
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.reserve_apr = 0.30  # type: ignore[misc]


def test_leaked_info_frozen_and_empty_default() -> None:
    leaked = LeakedInfo()
    assert leaked.is_empty
    with pytest.raises(dataclasses.FrozenInstanceError):
        leaked.competitor_bids = (Bid("x", 0.10),)  # type: ignore[misc]
