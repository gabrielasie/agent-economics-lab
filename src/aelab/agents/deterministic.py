"""Deterministic bidding agents.

Pure core seam consumer. No LLM, no edge, no plotting. TruthfulAgent is the honest
baseline: it bids its funder's true cost of capital, the weakly dominant strategy the
auction's dominance property proved. Both agents ignore AuctionContext.leaked, so neither
exploits information a leakage attack might expose. PolicyAgent is the house funder
bidding a signed, deterministic policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from aelab.agents.base import AuctionContext
from aelab.models import Bid, Funder, PricingPolicy


@dataclass(frozen=True)
class TruthfulAgent:
    """Bids the funder's true cost of capital, regardless of the context."""

    party: Funder

    def bid(self, ctx: AuctionContext) -> Bid:
        return Bid(bidder_id=self.party.party_id, apr=self.party.true_cost_apr)


@dataclass(frozen=True)
class PolicyAgent:
    """A house funder bidding a signed, deterministic pricing policy, blind to competitors.

    Bids its true cost clamped to the policy bounds, ignoring ctx (including ctx.leaked).
    The bid is reproducible from the policy and the funder's cost: the structural
    separation the house-extraction defense relies on. (A margin-based quote is a future
    refinement; clamping the true cost is enough for an honest, bounded, no-peek bid.)
    """

    party: Funder
    policy: PricingPolicy

    def bid(self, ctx: AuctionContext) -> Bid:
        return Bid(bidder_id=self.party.party_id, apr=self.policy.clamp(self.party.true_cost_apr))
