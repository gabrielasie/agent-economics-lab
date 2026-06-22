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

    Bids the policy quote for the invoice (base APR plus risk loadings, clamped), ignoring
    ctx.leaked. The bid is reproducible from the policy and the invoice, and it carries the
    policy version and content hash as its rationale: the structural separation the
    house-extraction defense relies on. The bid does not depend on the funder's true cost.
    """

    party: Funder
    policy: PricingPolicy

    def bid(self, ctx: AuctionContext) -> Bid:
        return Bid(
            bidder_id=self.party.party_id,
            apr=self.policy.quote(ctx.invoice),
            rationale=f"policy {self.policy.version} {self.policy.content_hash}",
        )
