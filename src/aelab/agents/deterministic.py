"""Deterministic bidding agents.

Pure core seam consumer. No LLM, no edge, no plotting. TruthfulAgent is the honest
baseline: it bids its funder's true cost of capital, the weakly dominant strategy the
auction's dominance property proved. It ignores AuctionContext.leaked, so it never
exploits information a leakage attack might expose.

PolicyAgent (the house) is deferred until PricingPolicy.quote exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from aelab.agents.base import AuctionContext
from aelab.models import Bid, Funder


@dataclass(frozen=True)
class TruthfulAgent:
    """Bids the funder's true cost of capital, regardless of the context."""

    party: Funder

    def bid(self, ctx: AuctionContext) -> Bid:
        return Bid(bidder_id=self.party.party_id, apr=self.party.true_cost_apr)
