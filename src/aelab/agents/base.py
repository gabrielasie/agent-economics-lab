"""The BiddingAgent protocol and AuctionContext, the only seam agents reach through.

Pure core seam. No LLM, no edge, no plotting. The information an agent may see is
modeled explicitly here. Leakage attacks populate AuctionContext.leaked; nothing
reaches an agent through side channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aelab.models import Bid, Funder, Invoice


@dataclass(frozen=True)
class LeakedInfo:
    """Extra data an agent should not normally see.

    Empty in a clean sealed-bid run. This is the only channel through which an
    information-leakage attack exposes data such as sealed competitor bids.
    """

    competitor_bids: tuple[Bid, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.competitor_bids


@dataclass(frozen=True)
class AuctionRules:
    """The publicly known mechanism parameters a bidder may rely on."""

    reserve_apr: float


@dataclass(frozen=True)
class AuctionContext:
    """Exactly what a bidding agent may see for one invoice."""

    invoice: Invoice
    rules: AuctionRules
    leaked: LeakedInfo = field(default_factory=LeakedInfo)


@runtime_checkable
class BiddingAgent(Protocol):
    """The one seam: an agent exposes its party and bids given a context."""

    party: Funder

    def bid(self, ctx: AuctionContext) -> Bid: ...
