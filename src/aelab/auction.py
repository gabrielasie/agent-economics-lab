"""Sealed-bid second-price reverse auction with a reserve.

Pure core. No LLM, no edge, no plotting. Clearing happens entirely in APR space:
the winner and the clearing price depend only on the bids and the reserve, not on
the invoice size or tenor. Converting an outcome into money is the job of economics
and metrics.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from aelab.models import AuctionResult, Bid


def clear_auction(bids: Sequence[Bid], reserve_apr: float, rng: random.Random) -> AuctionResult:
    """Clear a sealed-bid second-price reverse auction.

    Eligible bids are those at or below the reserve. The lowest eligible APR wins.
    With two or more eligible bids the winner pays the second-lowest eligible APR;
    with exactly one eligible bid the winner pays the reserve; with none, no trade.

    Ties at the lowest APR are broken by rng over the tied bids sorted by bidder_id,
    so the outcome is reproducible from the seed and independent of input order.
    Bidder ids must be unique.
    """
    ids = [b.bidder_id for b in bids]
    if len(ids) != len(set(ids)):
        raise ValueError("bidder_id values must be unique")

    eligible = [b for b in bids if b.apr <= reserve_apr]
    if not eligible:
        return AuctionResult.no_trade(num_eligible=0)

    sorted_aprs = sorted(b.apr for b in eligible)
    winning_bid_apr = sorted_aprs[0]
    clearing_apr = reserve_apr if len(eligible) == 1 else sorted_aprs[1]

    tied = sorted((b for b in eligible if b.apr == winning_bid_apr), key=lambda b: b.bidder_id)
    winner = rng.choice(tied)

    return AuctionResult.cleared(
        winner_id=winner.bidder_id,
        clearing_apr=clearing_apr,
        winning_bid_apr=winning_bid_apr,
        num_eligible=len(eligible),
    )
