"""Runs auctions over a population and collects per-invoice outcomes.

Pure orchestration over the core. No LLM, no edge, no plotting. The engine never
constructs a bid itself: it builds the AuctionContext each bidder may see and asks
each funder's agent to bid through the BiddingAgent seam, then clears the auction.
Scoring the outcomes is the job of metrics. (Note: this module does not import Bid,
which is structural proof it cannot build one.)
"""

from __future__ import annotations

import random
from collections.abc import Callable

from aelab.agents.base import AuctionContext, AuctionRules, BiddingAgent
from aelab.auction import clear_auction
from aelab.metrics import InvoiceOutcome
from aelab.models import Funder
from aelab.populations import Population


def run(
    population: Population,
    agent_factory: Callable[[Funder], BiddingAgent],
    rng: random.Random,
) -> list[InvoiceOutcome]:
    """Run one auction per (invoice, supplier) pair and return the outcomes.

    The i-th invoice belongs to the i-th supplier, so the population must carry the
    same number of invoices and suppliers (zip is strict). One agent is built per
    funder and reused across invoices. The engine only calls agent.bid; it never
    builds a Bid. The same rng threads through every auction, so a run is reproducible
    from its seed.
    """
    agents = [agent_factory(funder) for funder in population.funders]
    outcomes: list[InvoiceOutcome] = []
    for invoice, supplier in zip(population.invoices, population.suppliers, strict=True):
        ctx = AuctionContext(
            invoice=invoice,
            rules=AuctionRules(reserve_apr=supplier.reservation_apr),
        )  # leaked defaults to empty: a clean sealed-bid run
        bids = [agent.bid(ctx) for agent in agents]
        result = clear_auction(bids, supplier.reservation_apr, rng)
        outcomes.append(
            InvoiceOutcome(
                invoice=invoice,
                supplier=supplier,
                funders=population.funders,
                result=result,
            )
        )
    return outcomes
