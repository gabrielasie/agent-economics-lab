"""Financier collusion: a ring parks bids to lift the second price.

A ring of financiers designates its lowest-cost member to bid truthfully (and win) while
the others park just under the reserve, withholding the low bids that would otherwise set
the clearing. As the ring grows, more low bids are withheld, the second price climbs toward
the reserve, and supplier share collapses. The same fair-rate index that catches house
extraction catches this too: one defense instrument, two threats. As with the house, the
extraction is on price, not efficiency -- the lowest-cost funder still wins.

Pure core attack: imports models, agents, auction, engine, metrics, populations, and the
shared fair-rate index. No edge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from aelab.agents.deterministic import TruthfulAgent
from aelab.attacks.house_extraction import fair_rate_index
from aelab.auction import clear_auction
from aelab.engine import run
from aelab.metrics import InvoiceOutcome, summarize
from aelab.models import Bid, Funder, FunderKind
from aelab.populations import Population

EPSILON = 1e-4  # parking margin: just under the reserve


@dataclass(frozen=True)
class CollusionPoint:
    """One ring size's market metrics, indexed against the no-collusion baseline."""

    ring_size: int
    supplier_share: float
    efficiency: float
    fair_rate_index_flags: int


def _ring(population: Population, ring_size: int) -> list[Funder]:
    financiers = sorted(
        (f for f in population.funders if f.kind is FunderKind.FINANCIER),
        key=lambda f: f.true_cost_apr,
    )
    return financiers[:ring_size]


def run_collusion(
    population: Population, ring_size: int, rng: random.Random
) -> list[InvoiceOutcome]:
    """Run the auction with a ring: its lowest-cost member bids truthfully, the rest park."""
    parked = {f.party_id for f in _ring(population, ring_size)[1:]}
    outcomes: list[InvoiceOutcome] = []
    for invoice, supplier in zip(population.invoices, population.suppliers, strict=True):
        reserve = supplier.reservation_apr
        bids = [
            Bid(
                bidder_id=funder.party_id,
                # Park toward the reserve, but never below cost: a colluder withholds its
                # low bid to lift the price, it does not bid below cost and win at a loss.
                apr=max(reserve - EPSILON, funder.true_cost_apr)
                if funder.party_id in parked
                else funder.true_cost_apr,
            )
            for funder in population.funders
        ]
        result = clear_auction(bids, reserve, rng)
        outcomes.append(
            InvoiceOutcome(
                invoice=invoice, supplier=supplier, funders=population.funders, result=result
            )
        )
    return outcomes


def sweep_ring_sizes(population: Population, rng: random.Random) -> list[CollusionPoint]:
    """Sweep ring sizes 1..n_financiers, indexing each against the all-truthful baseline."""
    baseline = run(population, TruthfulAgent, rng)  # all truthful = the honest benchmark
    n_financiers = sum(1 for f in population.funders if f.kind is FunderKind.FINANCIER)
    points: list[CollusionPoint] = []
    for ring_size in range(1, n_financiers + 1):
        outcomes = run_collusion(population, ring_size, rng)
        report = summarize(outcomes)
        points.append(
            CollusionPoint(
                ring_size=ring_size,
                supplier_share=report.supplier_share,
                efficiency=report.allocative_efficiency,
                fair_rate_index_flags=fair_rate_index(outcomes, baseline),
            )
        )
    return points
