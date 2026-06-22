"""Tests for the engine: one auction per (invoice, supplier), every bid routed through
the BiddingAgent seam. With TruthfulAgent the lowest-cost eligible funder wins and the
full market is allocatively efficient.
"""

import random

import pytest

from aelab.agents.base import AuctionContext
from aelab.agents.deterministic import TruthfulAgent
from aelab.engine import run
from aelab.metrics import summarize
from aelab.models import Bid, Funder, Invoice, Supplier
from aelab.populations import (
    FinancierConfig,
    InvoiceConfig,
    Population,
    PopulationConfig,
    SupplierConfig,
    generate_population,
)


class RecordingAgent:
    """A truthful agent that records every context it is asked to bid on."""

    def __init__(self, party: Funder) -> None:
        self.party = party
        self.contexts: list[AuctionContext] = []

    def bid(self, ctx: AuctionContext) -> Bid:
        self.contexts.append(ctx)
        return Bid(bidder_id=self.party.party_id, apr=self.party.true_cost_apr)


def _manual_population() -> Population:
    return Population(
        suppliers=(Supplier("S0", 0.30), Supplier("S1", 0.25)),
        funders=(Funder("F0", 0.10), Funder("F1", 0.15), Funder("F2", 0.20)),
        invoices=(Invoice("INV0", 100_000.0, 60), Invoice("INV1", 50_000.0, 30)),
    )


def test_one_bid_per_funder_per_invoice_with_empty_leaked() -> None:
    population = _manual_population()
    created: list[RecordingAgent] = []

    def factory(funder: Funder) -> RecordingAgent:
        agent = RecordingAgent(funder)
        created.append(agent)
        return agent

    run(population, factory, random.Random(0))

    assert len(created) == len(population.funders)  # one agent built per funder
    for agent in created:
        assert len(agent.contexts) == len(population.invoices)  # bids once per invoice
        assert all(ctx.leaked.is_empty for ctx in agent.contexts)  # clean sealed-bid run
        reserves = [ctx.rules.reserve_apr for ctx in agent.contexts]
        assert reserves == [s.reservation_apr for s in population.suppliers]


def test_lowest_cost_eligible_funder_wins() -> None:
    population = _manual_population()
    outcomes = run(population, TruthfulAgent, random.Random(0))
    assert len(outcomes) == len(population.invoices)
    for outcome in outcomes:
        assert outcome.result.traded
        eligible = [
            f for f in population.funders if f.true_cost_apr <= outcome.supplier.reservation_apr
        ]
        lowest = min(eligible, key=lambda f: f.true_cost_apr)
        assert outcome.result.winner_id == lowest.party_id


def test_full_market_efficiency_is_one_under_truthful_bidding() -> None:
    config = PopulationConfig(
        n_suppliers=20,
        n_financiers=6,
        n_invoices=20,  # one invoice per supplier
        include_buyer=True,
        buyer_cost_apr=0.07,
        supplier=SupplierConfig(
            comfortable_fraction=0.4, comfortable_apr=(0.05, 0.12), strapped_apr=(0.25, 0.60)
        ),
        financier=FinancierConfig(cost_apr=(0.08, 0.15)),
        invoice=InvoiceConfig(face_value=(10_000.0, 500_000.0), days_early=(15, 90)),
    )
    population = generate_population(config, random.Random(42))
    report = summarize(run(population, TruthfulAgent, random.Random(7)))
    assert report.allocative_efficiency == pytest.approx(1.0)


def test_reproducible_from_seed() -> None:
    population = _manual_population()
    assert run(population, TruthfulAgent, random.Random(3)) == run(
        population, TruthfulAgent, random.Random(3)
    )


def test_zip_strict_requires_matching_supplier_and_invoice_counts() -> None:
    population = Population(
        suppliers=(Supplier("S0", 0.30),),
        funders=(Funder("F0", 0.10),),
        invoices=(Invoice("INV0", 100_000.0, 60), Invoice("INV1", 50_000.0, 30)),
    )
    with pytest.raises(ValueError):
        run(population, TruthfulAgent, random.Random(0))
