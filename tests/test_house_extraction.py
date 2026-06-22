"""Tests for the house-extraction attack.

A controlled population places the house as the price-setting second-lowest bid in the
separated regime, so its withholding in the informed regime visibly lifts the clearing.
The informed regime lowers supplier share, the fair-rate index flags it, the separated
regime holds the honest baseline, and both stay efficient (extraction is on price).
"""

import random

import pytest

from aelab.agents.base import AuctionContext, AuctionRules, LeakedInfo
from aelab.attacks.house_extraction import InformedHouseAgent, run_regimes
from aelab.models import Bid, Funder, FunderKind, Invoice, PricingPolicy, Supplier
from aelab.populations import Population

POLICY = PricingPolicy(version="v1", min_apr=0.0, max_apr=1.0)  # wide: clamp does not distort


def _leaked_ctx(reserve: float, competitor_aprs: list[float]) -> AuctionContext:
    leaked = LeakedInfo(competitor_bids=tuple(Bid(f"c{i}", a) for i, a in enumerate(competitor_aprs)))
    return AuctionContext(
        invoice=Invoice("INV", 100_000.0, 60),
        rules=AuctionRules(reserve_apr=reserve),
        leaked=leaked,
    )


def _population() -> Population:
    # House cost 0.10 sits between the lowest competitor (0.08) and the next (0.20), so the
    # honest house is the price-setting second-lowest, and withholding lifts the clearing.
    return Population(
        suppliers=(Supplier("S0", 0.40), Supplier("S1", 0.40)),
        funders=(Funder("F0", 0.08), Funder("F1", 0.20), Funder("H", 0.10, FunderKind.HOUSE)),
        invoices=(Invoice("INV0", 100_000.0, 60), Invoice("INV1", 50_000.0, 30)),
    )


# --- InformedHouseAgent behavior ----------------------------------------------


def test_informed_house_undercuts_when_it_can_win() -> None:
    agent = InformedHouseAgent(Funder("H", 0.05, FunderKind.HOUSE))
    bid = agent.bid(_leaked_ctx(0.40, [0.10, 0.20]))  # lowest competitor 0.10 > cost 0.05
    assert 0.05 < bid.apr < 0.10  # undercuts the lowest, stays above cost


def test_informed_house_withholds_when_it_cannot_win() -> None:
    agent = InformedHouseAgent(Funder("H", 0.15, FunderKind.HOUSE))
    bid = agent.bid(_leaked_ctx(0.40, [0.10, 0.20]))  # lowest competitor 0.10 < cost 0.15
    assert 0.20 < bid.apr < 0.40  # withholds high, just under the reserve


def test_informed_house_with_no_competitors_bids_cost() -> None:
    agent = InformedHouseAgent(Funder("H", 0.12, FunderKind.HOUSE))
    assert agent.bid(_leaked_ctx(0.40, [])).apr == 0.12


# --- the two regimes ----------------------------------------------------------


def test_informed_lowers_supplier_share_vs_separated() -> None:
    reports = run_regimes(_population(), POLICY, random.Random(0))
    assert reports["informed"].supplier_share < reports["separated"].supplier_share


def test_index_flags_informed_not_separated() -> None:
    reports = run_regimes(_population(), POLICY, random.Random(0))
    assert reports["informed"].fair_rate_index_flags > 0
    assert reports["separated"].fair_rate_index_flags == 0


def test_both_regimes_stay_efficient() -> None:
    reports = run_regimes(_population(), POLICY, random.Random(0))
    # The lowest-cost funder still wins under both regimes: extraction is on price, not
    # efficiency. A perfectly efficient market can still route surplus away from suppliers.
    assert reports["separated"].efficiency == pytest.approx(1.0)
    assert reports["informed"].efficiency == pytest.approx(1.0)


def test_requires_exactly_one_house() -> None:
    no_house = Population(
        suppliers=(Supplier("S0", 0.40),),
        funders=(Funder("F0", 0.08),),
        invoices=(Invoice("INV0", 100_000.0, 60),),
    )
    with pytest.raises(ValueError):
        run_regimes(no_house, POLICY, random.Random(0))
