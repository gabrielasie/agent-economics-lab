"""Tests for the financier-collusion attack.

A controlled population of low-cost financiers shows supplier share collapsing as the ring
grows, the fair-rate index flagging any real ring, and efficiency holding (price, not
allocation, is what's extracted).
"""

import random

import pytest

from aelab.attacks.collusion import run_collusion, sweep_ring_sizes
from aelab.models import Funder, Invoice, Supplier
from aelab.populations import Population


def _population() -> Population:
    return Population(
        suppliers=(Supplier("S0", 0.40), Supplier("S1", 0.40)),
        funders=(Funder("F0", 0.08), Funder("F1", 0.09), Funder("F2", 0.10), Funder("F3", 0.11)),
        invoices=(Invoice("INV0", 100_000.0, 60), Invoice("INV1", 50_000.0, 30)),
    )


def test_run_collusion_parks_non_winners() -> None:
    # Ring {F0, F1}: F0 (lowest) wins truthfully, F1 parks, so the clearing rises to F2.
    outcome = run_collusion(_population(), ring_size=2, rng=random.Random(0))[0]
    assert outcome.result.winner_id == "F0"
    assert outcome.result.clearing_apr == pytest.approx(0.10)


def test_supplier_share_collapses_as_ring_grows() -> None:
    shares = [p.supplier_share for p in sweep_ring_sizes(_population(), random.Random(0))]
    assert all(a >= b for a, b in zip(shares, shares[1:], strict=False))  # weakly decreasing
    assert shares[-1] < shares[0]  # the full ring is well below no collusion


def test_index_flags_collusion_not_singleton_ring() -> None:
    points = {p.ring_size: p for p in sweep_ring_sizes(_population(), random.Random(0))}
    assert points[1].fair_rate_index_flags == 0  # a ring of one is no collusion
    assert points[max(points)].fair_rate_index_flags > 0


def test_collusion_stays_efficient() -> None:
    points = sweep_ring_sizes(_population(), random.Random(0))
    assert all(p.efficiency == pytest.approx(1.0) for p in points)  # extraction is on price
