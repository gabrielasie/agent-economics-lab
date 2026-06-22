"""Tests for the red-team harness: baseline plus the two house regimes, tabulated."""

import random

import pytest

from aelab.harness import format_table, run_harness
from aelab.models import Funder, FunderKind, Invoice, PricingPolicy, Supplier
from aelab.populations import Population

POLICY = PricingPolicy(version="v1", min_apr=0.0, max_apr=1.0)


def _population() -> Population:
    return Population(
        suppliers=(Supplier("S0", 0.40), Supplier("S1", 0.40)),
        funders=(Funder("F0", 0.08), Funder("F1", 0.20), Funder("H", 0.10, FunderKind.HOUSE)),
        invoices=(Invoice("INV0", 100_000.0, 60), Invoice("INV1", 50_000.0, 30)),
    )


def test_baseline_efficiency_is_one() -> None:
    rows = run_harness(_population(), POLICY, random.Random(0))
    baseline = next(r for r in rows if r.regime == "baseline")
    assert baseline.efficiency == pytest.approx(1.0)
    assert baseline.fair_rate_index_flags == 0


def test_rows_are_comparable() -> None:
    rows = run_harness(_population(), POLICY, random.Random(0))
    assert [r.regime for r in rows] == ["baseline", "separated", "informed", "collusion"]
    assert all(r.efficiency == pytest.approx(1.0) for r in rows)  # extraction is on price
    by = {r.regime: r for r in rows}
    # informed extracts relative to the honest house, and the index flags it
    assert by["informed"].supplier_share < by["separated"].supplier_share
    assert by["informed"].fair_rate_index_flags > 0
    # the honest house only helps the supplier; it never raises the rate above no-house
    assert by["separated"].supplier_share >= by["baseline"].supplier_share
    # the ring collapses supplier share and the index flags it too
    assert by["collusion"].supplier_share < by["baseline"].supplier_share
    assert by["collusion"].fair_rate_index_flags > 0


def test_format_table() -> None:
    rows = run_harness(_population(), POLICY, random.Random(0))
    table = format_table(rows)
    assert "regime" in table
    for regime in ("baseline", "separated", "informed", "collusion"):
        assert regime in table
    assert len(table.splitlines()) == 1 + len(rows)  # header plus one line per row
