"""Tests for the economics floor: APR/discount math, surplus, first-best, efficiency.

Property tests use hypothesis because generated adversarial inputs prove the
invariants more strongly than hand-rolled grids. Surplus conservation is the
headline invariant from CLAUDE.md and is checked over arbitrary valid inputs.
"""

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from aelab.economics import (
    DAYS_PER_YEAR,
    SurplusSplit,
    allocative_efficiency,
    apr_to_discount,
    discount_to_apr,
    financing_cost,
    first_best_surplus,
    gains_from_trade,
    surplus_split,
)

aprs = st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)
days = st.integers(min_value=1, max_value=3650)
discounts = st.floats(min_value=0.0, max_value=0.9999, allow_nan=False, allow_infinity=False)
faces = st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False)


# --- APR <-> discount ---------------------------------------------------------


@given(apr=aprs, n=days)
def test_apr_discount_round_trip(apr: float, n: int) -> None:
    d = apr_to_discount(apr, n)
    assert 0.0 <= d < 1.0
    assert math.isclose(discount_to_apr(d, n), apr, rel_tol=1e-9, abs_tol=1e-12)


@given(d=discounts, n=days)
def test_discount_apr_round_trip(d: float, n: int) -> None:
    apr = discount_to_apr(d, n)
    assert apr >= 0.0
    assert math.isclose(apr_to_discount(apr, n), d, rel_tol=1e-9, abs_tol=1e-12)


@given(apr=aprs, n=days)
def test_discount_in_unit_interval(apr: float, n: int) -> None:
    assert 0.0 <= apr_to_discount(apr, n) < 1.0


@given(a=aprs, b=aprs, n=days)
def test_discount_strictly_increasing_in_apr(a: float, b: float, n: int) -> None:
    assume(abs(a - b) > 1e-6)
    lo, hi = sorted((a, b))
    assert apr_to_discount(lo, n) < apr_to_discount(hi, n)


def test_known_conversion_values() -> None:
    assert DAYS_PER_YEAR == 365
    # period = 1.0 -> d = 0.5
    assert apr_to_discount(1.0, 365) == 0.5
    assert discount_to_apr(0.5, 365) == 1.0
    # zero APR is zero discount
    assert apr_to_discount(0.0, 100) == 0.0
    assert discount_to_apr(0.0, 100) == 0.0


def test_conversion_domain_errors() -> None:
    for bad_days in (0, -5):
        with pytest.raises(ValueError):
            apr_to_discount(0.1, bad_days)
        with pytest.raises(ValueError):
            discount_to_apr(0.1, bad_days)
    with pytest.raises(ValueError):
        apr_to_discount(-0.01, 90)
    with pytest.raises(ValueError):
        discount_to_apr(-0.01, 90)
    for bad_discount in (1.0, 1.5):
        with pytest.raises(ValueError):
            discount_to_apr(bad_discount, 90)


# --- financing cost -----------------------------------------------------------


@given(f=faces, n=days)
def test_financing_cost_zero_at_zero_apr(f: float, n: int) -> None:
    assert financing_cost(f, 0.0, n) == 0.0


@given(f=faces, a=aprs, b=aprs, n=days)
def test_financing_cost_monotonic_in_apr(f: float, a: float, b: float, n: int) -> None:
    assume(abs(a - b) > 1e-6)
    lo, hi = sorted((a, b))
    assert financing_cost(f, lo, n) < financing_cost(f, hi, n)


@given(f=faces, a=aprs, n=days)
def test_financing_cost_linear_in_face(f: float, a: float, n: int) -> None:
    assert math.isclose(
        financing_cost(2.0 * f, a, n), 2.0 * financing_cost(f, a, n), rel_tol=1e-9, abs_tol=1e-9
    )


# --- surplus ------------------------------------------------------------------


def test_surplus_split_returns_dataclass() -> None:
    s = surplus_split(1000.0, 0.20, 0.15, 0.10, 90)
    assert isinstance(s, SurplusSplit)


@given(f=faces, res=aprs, clr=aprs, cost=aprs, n=days)
def test_surplus_conservation(f: float, res: float, clr: float, cost: float, n: int) -> None:
    # The CLAUDE.md invariant: supplier_surplus + winner_rent == realized gains.
    split = surplus_split(f, res, clr, cost, n)
    assert math.isclose(
        split.realized_gains, split.supplier_surplus + split.winner_rent, rel_tol=1e-9, abs_tol=1e-6
    )


@given(f=faces, res=aprs, c1=aprs, c2=aprs, cost=aprs, n=days)
def test_realized_gains_independent_of_clearing(
    f: float, res: float, c1: float, c2: float, cost: float, n: int
) -> None:
    s1 = surplus_split(f, res, c1, cost, n)
    s2 = surplus_split(f, res, c2, cost, n)
    assert math.isclose(s1.realized_gains, s2.realized_gains, rel_tol=1e-9, abs_tol=1e-6)
    expected = gains_from_trade(f, res, cost, n)
    assert math.isclose(s1.realized_gains, expected, rel_tol=1e-9, abs_tol=1e-6)


@given(f=faces, a=aprs, b=aprs, c=aprs, n=days)
def test_surplus_nonnegative_under_feasible_order(
    f: float, a: float, b: float, c: float, n: int
) -> None:
    # Feasible second-price order: winner_cost <= clearing <= reservation.
    cost, clr, res = sorted((a, b, c))
    split = surplus_split(f, res, clr, cost, n)
    assert split.supplier_surplus >= -1e-9
    assert split.winner_rent >= -1e-9
    assert split.realized_gains >= -1e-9


@given(f=faces, apr=aprs, n=days)
def test_gains_zero_when_cost_equals_reservation(f: float, apr: float, n: int) -> None:
    assert math.isclose(gains_from_trade(f, apr, apr, n), 0.0, abs_tol=1e-9)


def test_gains_negative_when_cost_above_reservation() -> None:
    assert gains_from_trade(1000.0, 0.10, 0.20, 90) < 0.0


@given(f=faces, res=aprs, cost=aprs, n=days)
def test_first_best_is_clamped_gains(f: float, res: float, cost: float, n: int) -> None:
    fb = first_best_surplus(f, res, cost, n)
    assert fb >= 0.0
    assert math.isclose(fb, max(0.0, gains_from_trade(f, res, cost, n)), rel_tol=1e-9, abs_tol=1e-6)


# --- efficiency ---------------------------------------------------------------


def test_efficiency_one_when_realized_equals_first_best() -> None:
    assert allocative_efficiency(22.9, 22.9) == 1.0


def test_efficiency_one_when_no_pie() -> None:
    assert allocative_efficiency(0.0, 0.0) == 1.0
    assert allocative_efficiency(0.0, -5.0) == 1.0


def test_efficiency_partial_when_winner_not_lowest_cost() -> None:
    f, n = 1000.0, 90
    reservation, winner_cost, lowest_cost = 0.20, 0.10, 0.05
    realized = surplus_split(f, reservation, 0.15, winner_cost, n).realized_gains
    first_best = first_best_surplus(f, reservation, lowest_cost, n)
    eff = allocative_efficiency(realized, first_best)
    assert 0.0 < eff < 1.0
    assert math.isclose(eff, realized / first_best, rel_tol=1e-12)


@given(
    fb=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_efficiency_in_unit_interval(fb: float, frac: float) -> None:
    # Feasible allocation: 0 <= realized <= first_best.
    assert 0.0 <= allocative_efficiency(fb * frac, fb) <= 1.0
