"""APR/discount math, surplus accounting, and allocative efficiency.

Pure core. No LLM, no edge, no plotting. Every function is a deterministic
arithmetic mapping over scalars so the property tests can exercise them directly.

Surplus is measured in money as face_value * discount: the cash the supplier
gives up to finance early. Under that unit, surplus conservation is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

DAYS_PER_YEAR = 365


def apr_to_discount(apr: float, days_early: int) -> float:
    """Convert an annualized simple APR to a discount fraction d in [0, 1).

    period = apr * days_early / 365; d = period / (1 + period).
    """
    if days_early <= 0:
        raise ValueError("days_early must be positive")
    if apr < 0:
        raise ValueError("apr must be non-negative")
    period = apr * days_early / DAYS_PER_YEAR
    return period / (1.0 + period)


def discount_to_apr(discount: float, days_early: int) -> float:
    """Convert a discount fraction d in [0, 1) to an annualized simple APR.

    apr = (d / (1 - d)) * 365 / days_early. The inverse of apr_to_discount.
    """
    if days_early <= 0:
        raise ValueError("days_early must be positive")
    if not 0.0 <= discount < 1.0:
        raise ValueError("discount must be in [0, 1)")
    return (discount / (1.0 - discount)) * DAYS_PER_YEAR / days_early


def financing_cost(face_value: float, apr: float, days_early: int) -> float:
    """Money the supplier gives up to finance at this APR: face_value * d."""
    return face_value * apr_to_discount(apr, days_early)


def gains_from_trade(face_value: float, reservation_apr: float, cost_apr: float, days_early: int) -> float:
    """The pie from funding at cost_apr instead of the supplier's outside option.

    Cost of financing at the reservation APR minus the cost at cost_apr. Negative
    when cost_apr exceeds the reservation, meaning there is no positive surplus.
    """
    return financing_cost(face_value, reservation_apr, days_early) - financing_cost(
        face_value, cost_apr, days_early
    )


def first_best_surplus(
    face_value: float, reservation_apr: float, lowest_cost_apr: float, days_early: int
) -> float:
    """First-best surplus: the pie at the lowest true cost, floored at zero.

    First-best funds an invoice only when funding it creates positive surplus.
    """
    return max(0.0, gains_from_trade(face_value, reservation_apr, lowest_cost_apr, days_early))


@dataclass(frozen=True)
class SurplusSplit:
    """Realized surplus on one cleared invoice, split by who captures it."""

    supplier_surplus: float
    winner_rent: float
    realized_gains: float


def surplus_split(
    face_value: float,
    reservation_apr: float,
    clearing_apr: float,
    winner_cost_apr: float,
    days_early: int,
) -> SurplusSplit:
    """Split realized surplus into supplier surplus and winner rent.

    supplier_surplus = cost at reservation - cost at clearing.
    winner_rent      = cost at clearing    - cost at the winner's true cost.
    realized_gains   = their sum, which equals cost at reservation - cost at winner cost.

    The split holds for any inputs; the auction enforces the feasible ordering
    winner_cost <= clearing <= reservation that makes both parts non-negative.
    """
    cost_reservation = financing_cost(face_value, reservation_apr, days_early)
    cost_clearing = financing_cost(face_value, clearing_apr, days_early)
    cost_winner = financing_cost(face_value, winner_cost_apr, days_early)
    supplier_surplus = cost_reservation - cost_clearing
    winner_rent = cost_clearing - cost_winner
    return SurplusSplit(
        supplier_surplus=supplier_surplus,
        winner_rent=winner_rent,
        realized_gains=supplier_surplus + winner_rent,
    )


def allocative_efficiency(realized: float, first_best: float) -> float:
    """Realized surplus divided by first-best surplus.

    When first-best is zero or negative there is no pie to capture, so efficiency
    is defined as 1.0.
    """
    if first_best <= 0.0:
        return 1.0
    return realized / first_best
