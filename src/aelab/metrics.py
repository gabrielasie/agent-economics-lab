"""Per-invoice outcomes and the market-level efficiency and trust report.

Pure core. No LLM, no edge, no plotting. Depends only on models and economics, and
must not import engine. InvoiceOutcome is the shared per-invoice result type: engine
imports it from here, so it lives in metrics, not engine (which metrics may not import)
and not models (it is a scored result, not an auction primitive).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aelab.economics import allocative_efficiency, first_best_surplus, surplus_split
from aelab.models import AuctionResult, Funder, Invoice, Supplier


@dataclass(frozen=True)
class InvoiceOutcome:
    """One invoice's auction result plus the inputs needed to score it."""

    invoice: Invoice
    supplier: Supplier
    funders: tuple[Funder, ...]
    result: AuctionResult


@dataclass(frozen=True)
class MarketReport:
    """Market-level efficiency and trust metrics across many invoices."""

    first_best_total: float
    realized_total: float
    supplier_surplus_total: float
    allocative_efficiency: float
    efficiency_loss: float
    supplier_share: float


def _winner_true_cost(outcome: InvoiceOutcome) -> float:
    winner_id = outcome.result.winner_id
    for funder in outcome.funders:
        if funder.party_id == winner_id:
            return funder.true_cost_apr
    raise ValueError(f"winner {winner_id!r} is not in the funder pool")


def summarize(outcomes: Sequence[InvoiceOutcome]) -> MarketReport:
    """Aggregate per-invoice outcomes into market-level metrics.

    First-best allocates each invoice to its lowest true-cost funder. Realized surplus
    scores the actual winner. Supplier share is realized supplier surplus over realized
    total: how the realized pie is split, not how big the missing pie is.
    """
    first_best_total = 0.0
    realized_total = 0.0
    supplier_surplus_total = 0.0

    for outcome in outcomes:
        face = outcome.invoice.face_value
        days = outcome.invoice.days_early
        reservation = outcome.supplier.reservation_apr

        if outcome.funders:
            lowest_cost = min(f.true_cost_apr for f in outcome.funders)
            first_best_total += first_best_surplus(face, reservation, lowest_cost, days)

        if outcome.result.traded:
            clearing = outcome.result.clearing_apr
            assert clearing is not None  # guaranteed when traded
            winner_cost = _winner_true_cost(outcome)
            split = surplus_split(face, reservation, clearing, winner_cost, days)
            realized_total += split.realized_gains
            supplier_surplus_total += split.supplier_surplus

    efficiency = allocative_efficiency(realized_total, first_best_total)
    supplier_share = supplier_surplus_total / realized_total if realized_total > 0 else 0.0
    return MarketReport(
        first_best_total=first_best_total,
        realized_total=realized_total,
        supplier_surplus_total=supplier_surplus_total,
        allocative_efficiency=efficiency,
        efficiency_loss=1.0 - efficiency,
        supplier_share=supplier_share,
    )


@dataclass(frozen=True)
class DeviationStats:
    """How far a set of bids departs from truthful (bid minus true cost)."""

    n: int
    mean_signed: float  # > 0 means bidding above true cost on average
    mean_absolute: float
    fraction_within: float  # share of bids within epsilon of truthful


def deviation_stats(deviations: Sequence[float], epsilon: float) -> DeviationStats:
    """Summarize bid deviations from true cost. Empty input gives all-zero stats."""
    n = len(deviations)
    if n == 0:
        return DeviationStats(n=0, mean_signed=0.0, mean_absolute=0.0, fraction_within=0.0)
    mean_signed = sum(deviations) / n
    mean_absolute = sum(abs(d) for d in deviations) / n
    fraction_within = sum(1 for d in deviations if abs(d) <= epsilon) / n
    return DeviationStats(
        n=n,
        mean_signed=mean_signed,
        mean_absolute=mean_absolute,
        fraction_within=fraction_within,
    )
