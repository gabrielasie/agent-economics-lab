"""Tests for the market metrics: per-invoice outcomes and the aggregate report.

The load-bearing check is the supplier-share denominator: it must be realized
supplier surplus over realized total, not over first-best.
"""

import pytest

from aelab.economics import first_best_surplus, surplus_split
from aelab.metrics import InvoiceOutcome, summarize
from aelab.models import AuctionResult, Funder, Invoice, Supplier


def _invoice() -> Invoice:
    return Invoice(invoice_id="INV-1", face_value=100_000.0, days_early=60)


def test_efficiency_is_one_when_winner_is_lowest_cost() -> None:
    funders = (Funder("F0", 0.10), Funder("F1", 0.15))  # F0 is the lowest true cost
    outcome = InvoiceOutcome(
        invoice=_invoice(),
        supplier=Supplier("S", 0.30),
        funders=funders,
        result=AuctionResult.cleared(winner_id="F0", clearing_apr=0.15, winning_bid_apr=0.10, num_eligible=2),
    )
    report = summarize([outcome])
    assert report.allocative_efficiency == pytest.approx(1.0)
    assert report.efficiency_loss == pytest.approx(0.0, abs=1e-9)


def test_supplier_share_is_over_realized_not_first_best() -> None:
    # Winner F_win (cost 0.10) is NOT the lowest-cost funder (F_low at 0.05),
    # so realized < first-best and the denominator choice is unambiguous.
    funders = (Funder("F_low", 0.05), Funder("F_win", 0.10))
    outcome = InvoiceOutcome(
        invoice=_invoice(),
        supplier=Supplier("S", 0.30),
        funders=funders,
        result=AuctionResult.cleared(
            winner_id="F_win", clearing_apr=0.20, winning_bid_apr=0.10, num_eligible=2
        ),
    )
    report = summarize([outcome])

    face, days, res = 100_000.0, 60, 0.30
    first_best = first_best_surplus(face, res, 0.05, days)
    split = surplus_split(face, res, 0.20, 0.10, days)  # clearing 0.20, winner true cost 0.10

    assert report.first_best_total == pytest.approx(first_best)
    assert report.realized_total == pytest.approx(split.realized_gains)
    assert report.realized_total < report.first_best_total
    assert report.allocative_efficiency < 1.0

    assert report.supplier_share == pytest.approx(split.supplier_surplus / split.realized_gains)
    assert report.supplier_share != pytest.approx(split.supplier_surplus / first_best)


def test_no_trade_contributes_zero() -> None:
    funders = (Funder("F0", 0.40), Funder("F1", 0.50))  # all above the reservation
    outcome = InvoiceOutcome(
        invoice=_invoice(),
        supplier=Supplier("S", 0.20),
        funders=funders,
        result=AuctionResult.no_trade(num_eligible=0),
    )
    report = summarize([outcome])
    assert report.realized_total == 0.0
    assert report.supplier_surplus_total == 0.0
    assert report.supplier_share == 0.0
    assert report.first_best_total == 0.0  # lowest cost 0.40 > reservation 0.20: no positive pie
    assert report.allocative_efficiency == 1.0  # 0/0 convention
    assert report.efficiency_loss == 0.0


def test_winner_not_in_pool_raises() -> None:
    outcome = InvoiceOutcome(
        invoice=_invoice(),
        supplier=Supplier("S", 0.30),
        funders=(Funder("F0", 0.10),),
        result=AuctionResult.cleared(
            winner_id="GHOST", clearing_apr=0.30, winning_bid_apr=0.10, num_eligible=1
        ),
    )
    with pytest.raises(ValueError):
        summarize([outcome])


def test_empty_outcomes() -> None:
    report = summarize([])
    assert report.first_best_total == 0.0
    assert report.realized_total == 0.0
    assert report.supplier_surplus_total == 0.0
    assert report.allocative_efficiency == 1.0
    assert report.efficiency_loss == 0.0
    assert report.supplier_share == 0.0


def test_aggregates_across_invoices() -> None:
    funders = (Funder("F0", 0.10), Funder("F1", 0.15))
    traded = InvoiceOutcome(
        invoice=_invoice(),
        supplier=Supplier("S", 0.30),
        funders=funders,
        result=AuctionResult.cleared(winner_id="F0", clearing_apr=0.15, winning_bid_apr=0.10, num_eligible=2),
    )
    unviable = InvoiceOutcome(
        invoice=Invoice("INV-2", 50_000.0, 30),
        supplier=Supplier("S2", 0.05),
        funders=(Funder("G0", 0.40),),  # above reservation: no trade, no pie
        result=AuctionResult.no_trade(),
    )
    one = summarize([traded])
    both = summarize([traded, unviable])
    assert both.realized_total == pytest.approx(one.realized_total)
    assert both.first_best_total == pytest.approx(one.first_best_total)
    assert both.supplier_surplus_total == pytest.approx(one.supplier_surplus_total)
