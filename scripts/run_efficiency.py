"""Efficiency and supplier-share curves under truthful bidding.

A controlled sweep: the suppliers and invoices are generated once and held fixed; only
the financier pool grows across counts (the same seed makes each larger pool nest the
smaller one, so going from k to k+1 financiers adds one competitor and changes nothing
else). The supplier-share line therefore reflects competition alone, not a shifting
population. Efficiency stays allocatively perfect under truthful bidding regardless.
Deterministic: no network, no LLM. Run with: uv run python scripts/run_efficiency.py
"""

from __future__ import annotations

import random
from pathlib import Path

from aelab.agents.deterministic import TruthfulAgent
from aelab.engine import run
from aelab.metrics import summarize
from aelab.populations import (
    FinancierConfig,
    InvoiceConfig,
    Population,
    SupplierConfig,
    generate_financiers,
    generate_invoices,
    generate_suppliers,
    make_buyer,
)
from aelab.report import plot_efficiency_and_supplier_share

FINANCIER_COUNTS = [1, 2, 3, 4, 6, 8, 12]
N = 50  # suppliers and invoices, held fixed across the sweep
BUYER_COST_APR = 0.07
SUPPLIER_SEED = 1
INVOICE_SEED = 2
FINANCIER_SEED = 3
AUCTION_SEED = 4

SUPPLIER_CONFIG = SupplierConfig(
    comfortable_fraction=0.4, comfortable_apr=(0.05, 0.12), strapped_apr=(0.25, 0.60)
)
FINANCIER_CONFIG = FinancierConfig(cost_apr=(0.08, 0.15))
INVOICE_CONFIG = InvoiceConfig(face_value=(10_000.0, 500_000.0), days_early=(15, 90))


def main() -> None:
    # Generated once, identical at every financier count.
    suppliers = tuple(generate_suppliers(SUPPLIER_CONFIG, N, random.Random(SUPPLIER_SEED)))
    invoices = tuple(generate_invoices(INVOICE_CONFIG, N, random.Random(INVOICE_SEED)))

    counts: list[int] = []
    efficiency: list[float] = []
    supplier_share: list[float] = []

    print(f"{'financiers':>10}  {'efficiency':>10}  {'supplier share':>14}")
    for n in FINANCIER_COUNTS:
        # Same seed each count, so the pool nests: count n is count n-1 plus one financier.
        financiers = generate_financiers(FINANCIER_CONFIG, n, random.Random(FINANCIER_SEED))
        funders = (*financiers, make_buyer(BUYER_COST_APR))
        population = Population(suppliers=suppliers, funders=funders, invoices=invoices)
        report = summarize(run(population, TruthfulAgent, random.Random(AUCTION_SEED)))
        counts.append(n)
        efficiency.append(report.allocative_efficiency)
        supplier_share.append(report.supplier_share)
        print(f"{n:>10}  {report.allocative_efficiency:>10.3f}  {report.supplier_share:>14.3f}")

    out = Path("results/efficiency.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_efficiency_and_supplier_share(counts, efficiency, supplier_share, out)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
