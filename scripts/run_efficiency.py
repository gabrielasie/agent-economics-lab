"""Efficiency and supplier-share curves under truthful bidding.

Sweeps the number of financiers, runs the seeded auction over a population at each count,
summarizes the market, prints a table, and saves the plot. Deterministic: no network, no
LLM. Run with: uv run python scripts/run_efficiency.py
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
    PopulationConfig,
    SupplierConfig,
    generate_population,
)
from aelab.report import plot_efficiency_and_supplier_share

FINANCIER_COUNTS = [1, 2, 3, 4, 6, 8, 12]
SEED = 42


def _config(n_financiers: int) -> PopulationConfig:
    return PopulationConfig(
        n_suppliers=50,
        n_financiers=n_financiers,
        n_invoices=50,  # equal to suppliers: one invoice per supplier
        include_buyer=True,
        buyer_cost_apr=0.07,
        supplier=SupplierConfig(
            comfortable_fraction=0.4, comfortable_apr=(0.05, 0.12), strapped_apr=(0.25, 0.60)
        ),
        financier=FinancierConfig(cost_apr=(0.08, 0.15)),
        invoice=InvoiceConfig(face_value=(10_000.0, 500_000.0), days_early=(15, 90)),
    )


def main() -> None:
    counts: list[int] = []
    efficiency: list[float] = []
    supplier_share: list[float] = []

    print(f"{'financiers':>10}  {'efficiency':>10}  {'supplier share':>14}")
    for n in FINANCIER_COUNTS:
        population = generate_population(_config(n), random.Random(SEED))
        report = summarize(run(population, TruthfulAgent, random.Random(SEED)))
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
