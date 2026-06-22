"""Efficiency and supplier-share curves under truthful bidding, from a named scenario.

A controlled sweep: the suppliers and invoices are generated once and held fixed; only the
financier pool grows across counts (the same seed nests each larger pool). The supplier-
share line isolates competition; efficiency stays 1.0 under truthful bidding. Deterministic.
Run: uv run python scripts/run_efficiency.py [--scenario NAME]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from aelab.agents.deterministic import TruthfulAgent
from aelab.config import Scenario, load_scenario
from aelab.engine import run
from aelab.metrics import summarize
from aelab.populations import (
    Population,
    generate_financiers,
    generate_invoices,
    generate_suppliers,
    make_buyer,
)
from aelab.report import plot_efficiency_and_supplier_share


def run_efficiency(scenario: Scenario) -> None:
    suppliers = tuple(
        generate_suppliers(scenario.supplier, scenario.n_suppliers, random.Random(scenario.seed))
    )
    invoices = tuple(
        generate_invoices(scenario.invoice, scenario.n_invoices, random.Random(scenario.seed + 1))
    )
    counts: list[int] = []
    efficiency: list[float] = []
    supplier_share: list[float] = []

    print(f"{'financiers':>10}  {'efficiency':>10}  {'supplier share':>14}")
    for n in scenario.financier_counts:
        financiers = generate_financiers(scenario.financier, n, random.Random(scenario.seed + 2))
        buyer = (make_buyer(scenario.buyer_cost_apr),) if scenario.include_buyer else ()
        population = Population(suppliers=suppliers, funders=(*financiers, *buyer), invoices=invoices)
        report = summarize(run(population, TruthfulAgent, random.Random(scenario.seed + 3)))
        counts.append(n)
        efficiency.append(report.allocative_efficiency)
        supplier_share.append(report.supplier_share)
        print(f"{n:>10}  {report.allocative_efficiency:>10.3f}  {report.supplier_share:>14.3f}")

    out = Path("results/efficiency.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_efficiency_and_supplier_share(counts, efficiency, supplier_share, out)
    print(f"\nSaved plot to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficiency and supplier-share sweep.")
    parser.add_argument("--scenario", default="default")
    run_efficiency(load_scenario(parser.parse_args().scenario))


if __name__ == "__main__":
    main()
