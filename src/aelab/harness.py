"""Red-team harness: run the clean baseline and the house regimes, and tabulate.

Runs the truthful no-house baseline plus the two house regimes (separated and informed)
and formats a comparison table of efficiency, supplier share, and fair-rate-index flags.

The key read is informed vs separated, not informed vs baseline. An informed house that
withholds reverts the market to roughly the no-house baseline, so the baseline hides the
extraction: the honest (separated) house is the right benchmark, and the index flags the
gap between informed and separated.

Depends on engine and attacks. No LLM, no edge, no plotting.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from aelab.agents.deterministic import TruthfulAgent
from aelab.attacks.house_extraction import run_regimes
from aelab.engine import run
from aelab.metrics import summarize
from aelab.models import FunderKind, PricingPolicy
from aelab.populations import Population


@dataclass(frozen=True)
class HarnessRow:
    """One regime's headline metrics for the comparison table."""

    regime: str
    efficiency: float
    supplier_share: float
    fair_rate_index_flags: int


def _without_house(population: Population) -> Population:
    return Population(
        suppliers=population.suppliers,
        funders=tuple(f for f in population.funders if f.kind is not FunderKind.HOUSE),
        invoices=population.invoices,
    )


def run_harness(
    population: Population, policy: PricingPolicy, rng: random.Random
) -> list[HarnessRow]:
    """Run the truthful no-house baseline and both house regimes over the population."""
    baseline = summarize(run(_without_house(population), TruthfulAgent, rng))
    regimes = run_regimes(population, policy, rng)
    return [
        HarnessRow("baseline", baseline.allocative_efficiency, baseline.supplier_share, 0),
        HarnessRow(
            "separated",
            regimes["separated"].efficiency,
            regimes["separated"].supplier_share,
            regimes["separated"].fair_rate_index_flags,
        ),
        HarnessRow(
            "informed",
            regimes["informed"].efficiency,
            regimes["informed"].supplier_share,
            regimes["informed"].fair_rate_index_flags,
        ),
    ]


def format_table(rows: Sequence[HarnessRow]) -> str:
    """Format the harness rows as a fixed-width text table."""
    header = f"{'regime':<12}{'efficiency':>12}{'supplier_share':>16}{'index_flags':>13}"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.regime:<12}{row.efficiency:>12.3f}"
            f"{row.supplier_share:>16.3f}{row.fair_rate_index_flags:>13d}"
        )
    return "\n".join(lines)
