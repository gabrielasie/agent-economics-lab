"""Run configuration: a frozen Scenario loaded from a TOML file.

A Scenario bundles the population distributions, counts, the buyer and house, the pricing
policy, and the per-experiment knobs and seed. load_scenario reads scenarios/<name>.toml
with tomllib, so a run is fully described by its scenario file plus the response cache.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aelab.models import PricingPolicy
from aelab.populations import FinancierConfig, InvoiceConfig, SupplierConfig


@dataclass(frozen=True)
class Scenario:
    """A complete, reproducible description of a run."""

    name: str
    seed: int
    supplier: SupplierConfig
    financier: FinancierConfig
    invoice: InvoiceConfig
    n_suppliers: int
    n_invoices: int
    n_financiers: int
    include_buyer: bool
    buyer_cost_apr: float
    house_cost_apr: float
    policy: PricingPolicy
    financier_counts: tuple[int, ...]
    probe_n_funders: int
    probe_n_invoices: int
    epsilon: float

    def __post_init__(self) -> None:
        counts = (
            self.n_suppliers,
            self.n_invoices,
            self.n_financiers,
            self.probe_n_funders,
            self.probe_n_invoices,
        )
        if any(c < 0 for c in counts):
            raise ValueError("counts must be non-negative")
        if not self.financier_counts:
            raise ValueError("financier_counts must be non-empty")


def _pair(values: Any) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _int_pair(values: Any) -> tuple[int, int]:
    return (int(values[0]), int(values[1]))


def load_scenario(name: str, scenarios_dir: Path = Path("scenarios")) -> Scenario:
    """Load scenarios/<name>.toml into a validated Scenario."""
    with (scenarios_dir / f"{name}.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return Scenario(
        name=name,
        seed=int(data["seed"]),
        supplier=SupplierConfig(
            comfortable_fraction=float(data["supplier"]["comfortable_fraction"]),
            comfortable_apr=_pair(data["supplier"]["comfortable_apr"]),
            strapped_apr=_pair(data["supplier"]["strapped_apr"]),
        ),
        financier=FinancierConfig(cost_apr=_pair(data["financier"]["cost_apr"])),
        invoice=InvoiceConfig(
            face_value=_pair(data["invoice"]["face_value"]),
            days_early=_int_pair(data["invoice"]["days_early"]),
        ),
        n_suppliers=int(data["counts"]["suppliers"]),
        n_invoices=int(data["counts"]["invoices"]),
        n_financiers=int(data["counts"]["financiers"]),
        include_buyer=bool(data["buyer"]["include"]),
        buyer_cost_apr=float(data["buyer"]["cost_apr"]),
        house_cost_apr=float(data["house"]["cost_apr"]),
        policy=PricingPolicy(
            version=f"{name}-policy",
            min_apr=float(data["policy"]["min_apr"]),
            max_apr=float(data["policy"]["max_apr"]),
        ),
        financier_counts=tuple(int(c) for c in data["efficiency"]["financier_counts"]),
        probe_n_funders=int(data["truthfulness"]["n_funders"]),
        probe_n_invoices=int(data["truthfulness"]["n_invoices"]),
        epsilon=float(data["truthfulness"]["epsilon"]),
    )
