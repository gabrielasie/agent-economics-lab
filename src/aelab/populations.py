"""Seeded synthetic populations of suppliers, funders, and invoices.

Pure core. No LLM, no edge, no plotting. Every random draw goes through a single
injected random.Random, so a population is reproducible from its seed. This module
never calls the global random functions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from aelab.models import Funder, FunderKind, Invoice, Supplier


def _check_range(name: str, low: float, high: float) -> None:
    if low > high:
        raise ValueError(f"{name} low must not exceed high")


@dataclass(frozen=True)
class SupplierConfig:
    """Bimodal reservation APRs: a comfortable bank-line band and a strapped band."""

    comfortable_fraction: float
    comfortable_apr: tuple[float, float]
    strapped_apr: tuple[float, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.comfortable_fraction <= 1.0:
            raise ValueError("comfortable_fraction must be in [0, 1]")
        _check_range("comfortable_apr", *self.comfortable_apr)
        _check_range("strapped_apr", *self.strapped_apr)
        if self.comfortable_apr[0] < 0 or self.strapped_apr[0] < 0:
            raise ValueError("reservation APRs must be non-negative")


@dataclass(frozen=True)
class FinancierConfig:
    """Financier true costs of capital drawn from a competitive band."""

    cost_apr: tuple[float, float]

    def __post_init__(self) -> None:
        _check_range("cost_apr", *self.cost_apr)
        if self.cost_apr[0] < 0:
            raise ValueError("cost APRs must be non-negative")


@dataclass(frozen=True)
class InvoiceConfig:
    """Invoice face values and tenors drawn from explicit ranges."""

    face_value: tuple[float, float]
    days_early: tuple[int, int]

    def __post_init__(self) -> None:
        _check_range("face_value", *self.face_value)
        _check_range("days_early", *self.days_early)
        if self.face_value[0] <= 0:
            raise ValueError("face_value low must be positive")
        if self.days_early[0] < 1:
            raise ValueError("days_early low must be at least 1")


@dataclass(frozen=True)
class PopulationConfig:
    """The full scenario: counts, the buyer, and the per-entity distributions."""

    n_suppliers: int
    n_financiers: int
    n_invoices: int
    include_buyer: bool
    buyer_cost_apr: float
    supplier: SupplierConfig
    financier: FinancierConfig
    invoice: InvoiceConfig

    def __post_init__(self) -> None:
        if self.n_suppliers < 0 or self.n_financiers < 0 or self.n_invoices < 0:
            raise ValueError("counts must be non-negative")
        if self.buyer_cost_apr < 0:
            raise ValueError("buyer_cost_apr must be non-negative")


@dataclass(frozen=True)
class Population:
    """A generated scenario. funders is the bidder pool: financiers plus the buyer."""

    suppliers: tuple[Supplier, ...]
    funders: tuple[Funder, ...]
    invoices: tuple[Invoice, ...]


def generate_suppliers(config: SupplierConfig, n: int, rng: random.Random) -> list[Supplier]:
    """Draw n suppliers, each in the comfortable or strapped reservation band."""
    suppliers: list[Supplier] = []
    for i in range(n):
        if rng.random() < config.comfortable_fraction:
            apr = rng.uniform(*config.comfortable_apr)
        else:
            apr = rng.uniform(*config.strapped_apr)
        suppliers.append(Supplier(party_id=f"S{i}", reservation_apr=apr))
    return suppliers


def generate_financiers(config: FinancierConfig, n: int, rng: random.Random) -> list[Funder]:
    """Draw n third-party financiers with true costs in the competitive band."""
    funders: list[Funder] = []
    for i in range(n):
        cost = rng.uniform(*config.cost_apr)
        funders.append(Funder(party_id=f"F{i}", true_cost_apr=cost, kind=FunderKind.FINANCIER))
    return funders


def generate_invoices(config: InvoiceConfig, n: int, rng: random.Random) -> list[Invoice]:
    """Draw n invoices with positive face value and a tenor in days."""
    invoices: list[Invoice] = []
    for i in range(n):
        face = rng.uniform(*config.face_value)
        days = rng.randint(*config.days_early)
        invoices.append(Invoice(invoice_id=f"INV{i}", face_value=face, days_early=days))
    return invoices


def make_buyer(cost_apr: float, party_id: str = "BUYER") -> Funder:
    """Build the buyer as a self-funding Funder of kind BUYER. Deterministic, no draw."""
    return Funder(party_id=party_id, true_cost_apr=cost_apr, kind=FunderKind.BUYER)


def generate_population(config: PopulationConfig, rng: random.Random) -> Population:
    """Generate a full population, threading the one rng through every draw.

    Suppliers, then financiers, then invoices are drawn in a fixed order. The buyer
    is appended to the funder pool when requested and consumes no randomness.
    """
    suppliers = generate_suppliers(config.supplier, config.n_suppliers, rng)
    funders = generate_financiers(config.financier, config.n_financiers, rng)
    invoices = generate_invoices(config.invoice, config.n_invoices, rng)
    if config.include_buyer:
        funders.append(make_buyer(config.buyer_cost_apr))
    return Population(
        suppliers=tuple(suppliers),
        funders=tuple(funders),
        invoices=tuple(invoices),
    )
