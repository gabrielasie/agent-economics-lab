"""Tests for seeded synthetic population generation.

Determinism is the headline: a population must be reproducible from its seed, with
no hidden global random state. Ranges, the funder kind mix, and config validation
are covered alongside.
"""

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aelab.models import FunderKind
from aelab.populations import (
    FinancierConfig,
    InvoiceConfig,
    PopulationConfig,
    SupplierConfig,
    generate_financiers,
    generate_invoices,
    generate_population,
    generate_suppliers,
    make_buyer,
)

seeds = st.integers(min_value=0, max_value=2**32 - 1)


def _supplier_config() -> SupplierConfig:
    return SupplierConfig(
        comfortable_fraction=0.4,
        comfortable_apr=(0.05, 0.12),
        strapped_apr=(0.25, 0.60),
    )


def _financier_config() -> FinancierConfig:
    return FinancierConfig(cost_apr=(0.08, 0.15))


def _invoice_config() -> InvoiceConfig:
    return InvoiceConfig(face_value=(10_000.0, 500_000.0), days_early=(15, 90))


def _population_config(include_buyer: bool = True) -> PopulationConfig:
    return PopulationConfig(
        n_suppliers=20,
        n_financiers=5,
        n_invoices=12,
        include_buyer=include_buyer,
        buyer_cost_apr=0.07,
        supplier=_supplier_config(),
        financier=_financier_config(),
        invoice=_invoice_config(),
    )


# --- determinism --------------------------------------------------------------


@given(seed=seeds)
def test_population_reproducible_from_seed(seed: int) -> None:
    config = _population_config()
    assert generate_population(config, random.Random(seed)) == generate_population(
        config, random.Random(seed)
    )


def test_no_hidden_global_state() -> None:
    config = _population_config()
    r1 = generate_population(config, random.Random(7))
    generate_population(config, random.Random(999))  # would perturb any global random state
    r3 = generate_population(config, random.Random(7))
    assert r1 == r3


@given(seed=seeds)
def test_each_generator_reproducible(seed: int) -> None:
    sc = _supplier_config()
    assert generate_suppliers(sc, 15, random.Random(seed)) == generate_suppliers(
        sc, 15, random.Random(seed)
    )
    fc = _financier_config()
    assert generate_financiers(fc, 6, random.Random(seed)) == generate_financiers(
        fc, 6, random.Random(seed)
    )
    ic = _invoice_config()
    assert generate_invoices(ic, 6, random.Random(seed)) == generate_invoices(
        ic, 6, random.Random(seed)
    )


def test_different_seeds_differ() -> None:
    config = _population_config()
    assert generate_population(config, random.Random(1)) != generate_population(
        config, random.Random(2)
    )


# --- field ranges -------------------------------------------------------------


@given(seed=seeds)
def test_supplier_reservations_in_bands(seed: int) -> None:
    sc = _supplier_config()
    for s in generate_suppliers(sc, 50, random.Random(seed)):
        in_comfortable = sc.comfortable_apr[0] <= s.reservation_apr <= sc.comfortable_apr[1]
        in_strapped = sc.strapped_apr[0] <= s.reservation_apr <= sc.strapped_apr[1]
        assert in_comfortable or in_strapped


@given(seed=seeds)
def test_financiers_in_band_and_kind(seed: int) -> None:
    fc = _financier_config()
    for f in generate_financiers(fc, 30, random.Random(seed)):
        assert fc.cost_apr[0] <= f.true_cost_apr <= fc.cost_apr[1]
        assert f.kind is FunderKind.FINANCIER


@given(seed=seeds)
def test_invoices_in_range(seed: int) -> None:
    ic = _invoice_config()
    for inv in generate_invoices(ic, 30, random.Random(seed)):
        assert ic.face_value[0] <= inv.face_value <= ic.face_value[1]
        assert inv.face_value > 0
        assert ic.days_early[0] <= inv.days_early <= ic.days_early[1]
        assert inv.days_early >= 1


# --- counts, ids, and kind mix ------------------------------------------------


def test_counts_and_unique_ids() -> None:
    p = generate_population(_population_config(include_buyer=True), random.Random(0))
    assert len(p.suppliers) == 20
    assert len(p.funders) == 5 + 1  # financiers + buyer
    assert len(p.invoices) == 12
    assert len({s.party_id for s in p.suppliers}) == 20
    assert len({f.party_id for f in p.funders}) == 6
    assert len({inv.invoice_id for inv in p.invoices}) == 12


def test_kind_mix_with_buyer() -> None:
    p = generate_population(_population_config(include_buyer=True), random.Random(0))
    kinds = [f.kind for f in p.funders]
    assert kinds.count(FunderKind.FINANCIER) == 5
    assert kinds.count(FunderKind.BUYER) == 1
    assert kinds.count(FunderKind.HOUSE) == 0


def test_kind_mix_without_buyer() -> None:
    p = generate_population(_population_config(include_buyer=False), random.Random(0))
    kinds = [f.kind for f in p.funders]
    assert kinds.count(FunderKind.FINANCIER) == 5
    assert kinds.count(FunderKind.BUYER) == 0
    assert len(p.funders) == 5


def test_make_buyer() -> None:
    b = make_buyer(0.07)
    assert b.kind is FunderKind.BUYER
    assert b.true_cost_apr == 0.07
    assert b.party_id == "BUYER"


# --- config validation --------------------------------------------------------


def test_supplier_config_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        SupplierConfig(comfortable_fraction=1.5, comfortable_apr=(0.05, 0.12), strapped_apr=(0.25, 0.60))


def test_supplier_config_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        SupplierConfig(comfortable_fraction=0.4, comfortable_apr=(0.12, 0.05), strapped_apr=(0.25, 0.60))


def test_supplier_config_rejects_negative_apr() -> None:
    with pytest.raises(ValueError):
        SupplierConfig(comfortable_fraction=0.4, comfortable_apr=(-0.01, 0.12), strapped_apr=(0.25, 0.60))


def test_financier_config_rejects_negative_apr() -> None:
    with pytest.raises(ValueError):
        FinancierConfig(cost_apr=(-0.01, 0.15))


def test_invoice_config_rejects_nonpositive_face() -> None:
    with pytest.raises(ValueError):
        InvoiceConfig(face_value=(0.0, 500_000.0), days_early=(15, 90))


def test_invoice_config_rejects_days_below_one() -> None:
    with pytest.raises(ValueError):
        InvoiceConfig(face_value=(10_000.0, 500_000.0), days_early=(0, 90))


def test_population_config_rejects_negative_count() -> None:
    with pytest.raises(ValueError):
        PopulationConfig(
            n_suppliers=-1,
            n_financiers=5,
            n_invoices=12,
            include_buyer=True,
            buyer_cost_apr=0.07,
            supplier=_supplier_config(),
            financier=_financier_config(),
            invoice=_invoice_config(),
        )
