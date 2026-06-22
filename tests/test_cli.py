"""Tests for the CLI: the attack-population builder, the shared compute helpers, and the
attacks command end to end.
"""

import json
import random

import pytest
from typer.testing import CliRunner

from aelab.agents.llm import build_batch_requests, encode_custom_id
from aelab.cli import app, attack_population, compute_deviations, compute_efficiency
from aelab.config import load_scenario
from aelab.models import FunderKind
from aelab.populations import generate_financiers, generate_invoices

runner = CliRunner()


def test_attack_population_has_exactly_one_house() -> None:
    scenario = load_scenario("default")
    population = attack_population(scenario)
    houses = [f for f in population.funders if f.kind is FunderKind.HOUSE]
    assert len(houses) == 1
    assert houses[0].true_cost_apr == scenario.house_cost_apr
    expected = scenario.n_financiers + (1 if scenario.include_buyer else 0) + 1
    assert len(population.funders) == expected
    assert len(population.suppliers) == scenario.n_suppliers
    assert len(population.invoices) == scenario.n_invoices


def test_attacks_command_prints_all_regimes() -> None:
    result = runner.invoke(app, ["attacks", "--scenario", "default"])
    assert result.exit_code == 0
    for regime in ("baseline", "separated", "informed", "collusion"):
        assert regime in result.stdout


# --- shared compute helpers (used by the CLI and the Streamlit UI) ------------


def test_compute_efficiency_is_a_fully_efficient_sweep() -> None:
    scenario = load_scenario("default")
    counts, efficiency, supplier_share = compute_efficiency(scenario)
    assert counts == list(scenario.financier_counts)
    assert len(efficiency) == len(counts) == len(supplier_share)
    assert all(e == pytest.approx(1.0) for e in efficiency)  # truthful bidding is efficient
    assert all(0.0 <= s <= 1.0 for s in supplier_share)


def test_compute_deviations_scores_truthful_and_counts_failures() -> None:
    scenario = load_scenario("default")
    funders = generate_financiers(
        scenario.financier, scenario.probe_n_funders, random.Random(scenario.seed)
    )
    invoices = generate_invoices(
        scenario.invoice, scenario.probe_n_invoices, random.Random(scenario.seed + 1)
    )
    pairs = [(f, inv) for f in funders for inv in invoices]
    _, index_map = build_batch_requests(pairs)
    truthful_cost = index_map[0][0].true_cost_apr
    texts = {
        encode_custom_id(0): json.dumps({"apr": truthful_cost, "rationale": "x"}),
        encode_custom_id(1): "not json",
    }
    deviations, failures = compute_deviations(scenario, texts)
    assert failures == 1
    assert deviations == pytest.approx([0.0])
