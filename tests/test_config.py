"""Tests for scenario loading from TOML."""

from pathlib import Path

import pytest

from aelab.config import load_scenario

SCENARIO_TOML = """
name = "x"
seed = 11

[supplier]
comfortable_fraction = 0.4
comfortable_apr = [0.05, 0.12]
strapped_apr = [0.25, 0.60]

[financier]
cost_apr = [0.08, 0.15]

[invoice]
face_value = [10000.0, 500000.0]
days_early = [15, 90]

[counts]
suppliers = 50
invoices = 50
financiers = 5

[buyer]
include = true
cost_apr = 0.07

[house]
cost_apr = 0.09

[policy]
min_apr = 0.0
max_apr = 1.0

[efficiency]
financier_counts = [1, 2, 3]

[truthfulness]
n_funders = 6
n_invoices = 20
epsilon = 0.005
"""


def _write(tmp_path: Path, name: str, body: str) -> Path:
    (tmp_path / f"{name}.toml").write_text(body, encoding="utf-8")
    return tmp_path


def test_load_scenario_parses_fields(tmp_path: Path) -> None:
    scenario = load_scenario("s", _write(tmp_path, "s", SCENARIO_TOML))
    assert scenario.name == "s"
    assert scenario.seed == 11
    assert scenario.supplier.comfortable_apr == (0.05, 0.12)
    assert scenario.financier.cost_apr == (0.08, 0.15)
    assert scenario.invoice.days_early == (15, 90)
    assert (scenario.n_suppliers, scenario.n_invoices, scenario.n_financiers) == (50, 50, 5)
    assert scenario.include_buyer is True
    assert scenario.buyer_cost_apr == 0.07
    assert scenario.house_cost_apr == 0.09
    assert scenario.policy.clamp(2.0) == 1.0  # policy max_apr is 1.0
    assert scenario.financier_counts == (1, 2, 3)
    assert scenario.probe_n_funders == 6
    assert scenario.epsilon == 0.005


def test_load_scenario_validates_via_nested_configs(tmp_path: Path) -> None:
    bad = SCENARIO_TOML.replace("comfortable_apr = [0.05, 0.12]", "comfortable_apr = [0.12, 0.05]")
    with pytest.raises(ValueError):
        load_scenario("bad", _write(tmp_path, "bad", bad))


def test_default_scenario_file_loads() -> None:
    # The committed scenarios/default.toml parses, validates, and keeps the probe params
    # that match the saved truthfulness raw results.
    scenario = load_scenario("default")
    assert scenario.name == "default"
    assert scenario.seed == 11
    assert scenario.probe_n_funders == 6
    assert scenario.financier.cost_apr == (0.08, 0.15)
