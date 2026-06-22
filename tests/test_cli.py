"""Tests for the CLI: the attack-population builder and the attacks command end to end."""

from typer.testing import CliRunner

from aelab.cli import app, attack_population
from aelab.config import load_scenario
from aelab.models import FunderKind

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
