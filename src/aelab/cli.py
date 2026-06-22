"""Command-line interface: efficiency, truthfulness, and attacks over named scenarios.

The application/edge layer: it wires the core, agents, engine, attacks, and report into
three commands. It holds the orchestration the scripts also call, so there is one
implementation. (No `from __future__ import annotations` here: Typer resolves annotations
at runtime and that import breaks its inference.)
"""

import json
import random
from pathlib import Path
from typing import Annotated

import typer

from aelab.agents.cache import AnthropicBatchClient, BatchRequest
from aelab.agents.deterministic import TruthfulAgent
from aelab.agents.llm import (
    DEFAULT_MODEL,
    build_batch_requests,
    build_counterfactual_requests,
    counterfactual_deviations,
    decode_custom_id,
    parse_bid_apr,
)
from aelab.config import Scenario, load_scenario
from aelab.engine import run
from aelab.harness import format_table, run_harness
from aelab.metrics import deviation_stats, summarize
from aelab.models import Funder, FunderKind
from aelab.populations import (
    Population,
    generate_financiers,
    generate_invoices,
    generate_suppliers,
    make_buyer,
)
from aelab.report import plot_deviation_distribution, plot_efficiency_and_supplier_share

app = typer.Typer(help="Sealed-bid second-price reverse-auction lab.", no_args_is_help=True)

_RAW_PATH = Path("results/truthfulness_raw.json")
_COUNTERFACTUAL_RAW_PATH = Path("results/first_price_raw.json")
_EFFICIENCY_PLOT = Path("results/efficiency.png")
_TRUTHFULNESS_PLOT = Path("results/truthfulness.png")


def attack_population(scenario: Scenario) -> Population:
    """Build a population with exactly one HOUSE funder for the attack regimes."""
    suppliers = tuple(
        generate_suppliers(scenario.supplier, scenario.n_suppliers, random.Random(scenario.seed))
    )
    invoices = tuple(
        generate_invoices(scenario.invoice, scenario.n_invoices, random.Random(scenario.seed + 1))
    )
    financiers = list(
        generate_financiers(scenario.financier, scenario.n_financiers, random.Random(scenario.seed + 2))
    )
    funders = list(financiers)
    if scenario.include_buyer:
        funders.append(make_buyer(scenario.buyer_cost_apr))
    funders.append(Funder("HOUSE", scenario.house_cost_apr, FunderKind.HOUSE))
    return Population(suppliers=suppliers, funders=tuple(funders), invoices=invoices)


def run_efficiency(scenario: Scenario) -> None:
    """Controlled sweep: suppliers and invoices fixed, only the financier pool grows."""
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
    _EFFICIENCY_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plot_efficiency_and_supplier_share(counts, efficiency, supplier_share, _EFFICIENCY_PLOT)
    print(f"\nSaved plot to {_EFFICIENCY_PLOT}")


def _bid_texts(requests: list[BatchRequest], raw_path: Path) -> dict[str, str]:
    """Load cached raw bids from raw_path, or run a live batch once and persist them there."""
    if raw_path.exists():
        print(f"Loaded cached batch results from {raw_path}")
        cached: dict[str, str] = json.loads(raw_path.read_text(encoding="utf-8"))
        return cached
    print(f"Running a live Message Batch of {len(requests)} bids against {DEFAULT_MODEL}...")
    texts = AnthropicBatchClient().run(requests)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")
    print(f"Saved raw results to {raw_path}")
    return texts


def run_probe(scenario: Scenario, from_raw: Path | None) -> None:
    """Measure bid deviation from true cost over (funder, invoice) pairs via the batch path."""
    funders = generate_financiers(
        scenario.financier, scenario.probe_n_funders, random.Random(scenario.seed)
    )
    invoices = generate_invoices(
        scenario.invoice, scenario.probe_n_invoices, random.Random(scenario.seed + 1)
    )
    pairs = [(funder, invoice) for funder in funders for invoice in invoices]
    requests, index_map = build_batch_requests(pairs)
    if from_raw is not None:
        print(f"Parsing saved raw results from {from_raw} (no live batch)")
        texts: dict[str, str] = json.loads(from_raw.read_text(encoding="utf-8"))
    else:
        texts = _bid_texts(requests, _RAW_PATH)
    deviations: list[float] = []
    failures = 0
    for custom_id, text in texts.items():
        funder, _invoice = index_map[decode_custom_id(custom_id)]
        apr = parse_bid_apr(text)
        if apr is None:
            failures += 1
            continue
        deviations.append(apr - funder.true_cost_apr)
    stats = deviation_stats(deviations, scenario.epsilon)
    print()
    print(f"bids parsed:             {stats.n}")
    print(f"parse failures:          {failures}")
    print(f"missing from batch:      {len(requests) - len(texts)}")
    print(f"mean signed deviation:   {stats.mean_signed:+.5f} APR")
    print(f"mean absolute deviation: {stats.mean_absolute:.5f} APR")
    print(f"within {scenario.epsilon:.3f} of truthful:  {stats.fraction_within:.1%}")
    if deviations:
        _TRUTHFULNESS_PLOT.parent.mkdir(parents=True, exist_ok=True)
        plot_deviation_distribution(deviations, _TRUTHFULNESS_PLOT)
        print(f"\nSaved histogram to {_TRUTHFULNESS_PLOT}")


def run_counterfactual(scenario: Scenario, from_raw: Path | None) -> None:
    """Bid the same pairs under neutral second-price and first-price prompts; compare deviations.

    Same funders and invoices as the truthfulness probe, but the system prompt only states
    the payment rule and recommends no strategy. A positive mean signed deviation under first
    price (shading up) is the agent reasoning about the rule; near-zero under both is the
    agent following the rule statement, which is what the coaching probe could not rule out.
    """
    funders = generate_financiers(
        scenario.financier, scenario.probe_n_funders, random.Random(scenario.seed)
    )
    invoices = generate_invoices(
        scenario.invoice, scenario.probe_n_invoices, random.Random(scenario.seed + 1)
    )
    pairs = [(funder, invoice) for funder in funders for invoice in invoices]
    requests, index_map = build_counterfactual_requests(pairs)
    if from_raw is not None:
        print(f"Parsing saved raw results from {from_raw} (no live batch)")
        texts: dict[str, str] = json.loads(from_raw.read_text(encoding="utf-8"))
    else:
        texts = _bid_texts(requests, _COUNTERFACTUAL_RAW_PATH)
    second, first = counterfactual_deviations(texts, index_map)
    s = deviation_stats(second, scenario.epsilon)
    f = deviation_stats(first, scenario.epsilon)
    within = f"within {scenario.epsilon:.3f} of truthful"
    print()
    print(f"{'metric':28}{'second price':>14}{'first price':>14}")
    print(f"{'bids parsed':28}{s.n:>14d}{f.n:>14d}")
    print(f"{'mean signed deviation':28}{s.mean_signed:>+14.5f}{f.mean_signed:>+14.5f}")
    print(f"{'mean absolute deviation':28}{s.mean_absolute:>14.5f}{f.mean_absolute:>14.5f}")
    print(f"{within:28}{s.fraction_within:>14.1%}{f.fraction_within:>14.1%}")
    print()
    print("Shading up under first price (positive mean signed) is the agent reasoning about")
    print("the rule. Near-zero under both means it followed the prompt, not the incentive.")


def run_attacks(scenario: Scenario) -> None:
    """Run the baseline and house/collusion regimes and print the comparison table."""
    rows = run_harness(attack_population(scenario), scenario.policy, random.Random(scenario.seed + 3))
    print(format_table(rows))


@app.command()
def efficiency(scenario: str = "default") -> None:
    """Sweep the financier count; print and plot efficiency and supplier share."""
    run_efficiency(load_scenario(scenario))


@app.command()
def truthfulness(
    scenario: str = "default",
    from_raw: Annotated[Path | None, typer.Option(help="Use saved raw results; no live batch.")] = None,
) -> None:
    """Measure LLM bid deviation from true cost (the truthfulness probe)."""
    run_probe(load_scenario(scenario), from_raw)


@app.command()
def attacks(scenario: str = "default") -> None:
    """Run the baseline and house/collusion regimes; print the comparison table."""
    run_attacks(load_scenario(scenario))


@app.command()
def counterfactual(
    scenario: str = "default",
    from_raw: Annotated[Path | None, typer.Option(help="Use saved raw results; no live batch.")] = None,
) -> None:
    """First-price vs second-price under neutral prompts: do the agents shade up or echo?"""
    run_counterfactual(load_scenario(scenario), from_raw)
