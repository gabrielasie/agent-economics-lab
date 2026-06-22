"""Truthfulness probe: do LLM agents bid the dominant strategy (their true cost)?

Generates funders with known true costs and N invoices from a named scenario, asks the
model to bid every (funder, invoice) pair via the Message Batch path, and measures the
deviation bid_apr - true_cost_apr. Prints mean signed and absolute deviation and the
fraction within epsilon of truthful, and saves a histogram.

The probe parses raw completions and counts parse failures separately; it does NOT use
batch_bids' truthful fallback, which would record a malformed bid as a perfect zero
deviation. Runs live against Haiku once, caching the raw results so re-runs are instant.
Run: uv run python scripts/run_truthfulness_probe.py [--scenario NAME] [--from-raw PATH]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aelab.agents.cache import AnthropicBatchClient, BatchRequest
from aelab.agents.llm import DEFAULT_MODEL, build_batch_requests, decode_custom_id, parse_bid_apr
from aelab.config import Scenario, load_scenario
from aelab.metrics import deviation_stats
from aelab.populations import generate_financiers, generate_invoices
from aelab.report import plot_deviation_distribution

RAW_PATH = Path("results/truthfulness_raw.json")
PLOT_PATH = Path("results/truthfulness.png")


def _bid_texts(requests: list[BatchRequest]) -> dict[str, str]:
    """Load cached raw completions if present, else run the live batch and cache them."""
    if RAW_PATH.exists():
        print(f"Loaded cached batch results from {RAW_PATH}")
        return json.loads(RAW_PATH.read_text(encoding="utf-8"))
    print(f"Running a live Message Batch of {len(requests)} bids against {DEFAULT_MODEL}...")
    texts = AnthropicBatchClient().run(requests)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(texts, indent=2), encoding="utf-8")
    print(f"Saved raw results to {RAW_PATH}")
    return texts


def run_probe(scenario: Scenario, from_raw: Path | None) -> None:
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
        texts = json.loads(from_raw.read_text(encoding="utf-8"))
    else:
        texts = _bid_texts(requests)

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
        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plot_deviation_distribution(deviations, PLOT_PATH)
        print(f"\nSaved histogram to {PLOT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM truthfulness probe.")
    parser.add_argument("--scenario", default="default")
    parser.add_argument(
        "--from-raw",
        type=Path,
        default=None,
        help="Parse stats from a saved raw-results JSON instead of running a live batch.",
    )
    args = parser.parse_args()
    run_probe(load_scenario(args.scenario), args.from_raw)


if __name__ == "__main__":
    main()
