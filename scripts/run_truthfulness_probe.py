"""Truthfulness probe: do LLM agents bid the dominant strategy (their true cost)?

Human subjects famously overbid in Vickrey experiments. This asks whether a Claude agent
given a known private cost bids it. It generates funders with known true costs and N
invoices, asks the model to bid every (funder, invoice) pair via the Message Batch path,
and measures the deviation bid_apr - true_cost_apr. It prints the mean signed deviation,
mean absolute deviation, and the fraction within epsilon of truthful, and saves a
histogram.

The probe parses the raw completions and counts parse failures separately; it does NOT
use batch_bids' truthful fallback, which would record a malformed bid as a perfect zero
deviation and overstate truthfulness. Custom ids are index-based, joined back to funders
through the index map. Runs live against Haiku once, then caches the raw results to
results/ so re-runs are instant and reproducible.

Run: uv run python scripts/run_truthfulness_probe.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aelab.agents.cache import AnthropicBatchClient, BatchRequest
from aelab.agents.llm import DEFAULT_MODEL, build_batch_requests, decode_custom_id, parse_bid_apr
from aelab.metrics import deviation_stats
from aelab.populations import (
    FinancierConfig,
    InvoiceConfig,
    generate_financiers,
    generate_invoices,
)
from aelab.report import plot_deviation_distribution

N_FUNDERS = 6
N_INVOICES = 20
EPSILON = 0.005  # within 50 bps of truthful
FINANCIER_SEED = 11
INVOICE_SEED = 12
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


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM truthfulness probe.")
    parser.add_argument(
        "--from-raw",
        type=Path,
        default=None,
        help="Parse stats from a saved raw-results JSON instead of running a live batch.",
    )
    args = parser.parse_args()

    funders = generate_financiers(
        FinancierConfig(cost_apr=(0.08, 0.15)), N_FUNDERS, random.Random(FINANCIER_SEED)
    )
    invoices = generate_invoices(
        InvoiceConfig(face_value=(10_000.0, 500_000.0), days_early=(15, 90)),
        N_INVOICES,
        random.Random(INVOICE_SEED),
    )
    pairs = [(funder, invoice) for funder in funders for invoice in invoices]
    requests, index_map = build_batch_requests(pairs)

    if args.from_raw is not None:
        print(f"Parsing saved raw results from {args.from_raw} (no live batch)")
        texts = json.loads(args.from_raw.read_text(encoding="utf-8"))
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
    missing = len(requests) - len(texts)

    stats = deviation_stats(deviations, EPSILON)
    print()
    print(f"bids parsed:             {stats.n}")
    print(f"parse failures:          {failures}")
    print(f"missing from batch:      {missing}")
    print(f"mean signed deviation:   {stats.mean_signed:+.5f} APR")
    print(f"mean absolute deviation: {stats.mean_absolute:.5f} APR")
    print(f"within {EPSILON:.3f} of truthful:  {stats.fraction_within:.1%}")

    if deviations:
        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plot_deviation_distribution(deviations, PLOT_PATH)
        print(f"\nSaved histogram to {PLOT_PATH}")


if __name__ == "__main__":
    main()
