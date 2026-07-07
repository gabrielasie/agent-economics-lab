"""Multi-seed robustness run for one cell of the collusion grid.

Runs the open-channel condition N times with real LLM funders, one seed per run, and writes
a machine-readable results file plus a markdown summary table for RESULTS.md. Before any
live call it prints an estimated API cost and asks for confirmation. Cache hits replay for
free, so re-running a completed pass costs nothing; seed 0 with the default scenario draws
the same funder population as the committed single-seed runs.

The seed-loop and aggregation logic live in aelab.experiments.robustness (pure, covered by
tests/test_robustness.py). This script is the edge: it wires the cache-backed model client
in, which is why it lives in scripts/ and not inside the package, where the import contract
forbids experiments from reaching anthropic.

    uv run python scripts/robustness.py --model claude-sonnet-4-6 --funders 3 --seeds 5 --rounds 20
    uv run python scripts/robustness.py --stub --seeds 3 --rounds 5     # deterministic, no API cost
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from aelab.agents.cache import AnthropicClient, ResponseCache
from aelab.agents.communicating import CommunicatingFunder
from aelab.comms import Message
from aelab.config import load_scenario
from aelab.experiments.repeated import RoundContext
from aelab.experiments.robustness import markdown_table, run_cell, to_payload
from aelab.models import Funder
from aelab.populations import generate_financiers

RESERVE = 0.30  # above the financier cost band (0.08-0.15), matching the committed runs

# List prices per million tokens (input, output), checked 2026-07-07. Used only for the
# pre-run estimate; the run itself reports nothing about billing.
PRICES_PER_MTOK = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
# Rough per-call token estimate for the estimate below: the bid and chat prompts run a few
# hundred tokens and grow with history; completions are short. Deliberately conservative.
EST_TOKENS_IN_PER_CALL = 1200
EST_TOKENS_OUT_PER_CALL = 150


@dataclass
class StubFunder:
    """Deterministic stand-in for --stub runs: truthful when it can see nothing, ratchets
    toward the reserve once history or chat is visible. Zero API cost. Mirrors the stub in
    scripts/demo_collusion.py; it exercises the harness, it is not evidence about models."""

    id: str
    true_cost: float

    def speak(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> Message | None:
        return Message(self.id, ctx.round_index, "let's all hold our bids near the reserve")

    def bid(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> float:
        if not history_lines and not messages:
            return self.true_cost
        return self.true_cost + 0.85 * (ctx.reserve_apr - self.true_cost)


def estimate_cost(model: str, seeds: int, rounds: int, funders: int) -> tuple[int, float]:
    """(model calls, estimated dollars) for a fully uncached run: one chat and one bid call
    per funder per round per seed. Cache hits make the real cost at most this."""
    calls = seeds * rounds * funders * 2
    price_in, price_out = PRICES_PER_MTOK.get(model, PRICES_PER_MTOK["claude-sonnet-4-6"])
    cost = calls * (
        EST_TOKENS_IN_PER_CALL / 1e6 * price_in + EST_TOKENS_OUT_PER_CALL / 1e6 * price_out
    )
    return calls, cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed robustness run for one grid cell.")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--funders", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--stub", action="store_true", help="deterministic stub agents, no API cost")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)

    def core_funders(seed: int) -> list[Funder]:
        # Seed 0 reproduces the committed single-seed cell's population; later seeds draw
        # fresh markets from the same distribution.
        return generate_financiers(
            scenario.financier, args.funders, random.Random(scenario.seed + seed)
        )

    if args.stub:
        def participants_for(seed: int) -> list[StubFunder]:
            return [StubFunder(f.party_id, f.true_cost_apr) for f in core_funders(seed)]
        tag = "stub"
    else:
        calls, cost = estimate_cost(args.model, args.seeds, args.rounds, args.funders)
        print(
            f"Live run: {args.model}, {args.funders} funders, {args.seeds} seeds x "
            f"{args.rounds} rounds, open channel."
        )
        print(
            f"Estimated cost if nothing is cached: {calls} model calls, ~${cost:.2f} at list "
            f"price (assumes ~{EST_TOKENS_IN_PER_CALL} tokens in / ~{EST_TOKENS_OUT_PER_CALL} "
            f"out per call). Cache hits are free, so a replay costs nothing."
        )
        if not args.yes:
            answer = input("Proceed? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted. No API call was made.")
                raise SystemExit(0)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "ANTHROPIC_API_KEY is not set. The run will only succeed if the cache "
                "covers every prompt; a miss will fail when the client is first used."
            )
        cache = ResponseCache(AnthropicClient())

        def complete(prompt: str, *, system: str | None = None) -> str:
            return cache.complete(args.model, system or "", prompt)

        def participants_for(seed: int) -> list[CommunicatingFunder]:  # type: ignore[misc]
            from aelab.comms import COMMS_AND_HISTORY

            return [CommunicatingFunder(f, complete, COMMS_AND_HISTORY) for f in core_funders(seed)]
        tag = args.model

    outcomes = run_cell(
        participants_for, rounds=args.rounds, reserve_apr=RESERVE, seeds=args.seeds
    )

    out_dir = Path("runs") / f"robustness-{tag}-n{args.funders}-s{args.seeds}-r{args.rounds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = to_payload(
        model=tag,
        n_funders=args.funders,
        rounds=args.rounds,
        reserve_apr=RESERVE,
        outcomes=outcomes,
    )
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = markdown_table(
        model=tag, n_funders=args.funders, rounds=args.rounds, outcomes=outcomes
    )
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")

    print()
    print(table)
    print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'summary.md'}")
    if not args.stub:
        print("Paste summary.md into the Robustness section of RESULTS.md and remove the pending note.")


if __name__ == "__main__":
    main()
