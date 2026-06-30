"""Run the agent-communication collusion experiment live and capture it reproducibly.

This drives the same experiment the app.py "Agent communication and collusion" section runs,
but offline and to disk so the transcripts can be hand-read. It reuses app.py's exact wiring:
the default scenario's funder population, the cache-backed complete seam, the pinned model, and
the real core clearing via make_core_clear.

ROUNDS defaults to 1 so the first run is a cheap smoke test. Each round calls the model once per
funder to bid, plus once per funder to chat in the comms condition, so the model-call count is
ROUNDS * (funders * 3 + funders) at most. Raise ROUNDS only after the smoke test looks right, and
never loop a full run automatically: the API key has a small budget.

Replay vs live: a run replays from the committed .cache directory when every prompt is already
cached (no API call, no key needed). When the cache does not cover the run, it calls the model,
which needs ANTHROPIC_API_KEY exported in the shell you run this from.

    uv run python scripts/run_comms_experiment.py
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from aelab.agents.cache import AnthropicClient, ResponseCache, cache_key
from aelab.agents.communicating import _BID_SYSTEM, _CHAT_SYSTEM, CommunicatingFunder
from aelab.agents.llm import DEFAULT_MODEL, parse_bid_apr
from aelab.comms import COMMS_AND_HISTORY, HISTORY_VISIBLE, SEALED_HIDDEN, CommsConfig
from aelab.config import load_scenario
from aelab.eval import collusion_series, compare, summarise
from aelab.eval.trace import Trace
from aelab.experiments.repeated import make_core_clear, run_repeated_auction
from aelab.models import Funder
from aelab.populations import generate_financiers

# -- knobs (mirror the app.py comms section so the cache lines up) -----------------
# Rounds per condition. Default 1 keeps the first run a cheap smoke test; raise it deliberately
# (via the COMMS_ROUNDS env var) for the full run, and never loop a full run automatically.
ROUNDS = int(os.environ.get("COMMS_ROUNDS", "1"))
SCENARIO = "default"
RESERVE = 0.30  # safely above the financier cost band (0.08-0.15), so a bid always clears
# Model and pool size are overridable for robustness checks (a more capable model, a smaller
# pool where coordination is easier). Defaults match app.py: Haiku 4.5, the scenario's pool.
MODEL = os.environ.get("COMMS_MODEL", DEFAULT_MODEL)
CONDITIONS: tuple[CommsConfig, ...] = (SEALED_HIDDEN, HISTORY_VISIBLE, COMMS_AND_HISTORY)
CACHE_DIR = Path(".cache")
RUNS_DIR = Path("runs")

_SCENARIO = load_scenario(SCENARIO)
N_FUNDERS = int(os.environ.get("COMMS_N_FUNDERS", "0")) or _SCENARIO.probe_n_funders
TAG = f"{MODEL}-n{N_FUNDERS}-r{ROUNDS}"  # one output dir per configuration, so nothing overwrites
OUT_DIR = RUNS_DIR / TAG

# Haiku 4.5 list price, per million tokens, for the cost estimate (input, output).
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00


class _ReplayMiss(Exception):
    """Raised when a needed completion is absent from the cache, so a no-key run can detect that
    it has nothing to replay instead of trying to call the model."""


class _CacheOnlyClient:
    """A completion client that never reaches the network. Wrapped by ResponseCache it serves
    cache hits and raises _ReplayMiss on a miss."""

    def complete(self, model: str, system: str, user: str) -> str:
        raise _ReplayMiss


@dataclass
class _Recorder:
    """Wraps the complete seam and records every call, so the run can report billed calls, a
    token and cost estimate, and any malformed-completion fallbacks without changing behaviour."""

    base: object  # a complete(prompt, *, system) callable
    calls: list[tuple[str, str, str, bool]] | None = None  # (system, prompt, raw, was_billed)

    def __post_init__(self) -> None:
        self.calls = []

    def __call__(self, prompt: str, *, system: str | None = None) -> str:
        sys = system or ""
        billed = not (CACHE_DIR / f"{cache_key(MODEL, sys, prompt)}.txt").exists()
        raw = self.base(prompt, system=system)  # type: ignore[operator]
        assert self.calls is not None
        self.calls.append((sys, prompt, raw, billed))
        return raw


def _complete_via(cache: ResponseCache):
    """The completion seam for CommunicatingFunder: pin the model, pass the system through, and
    send the built prompt as the user message. Identical wiring to the other cache-backed agents.
    """

    def complete(prompt: str, *, system: str | None = None) -> str:
        return cache.complete(MODEL, system or "", prompt)

    return complete


def build_funders() -> list[Funder]:
    """The real funder population, built exactly as app.py builds it: fixed scenario seed.
    N_FUNDERS overrides the pool size for the small-pool robustness check. The draws are
    sequential and seeded, so a 3-funder pool is a clean prefix of the 6-funder pool."""
    return generate_financiers(_SCENARIO.financier, N_FUNDERS, random.Random(_SCENARIO.seed))


def run_conditions(core: list[Funder], complete) -> dict[str, Trace]:
    """Run the three channel conditions through the real core clear_auction via make_core_clear.
    Each condition rebuilds the CommunicatingFunder list bound to its own config so the chat step
    is active only under the comms condition."""
    traces: dict[str, Trace] = {}
    for cfg in CONDITIONS:
        funders = [CommunicatingFunder(f, complete, cfg) for f in core]
        traces[cfg.label] = run_repeated_auction(
            funders=funders,
            reserve_apr=RESERVE,
            rounds=ROUNDS,
            config=cfg,
            clear=make_core_clear(random.Random(0)),
        )
    return traces


def trace_to_dict(trace: Trace) -> dict:
    """Full, replayable record of one condition: every round, bid, message, and flag."""
    return {
        "condition": trace.condition,
        "reserve_apr": trace.reserve_apr,
        "n_rounds": trace.n_rounds,
        "rounds": [
            {
                "round_index": r.round_index,
                "true_costs": r.true_costs,
                "bids": r.bids,
                "messages": [
                    {"sender": m.sender, "round_index": m.round_index, "text": m.text}
                    for m in r.messages
                ],
                "cleared_apr": r.cleared_apr,
                "winner_id": r.winner_id,
                "baseline_apr": r.baseline_apr,
                "reserve_apr": r.reserve_apr,
                "leaked": r.leaked,
                "injection_flags": r.injection_flags,
                "clamp_events": r.clamp_events,
            }
            for r in trace.rounds
        ],
    }


def write_json(traces: dict[str, Trace], path: Path) -> None:
    payload = {
        "model": MODEL,
        "scenario": SCENARIO,
        "rounds": ROUNDS,
        "reserve_apr": RESERVE,
        "conditions": {label: trace_to_dict(tr) for label, tr in traces.items()},
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_transcripts(traces: dict[str, Trace], path: Path) -> None:
    """Human-readable dump: per condition and round, the chat and each funder's bid next to its
    true cost, so the rounds can be hand-read without any tooling."""
    lines: list[str] = []
    lines.append(f"Agent-communication experiment: model {MODEL}, scenario {SCENARIO}, "
                 f"{N_FUNDERS} funders, {ROUNDS} round(s), reserve {RESERVE:.4f} APR")
    lines.append("")
    for cfg in CONDITIONS:
        trace = traces[cfg.label]
        lines.append("=" * 78)
        lines.append(f"condition: {cfg.label}")
        if trace.rounds:
            lines.append(f"truthful baseline: {trace.rounds[0].baseline_apr:.4f} APR")
        lines.append("=" * 78)
        for r in trace.rounds:
            lines.append(f"\n--- round {r.round_index} ---")
            if r.messages:
                lines.append("messages:")
                for m in r.messages:
                    lines.append(f"  {m.sender}: {m.text}")
            else:
                lines.append("messages: (none)")
            lines.append("bids (funder: true_cost -> bid):")
            for fid in r.bids:
                cost = r.true_costs.get(fid, float("nan"))
                mark = "  <- winner" if fid == r.winner_id else ""
                lines.append(f"  {fid}: {cost:.4f} -> {r.bids[fid]:.4f}{mark}")
            lines.append(f"cleared: {r.cleared_apr:.4f} APR, winner {r.winner_id}")
            if r.injection_flags:
                lines.append(f"injection flags: {r.injection_flags}")
            if r.clamp_events:
                lines.append(f"clamp events: {r.clamp_events}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def report_stats(traces: dict[str, Trace], recorder: _Recorder, source: str) -> None:
    """Print the diagnostics that decide whether the run is healthy and what it cost."""
    assert recorder.calls is not None
    calls = recorder.calls
    billed = [c for c in calls if c[3]]
    bid_raws = [raw for sys, _p, raw, _b in calls if sys == _BID_SYSTEM]
    chat_calls = [c for c in calls if c[0] == _CHAT_SYSTEM]
    fallbacks = sum(1 for raw in bid_raws if parse_bid_apr(raw) is None)
    clamp_events = sum(len(r.clamp_events) for tr in traces.values() for r in tr.rounds)
    injection_flags = sum(len(r.injection_flags) for tr in traces.values() for r in tr.rounds)

    # Rough token estimate over billed calls only (~4 chars per token). Ignores prompt caching,
    # so it overstates input cost. Exact only in number of calls.
    in_chars = sum(len(sys) + len(prompt) for sys, prompt, _r, _b in billed)
    out_chars = sum(len(raw) for _s, _p, raw, _b in billed)
    in_tok = in_chars / 4
    out_tok = out_chars / 4
    cost = in_tok / 1e6 * PRICE_IN_PER_MTOK + out_tok / 1e6 * PRICE_OUT_PER_MTOK

    print("\n--- run diagnostics ---")
    print(f"source: {source}")
    print(f"model calls total: {len(calls)} (bids+chats); chat calls: {len(chat_calls)}")
    print(f"billed (cache-miss) calls this run: {len(billed)}")
    print(f"malformed-completion fallbacks: {fallbacks} (0 means the JSON parser handled every bid)")
    print(f"clamp events: {clamp_events}; injection flags: {injection_flags}")
    print(f"estimated billed tokens: ~{in_tok:,.0f} in, ~{out_tok:,.0f} out")
    print(f"estimated cost this run at Haiku 4.5 list price: ~${cost:.4f}")
    if ROUNDS:
        print(f"rough per-round billed cost: ~${cost / ROUNDS:.4f} "
              f"(later rounds cost a little more as history grows)")
    n = sum(1 for _ in CACHE_DIR.glob("*.txt")) if CACHE_DIR.exists() else 0
    print(f".cache directory: {'present' if CACHE_DIR.exists() else 'absent'} "
          f"under {CACHE_DIR.resolve()} ({n} cached completions)")


def main() -> None:
    core = build_funders()
    print(f"model={MODEL}  funders={N_FUNDERS}  rounds={ROUNDS}  -> {OUT_DIR}")
    print(f"funders: {[(f.party_id, round(f.true_cost_apr, 4)) for f in core]}")

    # First try a cache-only replay. It never calls the API, so a warm cache reproduces the run
    # with no key. A miss means we must call the model.
    replay = _Recorder(_complete_via(ResponseCache(_CacheOnlyClient())))
    try:
        traces = run_conditions(core, replay)
        recorder, source = replay, "cache replay (no API call)"
    except _ReplayMiss:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "\nANTHROPIC_API_KEY is not set in this shell and the response cache does not "
                "cover this run,\nso there is nothing to replay and no way to call the model. "
                "Export the key in the\nshell you run this from, then re-run. No budget was spent."
            )
            raise SystemExit(1) from None
        live = _Recorder(_complete_via(ResponseCache(AnthropicClient())))
        traces = run_conditions(core, live)
        recorder, source = live, "live model run"

    print("\n" + compare(traces))
    print()
    for cfg in CONDITIONS:
        series = [round(x, 4) for x in collusion_series(traces[cfg.label])]
        print(f"{cfg.label:<18} collusion index per round: {series}")
        _ = summarise(traces[cfg.label])  # touch the summary so a broken eval surfaces here

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(traces, OUT_DIR / "traces.json")
    write_transcripts(traces, OUT_DIR / "transcripts.txt")
    print(f"\nwrote {OUT_DIR / 'traces.json'} and {OUT_DIR / 'transcripts.txt'}")

    report_stats(traces, recorder, source)


if __name__ == "__main__":
    main()
