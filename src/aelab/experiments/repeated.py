"""Repeated reverse-auction runner.

Orchestration only. It runs N rounds of the reverse second-price auction, lets funders
optionally chat, exposes history per the visibility policy, applies the guardrails, and
emits a Trace for the eval layer. It owns the policy (who sees what) so the sealed-bid
invariant lives in one place you can audit, not scattered across agents.

The headline experiment: run the same funders and supplier under each of SEALED_HIDDEN,
HISTORY_VISIBLE, COMMS_AND_HISTORY and compare the cleared APR.

This module does not import anthropic. It calls agents, which call the injected cache.
Clearing routes through the real core auction (make_core_clear), so the experiment tests
the actual mechanism, not a reimplementation. A standalone reference rule is also provided
for running without the core.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aelab.auction import clear_auction
from aelab.comms import CommsConfig, HistoryVisibility, Message, Transcript
from aelab.eval.guardrails import clamp_apr, scan_message
from aelab.eval.trace import RoundRecord, Trace
from aelab.models import Bid


# -- interfaces the runner needs (structural; concrete types satisfy them) --
@runtime_checkable
class Participant(Protocol):
    id: str
    true_cost: float

    def speak(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> Message | None: ...

    def bid(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class RoundContext:
    reserve_apr: float
    round_index: int
    leaked: dict[str, object]


@dataclass(frozen=True, slots=True)
class ClearingResult:
    cleared_apr: float
    winner_id: str


# A clearing function takes (bids as (id, apr) pairs, reserve) and returns the outcome.
ClearFn = Callable[[list[tuple[str, float]], float], ClearingResult]


def reverse_second_price(bids: list[tuple[str, float]], reserve_apr: float) -> ClearingResult:
    """Standalone reference rule, so the runner works without the core. Lowest APR wins and
    is paid the second-lowest eligible APR (or the reserve). Ties break by input order."""
    ranked = sorted(bids, key=lambda b: b[1])
    winner_id, _winner_bid = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else reserve_apr
    return ClearingResult(cleared_apr=min(second, reserve_apr), winner_id=winner_id)


def make_core_clear(rng: random.Random) -> ClearFn:
    """Adapter over the real core clear_auction. Use this to test the actual mechanism.

    The core needs an rng for reproducible tie-breaking, so the adapter closes over one.
    A round with no eligible bid (every APR above the reserve) is an error here: the
    experiment is built so the reserve sits above every funder's cost.
    """

    def _clear(bids: list[tuple[str, float]], reserve_apr: float) -> ClearingResult:
        core_bids = [Bid(bidder_id=bidder_id, apr=apr) for bidder_id, apr in bids]
        result = clear_auction(core_bids, reserve_apr, rng)
        if not result.traded or result.clearing_apr is None or result.winner_id is None:
            raise ValueError("no eligible bid cleared; reserve is below every bid")
        return ClearingResult(cleared_apr=result.clearing_apr, winner_id=result.winner_id)

    return _clear


def _history_lines(trace: Trace, visibility: HistoryVisibility) -> list[str]:
    if visibility is HistoryVisibility.HIDDEN:
        return []
    lines: list[str] = []
    for r in trace.rounds:
        if visibility is HistoryVisibility.OUTCOMES:
            lines.append(
                f"round {r.round_index}: cleared {r.cleared_apr:.4f} APR, winner {r.winner_id}"
            )
        else:  # FULL_BIDS
            bids = ", ".join(f"{fid} {apr:.4f}" for fid, apr in r.bids.items())
            lines.append(f"round {r.round_index}: bids [{bids}], cleared {r.cleared_apr:.4f} APR")
    return lines


def run_repeated_auction(
    *,
    funders: Sequence[Participant],
    reserve_apr: float,
    rounds: int,
    config: CommsConfig,
    clear: ClearFn = reverse_second_price,
    floor_apr: float = 0.0,
) -> Trace:
    trace = Trace(condition=config.label, reserve_apr=reserve_apr)

    # Truthful baseline: what a fully competitive, truthful second-price auction clears at.
    # Computed with the same clearing function so the comparison is apples to apples. This
    # is the line the collusion index measures against.
    truthful_bids = [(f.id, f.true_cost) for f in funders]
    baseline = clear(truthful_bids, reserve_apr).cleared_apr

    for r in range(rounds):
        ctx = RoundContext(reserve_apr=reserve_apr, round_index=r, leaked={})
        history = _history_lines(trace, config.history)
        transcript = Transcript()
        injection_flags: list[str] = []

        # 1) optional chat rounds
        if config.enabled:
            for _ in range(config.rounds_of_chat):
                for f in funders:
                    msg = f.speak(
                        messages=transcript.visible_to(f.id, r),
                        history_lines=history,
                        ctx=ctx,
                    )
                    if msg and msg.text:
                        flags = scan_message(msg.text)  # guardrail
                        injection_flags.extend(f"{f.id}:{name}" for name in flags)
                        transcript.add(msg)

        # 2) sealed bids
        raw_bids: dict[str, float] = {}
        clamp_events: list[str] = []
        for f in funders:
            apr = f.bid(
                messages=transcript.visible_to(f.id, r),
                history_lines=history,
                ctx=ctx,
            )
            clamped, was_clamped = clamp_apr(apr, floor=floor_apr, reserve=reserve_apr)  # guardrail
            if was_clamped:
                clamp_events.append(f"{f.id}:{apr:.4f}->{clamped:.4f}")
            raw_bids[f.id] = clamped

        # 3) clear
        result = clear(list(raw_bids.items()), reserve_apr)

        trace.add(
            RoundRecord(
                round_index=r,
                true_costs={f.id: f.true_cost for f in funders},
                bids=raw_bids,
                messages=list(transcript.messages),
                cleared_apr=result.cleared_apr,
                winner_id=result.winner_id,
                baseline_apr=baseline,
                reserve_apr=reserve_apr,
                leaked=dict(ctx.leaked),
                injection_flags=injection_flags,
                clamp_events=clamp_events,
            )
        )

    return trace
