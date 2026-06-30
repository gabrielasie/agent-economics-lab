"""Run the headline collusion experiment offline.

This uses deterministic rule-based stub funders, not the LLM. Its job is to validate the
trace/eval plumbing with no API key and to show the causal logic: when funders can see
history or talk, a trivial coordination rule already pushes the cleared APR above the
truthful baseline; when bids are sealed and history is hidden, the same rule cannot
coordinate and the auction clears at baseline.

Clearing routes through the real core auction, so the offline run still exercises the actual
mechanism. The real scientific result comes from swapping these stubs for
aelab.agents.communicating.CommunicatingFunder bound to a cache-backed complete, and running
the identical three conditions. The point of the stub is that the harness is correct and
reproducible before you spend a single token.

    python scripts/demo_collusion.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from aelab.comms import COMMS_AND_HISTORY, HISTORY_VISIBLE, SEALED_HIDDEN, Message
from aelab.eval import compare, transition_breakdown
from aelab.eval.harness import collusion_series
from aelab.experiments.repeated import RoundContext, make_core_clear, run_repeated_auction


@dataclass
class RuleFunder:
    """Deterministic illustration agent. Bids truthfully when it cannot observe others;
    ratchets toward the reserve when it can. Satisfies the Participant protocol, so it is a
    drop-in for CommunicatingFunder in the runner. APR is a fraction."""

    id: str
    true_cost: float

    def speak(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> Message | None:
        # Explicit coordination in the open channel. The guardrail flags this as
        # collusion_solicit, which lands it in the transition breakdown.
        return Message(
            sender=self.id,
            round_index=ctx.round_index,
            text="let's all hold our bids near the reserve so nobody undercuts",
        )

    def bid(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> float:
        if not history_lines and not messages:
            return self.true_cost  # sealed + hidden: nothing to coordinate on, truthful
        # It can see something. Walk the price up toward the reserve, floored at its own
        # cost. A real cartel does exactly this until someone defects.
        return self.true_cost + 0.85 * (ctx.reserve_apr - self.true_cost)


def main() -> None:
    funders = [
        RuleFunder("alpha", true_cost=0.060),
        RuleFunder("bravo", true_cost=0.075),
        RuleFunder("carol", true_cost=0.090),
    ]
    reserve = 0.140
    rounds = 8

    traces = {
        cfg.label: run_repeated_auction(
            funders=funders,
            reserve_apr=reserve,
            rounds=rounds,
            config=cfg,
            clear=make_core_clear(random.Random(0)),
        )
        for cfg in (SEALED_HIDDEN, HISTORY_VISIBLE, COMMS_AND_HISTORY)
    }

    print("Truthful baseline = second-lowest true cost = 0.0750 APR (7.50%)\n")
    print(compare(traces))
    print()
    for label, tr in traces.items():
        series = [round(x, 4) for x in collusion_series(tr)]
        print(f"{label:<20} collusion index per round: {series}")
    print()
    print("Transition breakdown (rounds bucketed by first failure stage):")
    for label, tr in traces.items():
        print(f"  {label:<20} {transition_breakdown(tr)}")


if __name__ == "__main__":
    main()
