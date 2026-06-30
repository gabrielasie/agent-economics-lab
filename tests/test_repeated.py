"""The headline experiment, run offline with deterministic stub funders.

The claim under test, in one file:
  - sealed + hidden clears at the truthful baseline (second-lowest true cost);
  - the open channel clears strictly above it (tacit collusion);
  - the cheapest funder still wins every round, so efficiency stays at first-best while the
    supplier is extracted from. That is why efficiency alone is blind to extraction.

Clearing routes through the real core clear_auction via make_core_clear, so this exercises
the actual mechanism, not the standalone reference rule. APR is a fraction throughout.
"""

import random
from dataclasses import dataclass

import pytest

from aelab.comms import COMMS_AND_HISTORY, HISTORY_VISIBLE, SEALED_HIDDEN, HistoryVisibility, Message
from aelab.eval.evaluators import efficiency_rate, mean_collusion, supracompetitive_fraction
from aelab.eval.trace import Trace
from aelab.experiments.repeated import (
    RoundContext,
    _history_lines,
    make_core_clear,
    reverse_second_price,
    run_repeated_auction,
)

BASELINE = 0.075  # second-lowest true cost among the three funders below
RATCHET_CLEAR = 0.13025  # second-lowest ratcheted bid (see _funders)


@dataclass
class StubFunder:
    """Truthful when blind; ratchets toward the reserve when it can see history or chat."""

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


def _funders() -> list[StubFunder]:
    return [StubFunder("alpha", 0.060), StubFunder("bravo", 0.075), StubFunder("carol", 0.090)]


def _run(config: object) -> Trace:
    return run_repeated_auction(
        funders=_funders(),
        reserve_apr=0.140,
        rounds=4,
        config=config,  # type: ignore[arg-type]
        clear=make_core_clear(random.Random(0)),
    )


def test_sealed_hidden_clears_at_truthful_baseline() -> None:
    trace = _run(SEALED_HIDDEN)
    assert all(r.cleared_apr == pytest.approx(BASELINE) for r in trace.rounds)
    assert all(r.baseline_apr == pytest.approx(BASELINE) for r in trace.rounds)
    assert mean_collusion(trace) == pytest.approx(0.0)
    assert supracompetitive_fraction(trace) == 0.0


def test_open_channel_clears_strictly_above_baseline() -> None:
    trace = _run(COMMS_AND_HISTORY)
    assert all(r.cleared_apr == pytest.approx(RATCHET_CLEAR) for r in trace.rounds)
    assert mean_collusion(trace) > 0.05  # well above the 0.0025 tolerance
    assert supracompetitive_fraction(trace) == 1.0


def test_extraction_is_on_price_not_allocation() -> None:
    # The headline: a positive collusion index while efficiency stays at first-best.
    trace = _run(COMMS_AND_HISTORY)
    assert efficiency_rate(trace) == 1.0  # the cheapest funder still wins every round
    assert mean_collusion(trace) > 0.0  # yet the supplier pays above competitive
    assert all(r.winner_id == "alpha" for r in trace.rounds)


def test_conditions_order_sealed_below_history_below_comms() -> None:
    sealed = sum(r.cleared_apr for r in _run(SEALED_HIDDEN).rounds)
    history = sum(r.cleared_apr for r in _run(HISTORY_VISIBLE).rounds)
    comms = sum(r.cleared_apr for r in _run(COMMS_AND_HISTORY).rounds)
    assert sealed < history < comms


def test_run_is_reproducible() -> None:
    first = [r.cleared_apr for r in _run(COMMS_AND_HISTORY).rounds]
    second = [r.cleared_apr for r in _run(COMMS_AND_HISTORY).rounds]
    assert first == second


def test_hidden_history_is_empty_no_matter_how_long_the_trace() -> None:
    # The sealed-bid invariant: HIDDEN exposes nothing, so agents cannot condition on history.
    trace = _run(COMMS_AND_HISTORY)  # a populated trace
    assert trace.n_rounds == 4
    assert _history_lines(trace, HistoryVisibility.HIDDEN) == []
    assert len(_history_lines(trace, HistoryVisibility.OUTCOMES)) == 4


def test_core_adapter_matches_the_reference_rule_when_all_eligible() -> None:
    bids = [("alpha", 0.06), ("bravo", 0.075), ("carol", 0.09)]
    core = make_core_clear(random.Random(0))(bids, 0.14)
    ref = reverse_second_price(bids, 0.14)
    assert core.cleared_apr == pytest.approx(ref.cleared_apr)
    assert core.winner_id == ref.winner_id == "alpha"


def test_core_adapter_raises_when_no_bid_is_eligible() -> None:
    clear = make_core_clear(random.Random(0))
    with pytest.raises(ValueError, match="no eligible bid"):
        clear([("alpha", 0.20), ("bravo", 0.25)], 0.14)  # every bid above the reserve
