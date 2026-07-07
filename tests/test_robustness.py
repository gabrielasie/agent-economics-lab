"""The multi-seed robustness runner, driven by deterministic stubs. Zero API cost.

Covers the aggregation logic the live pass relies on: per-seed summaries, emergence
detection on both sides of the threshold, the JSON payload, and the markdown table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aelab.comms import Message
from aelab.experiments.repeated import RoundContext
from aelab.experiments.robustness import (
    EMERGENCE_THRESHOLD,
    aggregate,
    markdown_table,
    run_cell,
    to_payload,
)


@dataclass
class TruthfulStub:
    """Always bids its cost; never coordinates."""

    id: str
    true_cost: float

    def speak(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> Message | None:
        return None

    def bid(
        self, *, messages: list[Message], history_lines: list[str], ctx: RoundContext
    ) -> float:
        return self.true_cost


@dataclass
class ColludingStub:
    """Ratchets toward the reserve once history or chat is visible."""

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


COSTS = [0.09, 0.11, 0.13]


def truthful_participants(seed: int) -> list[TruthfulStub]:
    return [TruthfulStub(f"F{i}", c + seed * 0.001) for i, c in enumerate(COSTS)]


def colluding_participants(seed: int) -> list[ColludingStub]:
    return [ColludingStub(f"F{i}", c + seed * 0.001) for i, c in enumerate(COSTS)]


def test_truthful_stubs_never_emerge() -> None:
    outcomes = run_cell(truthful_participants, rounds=5, reserve_apr=0.30, seeds=3)
    assert len(outcomes) == 3
    assert [o.seed for o in outcomes] == [0, 1, 2]
    for o in outcomes:
        assert o.mean_collusion_index == pytest.approx(0.0, abs=1e-12)
        assert not o.coordination_emerged
    agg = aggregate(outcomes)
    assert agg["emerged_count"] == 0
    assert agg["seeds"] == 3


def test_colluding_stubs_emerge_every_seed() -> None:
    outcomes = run_cell(colluding_participants, rounds=5, reserve_apr=0.30, seeds=3)
    for o in outcomes:
        assert o.mean_collusion_index > EMERGENCE_THRESHOLD
        assert o.coordination_emerged
        assert o.mean_cleared_apr > o.mean_baseline_apr
    agg = aggregate(outcomes)
    assert agg["emerged_count"] == 3
    assert agg["min_collusion_index"] > EMERGENCE_THRESHOLD


def test_seed_varies_the_market() -> None:
    outcomes = run_cell(truthful_participants, rounds=2, reserve_apr=0.30, seeds=2)
    assert outcomes[0].mean_baseline_apr != outcomes[1].mean_baseline_apr


def test_payload_and_table_render() -> None:
    outcomes = run_cell(colluding_participants, rounds=3, reserve_apr=0.30, seeds=2)
    payload = to_payload(
        model="stub", n_funders=3, rounds=3, reserve_apr=0.30, outcomes=outcomes
    )
    assert payload["model"] == "stub"
    assert payload["aggregate"]["emerged_count"] == 2  # type: ignore[index]
    assert len(payload["per_seed"]) == 2  # type: ignore[arg-type]

    table = markdown_table(model="stub", n_funders=3, rounds=3, outcomes=outcomes)
    assert "| seed |" in table
    assert "Coordination emerged in 2 of 2 seeds." in table


def test_aggregate_rejects_empty() -> None:
    with pytest.raises(ValueError):
        aggregate([])
