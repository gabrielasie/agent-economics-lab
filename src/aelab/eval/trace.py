"""The trace data model.

A trace is the complete, replayable record of one experimental run: every bid, every
message, the leaked fields, the clearing outcome, and the truthful baseline for each round.
Evaluation reads traces; it never reaches back into the live agents. This is the same "look
at the whole trace, find the first failure" discipline, specialised to an auction.

APR is stored as a fraction (0.075 is 7.5 percent APR), the same convention the core uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aelab.comms import Message


@dataclass(frozen=True, slots=True)
class RoundRecord:
    """Everything that happened in a single auction round."""

    round_index: int
    true_costs: dict[str, float]  # private values, for truthfulness scoring
    bids: dict[str, float]  # what each funder actually bid
    messages: list[Message]  # public chat this round (empty if comms off)
    cleared_apr: float  # price the winner is paid / supplier pays
    winner_id: str
    baseline_apr: float  # cleared APR if everyone had bid truthfully
    reserve_apr: float
    leaked: dict[str, object] = field(default_factory=dict)  # populated leakage, auditable
    injection_flags: list[str] = field(default_factory=list)
    clamp_events: list[str] = field(default_factory=list)


@dataclass
class Trace:
    """One condition (e.g. sealed_hidden) run over many rounds."""

    condition: str
    reserve_apr: float
    rounds: list[RoundRecord] = field(default_factory=list)

    def add(self, r: RoundRecord) -> None:
        self.rounds.append(r)

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)
