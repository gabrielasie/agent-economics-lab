"""Communication primitives for agent-to-agent negotiation.

Pure data and policy. No LLM, no anthropic import, so this is safe under the core-purity
import contract. This module exists so that "agents talking to each other" is a typed,
auditable channel you can switch on and off as an experimental variable, instead of ad hoc
string passing.

In a sealed-bid mechanism an open communication channel is an attack surface (a collusion
vector), not a feature. Modelling it explicitly is the point. You turn it on to make
collusion appear, then show the mechanism suppresses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, slots=True)
class Message:
    """One natural-language message a funder agent emits in a chat round."""

    sender: str
    round_index: int
    text: str


class HistoryVisibility(Enum):
    """How much of past rounds a funder agent can see when it bids.

    This is the single most important experimental knob. The collusion literature finds
    that supracompetitive behaviour in repeated auctions diminishes when bid history is
    hidden. HIDDEN is the Causa Prima design.
    """

    HIDDEN = "hidden"  # sealed-bid, no leakage: the Causa Prima design
    OUTCOMES = "outcomes"  # cleared APR and winner id revealed each round
    FULL_BIDS = "full_bids"  # every funder's bid revealed: worst case


@dataclass(frozen=True, slots=True)
class CommsConfig:
    """Switches that define one experimental condition."""

    enabled: bool = False  # may agents exchange messages at all
    rounds_of_chat: int = 1  # messaging rounds before bids, per auction
    history: HistoryVisibility = HistoryVisibility.HIDDEN
    max_chars: int = 500  # bound each message; the channel is finite

    @property
    def label(self) -> str:
        if not self.enabled and self.history is HistoryVisibility.HIDDEN:
            return "sealed_hidden"
        if not self.enabled:
            return f"history_{self.history.value}"
        return f"comms_{self.history.value}"


# The three conditions you actually compare in the headline result.
SEALED_HIDDEN = CommsConfig(enabled=False, history=HistoryVisibility.HIDDEN)
HISTORY_VISIBLE = CommsConfig(enabled=False, history=HistoryVisibility.OUTCOMES)
COMMS_AND_HISTORY = CommsConfig(enabled=True, history=HistoryVisibility.FULL_BIDS)


@dataclass
class Transcript:
    """Accumulates every message across all rounds. Auditable after the run."""

    messages: list[Message] = field(default_factory=list)

    def add(self, m: Message) -> None:
        self.messages.append(m)

    def for_round(self, r: int) -> list[Message]:
        return [m for m in self.messages if m.round_index == r]

    def visible_to(self, funder_id: str, current_round: int) -> list[Message]:
        """All chat up to and including the current round.

        Chat is symmetric (everyone hears everyone) but bids stay sealed. That asymmetry is
        the realistic threat model: parties can post in a public forum yet still submit
        private bids.
        """
        return [m for m in self.messages if m.round_index <= current_round]
