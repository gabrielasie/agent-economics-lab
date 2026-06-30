"""Tests for the communicating funder. Fully mocked: no network.

A fake complete callable stands in for the cache seam, so we can check the neutral prompts,
the wiring to the real Funder fields, the truthful fallback on a malformed completion, and
the feasibility clamp. APR is a fraction throughout.
"""

from aelab.agents.communicating import _BID_SYSTEM, _CHAT_SYSTEM, CommunicatingFunder
from aelab.comms import COMMS_AND_HISTORY, SEALED_HIDDEN, CommsConfig
from aelab.experiments.repeated import Participant, RoundContext
from aelab.models import Funder


class FakeComplete:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.text


def _ctx(reserve: float = 0.14, round_index: int = 0) -> RoundContext:
    return RoundContext(reserve_apr=reserve, round_index=round_index, leaked={})


def test_wraps_real_funder_fields() -> None:
    agent = CommunicatingFunder(Funder("F0", 0.06), FakeComplete("{}"), SEALED_HIDDEN)
    assert agent.id == "F0"
    assert agent.true_cost == 0.06


def test_satisfies_participant_protocol() -> None:
    agent = CommunicatingFunder(Funder("F0", 0.06), FakeComplete("{}"), SEALED_HIDDEN)
    assert isinstance(agent, Participant)


def test_bid_parses_decimal_apr() -> None:
    agent = CommunicatingFunder(
        Funder("F0", 0.06), FakeComplete('{"apr": 0.11, "reasoning": "x"}'), SEALED_HIDDEN
    )
    assert agent.bid(messages=[], history_lines=[], ctx=_ctx()) == 0.11


def test_bid_falls_back_to_true_cost_on_garbage() -> None:
    for bad in ["not json", "{}", '{"apr": "abc"}', '{"reasoning": "y"}']:
        agent = CommunicatingFunder(Funder("F0", 0.06), FakeComplete(bad), SEALED_HIDDEN)
        assert agent.bid(messages=[], history_lines=[], ctx=_ctx()) == 0.06


def test_bid_clamps_above_reserve() -> None:
    agent = CommunicatingFunder(
        Funder("F0", 0.06), FakeComplete('{"apr": 0.99}'), SEALED_HIDDEN
    )
    assert agent.bid(messages=[], history_lines=[], ctx=_ctx(reserve=0.14)) == 0.14


def test_bid_prompt_states_cost_and_reserve_as_fractions() -> None:
    fake = FakeComplete('{"apr": 0.06}')
    CommunicatingFunder(Funder("F0", 0.06), fake, SEALED_HIDDEN).bid(
        messages=[], history_lines=[], ctx=_ctx(reserve=0.14)
    )
    prompt, system = fake.calls[-1]
    assert system == _BID_SYSTEM
    assert "0.0600 APR" in prompt  # true cost as a fraction, not a percent
    assert "0.1400 APR" in prompt  # reserve


def test_speak_is_silent_when_comms_disabled() -> None:
    agent = CommunicatingFunder(Funder("F0", 0.06), FakeComplete("hello"), SEALED_HIDDEN)
    assert agent.speak(messages=[], history_lines=[], ctx=_ctx()) is None


def test_speak_emits_a_message_when_comms_enabled() -> None:
    fake = FakeComplete("we should all hold near the reserve")
    agent = CommunicatingFunder(Funder("F0", 0.06), fake, COMMS_AND_HISTORY)
    msg = agent.speak(messages=[], history_lines=[], ctx=_ctx(round_index=2))
    assert msg is not None
    assert msg.sender == "F0"
    assert msg.round_index == 2
    assert fake.calls[-1][1] == _CHAT_SYSTEM


def test_message_is_bounded_by_max_chars() -> None:
    long_text = "x" * 5000
    cfg = CommsConfig(enabled=True, max_chars=50)
    agent = CommunicatingFunder(Funder("F0", 0.06), FakeComplete(long_text), cfg)
    msg = agent.speak(messages=[], history_lines=[], ctx=_ctx())
    assert msg is not None
    assert len(msg.text) == 50


def test_prompts_do_not_coach_truth_or_collusion() -> None:
    banned = ["optimal", "truthful", "dominant", "shade", "collude", "collusion", "should bid"]
    for prompt in (_BID_SYSTEM, _CHAT_SYSTEM):
        low = prompt.lower()
        for word in banned:
            assert word not in low, f"prompt leaks coaching word {word!r}"
