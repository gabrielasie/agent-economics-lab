"""Tests for the communication primitives: the channel is typed and the labels are stable."""

from aelab.comms import (
    COMMS_AND_HISTORY,
    HISTORY_VISIBLE,
    SEALED_HIDDEN,
    CommsConfig,
    HistoryVisibility,
    Message,
    Transcript,
)


def test_config_labels_name_the_three_conditions() -> None:
    assert SEALED_HIDDEN.label == "sealed_hidden"
    assert HISTORY_VISIBLE.label == "history_outcomes"
    assert COMMS_AND_HISTORY.label == "comms_full_bids"


def test_config_defaults_are_the_causa_prima_design() -> None:
    cfg = CommsConfig()
    assert cfg.enabled is False
    assert cfg.history is HistoryVisibility.HIDDEN
    assert cfg.label == "sealed_hidden"


def test_transcript_accumulates_and_filters_by_round() -> None:
    t = Transcript()
    t.add(Message("alpha", 0, "hi"))
    t.add(Message("bravo", 1, "later"))
    assert [m.sender for m in t.for_round(0)] == ["alpha"]
    assert [m.sender for m in t.for_round(1)] == ["bravo"]


def test_visible_to_includes_up_to_current_round() -> None:
    t = Transcript()
    t.add(Message("alpha", 0, "r0"))
    t.add(Message("bravo", 1, "r1"))
    t.add(Message("carol", 2, "r2"))
    visible = t.visible_to("alpha", current_round=1)
    assert [m.text for m in visible] == ["r0", "r1"]  # round 2 not yet visible
