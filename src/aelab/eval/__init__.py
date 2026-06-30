"""Evaluation layer over auction traces.

Public surface:
    Trace, RoundRecord            (trace.py)
    EvalResult, evaluate_round    (evaluators.py)
    summarise, compare,
    collusion_series,
    transition_breakdown          (harness.py)
    scan_message, clamp_apr       (guardrails.py)
"""

from aelab.eval.evaluators import EvalResult, evaluate_round
from aelab.eval.guardrails import clamp_apr, scan_message
from aelab.eval.harness import (
    TraceSummary,
    collusion_series,
    compare,
    summarise,
    transition_breakdown,
)
from aelab.eval.trace import RoundRecord, Trace

__all__ = [
    "Trace",
    "RoundRecord",
    "EvalResult",
    "evaluate_round",
    "TraceSummary",
    "summarise",
    "compare",
    "collusion_series",
    "transition_breakdown",
    "scan_message",
    "clamp_apr",
]
