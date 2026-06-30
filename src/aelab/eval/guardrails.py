"""Inline, deterministic safety checks.

Guardrails are not evaluators. Guardrails run synchronously in the request path, are fast
and deterministic, and block or sanitise before anything acts on the input. Evaluators run
after the fact and measure quality. The two checks here are guardrails:

  1. injection scanning on the public message channel, because messages are
     attacker-influenced free text that flows toward a decision function;
  2. APR clamping, so an injected or malfunctioning agent cannot return a bid outside the
     feasible range.

In the live system these would sit on every untrusted channel feeding the bid decision
(invoice text, dispute notes, counterparty messages).
"""

from __future__ import annotations

import re

# Patterns that indicate an attempt to override the agent's instructions rather than to
# communicate commercially. Deliberately conservative: this is a guardrail, so false
# positives are treated as bugs, not silently tolerated.
_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(
        r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|earlier|above|all)\b", re.I
    ),
    "role_hijack": re.compile(r"\b(you are now|act as|pretend to be|new instructions?)\b", re.I),
    "system_probe": re.compile(r"\b(system prompt|your instructions|reveal|repeat your)\b", re.I),
    "bid_coercion": re.compile(r"\b(you must|always|never)\b.{0,20}\bbid\b", re.I),
    "collusion_solicit": re.compile(
        r"\b(all|everyone|let'?s all|we should)\b.{0,30}\b(bid|raise|keep|hold)\b", re.I
    ),
}


def scan_message(text: str) -> list[str]:
    """Return the names of any injection or collusion-solicitation patterns hit.

    collusion_solicit is not an attack on the agent, it is explicit coordination in the
    open channel. Flagging it lets you separate explicit collusion (caught here) from tacit
    collusion (only visible in the cleared-APR drift the evaluators measure).
    """
    return [name for name, pat in _INJECTION_PATTERNS.items() if pat.search(text)]


def clamp_apr(apr: float, *, floor: float, reserve: float) -> tuple[float, bool]:
    """Clamp a bid to the feasible range. Returns (value, was_clamped)."""
    clamped = max(floor, min(apr, reserve))
    return clamped, clamped != apr
