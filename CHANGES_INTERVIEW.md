# Interview polish: change log

One line per change, with the rationale. Newest phase last.

## Phase 0: audit

- Confirmed `agent-comms-eval` is fully merged into `main`; no merge needed. Baseline gate
  green: 245 tests, 3 import contracts kept, ruff and mypy clean.
- Noted the repo has no CI config; a minimal GitHub Actions workflow running the existing gate
  is added in Phase 2 so "CI green" is checkable, not just local.

## Phase 1: README overhaul

- Rewrote `README.md` to the interview skeleton: title and one-line description, the collusion
  boundary finding with exact numbers and the single-seed caveat, why it matters for a venue,
  a 5-minute reading path, a component table, a minimal quickstart, and honest limitations.
  Rationale: a product reader should land the finding in sixty seconds.
- Moved everything else (setting, architecture, economic model, mechanism, project structure,
  CLI, scenarios, cache and key details, testing gate) to `docs/LAB_GUIDE.md`, mostly verbatim.
  Rationale: keep the depth one click away instead of in front of the finding.
- Dropped the "start a cartel" framing from the old README lead. Rationale: plain and factual
  beats dramatic; the numbers carry the point.

## Phase 2: noise reduction

- Deleted `scripts/run_efficiency.py`, `scripts/run_truthfulness_probe.py`, and
  `scripts/run_first_price_counterfactual.py`. Rationale: thin wrappers duplicating the
  `aelab efficiency` / `aelab truthfulness` / `aelab counterfactual` CLI commands.
- Deleted `demo.ipynb` and its `nbconvert`/`ipykernel` dev dependencies and ruff exclusion.
  Rationale: the notebook reran the same three CLI commands the tested modules provide.
- Updated `SPEC.md` to stop referencing the deleted scripts and notebook. Rationale: keep the
  spec in sync with the code.
- Folded the app's "Prompt-injection defense" and "First-price test" tabs into one
  "Can the agents be trusted?" tab, and renamed "Incentive integrity" to "Can the venue cheat?"
  and "Agent arena" to "Run an auction". Rationale: four views total, named for a
  non-researcher; no code or tests behind the panels changed.
- Added `.github/workflows/ci.yml` running the exact local gate (ruff, mypy, lint-imports,
  pytest). Rationale: the repo had no CI; "the gate passes" should be checkable on GitHub.
