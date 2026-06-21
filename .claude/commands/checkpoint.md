---
description: Verify, then commit only if green
---

First run the full verification gate: `uv run ruff check .`, `uv run mypy`,
`uv run lint-imports`, `uv run pytest`.

If all four pass:
- Stage everything: `git add -A`
- Commit with one concise conventional-commits message describing what changed
  (for example: `feat(auction): second-price clearing + dominance property test`).
- Show the result: `git log --oneline -1`.

If anything fails:
- Do not commit.
- Show the failure and fix it, then re-run the gate before committing.

Each checkpoint is a state you could present as finished. Keep commits small and green.
