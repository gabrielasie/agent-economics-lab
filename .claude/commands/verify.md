---
description: Run the full verification gate (lint, types, imports, tests)
---

Run these four checks in order. Report results concisely. If a check fails, stop, show the
relevant output, and propose the smallest fix. Do not report success unless all four pass.

1. `uv run ruff check .`
2. `uv run mypy`
3. `uv run lint-imports`
4. `uv run pytest`

A passing gate is the only evidence of "done". Reading the code is not evidence.
