"""Truthfulness probe. Thin wrapper over aelab.cli.run_probe.

Run: uv run python scripts/run_truthfulness_probe.py [--scenario NAME] [--from-raw PATH]
(or: uv run aelab truthfulness --from-raw results/truthfulness_raw.json)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aelab.cli import run_probe
from aelab.config import load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM truthfulness probe.")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--from-raw", type=Path, default=None)
    args = parser.parse_args()
    run_probe(load_scenario(args.scenario), args.from_raw)


if __name__ == "__main__":
    main()
