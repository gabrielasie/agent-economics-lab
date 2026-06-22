"""Efficiency and supplier-share sweep. Thin wrapper over aelab.cli.run_efficiency.

Run: uv run python scripts/run_efficiency.py [--scenario NAME]   (or: uv run aelab efficiency)
"""

from __future__ import annotations

import argparse

from aelab.cli import run_efficiency
from aelab.config import load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficiency and supplier-share sweep.")
    parser.add_argument("--scenario", default="default")
    run_efficiency(load_scenario(parser.parse_args().scenario))


if __name__ == "__main__":
    main()
