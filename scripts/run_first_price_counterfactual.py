"""First-price counterfactual. Thin wrapper over aelab.cli.run_counterfactual.

Bids the same funders and invoices under neutral second-price and first-price prompts and
reports the two (bid minus true cost) means side by side. Shading up under first price means
the agent reasons about the rule; near-zero under both means it followed the prompt. Needs a
live API key on the first run; the raw bids are saved to results/first_price_raw.json so
--from-raw replays the analysis offline.

Run: uv run python scripts/run_first_price_counterfactual.py [--scenario NAME] [--from-raw PATH]
(or: uv run aelab counterfactual --from-raw results/first_price_raw.json)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aelab.cli import run_counterfactual
from aelab.config import load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="First-price vs second-price counterfactual.")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--from-raw", type=Path, default=None)
    args = parser.parse_args()
    run_counterfactual(load_scenario(args.scenario), args.from_raw)


if __name__ == "__main__":
    main()
