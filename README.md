# Agent Economics Lab

A simulator and red-team harness for the sealed-bid second-price reverse auction at the
core of invoice early-payment financing. Funding sources (third-party financiers, the
buyer self-funding, the house's own book) bid an APR to pay a supplier early; the lowest
APR wins and is paid the second-lowest eligible bid, with the supplier's reservation APR
as the reserve. That rule makes truthful bidding a dominant strategy for financiers. On
top of the auction sits a battery of attacks (house-venue extraction, financier collusion,
prompt injection on the LLM bidding agents) that measure how each one moves supplier
surplus and efficiency against a clean baseline.

## How to run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync                       # install dependencies and the package
uv run aelab efficiency       # efficiency and supplier-share curves
uv run aelab attacks          # baseline vs house and collusion regimes (a table)
uv run aelab truthfulness --from-raw results/truthfulness_raw.json
```

Every command takes `--scenario NAME`, loaded from `scenarios/NAME.toml` (default:
`scenarios/default.toml`). A run is reproducible from its scenario file plus the response
cache. The deterministic commands (`efficiency`, `attacks`) run with no network. The
`truthfulness` probe calls the Anthropic API (Claude Haiku) and needs `ANTHROPIC_API_KEY`;
its raw results are cached to `results/`, so `--from-raw` re-runs the analysis offline.

Verification gate (all four must be green):

```
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest
```

## Three results

**1. Efficiency and the supplier-share curve.** Under truthful bidding the lowest-cost
funder always wins, so allocative efficiency is 1.000 at every financier count. The
controlled sweep holds the suppliers and invoices fixed and grows only the financier pool;
the supplier share of surplus rises with competition (0.897 to 0.927 in the default
scenario). Efficiency is necessary but not sufficient: a market can be fully efficient and
still split the pie against the supplier, so supplier share is a first-class metric.

**2. Do LLM agents bid the dominant strategy?** Human subjects famously overbid in Vickrey
experiments. Given a known private cost, a Claude Haiku agent bids it: across 120 bids the
mean signed deviation is about zero and 100% land within 50 bps of truthful. No systematic
shading. (The tiny residual is the prompt's four-decimal rounding of the cost, not model
error; measuring shading below that floor needs more prompt precision and adversarial
framings.)

**3. House extraction, and a fair-rate index that catches it.** A house that bids its
signed policy blind helps suppliers by adding competition. A house that peeks at sealed
competitor bids extracts by withholding: when it cannot win profitably it bids just under
the reserve, removing the low bid that would have set the clearing, so the supplier pays
more. The extraction is on price, not allocation, so the market stays 100% efficient and a
no-house baseline cannot see it (the informed regime reverts to baseline). The fair-rate
index, benchmarked against the honest (structurally separated) house, flags it. The same
index flags financier collusion (a ring parking bids to lift the second price), and a
competitive in-house buyer disciplines that ring. For the LLM agents, clamping each bid to
the signed pricing policy is an output-validation defense that provably caps any prompt
injection inside the policy bounds, whatever the model returns.

## Architecture

Dependencies point inward to a pure, fully-seeded core (`economics`, `models`, `auction`,
`metrics`, `populations`); all nondeterminism (the LLM, the network) lives behind
`agents/cache.py`, only `report.py` imports matplotlib, and these boundaries are enforced
as build-failing `import-linter` contracts, not conventions.

## Caveats and open questions

This is a toy built to pressure-test the mechanism, not a production system. The
truthfulness result is for one model under a neutral prompt. The input-separation defense
(passing the memo as delimited data) is structural here; its effect on a live model is
measured, not proven, unlike the output-validation clamp. Open questions worth a
conversation: what makes truthful threshold-setting smart for suppliers and buyers (not
just financiers), how a fair-rate index is constructed against an independent benchmark a
supplier can trust, and where a signed deterministic policy should end and an LLM's
discretion begin.
