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
uv sync                                   # install dependencies and the package
uv run aelab efficiency                   # efficiency and supplier-share curve -> results/efficiency.png
uv run aelab attacks --scenario extraction  # the house-extraction harness (a table)
uv run aelab attacks                      # the default market, where competition disciplines extraction
uv run aelab truthfulness --from-raw results/truthfulness_raw.json
uv run aelab counterfactual               # first vs second price under neutral prompts (needs a key)
```

`demo.ipynb` runs all three results in sequence with context. Open it with Jupyter, or
execute it headless: `uv run jupyter nbconvert --to notebook --execute --inplace demo.ipynb`.

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
the supplier share of surplus rises with competition (0.885 to 0.917 in the default
scenario). Efficiency is necessary but not sufficient: a market can be fully efficient and
still split the pie against the supplier, so supplier share is a first-class metric. The
chart is `results/efficiency.png`.

**2. Do LLM agents bid the dominant strategy? (not yet a result)** Given a known private
cost, a Claude Haiku agent does not overbid the way humans do in Vickrey experiments: across
120 bids the mean signed deviation is about zero and 100% land within 50 bps of truthful (the
chart is `results/truthfulness.png`). This is deliberately not presented as a finding. The
probe's prompt states that truthful bidding is optimal, so the near-zero deviation measures
instruction-following, not reasoning: it is the number you would get from a model that echoed
the cost back. The test that separates the two is the first-price counterfactual (`aelab
counterfactual`): the same agent under a first-price auction with a neutral prompt, where
truthful is no longer optimal and a reasoner shades its bid up. That experiment is now built
and awaits one live run; until then the honest claim is only "the agent does not overbid."

**3. House extraction, and a fair-rate index that catches it.** Extraction is only
measurable where the house is the pivotal funder, so this runs on the `extraction` scenario:
a cheap buyer wins every invoice and the house is the price-setting second-lowest bid at its
honest cost. A house that bids that policy blind helps suppliers by adding competition
(supplier share 0.754 to 0.855). A house that peeks at sealed bids extracts by withholding:
it cannot beat the buyer, so it bids just under the reserve, removing the bid that set the
clearing, and supplier share falls back to 0.754 on all 50 invoices. The extraction is on
price, not allocation, so the market stays 100% efficient and a no-house baseline cannot see
it. The fair-rate index, benchmarked against the honest (structurally separated) house,
flags every extracted invoice; the same index flags financier collusion. The discipline that
defeats both is competition: in the `default` scenario a cheap in-house buyer and competitive
financiers sit below the house, so withholding and the ring move nothing and the index stays
quiet. For the LLM agents, clamping each bid to the signed pricing policy is an
output-validation defense that provably caps any prompt injection inside the policy bounds,
whatever the model returns.

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
