# Lab guide

The reference material behind the [README](../README.md): the setting, the architecture, how
to run everything (including live LLM runs), the economic model, the mechanism, the CLI, and
the reproducibility story. The findings themselves are in [RESULTS.md](../RESULTS.md) and
[memo/MEMO.md](../memo/MEMO.md).

## The setting

Can the operator of an invoice early-payment auction quietly extract from suppliers, and would
LLM agents bidding in that auction exploit the same opening? The mechanism under study is a
sealed-bid second-price reverse auction with competing funders: funding sources bid an APR to
pay a supplier early, the lowest APR wins and is paid the second-lowest eligible bid, and the
supplier's reservation APR is the reserve. This repo builds that mechanism, a deterministic
adversary that probes it, and a live experiment that puts real LLM agents in it. All APR
figures use the fractional-APR convention.

This is the same class of question studied by Fish, Shorrer, and Gonczarowski on algorithmic
and LLM collusion. Rather than asserting the effect is model- and prompt-dependent, the grid
in RESULTS.md observes that dependence directly: same mechanism, same prompts, collusion
present in one cell and absent in the others. It is not a replication of their work, and it is
not a claim that agents cannot collude.

## Architecture

A pure, deterministic core, with all nondeterminism (the LLM, the network) quarantined at the
outer edge. Dependencies point inward.

```
cli / scripts / app.py        entry points; app.py is the Streamlit UI, an edge outside the package
report                        presentation; the only matplotlib importer
harness   experiments         orchestration; attack regimes, and the repeated communication auction
engine    eval                engine runs one scenario; eval scores traces (binary evaluators, guardrails)
attacks   agents/llm  agents/cache  agents/communicating    edge; the only anthropic importers
          agents/deterministic  agents/base                 the BiddingAgent seam
========= pure, deterministic: no network, no LLM, no matplotlib =========
metrics   populations   auction   economics   models   comms
```

Three properties, checkable in a minute:

- **The core is isolated from the LLM edge, and it is enforced.** Three `import-linter`
  contracts fail the build on violation: the pure core (`economics, models, auction, metrics,
  populations`) may not import `anthropic`, `matplotlib`, or any edge, experiment, or eval
  module; only `report` imports `matplotlib`; only `agents/cache` and `agents/llm` import
  `anthropic`, and lazily, so the lab imports and runs without an API key. `comms.py` is pure
  data and policy (the typed communication channel), so it carries no LLM dependency either.
- **Record and replay reproducibility, via a content-addressed cache.** Every model call goes
  through `agents/cache.py`, keyed by a sha256 of model, system, and user. A run is
  reproducible from its seed plus the cache: a cache hit replays a stored completion with no
  network and no key, and a miss calls the model once and stores the result. Every stochastic
  draw threads a single injected `random.Random`; the code never touches the global `random`.
- **An eval layer with binary evaluators and guardrails.** Evaluators (`eval/evaluators.py`)
  return a pass or fail plus the number behind it, built from the failure modes that matter
  here (untruthful bidding, collusion, inefficiency, integrity), not generic quality scores.
  Guardrails (`eval/guardrails.py`) run inline in the request path: an injection and collusion
  scanner on the public channel, and an APR clamp so a malfunctioning or injected agent cannot
  bid outside the feasible range. The repeated runner (`experiments/repeated.py`) clears
  through the real core auction, so the experiment tests the actual mechanism, not a
  reimplementation.

Two seams carry everything. Agents reach the mechanism only through the `BiddingAgent`
protocol and the `AuctionContext` value, and what an agent may see is modeled explicitly, so a
leakage attack populates `AuctionContext.leaked` rather than passing a side channel.

## How to run

Setup: `uv sync`; add the UI with `uv sync --extra ui`.

**The deterministic demos run with no API key and no network.** They show the structural
possibility and the metrics that catch it:

```
uv run python scripts/demo_collusion.py            # stub funders under three channel conditions
uv run aelab attacks --scenario extraction         # the extraction attack; the index flags it
uv run aelab attacks                               # the competitive control, where the same attack moves nothing
uv run aelab efficiency                            # efficiency and supplier-share curve
```

**The live experiment needs a key.** It replays from the committed cache when one is present,
so a cache hit costs nothing; a miss calls the model and needs `ANTHROPIC_API_KEY` in the
shell.

```
# replays the committed run offline if the cache covers it; otherwise calls the model
uv run python scripts/run_comms_experiment.py

# scale the run; ROUNDS defaults to 1 (a cheap smoke test)
COMMS_ROUNDS=20 uv run python scripts/run_comms_experiment.py

# vary the model and pool size (the grid in RESULTS.md)
COMMS_MODEL=claude-sonnet-4-6 COMMS_N_FUNDERS=3 COMMS_ROUNDS=20 \
  uv run python scripts/run_comms_experiment.py

# multi-seed robustness for one cell; prints a cost estimate and asks before any live call
uv run python -m aelab.experiments.robustness --model claude-sonnet-4-6 --funders 3 --seeds 5 --rounds 20
```

The runner writes full traces and human-readable transcripts to `runs/` (gitignored except the
two featured 20-round transcripts). The other live CLI commands (`aelab truthfulness`,
`aelab counterfactual`) follow the same record-and-replay pattern and save their raw bids for
offline analysis.

**The web UI** presents the work for browsing. The deterministic panels run live and the LLM
panels replay committed results, so the deployed app needs no key.

```
uv run streamlit run app.py
```

## The economic model

All math is exact and lives in `economics.py`, with `DAYS_PER_YEAR = 365`. APR and discount
are exact inverses:

- `period = apr * days_early / 365`
- discount `d = period / (1 + period)`
- `apr = (d / (1 - d)) * 365 / days_early`

Money is the unit, so conservation is exact. `financing_cost(F, apr, days) = F * d` is the
cash the supplier gives up to finance early. From there:

- `gains_from_trade = financing_cost(reservation) - financing_cost(cost)` is the pie.
- `surplus_split` returns `supplier_surplus = fc(reservation) - fc(clearing)`,
  `winner_rent = fc(clearing) - fc(winner_cost)`, and their sum. The clearing term cancels, so
  realized gains reduce to `fc(reservation) - fc(winner_cost)`, independent of price. That
  cancellation is the conservation guarantee, and it is property-tested.
- `allocative_efficiency = realized / first_best`, with the convention `first_best <= 0 -> 1.0`.

**The four-party gap.** Second-price clearing makes truthful bidding weakly dominant, but only
for the bidders. It says nothing about whether the supplier sets an honest reserve, whether
the buyer reveals its true self-funding cost, or whether the house games its own venue. Those
need different machinery: structural separation, a fair-rate index, and the eval guardrails.

## The mechanism

Both auctions clear in APR space (price and tenor never enter the clearing), in `auction.py`.

- **Second-price** (`clear_auction`): among eligible bids (APR at or below the reserve), the
  lowest wins and is paid the second-lowest eligible bid. A single eligible bidder clears at
  the reserve, not at its own lone bid. No eligible bidder is no-trade. This makes truthful
  bidding weakly dominant for funders, proven with a `hypothesis` property test.
- **First-price** (`clear_first_price`): the lowest eligible bid wins and is paid its own bid.
  Truthful is no longer optimal, so a rational funder shades up. This is the counterfactual
  that separates strategic reasoning from prompt-following.

Ties at the lowest APR are broken by a seeded RNG over the tied bids sorted by bidder id, so
outcomes are reproducible and independent of input order.

## Project structure

```
src/aelab/
  models.py          frozen value types: Invoice, Funder, Supplier, Bid, AuctionResult, PricingPolicy
  economics.py       APR/discount math, surplus accounting, allocative efficiency
  auction.py         clear_auction (second-price) and clear_first_price
  metrics.py         InvoiceOutcome, MarketReport, summarize, deviation_stats
  populations.py     seeded synthetic suppliers, funders, invoices
  comms.py           the typed communication channel: Message, CommsConfig, the three conditions
  agents/
    base.py          BiddingAgent protocol, AuctionContext, AuctionRules, LeakedInfo
    deterministic.py TruthfulAgent, PolicyAgent (the house bidding a signed quote)
    cache.py         content-addressed response cache; an anthropic importer
    llm.py           LLMAgent, ValidatedLLMAgent, the batch path, the counterfactual probe
    communicating.py CommunicatingFunder: an agent that can chat before it bids
  attacks/           house_extraction (fair-rate index), collusion (ring), prompt_injection (clamp)
  eval/
    trace.py         the replayable trace: every bid, message, leaked field, clearing, baseline
    evaluators.py    binary evaluators with a measured value (collusion index, efficiency, integrity)
    guardrails.py    inline injection scan and APR clamp
    harness.py       per-condition summaries and the cross-condition compare table
  experiments/
    repeated.py      the repeated reverse-auction runner; clears through the real core
    robustness.py    multi-seed runner for one grid cell; cost estimate and confirmation first
  engine.py          runs auctions over a population via the agent seam
  harness.py         baseline vs house and collusion regimes, as a table
  report.py          the only matplotlib importer
  config.py          a frozen Scenario loaded from scenarios/*.toml
  cli.py             the aelab Typer app and the shared compute helpers
app.py               the Streamlit UI (edge, outside the package)
scenarios/           default.toml, extraction.toml
scripts/             demo_collusion.py (keyless), run_comms_experiment.py (live), make_figures.py
data/                committed bids and the collusion grid the UI and figures replay (keyless deploy)
docs/                this guide, the rendered figures, and the 90-second demo script
runs/                experiment traces and transcripts (gitignored except the two featured transcripts)
tests/               the test suite, including hypothesis property tests
memo/MEMO.md         the findings writeup
```

## Command-line interface

The `aelab` entry point (Typer). Every command takes `--scenario NAME`, loaded from
`scenarios/NAME.toml` (default: `default`).

| Command | What it does | Needs a key |
|---|---|---|
| `aelab efficiency` | sweep the financier count; print and plot efficiency and supplier share | no |
| `aelab attacks` | baseline vs house regimes and the collusion ring, as a table | no |
| `aelab truthfulness [--from-raw PATH]` | LLM bid deviation from true cost | live run only |
| `aelab counterfactual [--from-raw PATH]` | first vs second price under neutral prompts | live run only |

## Scenarios

- **`default.toml`** is a competitive market: bimodal suppliers, financiers, a buyer that
  self-funds at 0.07, and a house. A cheap buyer disciplines house withholding and collusion,
  so the attacks move nothing. This is the control, and the funder pool the live experiment
  draws from.
- **`extraction.toml`** is built so the house is the pivotal funder. Here the withholding
  attack is measurable and the fair-rate index fires.

## Reproducibility and the cache

A run is reproducible from its scenario file plus the response cache (`agents/cache.py`, keyed
by a sha256 of model, system, and user). The LLM commands save their raw bids and the comms
experiment writes through the cache, so the analysis replays with no network and no key once
the cache is committed.

## The API key

| Where | Key needed? |
|---|---|
| The deployed Streamlit app | **Optional.** Keyless by default (deterministic live, LLM replayed). A key (Streamlit secret) unlocks live agent bids, but on a public URL every visitor can spend it. |
| Local app, or the live CLI and experiment runs | **Yes**, on a cache miss: shell env, `.env`, or `.streamlit/secrets.toml`. |
| Anywhere in git | **Never.** `.env`, `.cache/` misses, `runs/`, and `.streamlit/secrets.toml` are gitignored. |

The key is read from the environment by `anthropic.Anthropic()`, lazily, so nothing needs it
unless a run hits a cache miss. For a shareable public demo, commit the cache and the saved
bids so the deployed app replays them with no key.

## Testing and the gate

No module is done until all four checks pass. CI runs the same gate on every push.

```
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest
```

The headline tests are `hypothesis` property tests: truthful bidding is weakly dominant, and
surplus conservation holds for all inputs.
