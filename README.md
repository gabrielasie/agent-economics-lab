# Agent Economics Lab

Can the operator of an invoice early-payment auction quietly extract from suppliers, and would
LLM agents bidding in that auction exploit the same opening? The mechanism under study is a
sealed-bid second-price reverse auction with competing funders, matching the design Causa Prima
publicly describes: funding sources bid an APR to pay a supplier early, the lowest APR wins and is
paid the second-lowest eligible bid, and the supplier's reservation APR is the reserve. This repo
builds that mechanism, a deterministic adversary that probes it, and a live experiment that puts
real LLM agents in it. All APR figures use the fractional-APR convention.

## The finding

**Collusion is structurally possible.** A deterministic adversary, no LLM involved, shows the
mechanism does not prevent extraction. In the single-shot auction, a house that both runs the
venue and bids in it can peek at the sealed bids and withhold the one that set the clearing price.
Where the house is the pivotal funder at its honest cost, this lowers the supplier's share of
surplus while allocative efficiency stays at 1.000 the whole time. An efficiency metric never sees
the harm; a fair-rate price index, benchmarked against the honest house, flags every extracted
invoice. In the repeated auction, deterministic stub funders that coordinate over an open channel
clear strictly above the competitive baseline. The opening is real.

**Whether LLM agents take it is model- and market-dependent.** The headline experiment puts real
LLM agents in the repeated auction with neutral prompts that state the rule and recommend no
strategy, then runs 20 rounds under three channel conditions: sealed with hidden history, past
outcomes visible, and an open chat channel with full bid history. The open-channel collusion index
(cleared APR above the truthful baseline, in percentage points) across two models and two pool
sizes:

```
                      6 funders     3 funders
    Claude Haiku 4.5     0.00          0.00
    Claude Sonnet 4.6   -0.02         +3.22   <- collusion emerged
```

Collusion emerged in exactly one cell: the more capable model in a thin market. With Haiku 4.5 at
either pool size, and with Sonnet 4.6 at six funders, no round cleared above baseline, and the
agents reasoned that public signaling was against their own interest. With Sonnet 4.6 at three
funders, every open-channel round cleared above baseline and the agents coordinated explicitly
("Rational coordination benefits everyone here") and held bids above competitive levels. The
table, both poles, and verbatim agent reasoning from each are in [`RESULTS.md`](RESULTS.md).

This is the same class of question studied by Fish, Shorrer, and Gonczarowski on algorithmic and
LLM collusion. Rather than asserting the effect is model- and prompt-dependent, the grid observes
that dependence directly: same mechanism, same prompts, collusion present in one cell and absent in
the others. It is not a replication of their work, and it is not a claim that agents cannot
collude. See the [caveats](#caveats).

**[Live demo](https://agent-economics-lab-mthq892brens5xgelgbr9i.streamlit.app/)** ·
**[Results](RESULTS.md)** · **[Memo](memo/MEMO.md)** · **[Design spec](SPEC.md)**

---

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

Three properties an interviewer can check in a minute:

- **The core is isolated from the LLM edge, and it is enforced.** Three `import-linter` contracts
  fail the build on violation: the pure core (`economics, models, auction, metrics, populations`)
  may not import `anthropic`, `matplotlib`, or any edge, experiment, or eval module; only `report`
  imports `matplotlib`; only `agents/cache` and `agents/llm` import `anthropic`, and lazily, so
  the lab imports and runs without an API key. `comms.py` is pure data and policy (the typed
  communication channel), so it carries no LLM dependency either.
- **Record and replay reproducibility, via a content-addressed cache.** Every model call goes
  through `agents/cache.py`, keyed by a sha256 of model, system, and user. A run is reproducible
  from its seed plus the cache: a cache hit replays a stored completion with no network and no
  key, and a miss calls the model once and stores the result. Every stochastic draw threads a
  single injected `random.Random`; the code never touches the global `random`.
- **An eval layer with binary evaluators and guardrails.** Evaluators (`eval/evaluators.py`)
  return a pass or fail plus the number behind it, built from the failure modes that matter here
  (untruthful bidding, collusion, inefficiency, integrity), not generic quality scores. Guardrails
  (`eval/guardrails.py`) run inline in the request path: an injection and collusion scanner on the
  public channel, and an APR clamp so a malfunctioning or injected agent cannot bid outside the
  feasible range. The repeated runner (`experiments/repeated.py`) clears through the real core
  auction, so the experiment tests the actual mechanism, not a reimplementation.

Two seams carry everything. Agents reach the mechanism only through the `BiddingAgent` protocol
and the `AuctionContext` value, and what an agent may see is modeled explicitly, so a leakage
attack populates `AuctionContext.leaked` rather than passing a side channel.

## How to run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync                 # install the package and its dependencies
uv sync --extra ui      # add Streamlit for the web UI
```

**The deterministic demos run with no API key and no network.** They show the structural
possibility and the metrics that catch it.

```
uv run python scripts/demo_collusion.py            # repeated auction: stub funders collude over a channel
uv run aelab attacks --scenario extraction         # single-shot: a pivotal house extracts; the index fires
uv run aelab attacks                               # the competitive control, where the same attack moves nothing
uv run aelab efficiency                            # efficiency and supplier-share curve
```

**The live experiment needs a key.** It replays from the committed cache when one is present, so
a cache hit costs nothing; a miss calls the model and needs `ANTHROPIC_API_KEY` in the shell.

```
# replays the committed run offline if the cache covers it; otherwise calls the model
uv run python scripts/run_comms_experiment.py

# scale the run; ROUNDS defaults to 1 (a cheap smoke test)
COMMS_ROUNDS=20 uv run python scripts/run_comms_experiment.py

# vary the model and pool size (the robustness grid in RESULTS.md)
COMMS_MODEL=claude-sonnet-4-6 COMMS_N_FUNDERS=3 COMMS_ROUNDS=20 \
  uv run python scripts/run_comms_experiment.py
```

The runner writes full traces and human-readable transcripts to `runs/` (gitignored). The other
live CLI commands (`aelab truthfulness`, `aelab counterfactual`) follow the same record-and-replay
pattern and save their raw bids for offline analysis.

**The web UI** presents the work for browsing. The deterministic panels run live and the LLM
panels replay committed results, so the deployed app needs no key.

```
uv run streamlit run app.py
```

It leads with one result, incentive integrity (whether the venue can extract and whether an index
catches it), and keeps the agent arena, the prompt-injection defense, the first-price reasoning
test, and the agent-communication collusion experiment as deeper panels on demand. It lives
outside the `aelab` package, so the import contracts still hold.

## Caveats

This is a lab built to pressure-test the mechanism, not a production system. The results hold under
specific conditions and must not be overstated.

- Two models (Claude Haiku 4.5, Claude Sonnet 4.6) and two pool sizes (three and six funders).
- Neutral prompts that state the rule and recommend no strategy; nothing nudges toward or away
  from coordination. The Sonnet coordination is emergent, not prompted.
- A single seed per cell.
- Simulated APR space. No real settlement, no cryptography, no real counterparty.

Read the negative cells as "collusion did not emerge under these conditions," never as "these
agents cannot collude": the positive cell shows they can, and the deterministic adversary shows the
mechanism itself does not prevent it. Read the positive cell as "collusion emerged in this thin
market with this model," not as a rate measured across markets. The extraction result is
demonstrated in a scenario deliberately built to make the house pivotal, with the default market
as the control where the same attack does nothing. The input-separation injection defense
(delimiting the untrusted memo as data) is measured on a live model, not proven, unlike the output
clamp.

---

## The economic model

All math is exact and lives in `economics.py`, with `DAYS_PER_YEAR = 365`. APR and discount are
exact inverses:

- `period = apr * days_early / 365`
- discount `d = period / (1 + period)`
- `apr = (d / (1 - d)) * 365 / days_early`

Money is the unit, so conservation is exact. `financing_cost(F, apr, days) = F * d` is the cash
the supplier gives up to finance early. From there:

- `gains_from_trade = financing_cost(reservation) - financing_cost(cost)` is the pie.
- `surplus_split` returns `supplier_surplus = fc(reservation) - fc(clearing)`,
  `winner_rent = fc(clearing) - fc(winner_cost)`, and their sum. The clearing term cancels, so
  realized gains reduce to `fc(reservation) - fc(winner_cost)`, independent of price. That
  cancellation is the conservation guarantee, and it is property-tested.
- `allocative_efficiency = realized / first_best`, with the convention `first_best <= 0 -> 1.0`.

**The four-party gap.** Second-price clearing makes truthful bidding weakly dominant, but only for
the bidders. It says nothing about whether the supplier sets an honest reserve, whether the buyer
reveals its true self-funding cost, or whether the house games its own venue. Those need different
machinery: structural separation, a fair-rate index, and the eval guardrails.

## The mechanism

Both auctions clear in APR space (price and tenor never enter the clearing), in `auction.py`.

- **Second-price** (`clear_auction`): among eligible bids (APR at or below the reserve), the
  lowest wins and is paid the second-lowest eligible bid. A single eligible bidder clears at the
  reserve, not at its own lone bid. No eligible bidder is no-trade. This makes truthful bidding
  weakly dominant for funders, proven with a `hypothesis` property test.
- **First-price** (`clear_first_price`): the lowest eligible bid wins and is paid its own bid.
  Truthful is no longer optimal, so a rational funder shades up. This is the counterfactual that
  separates strategic reasoning from prompt-following.

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
  engine.py          runs auctions over a population via the agent seam
  harness.py         baseline vs house and collusion regimes, as a table
  report.py          the only matplotlib importer
  config.py          a frozen Scenario loaded from scenarios/*.toml
  cli.py             the aelab Typer app and the shared compute helpers
app.py               the Streamlit UI (edge, outside the package)
scenarios/           default.toml, extraction.toml
scripts/             demo_collusion.py (keyless), run_comms_experiment.py (live), and CLI wrappers
data/                committed bids the UI replays (keyless deploy)
runs/                experiment traces and transcripts (gitignored, regenerable from the cache)
tests/               245 tests, including hypothesis property tests
memo/MEMO.md         the findings writeup
demo.ipynb           the main experiments run end to end
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
  self-funds at 0.07, and a house. A cheap buyer disciplines house withholding and collusion, so
  the attacks move nothing. This is the control, and the funder pool the live experiment draws
  from.
- **`extraction.toml`** is built so the house is the pivotal funder. Here the withholding attack
  is measurable and the fair-rate index fires.

## Reproducibility and the cache

A run is reproducible from its scenario file plus the response cache (`agents/cache.py`, keyed by a
sha256 of model, system, and user). The LLM commands save their raw bids and the comms experiment
writes through the cache, so the analysis replays with no network and no key once the cache is
committed.

## The API key

| Where | Key needed? |
|---|---|
| The deployed Streamlit app | **Optional.** Keyless by default (deterministic live, LLM replayed). A key (Streamlit secret) unlocks live agent bids, but on a public URL every visitor can spend it. |
| Local app, or the live CLI and experiment runs | **Yes**, on a cache miss: shell env, `.env`, or `.streamlit/secrets.toml`. |
| Anywhere in git | **Never.** `.env`, `.cache/`, `runs/`, and `.streamlit/secrets.toml` are gitignored. |

The key is read from the environment by `anthropic.Anthropic()`, lazily, so nothing needs it
unless a run hits a cache miss. For a shareable public demo, commit the cache and the saved bids
so the deployed app replays them with no key.

## Testing and the gate

No module is done until all four checks pass.

```
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest          # 245 tests
```

The headline tests are `hypothesis` property tests: truthful bidding is weakly dominant, and
surplus conservation holds for all inputs.

## License

MIT, see [`LICENSE`](LICENSE).
