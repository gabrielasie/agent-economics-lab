# Agent Economics Lab

A simulator and adversarial red-team harness for the **sealed-bid second-price reverse
auction** at the core of an invoice early-payment financing protocol. Funding sources bid an
APR to pay a supplier early; the lowest APR wins and is paid the second-lowest eligible bid,
with the supplier's reservation APR as the reserve. On top of that mechanism sit synthetic
populations, LLM bidding agents, and a battery of attacks that measure how the auction behaves
and how it gets gamed.

The thesis in one line: **second-price clearing buys truthfulness from funders and from no one
else.** The supplier who sets the reserve, the buyer who can self-fund, and the house that runs
the venue all sit outside that guarantee, and every experiment here probes that gap.

It is a pre-interview artifact for a Product Engineer, Agent Economics role, so it is judged on
the correctness of the mechanism, the sharpness of the findings, and the accompanying memo
([`memo/MEMO.md`](memo/MEMO.md)). The full design lives in [`SPEC.md`](SPEC.md).

**[Live demo](https://agent-economics-lab-mthq892brens5xgelgbr9i.streamlit.app/)** ·
**[Memo](memo/MEMO.md)** · **Where to start:** the **Agent arena** (agents bidding against each
other), then **Trust & integrity** (the fair-rate index).

---

## Contents

- [Quickstart](#quickstart)
- [The web UI](#the-web-ui)
- [Deployment](#deployment)
- [The economic model](#the-economic-model)
- [The mechanism](#the-mechanism)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Command-line interface](#command-line-interface)
- [Scenarios](#scenarios)
- [Results](#results)
- [Reproducibility and the cache](#reproducibility-and-the-cache)
- [The API key](#the-api-key)
- [Testing and the gate](#testing-and-the-gate)
- [Status and roadmap](#status-and-roadmap)
- [Caveats and open questions](#caveats-and-open-questions)
- [License](#license)

---

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync                       # install the package and its dependencies
uv run aelab efficiency       # efficiency and supplier-share curve -> results/efficiency.png
uv run aelab attacks --scenario extraction   # the house-extraction harness (a table)
uv run aelab attacks          # the default market, where competition disciplines extraction
uv run aelab truthfulness --from-raw results/truthfulness_raw.json
uv run aelab counterfactual   # first vs second price under neutral prompts (needs an API key)
```

The deterministic commands (`efficiency`, `attacks`) run with no network. The LLM commands
(`truthfulness`, `counterfactual`) call the Anthropic API on a live run and cache the raw bids
to `results/`, so `--from-raw` replays the analysis offline.

## The web UI

A [Streamlit](https://streamlit.io) app (`app.py`) presents the work in three views:

- **Agent arena** - Claude funders bid against each other on one invoice; their bids reveal one
  at a time, the auction clears to a winner and a price, and each agent's reasoning is shown.
- **Trust & integrity** - efficiency is not enough; the fair-rate index catches a house that
  extracts by withholding, and a fee-structure lever shows which fee base keeps the venue honest.
- **Do the agents reason?** - the truthfulness non-result beside the first-price counterfactual.

It lives outside the `aelab` package, so the pure core is untouched and the import contracts
still hold. The deterministic work runs live and the LLM panels replay committed bids in
`data/`, so it needs no API key; an optional key enables live agent bids.

```
uv sync --extra ui
uv run streamlit run app.py
```

## Deployment

Deploys free on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repo to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, choose **New app**.
3. Pick the repo, branch `main`, main file `app.py`, and **Deploy**.

The host installs `requirements.txt` (which is `.[ui]`, building the package from
`pyproject.toml`). **No secret is needed**: every panel is keyless. The app redeploys on each
push. To light up the first-price panel in the deployed app, run `uv run aelab counterfactual`
once with a key, then commit the resulting `data/first_price_raw.json`.

---

## The economic model

All math is exact and lives in `economics.py`. APR and discount are exact inverses, with
`DAYS_PER_YEAR = 365`:

- `period = apr * days_early / 365`
- discount `d = period / (1 + period)`
- `apr = (d / (1 - d)) * 365 / days_early`

Money is the unit, so conservation is exact: `financing_cost(F, apr, days) = F * d` is the cash
the supplier gives up to finance early. From there:

- `gains_from_trade = financing_cost(reservation) - financing_cost(cost)` is the pie.
- `surplus_split` returns `supplier_surplus = fc(reservation) - fc(clearing)`,
  `winner_rent = fc(clearing) - fc(winner_cost)`, and their sum `realized_gains`. The clearing
  term cancels, so `realized_gains = fc(reservation) - fc(winner_cost)`, independent of price.
  That algebraic cancellation **is** the conservation guarantee, and it is property-tested.
- `first_best` allocates each invoice to the lowest true-cost funder and floors at zero.
- `allocative_efficiency = realized / first_best`, with the convention `first_best <= 0 -> 1.0`.

**The four-party gap.** Second-price clearing makes truthful bidding weakly dominant, but only
for the bidders. It says nothing about whether the supplier sets an honest reserve, whether the
buyer reveals its true self-funding cost, or whether the house games its own venue. Those need
different machinery: structural separation and a fair-rate index.

## The mechanism

Both auctions clear in APR space (price and tenor never enter the clearing), in `auction.py`.

- **Second-price** (`clear_auction`): among eligible bids (APR at or below the reserve), the
  lowest wins and is paid the second-lowest eligible bid. Exactly one eligible bidder clears at
  the reserve (not at its own lone bid). No eligible bidder is no-trade. This makes truthful
  bidding weakly dominant for funders, proven numerically with a `hypothesis` property test.
- **First-price** (`clear_first_price`): the lowest eligible bid wins and is paid its own bid.
  Truthful is no longer optimal here, so a rational funder shades its bid up. This is the
  counterfactual that separates strategic reasoning from prompt-following.

Ties at the lowest APR are broken by a seeded RNG over the tied bids sorted by bidder id, so
outcomes are reproducible and independent of input order.

## Architecture

Dependencies point inward to a pure, deterministic core, with all nondeterminism (the LLM, the
network) quarantined at the outer edge.

```
cli / scripts / app.py   (entry, thin; app.py is the Streamlit UI, an edge outside the package)
   report                (presentation; the only matplotlib importer)
   harness               (orchestration; baseline vs attack regimes)
   engine                (runs one scenario -> list[InvoiceOutcome])
 attacks      agents/llm, agents/cache   (edge; the only anthropic importer)
   |          agents/deterministic, agents/base   (the BiddingAgent seam)
 ============ pure deterministic core (no network, no LLM, no matplotlib) ============
   metrics    populations    auction    economics    models
```

This is enforced, not aspirational, by three `import-linter` contracts that fail the build on
violation:

1. **Core stays pure** - `economics, models, auction, metrics, populations` may not import
   `anthropic`, `matplotlib`, or any edge module.
2. **Only report renders** - every module except `report` is forbidden from importing
   `matplotlib`.
3. **Nondeterminism is isolated** - only `agents/cache` imports `anthropic` (and lazily, so the
   lab imports and runs without an API key).

Two seams carry everything. Agents reach the mechanism only through the `BiddingAgent` protocol
and the `AuctionContext` value; and what an agent is allowed to see is modeled explicitly, so a
leakage attack populates `AuctionContext.leaked` rather than passing a side channel. The
Streamlit UI is a separate edge outside the package, so it never affects the contracts.

## Project structure

```
src/aelab/
  models.py          frozen value types: Invoice, Funder, Supplier, Bid, AuctionResult, PricingPolicy
  economics.py       APR/discount math, surplus accounting, allocative efficiency
  auction.py         clear_auction (second-price) and clear_first_price
  metrics.py         InvoiceOutcome, MarketReport, summarize, deviation_stats
  populations.py     seeded synthetic suppliers, funders, invoices
  agents/
    base.py          BiddingAgent protocol, AuctionContext, AuctionRules, LeakedInfo
    deterministic.py TruthfulAgent, PolicyAgent (the house bidding a signed quote)
    cache.py         content-addressed response cache; the only anthropic importer
    llm.py           LLMAgent, ValidatedLLMAgent, the batch path, the counterfactual probe
  attacks/
    house_extraction.py  informed vs separated house, the fair-rate index
    collusion.py         a financier ring parking bids to lift the second price
    prompt_injection.py  payloads in the memo, and the output-validation defense
  engine.py          runs auctions over a population via the agent seam
  harness.py         baseline vs house and collusion regimes, as a table
  report.py          the only matplotlib importer; efficiency and deviation plots
  config.py          a frozen Scenario loaded from scenarios/*.toml
  cli.py             the aelab Typer app and the shared compute helpers
app.py               the Streamlit UI (edge, outside the package)
scenarios/           default.toml, extraction.toml
scripts/             thin argparse wrappers over the cli orchestration
data/                committed bids the UI replays (keyless deploy)
tests/               207 tests, including hypothesis property tests
memo/MEMO.md         the findings writeup
demo.ipynb           the three results run end to end
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

The scripts in `scripts/` wrap the same orchestration for `uv run python scripts/...`.

## Scenarios

- **`default.toml`** - a competitive market: bimodal suppliers, five financiers, a buyer that
  self-funds at 0.07, and a house. Here a cheap buyer disciplines house withholding and
  collusion, so the attacks move nothing. This is the control.
- **`extraction.toml`** - a market built so the house is the pivotal funder (a cheap buyer wins
  every invoice, financiers sit entirely above the house cost, the policy is cost-priced). Here
  the withholding attack is measurable and the fair-rate index fires.

## Results

**1. Efficiency is necessary, not sufficient, so measure supplier share.** Under truthful
bidding the lowest-cost funder always wins, so allocative efficiency is 1.000 at every financier
count. The metric that moves is the supplier's share of surplus, which rises with competition
(0.885 to 0.917 in the default scenario). Efficiency tells you the pie is whole; supplier share
tells you who ate it.

**2. Do LLM agents bid the dominant strategy? (deliberately a non-result)** Given a known
private cost, a Claude Haiku agent does not overbid the way humans do in Vickrey experiments:
across 120 bids the mean signed deviation is about zero and 100% land within 50 bps of truthful.
This is **not** presented as a finding. The probe's prompt states that truthful bidding is
optimal, so the near-zero deviation measures instruction-following, not reasoning. The test that
separates the two is the first-price counterfactual (`aelab counterfactual`): the same agent
under a first-price auction with a neutral prompt, where truthful is no longer optimal and a
reasoner shades its bid up. That experiment is built and awaits one live run; until then the
honest claim is only "the agent does not overbid."

**3. House extraction, and a fair-rate index that catches it.** In the `extraction` scenario the
house is the pivotal second-lowest bid at its honest cost. A house that posts that policy blind
helps suppliers by adding competition (supplier share 0.754 to 0.855). A house that peeks at
sealed bids extracts by withholding: it bids just under the reserve, deletes the bid that set
the clearing, and pushes supplier share back to 0.754 on all 50 invoices. Efficiency stays 1.000
the whole time, so the harm is invisible to an efficiency metric and to a no-house baseline. The
fair-rate index, benchmarked against the honest separated house, flags every extracted invoice;
the same index flags financier collusion. The discipline that defeats both is competition: in
the `default` scenario a cheap buyer sits below the house, so withholding and the ring move
nothing. For the LLM agents, clamping each bid to the signed pricing policy is an output
validation that provably caps any prompt injection inside the policy bounds.

**4. Structural separation has an allocative price, and it is a cliff, not a slope.** When the
house posts a risk-priced quote (a base cost plus loadings) instead of bare cost, the markup is
allocatively free while it stays inside the house's cost advantage over the next funder, then
falls off a step the moment the loaded quote crosses that funder. In a measured case (house cost
0.08, next funder 0.10), efficiency holds at 1.000 through loadings of 0.02 and drops to 0.923 at
0.04. Setting policy margin is choosing how much allocative efficiency to spend.

## Reproducibility and the cache

Every stochastic path threads a single injected `random.Random`; the code never touches the
global `random`. The only other nondeterminism is the model call, isolated behind a
content-addressed cache (`agents/cache.py`, keyed by a sha256 of model, system, and user). A run
is reproducible from its scenario file plus the cache. The LLM commands save their raw bids, so
`--from-raw` replays the analysis with no network and no key.

## The API key

| Where | Key needed? |
|---|---|
| The deployed Streamlit app | **Optional.** Keyless by default (deterministic live, LLM replayed). A key (Streamlit secret) unlocks live agent bids, but on a public URL every visitor can spend it. |
| Local app, or the `truthfulness` / `counterfactual` CLI live runs | **Yes** - shell env, `.env`, or `.streamlit/secrets.toml` |
| Anywhere in git | **Never** - `.env` and `.streamlit/secrets.toml` are gitignored |

The key is read from the environment by `anthropic.Anthropic()`, lazily, so nothing needs it
unless you run a live command. For a shareable public demo, prefer the keyless path: run the LLM
experiments once locally and commit the saved bids (`data/*.json`), so the deployed app replays
them with no key on the public URL.

## Testing and the gate

No module is done until all four checks pass. Tests are written before the implementation, and a
passing gate is the only evidence of done.

```
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest          # 207 tests
```

The headline tests are `hypothesis` property tests: truthful bidding is weakly dominant, and
surplus conservation holds for all inputs.

## Status and roadmap

Built and green: the full core, both auctions, the agents and the cache, all three attacks, the
harness, the CLI, the Streamlit UI, `demo.ipynb`, and the memo.

Pending: one live run of the first-price counterfactual (it needs a key; the machinery is built
and tested), which would turn result 2 from "the agent does not overbid" into a claim about
reasoning; and live prompt-injection success rates per class.

## Caveats and open questions

This is a toy built to pressure-test the mechanism, not a production system. The truthfulness
result is one model under one coaching prompt, and is reported as a non-result for exactly that
reason. The input-separation defense (delimiting the untrusted memo as data) is measured on a
live model, not proven, unlike the output clamp. The extraction result is demonstrated in a
scenario deliberately built to make the house pivotal, with the default market as the control
where the same attack does nothing.

Open questions worth a conversation: what makes threshold-setting smart for the parties the
mechanism does not protect (the supplier and the buyer); how to construct a fair-rate benchmark a
supplier can actually trust, given that the honest-house counterfactual is something only the
operator can compute; and where a signed deterministic policy should end and an LLM's discretion
begin.

## License

MIT, see [`LICENSE`](LICENSE).
