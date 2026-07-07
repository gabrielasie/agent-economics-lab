# Agent Economics Lab

A simulator and red-team harness for a sealed-bid second-price reverse auction with LLM
bidding agents, built around invoice early payment: funders bid an APR to pay a supplier
early, the lowest APR wins and is paid the second-lowest eligible bid.

**The finding.** LLM bidding agents, given an open chat channel and neutral prompts that state
the auction rule and recommend no strategy, colluded in exactly one condition of a
2 models x 2 pool sizes grid: Claude Sonnet 4.6 with three funders, where every open-channel
round cleared above the truthful baseline, +3.22 percentage points on average over 20 rounds,
with explicit coordination in the chat ("Rational coordination benefits everyone here").
Claude Haiku 4.5 at both pool sizes, and Sonnet 4.6 with six funders, stayed competitive.
One seed per cell: this demonstrates a boundary where collusion can emerge, not a rate.

![When do LLM bidding agents collude?](docs/figures/collusion_grid.png)

**Why it matters for a real venue.** Efficiency metrics are blind to venue-side extraction: in
the deterministic extraction attack, allocative efficiency stays at 1.000 while the venue takes
supplier surplus. A fair-rate index computed from auction outcomes flags every extracted
invoice, so detection is possible without trusting the operator. And pool depth is a product
constraint, not a growth detail: coordination emerged only in the thin three-funder market.

## Read this in 5 minutes

1. The results grid and both poles: [RESULTS.md](RESULTS.md).
2. The verbatim agent coordination quotes:
   [the positive pole](RESULTS.md#the-positive-pole-claude-sonnet-46-three-funders), or the
   full committed transcript in
   [`runs/claude-sonnet-4-6-n3-r20/transcripts.txt`](runs/claude-sonnet-4-6-n3-r20/transcripts.txt).
3. The one-page findings memo: [memo/MEMO.md](memo/MEMO.md).
4. The [live demo](https://agent-economics-lab-mthq892brens5xgelgbr9i.streamlit.app/)
   (keyless; LLM results replay from committed runs).

## What is in here

| Component | Purpose |
|---|---|
| Mechanism core (`economics`, `auction`, `metrics`) | Pure, seeded second-price clearing and surplus accounting; property-tested (truthful bidding dominant, surplus conservation) |
| Adversarial suite (`attacks/`) | House extraction with the fair-rate index, a deterministic collusion ring, prompt injection |
| Collusion experiment (`comms`, `experiments/`, `eval/`) | Repeated auction with a chat channel, real LLM funders, guardrail scanners, the boundary grid |
| Truthfulness eval (`agents/llm`) | Do agents bid true cost under second price and shade up under first price? |
| Injection scan (`eval/guardrails`) | Channel scanner plus an APR clamp that provably bounds an injected bid |
| UI (`app.py`) | Streamlit app replaying the committed runs; deterministic panels run live, keyless |

## Run it

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python scripts/demo_collusion.py       # the collusion mechanics, deterministic, no API key
uv run aelab attacks --scenario extraction    # the extraction attack and the fair-rate index, no key
uv run pytest                                 # the full test suite
uv sync --extra ui && uv run streamlit run app.py   # the app, keyless
```

Everything above runs offline. Live LLM runs, the response cache, the CLI, and the architecture
contracts are documented in [docs/LAB_GUIDE.md](docs/LAB_GUIDE.md).

## Limitations

- One seed per cell in the collusion grid (a multi-seed runner exists; see
  [RESULTS.md](RESULTS.md#robustness)). Read the positive cell as "collusion emerged here",
  not a rate; read negative cells as "did not emerge under these conditions", not "cannot".
- Synthetic invoice and funder distributions drawn from fixed ranges, in simulated APR space.
  No real settlement, no real counterparty.
- Bids are sealed by convention in the simulator, not by cryptography. There is no MPC or
  commitment scheme; the "informed house" attack models exactly this trust gap.
- Models tested: Claude Haiku 4.5 and Claude Sonnet 4.6, runs from June-July 2026. Other
  models, prompts, or market shapes may behave differently.
- The extraction result uses a scenario built so the house is the pivotal funder; the default
  competitive market is the control where the same attack moves nothing.

## License

MIT, see [`LICENSE`](LICENSE).
