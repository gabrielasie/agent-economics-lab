# CLAUDE.md

Project constitution. These rules hold in every session. Read SPEC.md for the full design.

## Architecture (non-negotiable)

- Dependency direction points inward. The pure core (`economics`, `models`, `auction`,
  `metrics`, `populations`) MUST NOT import `anthropic`, `aelab.agents`, `aelab.attacks`,
  `aelab.engine`, `aelab.harness`, or `matplotlib`. `lint-imports` enforces this and a
  violation is a build failure, not a style note.
- Agents reach the mechanism only through the `BiddingAgent` protocol and the
  `AuctionContext` type. Nothing else is the seam.
- All nondeterminism (the LLM, the network) lives behind `agents/cache.py`. The core is
  pure and fully seeded.
- Compute and presentation are separate. Only `report.py` imports `matplotlib`.
- Information an agent can see is modeled explicitly in `AuctionContext`. Leakage attacks
  populate `AuctionContext.leaked`; they never pass side-channel kwargs.

## Economic invariants (do not "simplify" these)

- Discount/APR: `period = apr * days_early / 365`; `d = period / (1 + period)`;
  `apr = (d / (1 - d)) * 365 / days_early`.
- Auction: sealed-bid second-price reverse. Lowest APR wins and pays the second-lowest
  eligible bid. Exactly one eligible bidder clears at the reserve. No eligible bidder means
  no trade. The reserve is the supplier's reservation APR.
- Conservation: `supplier_surplus + winner_rent == realized gains_from_trade`. Test it.
- First-best allocates every positive-surplus invoice to the lowest-true-cost funder and
  realizes the full pie. Allocative efficiency = realized / first-best.

## Definition of done

- No module is done until `ruff`, `mypy --strict`, `lint-imports`, and `pytest` are all green.
- Write or confirm the test BEFORE implementing. Implement until the test passes.
- Run `/verify` before claiming success. Do not report green on the strength of reading the
  code; report green on the strength of the gate passing.

## How to work

- One module per session. Build inward-out: `models, economics` first, then `auction`,
  `metrics`, `populations`, the agent seam, agents, attacks, orchestration, report, cli.
- Plan before code. Propose the interface and the test list, wait for approval, then write.
- Use `hypothesis` for the core property tests (truthful bidding is dominant; surplus
  conservation). Generated adversarial inputs are stronger than hand-rolled grids.

## Style

- Typed everything. Frozen dataclasses for data. `typing.Protocol` for seams.
- Comments and docstrings: plain English, short sentences. No em dashes. No filler words
  ("genuinely", "honestly", "actually"). Say what the code does and why, not how it feels.
- Seed every stochastic path. A run must be reproducible from its seed plus the response cache.
