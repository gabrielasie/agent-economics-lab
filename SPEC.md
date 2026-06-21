# Agent Economics Lab — Project Description

A simulator and adversarial red-team harness for the sealed-bid second-price reverse auction at the core of Causa Prima's protocol. Built as a pre-interview project for the Product Engineer, Agent Economics role.

---

## 1. What this is, in one paragraph

A small Python lab that recreates Causa Prima's auction in miniature: competing financiers bid to fund an approved invoice early, the auction clears as a sealed-bid second-price (Vickrey) reverse auction, and synthetic populations of suppliers, buyers, and financiers let you measure how the mechanism behaves. On top of the simulator sits a red-team harness that runs a battery of attacks (house-venue extraction, financier collusion, prompt injection on the bidding agents) and reports how each one moves supplier surplus and efficiency against a clean baseline. The output is not a paper. It is a working tool of the kind the agent-economics function actually maintains: a standing test that answers "how does this mechanism get gamed?" before "how does it scale?"

## 2. Why this project and not something else

The role is mechanism design, not generic product engineering. They assume you can ship a feature. What they cannot easily assess in an interview, and therefore most need to verify, is whether you can reason rigorously about a multi-party auction with AI agents, find where it breaks, and turn that into design decisions. So the project leads with the rarest required skill, not the baseline one.

A "usable feature" or a "diagnosed pain point" was rejected for three reasons. You cannot diagnose their real internal bottlenecks from outside, so any guess is likely wrong and signals you do not understand the business. Anything genuinely usable by them touches real money, cryptography, underwriting, and accounting treatment, none of which is buildable in five days from outside without looking naive to founders who moved $50B in this exact domain. And a generic feature does not demonstrate agent-economics judgment, which is the one thing the role is uncertain about.

The red-team harness threads all three needles. It demonstrates mechanism understanding, hands-on agentic AI, and the adversarial mind the JD weights most heavily. It uses only inputs you actually have (synthetic scenarios and agent prompts, not their order flow), so it dodges the "you do not have our data" critique. And it is plausibly something their function would run.

## 3. The mechanism, stated correctly

Get this right above everything. Polished output on wrong theory is worse than no project.

**Setup.** A buyer approves an invoice of face value F, due in N days. Funding sources compete to pay the supplier early at day t, so the financing covers `days_early = N - t`. The bidders are all funding sources: third-party underwriters, Causa Prima's in-house book, and the buyer itself. The buyer self-funding from its own cash to capture the discount is just one more bidder. That single observation unifies two product lineages the JD's "nice to have" list names: classic dynamic discounting (Taulia) is the case where the buyer self-funds, and reverse factoring or multi-financier marketplaces (CRX, TReDS) are the case where third parties fund. Their auction is the generalization where both compete on one venue.

**APR is the unit, not the raw discount.** A discount fraction d on face value maps to an annualized simple APR. If a financier outlays F(1-d) now and is repaid F after `days_early` days, the period return is d/(1-d), so:

- `period = apr * days_early / 365`
- `discount d = period / (1 + period)`
- `apr = (d / (1 - d)) * 365 / days_early`

APR is what a CFO and a financier both reason in, and what makes invoices of different sizes and tenors comparable.

**The auction.** Sealed-bid, second-price, reverse. Financiers bid an APR. Lowest APR wins. The winner is paid the second-lowest eligible bid. The supplier's reservation APR (their best outside funding option) is the auction reserve: bids above it are ineligible. Conventions: with two or more eligible bids, clear at the second-lowest; with exactly one eligible bid, clear at the reserve; with none, no trade.

**Why second-price.** It makes truthful bidding a dominant strategy for financiers. Your bid decides whether you win, not the price you clear at, so shading your true cost of capital can only cost you profitable deals. This property has a name, dominant-strategy incentive compatibility, and naming it correctly is table stakes with these founders.

**The sharp observation to surface.** The JD says truthful behavior must be smart for "suppliers, buyers, financiers, and us." Second-price delivers financier truthfulness cleanly and for free. It says almost nothing about the other three. Supplier and buyer truthfulness is about whether their threshold-setting (reserve, hurdle rate) is honest. House truthfulness is about whether the venue operator games its own auction. Those are separate, harder claims that need different machinery. Articulating that the elegant guarantee covers only one of the four parties shows you read the mechanism critically rather than reciting Vickrey.

**Surplus and the headline metric.** On one invoice, the gains from trade are the money difference between financing at the supplier's reservation APR and financing at the lowest financier's true cost. First-best allocates every positive-surplus invoice to the lowest-cost funder and realizes the full pie. Allocative efficiency is realized surplus divided by first-best surplus. Efficiency loss is one minus that. Under truthful bidding with the lowest-cost winner, efficiency is ~1.0; loss appears from non-truthful behavior, collusion, thin markets, or agents misbehaving.

**The trust metric, equal billing with efficiency.** Split the realized surplus into supplier surplus (reservation minus clearing rate) and winner rent (clearing rate minus winner's true cost). A mechanism can be perfectly efficient and still route the entire pie to financiers and the house, at which point suppliers conclude the venue is an extraction tool and leave. The JD treats that as existential, so supplier share of surplus is a first-class metric, not a footnote.

## 4. Architecture and stack

Python, because the simulation, mechanism-design, and data-analysis ecosystem lives there, and because it is the natural home for the Anthropic SDK.

- **Core sim:** pure Python plus the standard library. Dataclasses for typed, serializable, deterministic models. No heavy dependencies in the floor.
- **Analysis and plots:** matplotlib for the efficiency and supplier-share curves.
- **LLM agents:** the Anthropic Python SDK, with Claude as the bidding agents. Built and iterated in Claude Code.
- **Tests:** pytest. The point is to prove mechanism properties numerically, not assert them.
- **Pricing policy:** modeled as a versioned, content-hashed, pure-function object, mirroring the JD's "signed, versioned, deterministic pricing policies."

Proposed layout:

```
agent-economics-lab/
  src/aelab/
    config.py         # run configuration: seeds and scenario parameters
    economics.py      # APR/discount math, surplus, first-best. Heaviest test coverage.
    models.py         # Invoice, Funder (+ FunderKind), Supplier, Bid, AuctionResult, PricingPolicy (versioned + hashed)
    auction.py        # sealed-bid second-price reverse auction with reserve
    populations.py    # synthetic supplier / buyer / financier generators
    metrics.py        # efficiency loss vs first-best, surplus split, supplier share
    engine.py         # runs auctions over populations and collects results
    harness.py        # red-team harness: runs attacks, diffs against baseline, reports
    report.py         # presentation layer: efficiency + supplier-share plots (only matplotlib importer)
    agents/
      base.py         # the BiddingAgent protocol and AuctionContext seam
      deterministic.py  # deterministic truthful + policy-driven (house) agents
      llm_agent.py    # Claude bidding agent (lazy client; rest of lab runs without a key)
    attacks/
      house_extraction.py   # centerpiece: venue conflict + fair-rate-index defense
      collusion.py          # stretch: financier ring inflating the second price
      prompt_injection.py   # agentic: malicious invoice memo flips a bid + defenses
  scripts/
    run_efficiency.py # Day 2 deliverable: efficiency + supplier-share curves
    run_attacks.py    # Day 4 deliverable: the red-team report you demo
  tests/
    test_auction.py   # proves truthful bidding is dominant, numerically
    test_economics.py
  memo/
    MEMO.md           # the one-page writeup
  README.md
```

## 5. The attacks, ranked

Go deep on one. A clean, correct, single result beats a broad shallow demo.

**1. House-venue conflict and the fair-rate index. The centerpiece.** Their strongest-worded fear: "if suppliers ever conclude we're a buyer-side extraction tool, the network fails. Your job is to make that structurally impossible." Model the house bidding its own auction. In the adversarial variant it peeks at sealed competitor bids and either undercuts to win or, when it cannot win, slots a bid just above the lowest to lift the second price toward the reserve and inflate what the supplier pays. Measure the supplier's extra cost. Then show the defense: structural separation (the house bids a signed, deterministic policy with no peek, so its bid is reproducible and any deviation is detectable) plus a fair-rate index (publish the clearing APR against the competitive benchmark that would have prevailed without the house; systematic excess is visible to everyone, which is what makes extraction unprofitable). This is the differentiator because it is not a textbook result. It is a market-structure judgment call, and most candidates will not think to build it.

**2. Financier collusion. Stretch.** Second-price auctions are famously fragile to collusion. A ring designates one member to bid near its true cost and the others to bid just under the reserve, lifting the second price without any colluder having an incentive to defect within the round. Sweep ring size and watch supplier share collapse. Show the same fair-rate index also flags collusion: one defense instrument, two threats. Strong because it demonstrates real auction theory, the kind of "where classical assumptions stop holding" the JD asks for, even before agents enter.

**3. Prompt injection on the bidding agent. Agentic, only if rigorous.** The invoice carries an untrusted free-text memo. A malicious counterparty embeds instructions in it to move an LLM agent's bid. Run a battery of payloads and measure the bid deviation versus the clean baseline. Then quantify defenses: input separation (memo passed as clearly delimited data, never as instructions), and output validation (clamp the agent's bid to the deterministic policy bounds, so even a fully compromised reasoning step cannot produce an out-of-policy bid). The validation defense is the strongest claim. Skip this entirely before shipping it half-baked; a single screenshot is not a result.

## 6. The result that would land hardest

Build at least the LLM bidding agent and run one experiment classical theory does not answer: do LLM agents actually bid the dominant strategy? Human subjects famously overbid in Vickrey experiments. Give a Claude agent a known private cost, run many auctions against known competitors, and plot the gap between its bids and its true cost. If the agents deviate systematically, the truthful-is-dominant guarantee that justifies the mechanism does not hold for the players who will actually use it. That is the literal intersection of the role: mechanism design meeting agentic AI, measured rather than asserted.

## 7. Five-day plan, floor-first

Sequence so a complete, impressive artifact exists at every checkpoint. Building it this way is itself the signal the JD wants ("bias to shipped decisions, turn ambiguity into a default proposal, an owner, and a deadline"). Make that explicit when you present it.

- **Day 1. The memo and the spec.** Write the one page first: the mechanism, why it is truthful on the financier side only, the four-party gap, and three to four sharp questions for the founders. Thinking before building.
- **Day 2. The floor.** Deterministic auction core, the economics module, the efficiency and supplier-share curves, and the pytest suite that proves truthfulness. This alone is a correct, on-target artifact. If everything after collapses, you still have this.
- **Day 3. The agents.** LLM bidding agents on the deterministic core. Run the truthfulness probe from section 6. This is your headline experiment.
- **Day 4. The centerpiece attack.** House extraction plus the fair-rate-index defense, wired into the harness. Add collusion only if it is clean.
- **Day 5. The rest of the loop.** Founder backgrounds, your own story, behavioral prep. Protect this day. The project is a multiplier on a good interview, not a substitute for one.

Do not chase a polished UI. With these people, a correct efficiency curve and one rigorous extraction result in a plain notebook beats a beautiful dashboard on shaky theory.

## 8. How to present it

Do not say you rebuilt their protocol. Say you wanted to understand the mechanism well enough to have a real conversation, so you built a toy to pressure-test it, and here is what you found and where you are unsure. The open questions carry as much weight as the demo, because they prove you know the gap between a weekend simulation and their production system. With a small founding team, that posture reads as exactly the hire they want.

## 9. Open questions for the founders (seed list)

These signal you know what you do not know. Refine to three or four.

- The dominant-strategy guarantee is clean for financiers. What makes truthful threshold-setting smart for suppliers and buyers, and what keeps the house honest, given it both runs and bids in the venue?
- Is the in-house book's information access actually walled off from the auction, or is separation enforced by policy and audit rather than architecture?
- How is the fair-rate index constructed, and against what independent benchmark, such that a supplier can trust it rather than trusting Causa Prima?
- In repeated play with a stable set of financiers, how much does sealed-bid privacy actually buy you before tacit collusion or bid-shading erodes it?
- When a bidding agent is an LLM, where does the deterministic, signed policy end and the model's discretion begin? What is the validation boundary that caps how badly an agent can misbehave?

## 10. Scope discipline

The full vision is more than five days. The cut list, in priority order: auction core and efficiency metric (non-negotiable floor), LLM agent and truthfulness probe, house-extraction result, collusion, prompt injection. Stop adding the moment a day runs out, and present what is finished as finished. A smaller, correct, well-explained artifact wins.
