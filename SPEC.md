# SPEC: Agent Economics Lab

The design source of truth. Operating rules (the gate, tests-first, the dependency
contracts) live in CLAUDE.md; this document describes the system those rules build. Where
the two ever disagree, fix the disagreement, do not pick one silently.

## 1. What this is

A command-line simulation lab for the sealed-bid second-price reverse auction at the core
of an invoice early-payment protocol. Competing funders bid to finance an approved invoice;
the auction clears; synthetic populations and an adversarial harness let us measure how the
mechanism behaves and how it gets gamed. It produces tables and charts, not a UI or a
deployed service. It is a pre-interview artifact for a Product Engineer, Agent Economics
role, so it is judged on the correctness of the mechanism, the sharpness of the findings,
and the accompanying memo.

## 2. The mechanism and the economics (invariants)

These are exact and must not be "simplified."

APR and discount are exact inverses, with DAYS_PER_YEAR = 365:

- period = apr * days_early / 365
- discount d = period / (1 + period)
- apr = (d / (1 - d)) * 365 / days_early

Money: financing_cost(F, apr, days_early) = F * discount(apr, days_early). This is the cash
the supplier gives up, the unit everything is measured in.

Surplus, measured in money so conservation is exact:

- gains_from_trade(F, reservation, cost, days) = financing_cost(reservation) -
  financing_cost(cost). The pie. May be negative if cost exceeds reservation.
- first_best_surplus = max(0, gains_from_trade(reservation, lowest_cost)). First-best funds
  only positive-surplus invoices, allocated to the lowest-cost funder.
- surplus_split returns SurplusSplit(supplier_surplus, winner_rent, realized_gains) where
  supplier_surplus = fc(reservation) - fc(clearing), winner_rent = fc(clearing) -
  fc(winner_cost), and realized_gains = supplier_surplus + winner_rent. Because the clearing
  term cancels, realized_gains collapses to fc(reservation) - fc(winner_cost) and is
  independent of the clearing price. Conservation holds for all inputs.
- allocative_efficiency = realized / first_best, with the convention that first_best <= 0
  returns 1.0.

The second-price reverse auction: among eligible bids (apr <= supplier reservation), the
lowest APR wins and is paid the second-lowest eligible bid. Exactly one eligible bidder
clears at the reserve, not at the lone bid. No eligible bidder is no-trade. The reserve is
the supplier's reservation APR. Ties at the lowest bid resolve by seeded RNG. This makes
truthful bidding a dominant strategy for funders, the property proven numerically in the
tests.

The four-party truthfulness gap, the sharp point: second-price gives truthfulness for free
only to funders. It says nothing about whether suppliers and buyers set their thresholds
honestly, or whether the house games its own venue. Those need different machinery
(structural separation, the fair-rate index).

The first-price reverse auction (counterfactual): lowest eligible APR wins and is paid ITS
OWN bid. Truthful bidding is not optimal here; a rational funder shades up (bids above true
cost) to earn margin. Used to test whether an LLM agent reasons about the mechanism or
merely follows coaching.

The unifying insight: funding is heterogeneous. Third-party financiers, the buyer
self-funding to capture the discount, and the house's own book are all funders that bid.
Dynamic discounting is the buyer-self-funds case; reverse factoring is the third-party case;
this auction is the generalization.

## 3. Architecture

A pure deterministic core with the LLM isolated at the outermost edge. Dependencies point
inward.

```
cli / scripts            (entry, thin)
   report                (presentation; the only matplotlib importer)
   harness               (orchestration; runs scenarios, diffs vs baseline)
   engine                (runs one scenario -> list[InvoiceOutcome])
 attacks      agents/llm, agents/cache   (edge; the only anthropic importers)
   |          agents/deterministic, agents/base   (the BiddingAgent seam)
 ---- pure deterministic core (no network, no LLM, no matplotlib) ----
   metrics    populations    auction    economics    models
```

The seam: agents reach the mechanism only through the BiddingAgent protocol and
AuctionContext. Information an agent may see is modeled explicitly; leakage attacks populate
AuctionContext.leaked rather than passing side channels.

Reproducibility: the only nondeterminism is the model call, isolated behind agents/cache.py
(content-addressed by hash of model, system, user). A run is reproducible from its seed plus
the cache, and a reviewer can regenerate results offline. All stochastic draws thread a
single injected random.Random; the code never touches the global random.

Enforcement (import-linter, with include_external_packages = true): the core imports no
edge, LLM, or matplotlib; only report imports matplotlib; only agents/cache and agents/llm
import anthropic. A violation fails the gate.

## 4. Type vocabulary

FunderKind enum: FINANCIER, BUYER, HOUSE.

Invoice(invoice_id, face_value>0, days_early>0, memo="", buyer_credit=0.0,
dilution_risk=0.0). buyer_credit and dilution_risk are risk scores in [0, 1] that feed
PricingPolicy.quote and never touch the auction directly. memo is untrusted free text,
stored verbatim, never interpreted.

Funder(party_id, true_cost_apr>=0, kind=FINANCIER). Private cost of capital.

Supplier(party_id, reservation_apr>=0). Becomes the auction reserve. A distinct type from
Funder so a supplier can never be passed where a bidder is expected.

Bid(bidder_id, apr>=0, rationale=""). rationale is optional free text: deterministic agents
fill it with the policy version and content hash, LLM agents with the model's stated
reasoning. It never affects clearing, which depends only on bidder_id and apr.

AuctionResult(traded, winner_id, clearing_apr, winning_bid_apr, num_eligible) with
no_trade() and cleared(...) constructors and a traded iff outcome-fields-non-None invariant.

PricingPolicy(version, min_apr, max_apr, base_apr=0.0, buyer_credit_loading=0.0,
dilution_loading=0.0) with content_hash (sha256 over every field, the signature stand-in),
clamp(apr), and quote(invoice) -> float (base_apr plus the risk loadings times the invoice
risk scores, clamped; monotonic in each risk input).

AuctionContext(invoice, rules, leaked), AuctionRules(reserve_apr), LeakedInfo(competitor_bids,
is_empty) (empty by default).

SurplusSplit(supplier_surplus, winner_rent, realized_gains) in economics.

InvoiceOutcome(invoice, supplier, funders, result) in metrics, the shared per-invoice result
type. It lives in metrics because the contract forbids metrics from importing engine, and
engine imports it from metrics.

MarketReport in metrics: first-best total, realized total, allocative efficiency, efficiency
loss, supplier share of surplus.

## 5. Module map and status

Done and committed:

models, economics, auction (second-price clear_auction), populations, metrics, engine,
report. agents/base, agents/deterministic (TruthfulAgent and PolicyAgent, the house bidding
PricingPolicy.quote with the policy version and hash as its rationale), agents/cache
(single-call and batch clients), agents/llm (LLMAgent with a fence-tolerant JSON parser,
ValidatedLLMAgent, and the batch path with index-based custom_ids). attacks/house_extraction
(informed vs separated house plus the fair-rate index), attacks/collusion (ring sweep),
attacks/prompt_injection (payloads, success-rate measurement, output-validation defense).
harness (baseline plus house regimes plus a collusion row, formatted as a table). config
(Scenario loaded from scenarios/*.toml) and cli (the aelab Typer app: efficiency,
truthfulness, attacks). scripts run_efficiency.py and run_truthfulness_probe.py, both
scenario-driven, the probe with --from-raw. README and demo.ipynb (the three results run in
sequence). scenarios/extraction.toml: a population built so the house is the pivotal funder,
where the withholding attack is measurable, with an honest cost-priced policy (zero loadings).

The parser fix (strip the json fence, extract the first balanced JSON object) and the probe
--from-raw mode are done. The run_efficiency controlled-sweep fix is done: suppliers and
invoices are generated once and only the financier pool grows, so the supplier-share line is
a clean controlled comparison.

Remaining (designed in sections 2, 4, and 6, not yet built):

- First-price counterfactual: clear_first_price in auction.py (lowest eligible wins, paid
  its own bid); a NEUTRAL prompt variant in agents/llm.py that states the rules and
  recommends no strategy; scripts/run_first_price_counterfactual.py.
- Live numbers that need an API key: the truthfulness probe has a result from saved raw bids;
  the first-price probe and the prompt-injection live success rates per class are not yet run.

Disagreement to resolve, not blocking: the truthfulness probe's system prompt currently
coaches ("bidding your true cost of capital is optimal"), which section 7 forbids for
strategy-reasoning experiments. So the 100%-within-epsilon probe result may be
prompt-following, not strategic reasoning. The first-price counterfactual with the neutral
prompt is the test that distinguishes the two; build it before drawing a strong conclusion
from the probe.

## 6. Experiments the tool produces

- Efficiency and supplier share vs financier count, with TruthfulAgent. Efficiency is ~1.0
  (the truthful-bidding prediction); supplier share depends on the reservation-to-cost gap.
- Truthfulness probe: distribution of (LLM bid minus true cost) under second-price. Whether
  LLM agents play the dominant strategy.
- First-price counterfactual: the same agents under first-price with the neutral prompt.
  Shading up means strategic reasoning; near-truthful in both auction types means
  prompt-following.
- House extraction: informed vs separated house, reporting supplier share, efficiency, and
  fair-rate-index flags. The argument for structural separation. Measurable only where the
  house is the pivotal funder (scenarios/extraction.toml); in the default market a competitive
  buyer disciplines it, so the same attack moves nothing. That contrast is the result.

## 7. Locked conventions

- Funder (with kind) plus separate Supplier. No overloaded Party.
- Surplus is money, F times discount. Conservation is exact.
- Second-price: lowest wins, paid second-lowest eligible; single eligible clears at reserve;
  none is no-trade. First-price: lowest wins, paid own bid.
- One injected RNG through every draw; never the global random.
- Nondeterminism isolated behind the cache; reproducible from seed plus cache.
- Batch custom_ids are index-based and match ^[a-zA-Z0-9_-]{1,64}$.
- Strategy-reasoning experiments use neutral prompts that state rules but recommend no
  strategy.
- Comments and docstrings: plain English, short sentences, no em dashes, no filler words.

## 8. Definition of done (per module)

ruff, mypy --strict, lint-imports, and pytest all green. Tests written or confirmed before
implementation. Commit at every green. A passing gate is the only evidence of done; reading
the code is not.
