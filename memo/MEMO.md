# Memo: what the auction lab shows

This is a simulation and red-team harness for the sealed-bid second-price reverse auction at
the core of an invoice early-payment protocol. Funders bid an APR to pay a supplier early,
the lowest APR wins, and the winner is paid the second-lowest eligible bid, with the
supplier's reservation APR as the reserve. The point of the artifact is not the auction,
which is textbook. It is to find where the textbook stops applying once you add a house that
runs the venue, a buyer who can self-fund, and agents that are language models.

## The spine: truthfulness is a funder-only property

Second-price clearing makes truthful bidding a weakly dominant strategy. That guarantee is
real and the tests prove it numerically. But it is a guarantee for one party. It says
nothing about whether the supplier sets an honest reserve, whether the buyer reveals its
true cost of self-funding, or whether the house games the venue it operates. The mechanism
buys honesty from the bidders and nothing from anyone else. Every finding below lives in that
gap.

## Findings

**1. Efficiency is necessary, not sufficient, so measure supplier share.** Under truthful
bidding the lowest-cost funder always wins, so allocative efficiency is 1.000 at every
financier count. That number is a trap: a market can be perfectly efficient and still route
the surplus away from the supplier. The metric that moves is the supplier's share of the
realized surplus, which rises with competition (0.885 to 0.917 in the default scenario as the
financier pool grows). Efficiency tells you the pie is whole. Supplier share tells you who
ate it.

**2. The LLM does not overbid, but that table is a non-result, not a finding.** Given a
known private cost, a Claude Haiku agent does not overbid the way humans do in Vickrey
experiments: across 120 bids the mean signed deviation is about zero and 100% land within 50
bps of truthful. Presenting that as evidence the agent plays the dominant strategy would be a
mistake, and a sharp reader would catch it in one question: did the prompt name the optimal
strategy? It did. The probe's system prompt states that truthful bidding is optimal, so the
near-zero deviation measures instruction-following, not reasoning. It is exactly the number
you would get from a model that read its cost out of the prompt and echoed it back. The test
that separates the two is the first-price counterfactual: the same agent under a first-price
auction with a neutral prompt that states the payment rule and recommends nothing. Truthful
is no longer optimal there, so a reasoner shades its bid up to capture margin and a
prompt-follower stays put. That experiment is now built (clear_first_price, the neutral
prompts, the side-by-side probe) and awaits one live run. Until those two numbers sit beside
each other the honest claim is only "the agent does not overbid." Shaded-up-under-first means
the agent reasons and the project lands; truthful-under-both means it followed coaching, which
is itself a finding worth reporting honestly. Either is worth more than the table alone.

**3. A house that runs the venue extracts on price, and a fair-rate index catches it.** This
only shows up where the house is the marginal funder, so it runs on a scenario built for it:
a cheap buyer wins every invoice and the house is the price-setting second-lowest bid at its
honest cost. A house that posts that policy blind helps suppliers by adding competition
(supplier share 0.754 to 0.855). A house that peeks at sealed bids extracts by withholding:
it cannot beat the buyer, so it bids just under the reserve, deletes the bid that set the
clearing, and pushes supplier share back to 0.754 on all 50 invoices. Efficiency stays 1.000
the whole time, because the buyer still wins. This is the dangerous case: the harm is
invisible to an efficiency metric and invisible to a no-house baseline, which the withholding
reverts to. The defense is a benchmark, not a rule. The fair-rate index compares the clearing
to what a structurally separated, honest house would have produced and flags every one of the
50 extracted invoices. The same index flags a financier ring. And the discipline that
defeats both is competition itself: in the default market a cheap buyer sits below the house,
so withholding and collusion move nothing and the index stays quiet.

**4. Structural separation has an allocative price, and it is a cliff, not a slope.** The
house-extraction defense asks the house to post a signed, deterministic policy instead of
bidding freely. If that policy is risk-priced (a base cost plus loadings) rather than bare
cost, the markup has an allocative cost, and the shape of that cost is the interesting part.
In a market where the house is the lowest-cost funder, the markup is free as long as the
quoted APR stays inside the house's cost advantage over the next funder: the house still wins
every invoice it should, and efficiency holds at 1.000. The moment the loaded quote crosses
the next-cheapest funder, the house loses those invoices to a higher-cost financier and
efficiency falls off a step. In a measured case (house cost 0.08, next funder 0.10),
efficiency stays at 1.000 through loadings up to 0.02 and drops to 0.923 at 0.04. So the price
of separation is not a smooth tax you can wave off; it is zero up to a threshold and then a
discrete drop. Setting policy margin is choosing how much allocative efficiency to spend, and
the safe region is bounded by the house's true cost advantage, which the operator may not even
know. This is the second-order effect the role exists to reason about: a defense (separation)
that is sound on price has a hidden, non-linear cost on allocation.

## What I would build next, and the questions that matter

The first-price counterfactual is built; the single thing standing between this project and a
real claim about agent reasoning is one live run of it, reading the two deviation means side
by side. After that the open questions are not modules, they are decisions. What makes threshold-setting smart for the parties the mechanism
does not protect, the supplier and the buyer? How do you construct a fair-rate benchmark that
a supplier can actually trust, given that the honest-house counterfactual here is something
only the operator can compute? And where should a signed deterministic policy end and an
LLM's discretion begin, given that the output-validation clamp provably caps an injected bid
but says nothing about whether the bid inside the bounds is the right one? Those are the
questions worth a conversation. The auction is solved; the governance around it is not.

## Limits

This is a toy built to pressure-test the mechanism, not a production system. The truthfulness
result is one model under one coaching prompt. The input-separation defense (delimiting the
untrusted memo as data) is structural here; its effect on a live model is measured, not
proven, unlike the output clamp. The extraction result is demonstrated in a scenario
deliberately built to make the house pivotal, which is the honest way to show a mechanism: the
default market is the control where the same attack does nothing.
