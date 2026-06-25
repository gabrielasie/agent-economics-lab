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

**2. The LLM agents reason about the mechanism, not just the prompt.** Two experiments make
this airtight. First, the coaching probe: told that truthful bidding is optimal, a Claude
Haiku agent bids its cost almost exactly, mean deviation about zero across 120 bids. On its
own that is a non-result, because the prompt named the answer; a sharp reader would dismiss it
in one question. The first-price counterfactual removes the coaching and isolates the
variable. The same agents bid under a neutral prompt that states only the payment rule, once
for second-price and once for first-price. Under second-price they stay truthful (mean signed
deviation -0.00001). Under first-price, where bidding true cost earns no margin, they shade
their bids up by +0.034 APR on average. The only thing that changed between the two runs is
the payment rule, and the agents responded to it correctly, shading exactly where truthful
stops being optimal and nowhere else. That is a strategic response to the rule, not an echo of
a coached answer: these agents reason about the mechanism. It is the claim the truthfulness
table alone could never support.

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

**5. The fee base decides whether the venue's own incentive fights extraction or funds it.**
The venue has to charge a fee, and the base it charges on is a design choice with teeth.
Conservation pins the two sides: on each invoice, supplier share plus winner-rent share equals
one. So a fee on the supplier's surplus and a fee on the spread move in opposite directions
when the house withholds. Charge on the supplier's surplus and the fee shrinks exactly when the
supplier is squeezed, so withholding to lift the clearing costs the venue its own revenue:
extraction is self-defeating. Charge on the spread and the fee grows when the supplier is
squeezed, so the fee structure literally pays the venue to extract. The point of "make
buyer-side extraction structurally impossible" is not a promise or a disclosure; it is choosing
the fee base so the venue earns most when the supplier does.

## What I would build next, and the questions that matter

The first-price counterfactual is run and the agents reason, so the headline question is
answered. The open questions now are not modules, they are decisions. What makes
threshold-setting smart for the parties the mechanism
does not protect, the supplier and the buyer? How do you construct a fair-rate benchmark that
a supplier can actually trust, given that the honest-house counterfactual here is something
only the operator can compute? And where should a signed deterministic policy end and an
LLM's discretion begin, given that the output-validation clamp provably caps an injected bid
but says nothing about whether the bid inside the bounds is the right one? Those are the
questions worth a conversation. The auction is solved; the governance around it is not.

## Limits

This is a toy built to pressure-test the mechanism, not a production system. The reasoning
result is one model (Claude Haiku) under one neutral prompt: it shows this agent responds to
the payment rule, not that every agent or framing would. The input-separation defense
(delimiting the untrusted memo as data) is structural here; its effect on a live model is
measured, not proven, unlike the output clamp. The extraction result is demonstrated in a scenario
deliberately built to make the house pivotal, which is the honest way to show a mechanism: the
default market is the control where the same attack does nothing.
