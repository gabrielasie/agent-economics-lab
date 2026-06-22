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

## Three findings

**1. Efficiency is necessary, not sufficient, so measure supplier share.** Under truthful
bidding the lowest-cost funder always wins, so allocative efficiency is 1.000 at every
financier count. That number is a trap: a market can be perfectly efficient and still route
the surplus away from the supplier. The metric that moves is the supplier's share of the
realized surplus, which rises with competition (0.885 to 0.917 in the default scenario as the
financier pool grows). Efficiency tells you the pie is whole. Supplier share tells you who
ate it.

**2. The LLM bids the dominant strategy, but I cannot yet say it reasons.** Humans famously
overbid in Vickrey experiments. Given a known private cost, a Claude Haiku agent does not:
across 120 bids the mean signed deviation is about zero and 100% land within 50 bps of
truthful. The honest caveat is the whole story here. The probe's prompt tells the model that
truthful bidding is optimal, which means the result is consistent with reasoning and equally
consistent with following instructions. The test that separates them is the first-price
counterfactual: the same agent under a first-price auction with a neutral prompt, where
truthful is no longer optimal and a reasoner shades its bid up. That experiment is designed
and not yet run. Until it is, the claim is "the agent does not overbid," not "the agent
understands the mechanism."

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

## One thing the modeling surfaced

When the house posts a risk-priced quote with margin instead of its bare cost, the margin can
price it out of trades it would have won at cost, and allocative efficiency drops below 1.000
in the regimes where the house is competitive. Margin and allocation trade off. It is a small
effect here, but it is the seam where "a signed deterministic policy" stops being free: every
basis point of policy margin is a basis point of allocative risk, and that is a design dial,
not a constant.

## What I would build next, and the questions that matter

The first-price counterfactual is the next build, because it converts finding 2 from a
behavioral observation into a claim about reasoning. After that the open questions are not
modules, they are decisions. What makes threshold-setting smart for the parties the mechanism
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
