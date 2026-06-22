"""House-venue extraction: the centerpiece attack and its fair-rate-index defense.

A HOUSE funder competes in its own auction under two regimes:

- Separated: the house bids a signed, deterministic policy, blind to competitors
  (structural separation). It bids its clamped true cost and ignores ctx.leaked.
- Informed: the house peeks at sealed competitor bids via ctx.leaked. When it can win
  profitably it undercuts the lowest competitor; otherwise it withholds its competitive
  bid (bids just under the reserve) so the clearing rises toward the reserve and the
  supplier pays more.

The fair-rate index is the defense. In a reverse second-price auction an extra bid can
only lower or hold the clearing, so the house can never push the rate above the honest
benchmark by *adding* a bid; it extracts only by *withholding*. The index counts invoices
where a regime's clearing exceeds the clearing the honest (separated) house would have
produced. The sharp result: extraction is on price, not efficiency. The lowest-cost funder
still wins, so the market stays ~100% efficient while supplier surplus is routed away.

Pure core attack: imports models, agents, auction, engine, metrics, populations. No edge.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from aelab.agents.base import AuctionContext, AuctionRules, BiddingAgent, LeakedInfo
from aelab.agents.deterministic import PolicyAgent, TruthfulAgent
from aelab.auction import clear_auction
from aelab.engine import run
from aelab.metrics import InvoiceOutcome, summarize
from aelab.models import Bid, Funder, FunderKind, PricingPolicy
from aelab.populations import Population

EPSILON = 1e-4  # the "just under / just below" margin for the informed house


@dataclass(frozen=True)
class InformedHouseAgent:
    """Adversarial house that peeks at sealed competitor bids via ctx.leaked.

    Undercuts the lowest eligible competitor when it can win profitably; otherwise bids
    just under the reserve, withholding its competitive bid so the clearing rises.
    """

    party: Funder
    epsilon: float = EPSILON

    def bid(self, ctx: AuctionContext) -> Bid:
        reserve = ctx.rules.reserve_apr
        cost = self.party.true_cost_apr
        eligible = [b.apr for b in ctx.leaked.competitor_bids if b.apr <= reserve]
        if not eligible:
            apr = cost  # only funder: bidding the true cost clears at the reserve
        elif min(eligible) > cost:
            apr = max(0.0, min(eligible) - self.epsilon)  # undercut to win profitably
        else:
            apr = max(0.0, reserve - self.epsilon)  # withhold to lift the clearing
        return Bid(bidder_id=self.party.party_id, apr=apr)


@dataclass(frozen=True)
class RegimeReport:
    """Market metrics for one house regime, with the fair-rate-index flag count."""

    regime: str
    efficiency: float
    supplier_share: float
    fair_rate_index_flags: int


def _the_house(population: Population) -> Funder:
    houses = [f for f in population.funders if f.kind is FunderKind.HOUSE]
    if len(houses) != 1:
        raise ValueError("population must contain exactly one HOUSE funder")
    return houses[0]


def _separated_factory(policy: PricingPolicy) -> Callable[[Funder], BiddingAgent]:
    def factory(funder: Funder) -> BiddingAgent:
        if funder.kind is FunderKind.HOUSE:
            return PolicyAgent(funder, policy)
        return TruthfulAgent(funder)

    return factory


def run_separated(
    population: Population, policy: PricingPolicy, rng: random.Random
) -> list[InvoiceOutcome]:
    """The honest regime: the house bids its signed policy, blind, like any other funder."""
    return run(population, _separated_factory(policy), rng)


def run_informed(population: Population, rng: random.Random) -> list[InvoiceOutcome]:
    """The adversarial regime: competitor bids are leaked to the house before it bids."""
    house_agent = InformedHouseAgent(_the_house(population))
    competitors = [f for f in population.funders if f.kind is not FunderKind.HOUSE]
    competitor_agents = [TruthfulAgent(f) for f in competitors]
    outcomes: list[InvoiceOutcome] = []
    for invoice, supplier in zip(population.invoices, population.suppliers, strict=True):
        rules = AuctionRules(reserve_apr=supplier.reservation_apr)
        clean = AuctionContext(invoice=invoice, rules=rules)
        competitor_bids = [agent.bid(clean) for agent in competitor_agents]
        leaked_ctx = AuctionContext(
            invoice=invoice,
            rules=rules,
            leaked=LeakedInfo(competitor_bids=tuple(competitor_bids)),
        )
        bids = [*competitor_bids, house_agent.bid(leaked_ctx)]
        result = clear_auction(bids, supplier.reservation_apr, rng)
        outcomes.append(
            InvoiceOutcome(
                invoice=invoice, supplier=supplier, funders=population.funders, result=result
            )
        )
    return outcomes


def fair_rate_index(
    outcomes: Sequence[InvoiceOutcome], baseline: Sequence[InvoiceOutcome]
) -> int:
    """Count invoices where a regime's clearing exceeds the honest (baseline) clearing."""
    flags = 0
    for outcome, base in zip(outcomes, baseline, strict=True):
        clearing = outcome.result.clearing_apr
        honest = base.result.clearing_apr
        if clearing is not None and honest is not None and clearing > honest + EPSILON:
            flags += 1
    return flags


def run_regimes(
    population: Population, policy: PricingPolicy, rng: random.Random
) -> dict[str, RegimeReport]:
    """Run both house regimes and report metrics, indexing each against the honest house."""
    separated = run_separated(population, policy, rng)
    informed = run_informed(population, rng)
    sep, inf = summarize(separated), summarize(informed)
    return {
        "separated": RegimeReport(
            "separated", sep.allocative_efficiency, sep.supplier_share,
            fair_rate_index(separated, separated),
        ),
        "informed": RegimeReport(
            "informed", inf.allocative_efficiency, inf.supplier_share,
            fair_rate_index(informed, separated),
        ),
    }
