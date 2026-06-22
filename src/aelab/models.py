"""Typed, frozen data models for the auction.

Pure core. No LLM, no edge, no plotting. These are the value types the auction,
economics, metrics, and population code all share.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum


class FunderKind(Enum):
    """What kind of funding source a bidder is.

    Funding is heterogeneous: third-party financiers, the buyer self-funding to
    capture the discount, and the house's own book all bid on one venue. The kind
    is load-bearing: attacks single out the house and metrics attribute surplus by it.
    """

    FINANCIER = "financier"
    BUYER = "buyer"
    HOUSE = "house"


@dataclass(frozen=True)
class Invoice:
    """An approved invoice offered for early payment.

    days_early is N - t: the number of days the financing covers. The memo is
    untrusted free text from a counterparty. It is stored verbatim and is never
    interpreted as instructions by the core.

    buyer_credit and dilution_risk are risk scores in [0, 1]. Higher means more risk:
    a weaker buyer (buyer_credit) or a higher chance the invoice is reduced after funding
    (dilution_risk). They feed PricingPolicy.quote and never touch the auction directly;
    each adds to a policy's quoted APR. Both default to 0, meaning no risk loading.
    """

    invoice_id: str
    face_value: float
    days_early: int
    memo: str = ""
    buyer_credit: float = 0.0
    dilution_risk: float = 0.0

    def __post_init__(self) -> None:
        if self.face_value <= 0:
            raise ValueError("face_value must be positive")
        if self.days_early <= 0:
            raise ValueError("days_early must be positive")
        if not 0.0 <= self.buyer_credit <= 1.0:
            raise ValueError("buyer_credit must be in [0, 1]")
        if not 0.0 <= self.dilution_risk <= 1.0:
            raise ValueError("dilution_risk must be in [0, 1]")


@dataclass(frozen=True)
class Funder:
    """A funding source that bids in the auction.

    true_cost_apr is the funder's private cost of capital, the APR a truthful
    bidder would submit.
    """

    party_id: str
    true_cost_apr: float
    kind: FunderKind = FunderKind.FINANCIER

    def __post_init__(self) -> None:
        if self.true_cost_apr < 0:
            raise ValueError("true_cost_apr must be non-negative")


@dataclass(frozen=True)
class Supplier:
    """The party selling the invoice. Sets the reserve, never bids.

    reservation_apr is the supplier's best outside funding option. It becomes the
    auction reserve: bids above it are ineligible.
    """

    party_id: str
    reservation_apr: float

    def __post_init__(self) -> None:
        if self.reservation_apr < 0:
            raise ValueError("reservation_apr must be non-negative")


@dataclass(frozen=True)
class Bid:
    """A sealed bid: a funder offering to fund at this APR.

    rationale is optional free text explaining the bid. Deterministic agents fill it with
    the policy version and content hash; LLM agents fill it with the model's stated
    reasoning. It never affects clearing, which depends only on bidder_id and apr.
    """

    bidder_id: str
    apr: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.apr < 0:
            raise ValueError("apr must be non-negative")


@dataclass(frozen=True)
class AuctionResult:
    """The outcome of clearing one auction.

    Under second-price, clearing_apr is what the winner is paid and differs from
    the winner's own bid (winning_bid_apr). When no eligible bidder clears, traded
    is False and the optional fields are None.
    """

    traded: bool
    winner_id: str | None
    clearing_apr: float | None
    winning_bid_apr: float | None
    num_eligible: int

    def __post_init__(self) -> None:
        outcome = (self.winner_id, self.clearing_apr, self.winning_bid_apr)
        if self.traded and any(x is None for x in outcome):
            raise ValueError("a trade must have a winner, clearing APR, and winning bid")
        if not self.traded and any(x is not None for x in outcome):
            raise ValueError("no trade must leave winner, clearing APR, and winning bid unset")

    @classmethod
    def no_trade(cls, num_eligible: int = 0) -> AuctionResult:
        """No eligible bidder cleared."""
        return cls(
            traded=False,
            winner_id=None,
            clearing_apr=None,
            winning_bid_apr=None,
            num_eligible=num_eligible,
        )

    @classmethod
    def cleared(
        cls, winner_id: str, clearing_apr: float, winning_bid_apr: float, num_eligible: int
    ) -> AuctionResult:
        """A winner cleared at clearing_apr."""
        return cls(
            traded=True,
            winner_id=winner_id,
            clearing_apr=clearing_apr,
            winning_bid_apr=winning_bid_apr,
            num_eligible=num_eligible,
        )


@dataclass(frozen=True)
class PricingPolicy:
    """A versioned, content-hashed pricing policy.

    Mirrors the protocol's signed, versioned, deterministic pricing policies. quote turns
    an invoice into a bid: a base APR plus additive risk loadings, clamped to the output
    bounds. The bounds are the validation boundary: clamp caps how far any bid, including a
    compromised agent's, can move. content_hash is the signature stand-in over every field.
    """

    version: str
    min_apr: float
    max_apr: float
    base_apr: float = 0.0
    buyer_credit_loading: float = 0.0
    dilution_loading: float = 0.0

    def __post_init__(self) -> None:
        if self.min_apr < 0:
            raise ValueError("min_apr must be non-negative")
        if self.max_apr <= self.min_apr:
            raise ValueError("max_apr must be greater than min_apr")
        if self.base_apr < 0:
            raise ValueError("base_apr must be non-negative")
        if self.buyer_credit_loading < 0 or self.dilution_loading < 0:
            raise ValueError("risk loadings must be non-negative")

    @property
    def content_hash(self) -> str:
        """Stable sha256 over the defining fields. Independent of object identity."""
        canonical = json.dumps(
            {
                "version": self.version,
                "min_apr": self.min_apr,
                "max_apr": self.max_apr,
                "base_apr": self.base_apr,
                "buyer_credit_loading": self.buyer_credit_loading,
                "dilution_loading": self.dilution_loading,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def clamp(self, apr: float) -> float:
        """Clamp an APR into [min_apr, max_apr]. The output-validation boundary."""
        return min(max(apr, self.min_apr), self.max_apr)

    def quote(self, invoice: Invoice) -> float:
        """Deterministic APR for an invoice: base plus risk loadings, then clamped.

        The signed-policy bid a structurally separated house posts, blind to competitors.
        Monotonic in each risk input because the loadings are non-negative.
        """
        raw = (
            self.base_apr
            + self.buyer_credit_loading * invoice.buyer_credit
            + self.dilution_loading * invoice.dilution_risk
        )
        return self.clamp(raw)
