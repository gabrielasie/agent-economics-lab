"""Tests for the sealed-bid second-price reverse auction.

The headline is a hypothesis property proving truthful bidding is weakly dominant
for financiers, the guarantee that justifies the mechanism. The mechanics and the
seeded tie-break are covered alongside it.
"""

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aelab.auction import clear_auction, clear_first_price
from aelab.economics import financing_cost
from aelab.models import Bid

aprs = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
reserves = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
faces = st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)
tenors = st.integers(min_value=1, max_value=365)
seeds = st.integers(min_value=0, max_value=2**32 - 1)


def _bids(apr_list: list[float]) -> list[Bid]:
    return [Bid(bidder_id=f"c{i}", apr=a) for i, a in enumerate(apr_list)]


# --- mechanics ----------------------------------------------------------------


def test_empty_bids_no_trade() -> None:
    r = clear_auction([], 0.20, random.Random(0))
    assert r.traded is False
    assert r.num_eligible == 0


def test_all_above_reserve_no_trade() -> None:
    r = clear_auction(_bids([0.30, 0.40, 0.50]), 0.20, random.Random(0))
    assert r.traded is False
    assert r.num_eligible == 0


def test_single_eligible_clears_at_reserve() -> None:
    bids = [Bid("a", 0.10), Bid("b", 0.30), Bid("c", 0.40)]
    r = clear_auction(bids, 0.20, random.Random(0))
    assert r.traded is True
    assert r.winner_id == "a"
    assert r.winning_bid_apr == 0.10
    assert r.clearing_apr == 0.20  # clears at the reserve
    assert r.num_eligible == 1


def test_second_price_among_eligible() -> None:
    bids = [Bid("a", 0.10), Bid("b", 0.15), Bid("c", 0.20)]
    r = clear_auction(bids, 0.25, random.Random(0))
    assert r.winner_id == "a"
    assert r.winning_bid_apr == 0.10
    assert r.clearing_apr == 0.15  # second-lowest eligible
    assert r.num_eligible == 3


def test_bid_at_reserve_is_eligible() -> None:
    r = clear_auction([Bid("a", 0.20)], 0.20, random.Random(0))
    assert r.traded is True
    assert r.winner_id == "a"
    assert r.clearing_apr == 0.20
    assert r.num_eligible == 1


@given(apr_list=st.lists(aprs, min_size=0, max_size=6), reserve=reserves)
def test_num_eligible_counts_bids_at_or_below_reserve(apr_list: list[float], reserve: float) -> None:
    r = clear_auction(_bids(apr_list), reserve, random.Random(0))
    assert r.num_eligible == sum(1 for a in apr_list if a <= reserve)


def test_duplicate_bidder_ids_raise() -> None:
    with pytest.raises(ValueError):
        clear_auction([Bid("dup", 0.10), Bid("dup", 0.15)], 0.20, random.Random(0))


# --- invariants when traded ---------------------------------------------------


@given(apr_list=st.lists(aprs, min_size=1, max_size=6), reserve=reserves, seed=seeds)
def test_traded_invariants(apr_list: list[float], reserve: float, seed: int) -> None:
    r = clear_auction(_bids(apr_list), reserve, random.Random(seed))
    eligible = [a for a in apr_list if a <= reserve]
    if not eligible:
        assert r.traded is False
        return
    assert r.traded is True
    assert r.clearing_apr is not None and r.winning_bid_apr is not None
    assert r.clearing_apr <= reserve + 1e-12
    assert r.winning_bid_apr == min(eligible)
    assert r.winning_bid_apr <= r.clearing_apr + 1e-12


# --- determinism and tie-breaking ---------------------------------------------


@given(apr_list=st.lists(aprs, min_size=1, max_size=6), reserve=reserves, seed=seeds)
def test_reproducible_given_seed(apr_list: list[float], reserve: float, seed: int) -> None:
    bids = _bids(apr_list)
    assert clear_auction(bids, reserve, random.Random(seed)) == clear_auction(
        bids, reserve, random.Random(seed)
    )


def test_tie_winner_is_one_of_tied_and_clears_at_that_apr() -> None:
    bids = [Bid("a", 0.10), Bid("b", 0.10), Bid("c", 0.20)]
    winners = set()
    for seed in range(50):
        r = clear_auction(bids, 0.25, random.Random(seed))
        assert r.winner_id in {"a", "b"}
        assert r.winning_bid_apr == 0.10
        assert r.clearing_apr == 0.10  # second-lowest equals the lowest under a tie
        winners.add(r.winner_id)
    assert winners == {"a", "b"}  # both tied bidders can win, depending on the seed


@given(seed=seeds)
def test_tie_break_independent_of_input_order(seed: int) -> None:
    forward = [Bid("a", 0.10), Bid("b", 0.10), Bid("c", 0.20)]
    shuffled = [Bid("c", 0.20), Bid("b", 0.10), Bid("a", 0.10)]
    assert clear_auction(forward, 0.25, random.Random(seed)) == clear_auction(
        shuffled, 0.25, random.Random(seed)
    )


# --- first-price reverse auction ----------------------------------------------


def test_first_price_winner_pays_own_bid() -> None:
    bids = [Bid("a", 0.10), Bid("b", 0.15), Bid("c", 0.20)]
    r = clear_first_price(bids, 0.25, random.Random(0))
    assert r.winner_id == "a"
    assert r.winning_bid_apr == 0.10
    assert r.clearing_apr == 0.10  # paid its own bid, not the second-lowest
    assert r.num_eligible == 3


def test_first_price_single_eligible_pays_own_bid_not_reserve() -> None:
    # The distinguishing case: under second-price this clears at the reserve (0.20),
    # under first-price the lone winner is paid its own bid (0.10).
    bids = [Bid("a", 0.10), Bid("b", 0.30), Bid("c", 0.40)]
    r = clear_first_price(bids, 0.20, random.Random(0))
    assert r.winner_id == "a"
    assert r.winning_bid_apr == 0.10
    assert r.clearing_apr == 0.10
    assert r.num_eligible == 1


def test_first_price_no_eligible_no_trade() -> None:
    r = clear_first_price(_bids([0.30, 0.40]), 0.20, random.Random(0))
    assert r.traded is False
    assert r.num_eligible == 0


def test_first_price_empty_no_trade() -> None:
    r = clear_first_price([], 0.20, random.Random(0))
    assert r.traded is False


def test_first_price_bid_at_reserve_is_eligible() -> None:
    r = clear_first_price([Bid("a", 0.20)], 0.20, random.Random(0))
    assert r.traded is True
    assert r.clearing_apr == 0.20  # own bid, which happens to equal the reserve here


def test_first_price_duplicate_bidder_ids_raise() -> None:
    with pytest.raises(ValueError):
        clear_first_price([Bid("dup", 0.10), Bid("dup", 0.15)], 0.20, random.Random(0))


@given(apr_list=st.lists(aprs, min_size=1, max_size=6), reserve=reserves, seed=seeds)
def test_first_price_clearing_equals_winning_bid(
    apr_list: list[float], reserve: float, seed: int
) -> None:
    r = clear_first_price(_bids(apr_list), reserve, random.Random(seed))
    if r.traded:
        assert r.clearing_apr == r.winning_bid_apr  # the defining first-price property
        assert r.winning_bid_apr == min(a for a in apr_list if a <= reserve)
        assert r.clearing_apr is not None and r.clearing_apr <= reserve + 1e-12


@given(apr_list=st.lists(aprs, min_size=1, max_size=6), reserve=reserves, seed=seeds)
def test_first_price_reproducible_given_seed(
    apr_list: list[float], reserve: float, seed: int
) -> None:
    bids = _bids(apr_list)
    assert clear_first_price(bids, reserve, random.Random(seed)) == clear_first_price(
        bids, reserve, random.Random(seed)
    )


def test_first_price_tie_winner_is_one_of_tied() -> None:
    bids = [Bid("a", 0.10), Bid("b", 0.10), Bid("c", 0.20)]
    winners = set()
    for seed in range(50):
        r = clear_first_price(bids, 0.25, random.Random(seed))
        assert r.winner_id in {"a", "b"}
        assert r.clearing_apr == 0.10
        winners.add(r.winner_id)
    assert winners == {"a", "b"}


# --- headline: truthful bidding is weakly dominant ----------------------------


def _focal_profit(
    cost: float,
    bid: float,
    competitors: list[float],
    reserve: float,
    face: float,
    days: int,
    seed: int,
) -> float:
    """Profit to a focal funder with true cost `cost` when it submits `bid`.

    The funder earns the discount at the clearing APR and pays its true cost, so
    profit is financing_cost(clearing) - financing_cost(cost) when it wins, else 0.
    """
    bids = [Bid("focal", bid)] + [Bid(f"c{i}", a) for i, a in enumerate(competitors)]
    r = clear_auction(bids, reserve, random.Random(seed))
    if r.traded and r.winner_id == "focal":
        assert r.clearing_apr is not None
        return financing_cost(face, r.clearing_apr, days) - financing_cost(face, cost, days)
    return 0.0


@given(
    cost=aprs,
    deviation=aprs,
    competitors=st.lists(aprs, min_size=0, max_size=5),
    reserve=reserves,
    face=faces,
    days=tenors,
    seed=seeds,
)
def test_truthful_bidding_is_weakly_dominant(
    cost: float,
    deviation: float,
    competitors: list[float],
    reserve: float,
    face: float,
    days: int,
    seed: int,
) -> None:
    truthful = _focal_profit(cost, cost, competitors, reserve, face, days, seed)
    deviated = _focal_profit(cost, deviation, competitors, reserve, face, days, seed)
    assert deviated <= truthful + 1e-9


@given(
    cost=aprs,
    competitors=st.lists(aprs, min_size=0, max_size=5),
    reserve=reserves,
    face=faces,
    days=tenors,
    seed=seeds,
)
def test_truthful_profit_never_negative(
    cost: float,
    competitors: list[float],
    reserve: float,
    face: float,
    days: int,
    seed: int,
) -> None:
    assert _focal_profit(cost, cost, competitors, reserve, face, days, seed) >= -1e-9
