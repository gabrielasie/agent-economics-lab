"""Streamlit UI for the Agent Economics Lab.

A thin presentation edge over the aelab package. It runs the deterministic experiments live
(efficiency, house extraction, the interactive auction, the plain-English terms calculator) and
replays the LLM experiments from saved bids, so it needs no API key. It imports only the public
aelab orchestration; the pure core is untouched, and because this file lives outside the aelab
package the import contracts still hold.

Run locally:  uv sync --extra ui && uv run streamlit run app.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from aelab.agents.llm import build_counterfactual_requests, counterfactual_deviations
from aelab.auction import clear_auction, clear_first_price
from aelab.cli import arena_bids, attack_population, compute_deviations, compute_efficiency
from aelab.config import load_scenario
from aelab.economics import apr_to_discount, financing_cost, surplus_split
from aelab.harness import run_harness
from aelab.metrics import deviation_stats
from aelab.models import Bid, Funder, Invoice
from aelab.populations import generate_financiers, generate_invoices

st.set_page_config(page_title="Agent Economics Lab", page_icon="📈", layout="wide")

SCENARIOS_DIR = Path("scenarios")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PROBE_SCENARIO = "default"  # the scenario the saved truthfulness/first-price bids were generated under
ACCENT = "#4f46e5"


def scenario_names() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.toml"))


def find_raw(filename: str) -> Path | None:
    """Prefer a committed bundle in data/, fall back to a local run in results/."""
    for base in (DATA_DIR, RESULTS_DIR):
        path = base / filename
        if path.exists():
            return path
    return None


def get_api_key() -> str | None:
    """Read the Anthropic key from the environment or Streamlit secrets, if either is set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def efficiency_frame(scenario_name: str) -> pd.DataFrame:
    counts, efficiency, supplier_share = compute_efficiency(load_scenario(scenario_name))
    return pd.DataFrame(
        {
            "financiers": counts,
            "allocative efficiency": efficiency,
            "supplier share": supplier_share,
        }
    ).set_index("financiers")


@st.cache_data(show_spinner=False)
def harness_frame(scenario_name: str) -> pd.DataFrame:
    scenario = load_scenario(scenario_name)
    rows = run_harness(attack_population(scenario), scenario.policy, random.Random(scenario.seed + 3))
    return pd.DataFrame(
        [
            {
                "regime": r.regime,
                "efficiency": r.efficiency,
                "supplier share": r.supplier_share,
                "index flags": r.fair_rate_index_flags,
            }
            for r in rows
        ]
    ).set_index("regime")


# --- header -------------------------------------------------------------------

st.title("Agent Economics Lab")
st.markdown(
    "An MVP of agent-to-agent invoice early-payment: when a buyer approves an invoice, the "
    "supplier's agent and competing underwriters enter a **sealed-bid, second-price reverse "
    "auction** that clears in seconds. The mechanism's job is to make truthful behaviour the "
    "smart play for every party. This lab calibrates it, prices the house's risk policy, and "
    "tries to break the incentives before they ship."
)

cards = [
    (
        "Mechanism",
        "Sealed-bid second-price reverse auction. Truthful bidding is the dominant strategy "
        "for funders, proven numerically.",
    ),
    (
        "Incentive integrity",
        "A fair-rate index, benchmarked against an honest separated house, catches the venue "
        "extracting from suppliers.",
    ),
    (
        "Agentic AI",
        "LLM funders (Claude) bid through a typed seam. A first-price counterfactual tests "
        "real reasoning against prompt-following.",
    ),
    (
        "Adversarial",
        "Information leakage, financier collusion, and prompt injection, each measured "
        "against a defense.",
    ),
]
card_cols = st.columns(4)
for col, (heading, body) in zip(card_cols, cards, strict=True):
    with col.container(border=True):
        st.markdown(f"**{heading}**")
        st.caption(body)

st.divider()

API_KEY = get_api_key()
if API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = API_KEY

with st.sidebar:
    st.header("Controls")
    names = scenario_names()
    scenario_name = st.selectbox(
        "Scenario (efficiency and extraction panels)",
        names,
        index=names.index("default") if "default" in names else 0,
    )
    st.caption(
        "The truthfulness and first-price panels are pinned to the `default` scenario, the "
        "population the saved bids were generated under."
    )
    st.divider()
    if API_KEY:
        st.success("API key detected: live LLM runs are enabled.")
    else:
        st.caption(
            "No API key. Deterministic panels run live; LLM panels replay committed bids. To "
            "enable live runs, set ANTHROPIC_API_KEY (see .streamlit/secrets.toml.example)."
        )

tab_arena, tab_play, tab_eff, tab_house, tab_truth, tab_first, tab_terms = st.tabs(
    [
        "Agent arena",
        "Try the auction",
        "Efficiency",
        "Incentive integrity",
        "Truthfulness",
        "First-price",
        "Plain-English terms",
    ]
)

# --- agent arena: Claude funders bid against each other ------------------------

with tab_arena:
    st.subheader("Claude agents bid against each other")
    st.write(
        "Several funders, each a Claude agent, privately bid on one invoice. This is a "
        "sealed-bid auction, so they never see each other: every agent gets only its own cost of "
        "capital and the invoice, then submits one APR with a rationale. Below is a real field of "
        "Claude bids; the auction clears to a winner and a price, and you can read each agent's "
        "reasoning."
    )
    arena_raw = find_raw("truthfulness_raw.json")
    arena_scenario = load_scenario(PROBE_SCENARIO)
    if arena_raw is None:
        st.info("No committed Claude bids found (data/truthfulness_raw.json).")
    else:
        arena_texts = json.loads(arena_raw.read_text(encoding="utf-8"))
        pick = st.columns([1, 1, 2])
        inv_idx = pick[0].slider("Invoice", 0, arena_scenario.probe_n_invoices - 1, 0)
        reserve_a = pick[1].slider("Supplier reserve (APR %)", 5.0, 60.0, 40.0, 1.0) / 100
        invoice, field = arena_bids(arena_scenario, arena_texts, inv_idx)
        result = clear_auction([bid for _, bid in field], reserve_a, random.Random(0))

        table = pd.DataFrame(
            [
                {
                    "agent": funder.party_id,
                    "cost": funder.true_cost_apr,
                    "bid": bid.apr,
                    "winner": bool(result.traded and result.winner_id == funder.party_id),
                }
                for funder, bid in field
            ]
        ).set_index("agent")
        chart_col, win_col = st.columns([3, 2])
        chart_col.bar_chart(table[["bid"]], height=300)
        with win_col:
            if result.traded:
                assert result.clearing_apr is not None and result.winning_bid_apr is not None
                split = surplus_split(
                    invoice.face_value,
                    reserve_a,
                    result.clearing_apr,
                    result.winning_bid_apr,
                    invoice.days_early,
                )
                st.metric("Winner", f"Agent {result.winner_id}")
                st.metric("Clearing APR (paid)", f"{result.clearing_apr:.1%}")
                m = st.columns(2)
                m[0].metric("Supplier surplus", f"EUR {split.supplier_surplus:,.0f}")
                m[1].metric("Winner rent", f"EUR {split.winner_rent:,.0f}")
            else:
                st.warning("No bid cleared under this reserve.")
        st.dataframe(table.style.format({"cost": "{:.1%}", "bid": "{:.1%}"}), width="stretch")

        with st.expander("Read each agent's reasoning"):
            for funder, bid in field:
                crown = " (winner)" if result.traded and result.winner_id == funder.party_id else ""
                st.markdown(f"**Agent {funder.party_id}**{crown} bid {bid.apr:.2%}")
                st.caption(bid.rationale or "(no rationale returned)")

        st.caption(
            "These are real Claude Haiku bids from the committed probe, made under a prompt that "
            "names truthful bidding, so the agents bid close to their cost and the lowest-cost "
            "agent wins. Whether they would still do that under a neutral first-price prompt is "
            "what the First-price tab tests."
        )

    if API_KEY:
        with st.expander("Run a fresh live arena with custom costs (uses your API key)"):
            live = st.columns(4)
            costs_live = [
                live[0].slider("Agent A cost (%)", 1.0, 40.0, 7.0, 0.5, key="arena_a") / 100,
                live[1].slider("Agent B cost (%)", 1.0, 40.0, 10.0, 0.5, key="arena_b") / 100,
                live[2].slider("Agent C cost (%)", 1.0, 40.0, 14.0, 0.5, key="arena_c") / 100,
            ]
            face_live = live[3].number_input(
                "Face value (EUR)", 1_000, 5_000_000, 100_000, 1_000, key="arena_face"
            )
            reserve_live = st.slider("Reserve (APR %)", 5.0, 60.0, 40.0, 1.0, key="arena_res") / 100
            if st.button("Run live arena"):
                from aelab.agents.base import AuctionContext, AuctionRules
                from aelab.agents.cache import AnthropicClient, ResponseCache
                from aelab.agents.llm import LLMAgent

                client = ResponseCache(AnthropicClient())
                ctx = AuctionContext(
                    invoice=Invoice("ARENA", float(face_live), 60),
                    rules=AuctionRules(reserve_apr=reserve_live),
                )
                live_field: list[tuple[Funder, Bid]] = []
                with st.spinner("Asking each Claude agent for a bid..."):
                    for i, cost in enumerate(costs_live):
                        funder = Funder(f"L{i}", cost)
                        live_field.append((funder, LLMAgent(funder, client).bid(ctx)))
                live_result = clear_auction([b for _, b in live_field], reserve_live, random.Random(0))
                if live_result.traded:
                    st.success(f"Winner: Agent {live_result.winner_id} at {live_result.clearing_apr:.1%}")
                for funder, bid in live_field:
                    st.markdown(
                        f"**Agent {funder.party_id}** cost {funder.true_cost_apr:.1%}, "
                        f"bid {bid.apr:.2%}"
                    )
                    st.caption(bid.rationale or "(no rationale returned)")

# --- interactive auction ------------------------------------------------------

with tab_play:
    st.subheader("Clear one auction, live")
    st.write(
        "Set each funder's cost of capital and the supplier's reserve. Funders bid truthfully "
        "(the dominant strategy), and the auction clears. See who wins, what they are paid under "
        "the second-price rule, and how the surplus splits between the supplier and the winner."
    )
    controls, outcome = st.columns([2, 3])
    with controls:
        cost_a = st.slider("Funder A cost (APR %)", 1.0, 40.0, 8.0, 0.5)
        cost_b = st.slider("Funder B cost (APR %)", 1.0, 40.0, 11.0, 0.5)
        cost_c = st.slider("Funder C cost (APR %)", 1.0, 40.0, 16.0, 0.5)
        reserve_pct = st.slider("Supplier reserve (APR %)", 1.0, 60.0, 30.0, 0.5)
        face = st.number_input("Invoice face value (EUR)", 1_000, 5_000_000, 100_000, 1_000)
        days = st.slider("Days paid early", 7, 180, 60, 1)

    costs = {"A": cost_a / 100, "B": cost_b / 100, "C": cost_c / 100}
    reserve = reserve_pct / 100
    bids = [Bid(name, apr) for name, apr in costs.items()]
    second = clear_auction(bids, reserve, random.Random(0))
    first = clear_first_price(bids, reserve, random.Random(0))

    with outcome:
        if not second.traded:
            st.error("No trade: every funder's cost is above the supplier's reserve.")
        else:
            assert second.clearing_apr is not None and second.winning_bid_apr is not None
            split = surplus_split(face, reserve, second.clearing_apr, second.winning_bid_apr, days)
            m = st.columns(3)
            m[0].metric("Winner", f"Funder {second.winner_id}")
            m[1].metric("Clearing APR (paid)", f"{second.clearing_apr:.1%}")
            m[2].metric("Winner's own bid", f"{second.winning_bid_apr:.1%}")
            m2 = st.columns(2)
            m2[0].metric("Supplier surplus", f"EUR {split.supplier_surplus:,.0f}")
            m2[1].metric("Winner rent", f"EUR {split.winner_rent:,.0f}")
            assert first.clearing_apr is not None
            st.caption(
                f"Under a first-price rule the winner would instead be paid its own bid "
                f"({first.clearing_apr:.1%}), so truthful bidding would stop being optimal and "
                f"funders would shade up. That is what the First-price panel tests."
            )

    df = pd.DataFrame(
        {
            "funder": list(costs),
            "bid": list(costs.values()),
            "eligible": [c <= reserve for c in costs.values()],
        }
    )
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("funder:N", title="funder"),
            y=alt.Y("bid:Q", title="bid (APR)", axis=alt.Axis(format="%")),
            color=alt.Color(
                "eligible:N",
                scale=alt.Scale(domain=[True, False], range=[ACCENT, "#c7c9d9"]),
                legend=alt.Legend(title="at or below reserve"),
            ),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"reserve": [reserve]}))
        .mark_rule(color="#dc2626", strokeDash=[6, 4])
        .encode(y="reserve:Q")
    )
    st.altair_chart((bars + rule).properties(height=300), width="stretch")

# --- efficiency ---------------------------------------------------------------

with tab_eff:
    st.subheader("Efficiency and the supplier-share curve")
    st.write(
        "Under truthful bidding the lowest-cost funder always wins, so allocative efficiency is "
        "1.000 at every financier count. The metric that moves is the supplier's share of "
        "surplus, which rises with competition. Efficiency tells you the pie is whole; supplier "
        "share tells you who ate it."
    )
    frame = efficiency_frame(scenario_name)
    chart_col, stat_col = st.columns([3, 1])
    chart_col.line_chart(frame, height=360)
    stat_col.metric("Allocative efficiency", f"{frame['allocative efficiency'].mean():.3f}")
    stat_col.metric(
        "Supplier share (range)",
        f"{frame['supplier share'].min():.3f}",
        delta=f"to {frame['supplier share'].max():.3f}",
    )
    stat_col.caption("More financiers, more competition, a bigger slice for the supplier.")
    st.dataframe(frame.style.format("{:.3f}"), width="stretch")

# --- incentive integrity (house extraction) -----------------------------------

with tab_house:
    st.subheader("Incentive integrity: can the venue extract, and would a supplier see it?")
    st.write(
        "A house that runs the venue can extract by withholding: it peeks at sealed bids and, "
        "when it cannot win, bids just under the reserve to lift the clearing. The extraction is "
        "on price, not allocation, so efficiency stays 1.000 and a no-house baseline cannot see "
        "it. The fair-rate index, benchmarked against the honest separated house, flags it. "
        "Switch the sidebar to `extraction` for the attack and `default` for the market where a "
        "competitive buyer disciplines it."
    )
    frame = harness_frame(scenario_name)
    by = frame.to_dict("index")
    cards2 = st.columns(4)
    for col, regime in zip(cards2, ["baseline", "separated", "informed", "collusion"], strict=False):
        if regime in by:
            with col.container(border=True):
                st.markdown(f"**{regime}**")
                st.metric("supplier share", f"{by[regime]['supplier share']:.3f}")
                st.caption(f"index flags: {int(by[regime]['index flags'])}")
    left, right = st.columns(2)
    left.caption("Supplier share by regime")
    left.bar_chart(frame[["supplier share"]], height=280)
    right.caption("Fair-rate-index flags by regime")
    right.bar_chart(frame[["index flags"]], height=280)
    st.dataframe(
        frame.style.format(
            {"efficiency": "{:.3f}", "supplier share": "{:.3f}", "index flags": "{:d}"}
        ),
        width="stretch",
    )

    st.divider()
    st.markdown("**Fee structure: the base decides the incentive**")
    st.write(
        "The venue has to charge a fee, but the base it charges on sets its incentive. A fee on "
        "the supplier's surplus shrinks when the supplier is squeezed, so withholding costs the "
        "venue its own revenue. A fee on the spread (the winner's rent) grows when the supplier "
        "is squeezed, so that base literally pays the venue to extract. Switch the sidebar to "
        "`extraction` to see the two bases diverge."
    )
    fee_left, fee_right = st.columns([1, 2])
    fee_rate = fee_left.slider("Fee rate (%)", 0.0, 30.0, 10.0, 1.0) / 100
    fee_base = fee_left.radio("Fee base", ["supplier surplus (aligned)", "the spread (extractive)"])
    aligned = fee_base.startswith("supplier")
    fee_df = frame.copy()
    # Conservation: supplier share + winner-rent share = 1 of the realized pie on each invoice.
    fee_df["winner rent share"] = 1.0 - fee_df["supplier share"]
    base_share = fee_df["supplier share"] if aligned else fee_df["winner rent share"]
    fee_df["venue fee"] = fee_rate * base_share
    fee_df["supplier net"] = fee_df["supplier share"] - (fee_df["venue fee"] if aligned else 0.0)
    fee_right.caption("Venue fee revenue by regime (share of the realized pie)")
    fee_right.bar_chart(fee_df[["venue fee"]], height=240)
    if "separated" in fee_df.index and "informed" in fee_df.index:
        delta = float(fee_df.loc["informed", "venue fee"] - fee_df.loc["separated", "venue fee"])
        if aligned:
            st.success(
                f"A surplus-based fee tracks the supplier: going from the honest house to "
                f"withholding moves the venue's fee by {delta:+.4f} of the pie. The venue earns "
                f"most when the supplier does, so it has no reason to withhold. Extraction is "
                f"self-defeating, which is what makes the separation externally defensible."
            )
        else:
            st.error(
                f"A spread-based fee tracks the squeeze: going from the honest house to "
                f"withholding moves the venue's fee by {delta:+.4f} of the pie. This base can pay "
                f"the venue to extract, so do not price the underwriting book on the spread."
            )

# --- truthfulness -------------------------------------------------------------

with tab_truth:
    st.subheader("Do LLM agents bid the dominant strategy?")
    st.warning(
        "This is deliberately a NON-RESULT. The probe's prompt states that truthful bidding is "
        "optimal, so the near-zero deviation measures instruction-following, not reasoning. It is "
        "the number you would get from a model that echoed its cost back. The first-price "
        "counterfactual (next tab) is the test that tells the two apart."
    )
    raw = find_raw("truthfulness_raw.json")
    if raw is None:
        st.info(
            "No saved bids found. Run `uv run aelab truthfulness` with an API key, or commit "
            "`data/truthfulness_raw.json`."
        )
    else:
        scenario = load_scenario(PROBE_SCENARIO)
        texts = json.loads(raw.read_text(encoding="utf-8"))
        deviations, failures = compute_deviations(scenario, texts)
        stats = deviation_stats(deviations, scenario.epsilon)
        cols = st.columns(3)
        cols[0].metric("Bids parsed", stats.n)
        cols[1].metric("Mean signed deviation", f"{stats.mean_signed:+.5f}")
        cols[2].metric(f"Within {scenario.epsilon:.3f}", f"{stats.fraction_within:.0%}")
        if deviations:
            df = pd.DataFrame({"deviation": deviations})
            chart = (
                alt.Chart(df)
                .mark_bar(color=ACCENT)
                .encode(
                    alt.X("deviation:Q", bin=alt.Bin(maxbins=40), title="bid APR minus true cost"),
                    alt.Y("count()", title="bids"),
                )
                .properties(height=320)
            )
            st.altair_chart(chart, width="stretch")
        st.caption(f"Parsed {stats.n} bids with {failures} parse failures, from {raw}.")

# --- first-price counterfactual -----------------------------------------------

with tab_first:
    st.subheader("First-price counterfactual: reasoning or coaching?")
    st.write(
        "The same agents under a first-price auction with a neutral prompt that states the rule "
        "and recommends nothing. Truthful is no longer optimal there, so a reasoner shades its "
        "bid up and a prompt-follower stays put. Shaded-up-under-first means the agent reasons; "
        "truthful-under-both means it followed coaching all along."
    )
    raw = find_raw("first_price_raw.json")
    if raw is None and API_KEY:
        st.caption(
            "Live runs are enabled. Generating bids calls the Anthropic API and spends your key; "
            "on a public deployment every visitor can trigger it. For a shareable demo, prefer "
            "running it once locally and committing data/first_price_raw.json."
        )
        if st.button("Generate first-price bids live (uses your API key)"):
            from aelab.agents.cache import AnthropicBatchClient

            scenario = load_scenario(PROBE_SCENARIO)
            funders = generate_financiers(
                scenario.financier, scenario.probe_n_funders, random.Random(scenario.seed)
            )
            invoices = generate_invoices(
                scenario.invoice, scenario.probe_n_invoices, random.Random(scenario.seed + 1)
            )
            pairs = [(funder, invoice) for funder in funders for invoice in invoices]
            requests, _ = build_counterfactual_requests(pairs)
            with st.spinner("Running a live Message Batch. This can take a few minutes..."):
                texts = AnthropicBatchClient().run(requests)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (RESULTS_DIR / "first_price_raw.json").write_text(
                    json.dumps(texts, indent=2), encoding="utf-8"
                )
            st.rerun()
    elif raw is None:
        st.info(
            "Pending a live run. Either set ANTHROPIC_API_KEY to generate the bids from here, or "
            "run `uv run aelab counterfactual` locally and commit data/first_price_raw.json."
        )
    else:
        scenario = load_scenario(PROBE_SCENARIO)
        funders = generate_financiers(
            scenario.financier, scenario.probe_n_funders, random.Random(scenario.seed)
        )
        invoices = generate_invoices(
            scenario.invoice, scenario.probe_n_invoices, random.Random(scenario.seed + 1)
        )
        pairs = [(funder, invoice) for funder in funders for invoice in invoices]
        _, index_map = build_counterfactual_requests(pairs)
        texts = json.loads(raw.read_text(encoding="utf-8"))
        second_dev, first_dev = counterfactual_deviations(texts, index_map)
        s_stats = deviation_stats(second_dev, scenario.epsilon)
        f_stats = deviation_stats(first_dev, scenario.epsilon)
        cols = st.columns(2)
        cols[0].metric("Second price: mean signed", f"{s_stats.mean_signed:+.5f}")
        cols[1].metric(
            "First price: mean signed",
            f"{f_stats.mean_signed:+.5f}",
            delta=f"{f_stats.mean_signed - s_stats.mean_signed:+.5f} vs second",
        )
        if f_stats.mean_signed > s_stats.mean_signed + scenario.epsilon:
            st.success("Shading up under first price: the agents reason about the rule.")
        else:
            st.warning("Near-zero under both: the agents followed the prompt, not the incentive.")

# --- plain-English terms (the SMB translation) --------------------------------

with tab_terms:
    st.subheader("What a cleared APR means to the supplier")
    st.write(
        "A cleared APR is meaningless to an SMB owner. This translates it into the terms they "
        "act on: cash today versus cash later, the cost of getting paid early, and whether it "
        "beats their outside funding. This is the supplier-facing expression layer the protocol "
        "still needs."
    )
    left, right = st.columns([2, 3])
    with left:
        face_t = st.number_input(
            "Invoice face value (EUR)", 1_000, 5_000_000, 100_000, 1_000, key="terms_face"
        )
        days_t = st.slider("Days paid early", 7, 180, 60, 1, key="terms_days")
        cleared_pct = st.slider("Cleared APR (%)", 1.0, 60.0, 11.0, 0.5)
        outside_pct = st.slider("Supplier's outside funding APR (%)", 1.0, 60.0, 18.0, 0.5)

    cleared = cleared_pct / 100
    outside = outside_pct / 100
    discount = apr_to_discount(cleared, days_t)
    cost = financing_cost(face_t, cleared, days_t)
    receive = face_t - cost
    outside_cost = financing_cost(face_t, outside, days_t)
    savings = outside_cost - cost

    with right:
        m = st.columns(2)
        m[0].metric("You receive today", f"EUR {receive:,.0f}")
        m[1].metric("Cost of early payment", f"EUR {cost:,.0f}", delta=f"{discount:.2%} of face")
        st.markdown(
            f"You get **EUR {receive:,.0f}** now instead of **EUR {face_t:,.0f}** in "
            f"**{days_t} days**. Paying early costs **EUR {cost:,.0f}**, a **{discount:.2%}** "
            f"discount on the invoice, which is **{cleared:.1%} APR** annualised."
        )
        if savings > 0:
            st.success(
                f"Cheaper than the supplier's outside option by **EUR {savings:,.0f}** "
                f"({outside:.1%} APR would cost EUR {outside_cost:,.0f})."
            )
        else:
            st.warning(
                f"The supplier's own funding at {outside:.1%} APR is cheaper here, by "
                f"EUR {-savings:,.0f}. They would decline."
            )
