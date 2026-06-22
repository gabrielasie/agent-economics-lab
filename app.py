"""Streamlit UI for the Agent Economics Lab.

A thin presentation edge over the aelab package. It runs the deterministic experiments live
(efficiency, house extraction) and replays the LLM experiments from saved bids, so it needs
no API key. It imports only the public aelab orchestration; the pure core is untouched, and
because this file lives outside the aelab package the import contracts still hold.

Run locally:  uv sync --extra ui && uv run streamlit run app.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from aelab.agents.llm import build_counterfactual_requests, counterfactual_deviations
from aelab.cli import attack_population, compute_deviations, compute_efficiency
from aelab.config import load_scenario
from aelab.harness import run_harness
from aelab.metrics import deviation_stats
from aelab.populations import generate_financiers, generate_invoices

st.set_page_config(page_title="Agent Economics Lab", page_icon="📈", layout="wide")

SCENARIOS_DIR = Path("scenarios")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PROBE_SCENARIO = "default"  # the scenario the saved truthfulness/first-price bids were generated under


def scenario_names() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.toml"))


def find_raw(filename: str) -> Path | None:
    """Prefer a committed bundle in data/, fall back to a local run in results/."""
    for base in (DATA_DIR, RESULTS_DIR):
        path = base / filename
        if path.exists():
            return path
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


st.title("Agent Economics Lab")
st.caption(
    "A sealed-bid second-price reverse auction for invoice early-payment, with LLM bidding "
    "agents and an adversarial red-team harness. Second-price clearing buys truthfulness from "
    "funders and from no one else; every panel here probes that gap."
)

names = scenario_names()
scenario_name = st.sidebar.selectbox(
    "Scenario (efficiency and extraction panels)",
    names,
    index=names.index("default") if "default" in names else 0,
)
st.sidebar.caption(
    "The truthfulness and first-price panels are pinned to the `default` scenario, the "
    "population the saved bids were generated under."
)
st.sidebar.divider()
st.sidebar.caption(
    "Deterministic and reproducible. The LLM panels replay committed bids, so this app uses "
    "no API key."
)

tab_eff, tab_attack, tab_truth, tab_first = st.tabs(
    ["Efficiency", "House extraction", "Truthfulness (non-result)", "First-price counterfactual"]
)

with tab_eff:
    st.subheader("Efficiency and the supplier-share curve")
    st.write(
        "Under truthful bidding the lowest-cost funder always wins, so allocative efficiency is "
        "1.000 at every financier count. The metric that moves is the supplier's share of "
        "surplus, which rises with competition. Efficiency tells you the pie is whole; supplier "
        "share tells you who ate it."
    )
    frame = efficiency_frame(scenario_name)
    st.line_chart(frame, height=360)
    left, right = st.columns(2)
    left.metric("Allocative efficiency", f"{frame['allocative efficiency'].mean():.3f}")
    right.metric(
        "Supplier share (range)",
        f"{frame['supplier share'].min():.3f} to {frame['supplier share'].max():.3f}",
    )
    st.dataframe(frame.style.format("{:.3f}"), width="stretch")

with tab_attack:
    st.subheader("House extraction, and a fair-rate index that catches it")
    st.write(
        "A house that runs the venue can extract by withholding: it peeks at sealed bids and, "
        "when it cannot win, bids just under the reserve to lift the clearing. The extraction is "
        "on price, not allocation, so efficiency stays 1.000 and a no-house baseline cannot see "
        "it. The fair-rate index, benchmarked against the honest separated house, flags it. It is "
        "only measurable where the house is the pivotal funder: pick the `extraction` scenario "
        "for the attack, and `default` for the market where competition disciplines it."
    )
    frame = harness_frame(scenario_name)
    st.dataframe(
        frame.style.format(
            {"efficiency": "{:.3f}", "supplier share": "{:.3f}", "index flags": "{:d}"}
        ),
        width="stretch",
    )
    left, right = st.columns(2)
    left.caption("Supplier share by regime")
    left.bar_chart(frame[["supplier share"]], height=300)
    right.caption("Fair-rate-index flags by regime")
    right.bar_chart(frame[["index flags"]], height=300)

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
                .mark_bar()
                .encode(
                    alt.X("deviation:Q", bin=alt.Bin(maxbins=40), title="bid APR minus true cost"),
                    alt.Y("count()", title="bids"),
                )
                .properties(height=320)
            )
            st.altair_chart(chart, width="stretch")
        st.caption(f"Parsed {stats.n} bids with {failures} parse failures, from {raw}.")

with tab_first:
    st.subheader("First-price counterfactual: reasoning or coaching?")
    st.write(
        "The same agents under a first-price auction with a neutral prompt that states the rule "
        "and recommends nothing. Truthful is no longer optimal there, so a reasoner shades its "
        "bid up and a prompt-follower stays put. Shaded-up-under-first means the agent reasons; "
        "truthful-under-both means it followed coaching all along."
    )
    raw = find_raw("first_price_raw.json")
    if raw is None:
        st.info(
            "Pending a live run. Run `uv run aelab counterfactual` with an API key, then commit "
            "`data/first_price_raw.json`. The deterministic panels above need no key."
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
        second, first = counterfactual_deviations(texts, index_map)
        s_stats = deviation_stats(second, scenario.epsilon)
        f_stats = deviation_stats(first, scenario.epsilon)
        cols = st.columns(2)
        cols[0].metric("Second price: mean signed", f"{s_stats.mean_signed:+.5f}")
        cols[1].metric(
            "First price: mean signed",
            f"{f_stats.mean_signed:+.5f}",
            delta=f"{f_stats.mean_signed - s_stats.mean_signed:+.5f} vs second",
        )
        reasons = f_stats.mean_signed > s_stats.mean_signed + scenario.epsilon
        if reasons:
            st.success("Shading up under first price: the agents reason about the rule.")
        else:
            st.warning("Near-zero under both: the agents followed the prompt, not the incentive.")
