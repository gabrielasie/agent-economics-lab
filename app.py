"""Streamlit UI for the Agent Economics Lab.

A thin presentation edge over the aelab package. Three focused views: an agent arena, the
incentive-integrity story, and the open question of whether the agents reason. It runs the
deterministic work live and replays the LLM work from committed bids, so it needs no API key.
It imports only the public aelab orchestration; the pure core is untouched, and because this
file lives outside the aelab package the import contracts still hold.

Run locally:  uv sync --extra ui && uv run streamlit run app.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from aelab.agents.llm import build_counterfactual_requests, counterfactual_deviations
from aelab.auction import clear_auction
from aelab.cli import arena_bids, attack_population, compute_deviations, compute_efficiency
from aelab.config import load_scenario
from aelab.economics import financing_cost, surplus_split
from aelab.harness import run_harness
from aelab.metrics import deviation_stats
from aelab.models import Bid, Funder, Invoice
from aelab.populations import generate_financiers, generate_invoices

st.set_page_config(page_title="Agent Economics Lab", page_icon="⚖️", layout="wide")

ACCENT = "#4f46e5"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
SCENARIOS_DIR = Path("scenarios")
PROBE_SCENARIO = "default"


def find_raw(filename: str) -> Path | None:
    for base in (DATA_DIR, RESULTS_DIR):
        path = base / filename
        if path.exists():
            return path
    return None


def get_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        return None


def scenario_names() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.toml"))


@st.cache_data(show_spinner=False)
def efficiency_frame(name: str) -> pd.DataFrame:
    counts, efficiency, supplier_share = compute_efficiency(load_scenario(name))
    return pd.DataFrame(
        {"financiers": counts, "efficiency": efficiency, "supplier share": supplier_share}
    ).set_index("financiers")


@st.cache_data(show_spinner=False)
def harness_frame(name: str) -> pd.DataFrame:
    scenario = load_scenario(name)
    rows = run_harness(attack_population(scenario), scenario.policy, random.Random(scenario.seed + 3))
    return pd.DataFrame(
        [
            {
                "regime": r.regime,
                "efficiency": r.efficiency,
                "supplier share": r.supplier_share,
                "flags": r.fair_rate_index_flags,
            }
            for r in rows
        ]
    ).set_index("regime")


# Each agent gets a stable identity so the field reads as characters, not table rows.
AGENT_COLORS = ["violet", "blue", "green", "orange", "red", "gray"]
AGENT_AVATARS = ["\U0001f7e3", "\U0001f535", "\U0001f7e2", "\U0001f7e0", "\U0001f534", "⚪"]
AGENT_NAMES = ["Vega", "Orion", "Lyra", "Nova", "Atlas", "Sol"]
PERSONAS = ["lean", "keen", "balanced", "measured", "cautious", "premium"]


def identities_for(field: list[tuple[Funder, Bid]]) -> list[tuple[str, str, str, str]]:
    """Assign each agent a colour, avatar, name, and a one-word persona by cost rank."""
    order = sorted(range(len(field)), key=lambda i: field[i][0].true_cost_apr)
    rank = {i: r for r, i in enumerate(order)}
    return [
        (
            AGENT_COLORS[i % len(AGENT_COLORS)],
            AGENT_AVATARS[i % len(AGENT_AVATARS)],
            AGENT_NAMES[i % len(AGENT_NAMES)],
            PERSONAS[min(rank[i], len(PERSONAS) - 1)],
        )
        for i in range(len(field))
    ]


def render_card(slot, funder: Funder, bid: Bid, ident: tuple[str, str, str, str], won: bool) -> None:
    """Render one agent as a bordered card with its identity, cost, and bid."""
    color, avatar, name, persona = ident
    with slot.container(border=True):
        st.markdown(f":{color}[{avatar} **{name}**]")
        st.caption(f"_{persona}_ | cost {funder.true_cost_apr:.1%} -> bid **{bid.apr:.1%}**")
        if won:
            st.markdown(":green[\U0001f3c6 **winner**]")


def render_banner(slot, result, identities: list[tuple[str, str, str, str]], field) -> None:
    """The cleared-price highlight: who won and at what APR, in the winner's colour."""
    if not result.traded:
        slot.warning("No agent's bid was at or below the reserve, so nothing cleared.")
        return
    widx = next(i for i, (f, _b) in enumerate(field) if f.party_id == result.winner_id)
    color, avatar, name, _persona = identities[widx]
    slot.markdown(f"### :{color}[{avatar} {name} wins, cleared at {result.clearing_apr:.1%} APR]")


API_KEY = get_api_key()
if API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = API_KEY

# --- header -------------------------------------------------------------------

st.title("Agent Economics Lab")
st.markdown(
    "##### Sealed-bid auctions where AI agents finance invoices, and the mechanism that keeps "
    "every party honest"
)
with st.expander("About this lab"):
    st.write(
        "When a buyer approves an invoice, competing funder agents enter a sealed-bid, "
        "second-price auction that clears in seconds. This lab shows the agents bidding, checks "
        "that the mechanism keeps them honest, and stress-tests where that breaks."
    )
    pillars = st.columns(3)
    pillars[0].markdown(
        "**Agents, not spreadsheets**  \nClaude funders submit private bids with reasoning."
    )
    pillars[1].markdown(
        "**Incentive integrity**  \nA fair-rate index shows the venue cannot quietly extract."
    )
    pillars[2].markdown(
        "**Tested honestly**  \nIncluding where LLM agents stop actually reasoning."
    )

arena_tab, trust_tab, reason_tab = st.tabs(
    ["Agent arena", "Trust & integrity", "Do the agents reason?"]
)

# --- arena --------------------------------------------------------------------

with arena_tab:
    st.subheader("Claude agents bid against each other")
    st.write(
        "Each funder is a Claude agent. The auction is sealed-bid, so they never see each other: "
        "an agent gets only its own cost of capital and the invoice, then submits one APR. The "
        "lowest bid wins and is paid the second-lowest, which makes bidding your true cost the "
        "smart move. Watch the field clear, then read what each agent was thinking."
    )
    raw = find_raw("truthfulness_raw.json")
    scenario = load_scenario(PROBE_SCENARIO)
    if raw is None:
        st.info("No committed Claude bids found (data/truthfulness_raw.json).")
    else:
        texts = json.loads(raw.read_text(encoding="utf-8"))
        top = st.columns([1, 1, 1])
        inv_idx = top[0].slider("Invoice", 0, scenario.probe_n_invoices - 1, 0)
        reserve = top[1].slider("Supplier reserve (APR %)", 5.0, 60.0, 40.0, 1.0) / 100
        invoice, field = arena_bids(scenario, texts, inv_idx)
        result = clear_auction([bid for _, bid in field], reserve, random.Random(0))
        identities = identities_for(field)

        top[2].write("")
        reveal = top[2].button("▶ Reveal the bids")
        grid = st.columns(3)
        slots = [grid[i % 3].empty() for i in range(len(field))]
        banner = st.empty()

        if reveal and result.traded:
            for i in sorted(range(len(field)), key=lambda j: -field[j][1].apr):  # winner revealed last
                render_card(slots[i], field[i][0], field[i][1], identities[i], won=False)
                time.sleep(0.3)
            widx = next(i for i, (f, _b) in enumerate(field) if f.party_id == result.winner_id)
            time.sleep(0.2)
            render_card(slots[widx], field[widx][0], field[widx][1], identities[widx], won=True)
            time.sleep(0.35)
            render_banner(banner, result, identities, field)
        else:
            for i in range(len(field)):
                won = result.traded and field[i][0].party_id == result.winner_id
                render_card(slots[i], field[i][0], field[i][1], identities[i], won)
            render_banner(banner, result, identities, field)

        if result.traded:
            assert result.clearing_apr is not None and result.winning_bid_apr is not None
            split = surplus_split(
                invoice.face_value, reserve, result.clearing_apr, result.winning_bid_apr, invoice.days_early
            )
            cost = financing_cost(invoice.face_value, result.clearing_apr, invoice.days_early)
            money = st.columns(2)
            money[0].metric("Supplier surplus", f"EUR {split.supplier_surplus:,.0f}")
            money[1].metric("Winner rent", f"EUR {split.winner_rent:,.0f}")
            st.success(
                f"In plain terms: the supplier receives **EUR {invoice.face_value - cost:,.0f}** "
                f"today instead of **EUR {invoice.face_value:,.0f}** in {invoice.days_early} days. "
                f"Paying early costs **EUR {cost:,.0f}** at **{result.clearing_apr:.1%}** APR."
            )

        with st.expander("Read each agent's reasoning"):
            for i, (funder, bid) in enumerate(field):
                crown = " (winner)" if result.traded and result.winner_id == funder.party_id else ""
                st.markdown(f"**{identities[i][2]}**{crown} bid {bid.apr:.2%}")
                st.caption(bid.rationale or "(no rationale returned)")

        st.caption(
            "Real Claude Haiku bids from the committed probe. Under this prompt the agents bid "
            "close to their cost and the cheapest wins, exactly as the mechanism intends. Whether "
            "that holds when the prompt does not coach is the third view."
        )

    if API_KEY:
        with st.expander("Run a fresh live arena with your own funder costs"):
            live = st.columns(3)
            costs_live = [
                live[0].slider("Agent A cost (%)", 1.0, 40.0, 7.0, 0.5, key="a_a") / 100,
                live[1].slider("Agent B cost (%)", 1.0, 40.0, 10.0, 0.5, key="a_b") / 100,
                live[2].slider("Agent C cost (%)", 1.0, 40.0, 14.0, 0.5, key="a_c") / 100,
            ]
            reserve_live = st.slider("Reserve (APR %)", 5.0, 60.0, 40.0, 1.0, key="a_res") / 100
            if st.button("Ask Claude agents to bid live"):
                from aelab.agents.base import AuctionContext, AuctionRules
                from aelab.agents.cache import AnthropicClient, ResponseCache
                from aelab.agents.llm import LLMAgent

                client = ResponseCache(AnthropicClient())
                ctx = AuctionContext(Invoice("ARENA", 100_000.0, 60), AuctionRules(reserve_live))
                live_field: list[tuple[Funder, Bid]] = []
                with st.spinner("Asking each Claude agent for a bid..."):
                    for i, cost in enumerate(costs_live):
                        funder = Funder(f"L{i}", cost)
                        live_field.append((funder, LLMAgent(funder, client).bid(ctx)))
                live_result = clear_auction([b for _, b in live_field], reserve_live, random.Random(0))
                live_ids = identities_for(live_field)
                live_cols = st.columns(len(live_field))
                for i, (funder, bid) in enumerate(live_field):
                    won = live_result.traded and funder.party_id == live_result.winner_id
                    render_card(live_cols[i], funder, bid, live_ids[i], won)
                render_banner(st.empty(), live_result, live_ids, live_field)
                for i, (_funder, bid) in enumerate(live_field):
                    st.caption(f"**{live_ids[i][2]}**: {bid.rationale or '(none)'}")

# --- trust & integrity --------------------------------------------------------

with trust_tab:
    st.subheader("Can the venue extract, and would a supplier see it?")
    labels = {"default": "Competitive market", "extraction": "House is pivotal"}
    names = scenario_names()
    options = [labels.get(n, n) for n in names]
    choice = st.radio("Market", options, horizontal=True, label_visibility="collapsed")
    scenario_name = names[options.index(choice)]
    st.caption(
        "**Competitive market**: a cheap in-house buyer and rival financiers sit below the house, "
        "so it cannot pivot the price. **House is pivotal**: the house is the marginal funder, so "
        "withholding bites. The contrast is the finding: competition, not a rule, is the defense."
    )
    st.write(
        "The hard part is not efficiency. Under truthful bidding the lowest-cost funder always "
        "wins, so the market is fully efficient. What matters is how the surplus is split, and "
        "whether the house that runs the venue can quietly take more of it. If suppliers ever "
        "conclude the venue extracts, the network dies."
    )

    eff = efficiency_frame(scenario_name)
    har = harness_frame(scenario_name).to_dict("index")
    sep = har.get("separated", {})
    inf = har.get("informed", {})

    cols = st.columns(3)
    cols[0].metric("Allocative efficiency", f"{eff['efficiency'].mean():.3f}", help="Always near 1.0")
    cols[1].metric(
        "Supplier share, honest house",
        f"{sep.get('supplier share', 0):.3f}",
        help="A house that bids its signed policy blind adds competition and helps the supplier.",
    )
    cols[2].metric(
        "Supplier share, house peeks",
        f"{inf.get('supplier share', 0):.3f}",
        delta=f"{inf.get('supplier share', 0) - sep.get('supplier share', 0):+.3f}",
        delta_color="normal",
        help="A house that peeks at sealed bids and withholds extracts from the supplier.",
    )

    chart_cols = st.columns([3, 2])
    with chart_cols[0]:
        st.caption("Supplier share of surplus, by regime")
        st.bar_chart(harness_frame(scenario_name)[["supplier share"]], height=260, color=ACCENT)
    with chart_cols[1]:
        flags = int(inf.get("flags", 0))
        if flags > 0:
            st.error(
                f"The fair-rate index flags **{flags}** invoices where the informed house lifted "
                f"the clearing above the honest benchmark. The extraction is caught."
            )
        else:
            st.success(
                "The fair-rate index is quiet: in this market a competitive buyer sits below the "
                "house, so withholding moves nothing. Competition is the discipline."
            )
        st.caption(
            "Switch the scenario in the sidebar to `extraction` (house is pivotal) versus "
            "`default` (a competitive buyer disciplines it)."
        )

    with st.expander("Fee structure: the base decides the incentive"):
        st.write(
            "The venue must charge a fee, but the base sets its incentive. A fee on the supplier's "
            "surplus shrinks when the supplier is squeezed, so withholding costs the venue its own "
            "revenue. A fee on the spread grows when the supplier is squeezed, so it pays the venue "
            "to extract."
        )
        fee_cols = st.columns([1, 2])
        fee_rate = fee_cols[0].slider("Fee rate (%)", 0.0, 30.0, 10.0, 1.0) / 100
        base = fee_cols[0].radio("Fee base", ["supplier surplus", "the spread"])
        aligned = base == "supplier surplus"
        fee_df = harness_frame(scenario_name).copy()
        fee_df["winner rent share"] = 1.0 - fee_df["supplier share"]
        base_share = fee_df["supplier share"] if aligned else fee_df["winner rent share"]
        fee_df["venue fee"] = fee_rate * base_share
        fee_cols[1].caption("Venue fee revenue by regime (share of the pie)")
        fee_cols[1].bar_chart(fee_df[["venue fee"]], height=240, color=ACCENT)
        if "separated" in fee_df.index and "informed" in fee_df.index:
            delta = float(fee_df.loc["informed", "venue fee"] - fee_df.loc["separated", "venue fee"])
            if aligned:
                st.success(
                    f"Withholding moves the venue's fee by {delta:+.4f} of the pie: a surplus-based "
                    f"fee makes extraction self-defeating, which is what makes separation defensible."
                )
            else:
                st.error(
                    f"Withholding moves the venue's fee by {delta:+.4f} of the pie: a spread-based "
                    f"fee can pay the venue to extract. Do not price the book this way."
                )

    with st.expander("Supplier share rises with competition"):
        st.line_chart(eff, height=260)

# --- do the agents reason? ----------------------------------------------------

with reason_tab:
    st.subheader("Do the agents reason, or just follow the prompt?")
    scenario = load_scenario(PROBE_SCENARIO)

    st.markdown("**The truthfulness probe is a non-result.**")
    truth_raw = find_raw("truthfulness_raw.json")
    if truth_raw is not None:
        deviations, _failures = compute_deviations(scenario, json.loads(truth_raw.read_text()))
        stats = deviation_stats(deviations, scenario.epsilon)
        cols = st.columns([1, 2])
        cols[0].metric("Mean deviation from cost", f"{stats.mean_signed:+.5f}")
        cols[0].metric("Within tolerance", f"{stats.fraction_within:.0%}")
        cols[1].warning(
            "The agents bid their true cost almost exactly. But the prompt told them truthful "
            "bidding is optimal, so this measures instruction-following, not reasoning. It is the "
            "number you would get from a model that echoed its cost back."
        )

    st.divider()
    st.markdown("**The first-price counterfactual is the real test.**")
    st.write(
        "Run the same agents under a first-price auction with a neutral prompt that states the "
        "rule and recommends nothing. There, bidding your cost earns zero, so a reasoner shades "
        "its bid up and a prompt-follower stays put."
    )
    fp_raw = find_raw("first_price_raw.json")
    if fp_raw is None and API_KEY:
        if st.button("Generate the first-price bids live (uses your API key)"):
            from aelab.agents.cache import AnthropicBatchClient

            funders = generate_financiers(
                scenario.financier, scenario.probe_n_funders, random.Random(scenario.seed)
            )
            invoices = generate_invoices(
                scenario.invoice, scenario.probe_n_invoices, random.Random(scenario.seed + 1)
            )
            pairs = [(f, inv) for f in funders for inv in invoices]
            requests, _ = build_counterfactual_requests(pairs)
            with st.spinner("Running a live Message Batch. This can take a few minutes..."):
                out = AnthropicBatchClient().run(requests)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (RESULTS_DIR / "first_price_raw.json").write_text(json.dumps(out, indent=2))
            st.rerun()
    elif fp_raw is None:
        st.info(
            "Not yet run. Run `uv run aelab counterfactual` with a key and commit "
            "data/first_price_raw.json, or set a key to generate it from here."
        )
    else:
        funders = generate_financiers(
            scenario.financier, scenario.probe_n_funders, random.Random(scenario.seed)
        )
        invoices = generate_invoices(
            scenario.invoice, scenario.probe_n_invoices, random.Random(scenario.seed + 1)
        )
        pairs = [(f, inv) for f in funders for inv in invoices]
        _, index_map = build_counterfactual_requests(pairs)
        second, first = counterfactual_deviations(json.loads(fp_raw.read_text()), index_map)
        s_stats = deviation_stats(second, scenario.epsilon)
        f_stats = deviation_stats(first, scenario.epsilon)
        cols = st.columns(2)
        cols[0].metric("Second price: shading", f"{s_stats.mean_signed:+.5f}")
        cols[1].metric(
            "First price: shading",
            f"{f_stats.mean_signed:+.5f}",
            delta=f"{f_stats.mean_signed - s_stats.mean_signed:+.5f} vs second",
        )
        if f_stats.mean_signed > s_stats.mean_signed + scenario.epsilon:
            st.success("Shading up under first price: the agents reason about the rule.")
        else:
            st.warning("Near-zero under both: the agents followed the prompt, not the incentive.")

# --- footer -------------------------------------------------------------------

st.divider()
foot = st.columns([3, 1])
foot[0].caption(
    "Live LLM runs enabled."
    if API_KEY
    else "Running keyless: deterministic work is live; the LLM panels replay committed bids."
)
foot[1].caption("[Source on GitHub](https://github.com/gabrielasie/agent-economics-lab)")
