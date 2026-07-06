# 90-second demo script

A numbered walkthrough for screen-sharing. Before you start: open the
[live demo](https://agent-economics-lab-mthq892brens5xgelgbr9i.streamlit.app/) in a browser tab
and a terminal at the repo root (`uv sync` already run). No API key needed anywhere.

1. **The heatmap** (top of the app, or `docs/figures/collusion_grid.png`).
   Click: nothing; it is the landing view.
   Say: "Same mechanism, same neutral prompts, four conditions; the LLM agents collude in exactly
   one: the stronger model in the thin three-funder market, +3.22 points above the competitive
   baseline."

2. **The coordinating round** (the transcript viewer under the heatmap).
   Click: the Round slider is already on round 10; point at the red ⚑ flag.
   Say: "This is real agent chat, replayed from the committed run; 'Rational coordination benefits
   everyone here' is flagged by the repo's own guardrail, and the bids table shows everyone
   bidding above cost."

3. **Slide back to round 2.**
   Click: drag the Round slider to 2.
   Say: "Round 2 is the first flagged solicitation — 'let's keep bids rational' — and every
   open-channel round cleared above baseline, while sealed and history-only stayed exactly at it."

4. **The structural point, live in the terminal.**
   Run: `uv run aelab attacks --scenario extraction`
   Say: "No LLM here: a deterministic house that runs the venue and bids in it extracts from the
   supplier while allocative efficiency stays at 1.000; efficiency metrics are blind to this, the
   fair-rate index flags all 50 invoices."

5. **The caveats** (README `Caveats`, or the grid legend in the app).
   Click: scroll to the caveats section.
   Say: "Two models, two pool sizes, one seed per cell, simulated APR space; read the negative
   cells as 'did not emerge under these conditions', never as 'cannot collude'."

Fallback if the live app is down: `uv run streamlit run app.py` locally, or step through
`RESULTS.md`, which holds the same table and verbatim quotes.
