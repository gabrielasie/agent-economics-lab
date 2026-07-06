"""Render the presentation figures from committed result data.

One figure today: the model-by-pool-size collusion heatmap. It reads the grid from
data/collusion_grid.json, whose values are copied verbatim from RESULTS.md, and writes
docs/figures/collusion_grid.png at README width.

This script is an edge outside the aelab package, so importing matplotlib here keeps the
"only report renders" contract intact: lint-imports checks the package graph, not scripts.

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GRID_JSON = REPO / "data" / "collusion_grid.json"
OUT_PNG = REPO / "docs" / "figures" / "collusion_grid.png"

TITLE = "When do LLM bidding agents collude?"
SUBTITLE = (
    "Open chat channel in a sealed-bid second-price reverse auction.\n"
    "Cell = cleared APR above the truthful baseline, in percentage points over 20 rounds."
)


def render_heatmap(grid_path: Path, out_path: Path) -> None:
    """Draw the 2x2 collusion-index heatmap and save it as a PNG."""
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    models: list[str] = grid["models"]
    pools: list[str] = grid["pools"]
    values = [[float(grid["index"][m][p]) for p in pools] for m in models]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    # Anchor white at zero so the competitive cells stay pale and +3.22 saturates.
    image = ax.imshow(values, cmap="OrRd", vmin=-0.4, vmax=3.6, aspect="auto")

    ax.set_xticks(range(len(pools)), labels=pools, fontsize=15)
    ax.set_yticks(range(len(models)), labels=models, fontsize=15)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    emergent = (grid["emergent_cell"]["model"], grid["emergent_cell"]["pool"])
    for i, model in enumerate(models):
        for j, pool in enumerate(pools):
            value = values[i][j]
            is_emergent = (model, pool) == emergent
            label = f"{value:+.2f}" if value != 0 else "0.00"
            if is_emergent:
                label += "\ncollusion emerged"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=19 if is_emergent else 17,
                fontweight="bold" if is_emergent else "normal",
                color="white" if is_emergent else "#1f2937",
            )

    fig.suptitle(TITLE, fontsize=20, fontweight="bold", x=0.02, y=0.97, ha="left")
    fig.text(0.02, 0.845, SUBTITLE, fontsize=11.5, color="#4b5563", ha="left", va="top")

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("collusion index (pp above baseline)", fontsize=11)
    bar.ax.tick_params(labelsize=10)
    bar.outline.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.76))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render_heatmap(GRID_JSON, OUT_PNG)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
