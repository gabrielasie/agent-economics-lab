"""Presentation layer: efficiency and supplier-share plots.

This is the only module that imports matplotlib. It turns market metrics into a PNG and
does no computation of its own. Compute and presentation stay separate.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # headless: render to a file, never open a window


def plot_efficiency_and_supplier_share(
    counts: Sequence[int],
    efficiency: Sequence[float],
    supplier_share: Sequence[float],
    path: Path,
) -> None:
    """Plot allocative efficiency and supplier share against the financier count."""
    if not len(counts) == len(efficiency) == len(supplier_share):
        raise ValueError("counts, efficiency, and supplier_share must be the same length")
    fig, ax = plt.subplots()
    ax.plot(counts, efficiency, marker="o", label="allocative efficiency")
    ax.plot(counts, supplier_share, marker="s", label="supplier share of surplus")
    ax.set_xlabel("number of financiers")
    ax.set_ylabel("ratio")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Efficiency and supplier share vs financier count")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_deviation_distribution(deviations: Sequence[float], path: Path, bins: int = 20) -> None:
    """Histogram of bid deviation from true cost (bid_apr - true_cost_apr).

    The dashed line at zero marks truthful bidding; mass to the right is bidding above
    true cost, mass to the left is bidding below it.
    """
    if not deviations:
        raise ValueError("no deviations to plot")
    fig, ax = plt.subplots()
    ax.hist(deviations, bins=bins)
    ax.axvline(0.0, color="black", linestyle="--", label="truthful (bid = true cost)")
    ax.set_xlabel("bid_apr - true_cost_apr")
    ax.set_ylabel("count")
    ax.set_title("LLM bid deviation from true cost")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
