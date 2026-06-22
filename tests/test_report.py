"""Light tests for the report plotting layer: it writes a PNG and validates inputs."""

from pathlib import Path

import pytest

from aelab.report import plot_deviation_distribution, plot_efficiency_and_supplier_share


def test_plot_writes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "plot.png"
    plot_efficiency_and_supplier_share([1, 2, 3], [1.0, 1.0, 1.0], [0.2, 0.4, 0.6], out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_efficiency_and_supplier_share([1, 2], [1.0], [0.2, 0.4], tmp_path / "x.png")


def test_plot_deviation_writes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "dev.png"
    plot_deviation_distribution([0.0, 0.01, -0.02, 0.03], out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_deviation_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_deviation_distribution([], tmp_path / "x.png")
