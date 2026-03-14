from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab.research.factor_heatmap import compute_momentum_sweep_matrix, plot_heatmap


def _toy_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2020-01-01", periods=320, freq="B")
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        {
            "A": 100.0 * np.power(1.0010, np.arange(len(idx))),
            "B": 90.0 * np.power(1.0008, np.arange(len(idx))),
            "C": 110.0 * np.power(1.0005, np.arange(len(idx))),
            "D": 95.0 * np.power(1.0002, np.arange(len(idx))),
        },
        index=idx,
    )
    fwd = close.pct_change().shift(-1)
    return close, fwd


def test_compute_momentum_sweep_matrix_shape() -> None:
    close, fwd = _toy_panels()
    out = compute_momentum_sweep_matrix(
        close=close,
        future_returns=fwd,
        lookbacks=[21, 63, 126],
        metric="sharpe",
        period="year",
    )
    assert out.shape[0] == 3
    assert out.index.tolist() == [21, 63, 126]
    assert out.columns.tolist()


def test_plot_heatmap_writes_file(tmp_path: Path) -> None:
    mat = pd.DataFrame(
        [[-0.2, 0.0, 0.3], [0.1, -0.1, 0.2]],
        index=[21, 63],
        columns=["2021", "2022", "2023"],
    )
    out = tmp_path / "heatmap.png"
    path = plot_heatmap(mat, title="Toy", outpath=out)
    assert path.exists()
    assert path.stat().st_size > 0

