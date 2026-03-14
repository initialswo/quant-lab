from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.factors.normalize import preprocess_factor_scores


def test_preprocess_shape_and_percentile_bounds() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = ["A", "B", "C", "D"]
    scores = pd.DataFrame(
        [[1.0, 2.0, 3.0, 100.0], [4.0, 5.0, 6.0, 7.0], [1.0, np.nan, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]],
        index=idx,
        columns=cols,
    )
    out = preprocess_factor_scores(scores, use_factor_normalization=True, winsor_p=0.05)
    assert out.shape == scores.shape
    assert out.index.equals(scores.index)
    assert out.columns.equals(scores.columns)
    vals = out.stack().astype(float)
    assert float(vals.min()) >= -1e-12
    assert float(vals.max()) <= 1.0 + 1e-12


def test_preprocess_std_zero_row_becomes_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["A", "B", "C"]
    scores = pd.DataFrame([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]], index=idx, columns=cols)
    out = preprocess_factor_scores(scores, use_factor_normalization=True, winsor_p=0.05)
    assert out.loc[idx[0]].isna().all()
    assert out.loc[idx[1]].notna().all()


def test_preprocess_toggle_off_returns_raw() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B"]
    scores = pd.DataFrame([[1.0, 3.0], [2.0, 4.0], [np.nan, 5.0]], index=idx, columns=cols)
    out = preprocess_factor_scores(scores, use_factor_normalization=False, winsor_p=0.05)
    assert np.allclose(out.to_numpy(), scores.to_numpy(), equal_nan=True)
