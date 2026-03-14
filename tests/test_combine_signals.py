from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.factors.normalize import normalize_scores
from quant_lab.research.combine_signals import combine_factor_panels


def _toy_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2023-01-01", periods=8, freq="B")
    cols = ["A", "B", "C"]
    f1 = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.2, 2.1, 2.9], [1.3, 2.2, 2.8], [1.5, 2.3, 2.7], [1.4, 2.0, 2.6], [1.6, 2.4, 2.5], [1.8, 2.6, 2.4], [2.0, 2.8, 2.3]],
        index=idx,
        columns=cols,
    )
    f2 = pd.DataFrame(
        [[3.0, 2.0, 1.0], [2.9, 2.1, 1.1], [2.8, 2.2, 1.2], [2.7, 2.3, 1.3], [2.6, 2.4, 1.4], [2.5, 2.5, 1.5], [2.4, 2.6, 1.6], [2.3, 2.7, 1.7]],
        index=idx,
        columns=cols,
    )
    return f1, f2


def test_equal_weight_combination_matches_manual_average() -> None:
    f1, f2 = _toy_panels()
    out = combine_factor_panels({"f1": f1, "f2": f2}, weights=None, normalize="zscore")
    n1 = normalize_scores(f1, method="zscore")
    n2 = normalize_scores(f2, method="zscore")
    exp = 0.5 * n1 + 0.5 * n2
    assert np.allclose(out.to_numpy(), exp.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_custom_weight_combination() -> None:
    f1, f2 = _toy_panels()
    out = combine_factor_panels(
        {"f1": f1, "f2": f2},
        weights={"f1": 0.7, "f2": 0.3},
        normalize="zscore",
    )
    n1 = normalize_scores(f1, method="zscore")
    n2 = normalize_scores(f2, method="zscore")
    exp = 0.7 * n1 + 0.3 * n2
    assert np.allclose(out.to_numpy(), exp.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_weights_key_mismatch_raises() -> None:
    f1, f2 = _toy_panels()
    with pytest.raises(ValueError):
        combine_factor_panels({"f1": f1, "f2": f2}, weights={"f1": 1.0}, normalize="zscore")


def test_missing_values_handled_safely_and_shape_preserved() -> None:
    f1, f2 = _toy_panels()
    f2 = f2.copy()
    f2.iloc[2, 1] = np.nan
    f2.iloc[4, :] = np.nan
    out = combine_factor_panels({"f1": f1, "f2": f2}, weights=None, normalize="zscore")
    assert out.shape == f1.shape
    assert out.index.equals(f1.index)
    assert out.columns.equals(f1.columns)
    assert out.notna().any().any()


def test_single_factor_path_matches_normalized_panel() -> None:
    f1, _ = _toy_panels()
    out = combine_factor_panels({"f1": f1}, weights=None, normalize="zscore")
    exp = normalize_scores(f1, method="zscore")
    assert np.allclose(out.to_numpy(), exp.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)

