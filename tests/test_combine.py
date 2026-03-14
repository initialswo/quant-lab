from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.factors.combine import aggregate_factor_scores, combine_factor_scores


def _old_scalar_combine(scores: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    names = list(scores.keys())
    base = scores[names[0]]
    out = pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype=float)
    any_signal = pd.DataFrame(False, index=base.index, columns=base.columns)
    total = float(sum(weights.values()))
    w = {k: float(v) / total for k, v in weights.items()}
    for n in names:
        panel = scores[n]
        out = out + w[n] * panel.fillna(0.0)
        any_signal = any_signal | panel.notna()
    return out.where(any_signal).astype(float)


def test_combine_scalar_regression_matches_old_path() -> None:
    idx = pd.date_range("2022-01-01", periods=12, freq="B")
    cols = ["A", "B", "C"]
    s1 = pd.DataFrame(np.arange(36, dtype=float).reshape(12, 3), index=idx, columns=cols)
    s2 = pd.DataFrame(np.arange(36, 72, dtype=float).reshape(12, 3), index=idx, columns=cols)
    s1.iloc[3, 1] = np.nan
    s2.iloc[7, 2] = np.nan
    scores = {"f1": s1, "f2": s2}
    w = {"f1": 0.7, "f2": 0.3}

    got = combine_factor_scores(scores, w)
    exp = _old_scalar_combine(scores, w)
    assert np.allclose(got.to_numpy(), exp.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_combine_supports_per_date_weights() -> None:
    idx = pd.date_range("2022-01-01", periods=6, freq="B")
    cols = ["A", "B"]
    s1 = pd.DataFrame(1.0, index=idx, columns=cols)
    s2 = pd.DataFrame(3.0, index=idx, columns=cols)
    s2.iloc[2] = np.nan  # force dynamic renorm on this date

    w1 = pd.Series([1, 1, 1, 0, 0, 0], index=idx, dtype=float)
    w2 = pd.Series([0, 0, 0, 1, 1, 1], index=idx, dtype=float)
    out = combine_factor_scores({"f1": s1, "f2": s2}, {"f1": w1, "f2": w2})

    # Early dates f1 dominates, later f2 dominates.
    assert np.isclose(float(out.iloc[0, 0]), 1.0)
    assert np.isclose(float(out.iloc[5, 0]), 3.0)
    # Missing factor row falls back to available factor.
    assert np.isclose(float(out.iloc[2, 0]), 1.0)


def test_combine_require_all_factors_propagates_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B"]
    s1 = pd.DataFrame([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], index=idx, columns=cols)
    s2 = pd.DataFrame([[1.0, np.nan], [2.0, 2.0], [np.nan, 3.0]], index=idx, columns=cols)

    out = combine_factor_scores(
        {"f1": s1, "f2": s2},
        {"f1": 0.5, "f2": 0.5},
        require_all_factors=True,
    )

    assert np.isfinite(float(out.loc[idx[1], "A"]))
    assert np.isfinite(float(out.loc[idx[1], "B"]))
    assert pd.isna(out.loc[idx[0], "B"])
    assert pd.isna(out.loc[idx[2], "A"])


def test_aggregate_mean_rank_prefers_consensus() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="B")
    cols = ["A", "B", "C"]
    # B is strong on both factors -> consensus winner for rank aggregation.
    f1 = pd.DataFrame([[3.0, 2.0, 1.0]], index=idx, columns=cols)
    f2 = pd.DataFrame([[1.0, 3.0, 2.0]], index=idx, columns=cols)
    out = aggregate_factor_scores({"f1": f1, "f2": f2}, {"f1": 0.5, "f2": 0.5}, method="mean_rank")
    winner = out.loc[idx[0]].idxmax()
    assert winner == "B"


def test_aggregate_geometric_rank_requires_consistency() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="B")
    cols = ["A", "B", "C"]
    f1 = pd.DataFrame([[3.0, 2.0, 1.0]], index=idx, columns=cols)
    f2 = pd.DataFrame([[1.0, 3.0, 2.0]], index=idx, columns=cols)
    out = aggregate_factor_scores({"f1": f1, "f2": f2}, {"f1": 0.9, "f2": 0.1}, method="geometric_rank")
    winner = out.loc[idx[0]].idxmax()
    assert winner == "B"
