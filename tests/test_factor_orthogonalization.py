from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.factors.orthogonalize import maybe_orthogonalize_factor_scores


def _corr(a: pd.Series, b: pd.Series) -> float:
    return float(a.astype(float).corr(b.astype(float)))


def test_orthogonalization_reduces_cross_sectional_linear_overlap() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    cols = [f"T{i:03d}" for i in range(250)]

    f1_vals = rng.normal(0.0, 1.0, size=(len(idx), len(cols)))
    f2_vals = 1.8 * f1_vals + rng.normal(0.0, 0.1, size=(len(idx), len(cols)))
    f3_vals = -0.6 * f1_vals + 1.1 * f2_vals + rng.normal(0.0, 0.1, size=(len(idx), len(cols)))

    factors = {
        "momentum_12_1": pd.DataFrame(f1_vals, index=idx, columns=cols),
        "reversal_1m": pd.DataFrame(f2_vals, index=idx, columns=cols),
        "low_vol_20": pd.DataFrame(f3_vals, index=idx, columns=cols),
    }
    out = maybe_orthogonalize_factor_scores(
        factors,
        enabled=True,
        factor_order=["momentum_12_1", "reversal_1m", "low_vol_20"],
    )

    for dt in idx:
        c21 = _corr(out["reversal_1m"].loc[dt], out["momentum_12_1"].loc[dt])
        c31 = _corr(out["low_vol_20"].loc[dt], out["momentum_12_1"].loc[dt])
        c32 = _corr(out["low_vol_20"].loc[dt], out["reversal_1m"].loc[dt])
        assert abs(c21) < 1e-10
        assert abs(c31) < 1e-10
        assert abs(c32) < 1e-10


def test_orthogonalization_deterministic_and_no_nan_on_valid_data() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B", "C", "D", "E", "F"]
    a = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 5.0, 7.0, 11.0, 13.0], [3.0, 5.0, 8.0, 13.0, 21.0, 34.0]],
        index=idx,
        columns=cols,
    )
    b = 2.0 * a + 1.0
    c = -1.0 * a + 0.5 * b + 2.0
    factors = {"f1": a, "f2": b, "f3": c}

    out1 = maybe_orthogonalize_factor_scores(factors, enabled=True, factor_order=["f1", "f2", "f3"])
    out2 = maybe_orthogonalize_factor_scores(factors, enabled=True, factor_order=["f1", "f2", "f3"])

    for name in ["f1", "f2", "f3"]:
        assert np.allclose(out1[name].to_numpy(), out2[name].to_numpy(), equal_nan=True)
        assert out1[name].notna().all().all()


def test_orthogonalization_flag_off_is_noop() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    raw = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=["A", "B"])
    factors = {"x": raw, "y": raw * 2.0}

    out = maybe_orthogonalize_factor_scores(factors, enabled=False, factor_order=["x", "y"])
    assert np.allclose(out["x"].to_numpy(), factors["x"].to_numpy(), equal_nan=True)
    assert np.allclose(out["y"].to_numpy(), factors["y"].to_numpy(), equal_nan=True)

