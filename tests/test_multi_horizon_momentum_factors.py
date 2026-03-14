from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.factors.registry import compute_factor, list_factors


def test_momentum_3_1_formula() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    close = pd.DataFrame({"AAA": np.arange(100.0, 200.0)}, index=idx)
    out = compute_factor("momentum_3_1", close)
    t = idx[-1]
    exp = float(close.shift(21).loc[t, "AAA"] / close.shift(63).loc[t, "AAA"] - 1.0)
    assert np.isclose(float(out.loc[t, "AAA"]), exp, atol=1e-12)


def test_momentum_6_1_formula() -> None:
    idx = pd.date_range("2020-01-01", periods=180, freq="B")
    close = pd.DataFrame({"AAA": np.arange(100.0, 280.0)}, index=idx)
    out = compute_factor("momentum_6_1", close)
    t = idx[-1]
    exp = float(close.shift(21).loc[t, "AAA"] / close.shift(126).loc[t, "AAA"] - 1.0)
    assert np.isclose(float(out.loc[t, "AAA"]), exp, atol=1e-12)


def test_multi_horizon_momentum_registry_discovery() -> None:
    names = set(list_factors())
    assert "momentum_3_1" in names
    assert "momentum_6_1" in names

