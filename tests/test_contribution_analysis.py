from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.contribution import (
    compute_daily_ticker_contributions,
    summarize_contribution_concentration,
)


def test_daily_ticker_contributions_lagged_alignment() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.DataFrame(
        {"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 90.0]},
        index=idx,
    )
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
    c = compute_daily_ticker_contributions(close=close, realized_weights=w)
    # Day 2 uses day1 weights: fully A, +10%
    assert np.isclose(float(c.loc[idx[1], "A"]), 0.1, atol=1e-12)
    assert np.isclose(float(c.loc[idx[1], "B"]), 0.0, atol=1e-12)
    # Day 3 uses day2 weights: fully B, -10%
    assert np.isclose(float(c.loc[idx[2], "A"]), 0.0, atol=1e-12)
    assert np.isclose(float(c.loc[idx[2], "B"]), -0.1, atol=1e-12)


def test_concentration_summary_shares_and_hhi() -> None:
    s = pd.Series({"A": 0.5, "B": 0.3, "C": -0.2, "D": 0.0}, dtype=float)
    out = summarize_contribution_concentration(s)
    # abs shares: 0.5,0.3,0.2,0.0 => top1=0.5 top5=1.0
    assert np.isclose(float(out["top1_share_abs"]), 0.5, atol=1e-12)
    assert np.isclose(float(out["top5_share_abs"]), 1.0, atol=1e-12)
    # hhi = 0.5^2 + 0.3^2 + 0.2^2 = 0.38
    assert np.isclose(float(out["herfindahl_abs"]), 0.38, atol=1e-12)
    assert np.isclose(float(out["effective_n_abs"]), 1.0 / 0.38, atol=1e-12)

