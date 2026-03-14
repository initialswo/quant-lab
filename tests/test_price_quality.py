from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data.quality import compute_price_quality, flag_bad_tickers, summarize_price_quality


def test_price_quality_detects_stale_series() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.DataFrame(
        {
            "T1": np.full(len(idx), 100.0),
            "T2": 100.0 + np.cumsum(np.linspace(0.1, 0.5, len(idx))),
        },
        index=idx,
    )
    q = compute_price_quality(close, window_name="backtest")
    flagged = flag_bad_tickers(q, zero_ret_thresh=0.95, min_valid_frac=0.98)

    row_t1 = flagged.loc[flagged["ticker"] == "T1"].iloc[0]
    row_t2 = flagged.loc[flagged["ticker"] == "T2"].iloc[0]
    assert bool(row_t1["bad_zero"])
    assert float(row_t1["zero_ret_frac"]) >= 0.95
    assert not bool(row_t2["bad_zero"])


def test_price_quality_detects_low_valid_frac() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    t3 = pd.Series(100.0 + np.arange(len(idx), dtype=float), index=idx)
    t3.iloc[:10] = np.nan
    close = pd.DataFrame({"T3": t3, "T4": 100.0 + np.arange(len(idx), dtype=float)}, index=idx)

    q = compute_price_quality(close, window_name="wf_0")
    flagged = flag_bad_tickers(q, zero_ret_thresh=0.95, min_valid_frac=0.98)

    row_t3 = flagged.loc[flagged["ticker"] == "T3"].iloc[0]
    assert bool(row_t3["bad_valid"])
    assert bool(row_t3["is_bad"])


def test_summarize_price_quality_topk() -> None:
    df = pd.DataFrame(
        [
            {"ticker": "A", "valid_frac_close": 1.0, "zero_ret_frac": 0.99, "window_name": "w", "is_bad": True},
            {"ticker": "B", "valid_frac_close": 0.70, "zero_ret_frac": 0.10, "window_name": "w", "is_bad": True},
            {"ticker": "C", "valid_frac_close": 0.95, "zero_ret_frac": 0.80, "window_name": "w", "is_bad": False},
        ]
    )
    out = summarize_price_quality(df, topk=2)
    assert out["num_bad"] == 2
    assert len(out["worst_by_zero_ret"]) == 2
    assert out["worst_by_zero_ret"][0]["ticker"] == "A"
    assert out["worst_by_valid_frac"][0]["ticker"] == "B"
