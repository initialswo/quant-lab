from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.engine.runner import _prepare_close_panel, filter_bad_tickers
from quant_lab.factors.combine import combine_factor_scores


def test_price_fill_none_does_not_ffill() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    close_raw = pd.DataFrame(
        {
            "A": [100.0, 101.0, np.nan, np.nan, 104.0, 105.0],
            "B": [50.0, 50.5, 51.0, 51.2, 51.4, 51.8],
        },
        index=idx,
    )

    none_panel = _prepare_close_panel(close_raw, price_fill_mode="none")
    ffill_panel = _prepare_close_panel(close_raw, price_fill_mode="ffill")

    assert pd.isna(none_panel.loc[idx[2], "A"])
    assert pd.isna(none_panel.loc[idx[3], "A"])
    assert np.isclose(float(ffill_panel.loc[idx[2], "A"]), 101.0)
    assert np.isclose(float(ffill_panel.loc[idx[3], "A"]), 101.0)


def test_drop_bad_tickers_removes_stale() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.DataFrame(
        {
            "A": np.full(len(idx), 100.0),
            "B": 100.0 + np.cumsum(np.linspace(0.1, 0.5, len(idx))),
        },
        index=idx,
    )

    filtered, dropped, flagged = filter_bad_tickers(
        close=close,
        window_name="w",
        zero_ret_thresh=0.95,
        min_valid_frac=0.98,
        exempt=set(),
    )

    assert "A" in dropped
    assert "A" not in filtered.columns
    assert "B" in filtered.columns
    assert bool(flagged.loc[flagged["ticker"] == "A", "is_bad"].iloc[0])


def test_drop_bad_tickers_exempt_spy() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    close = pd.DataFrame(
        {
            "SPY": np.full(len(idx), 400.0),
            "X": np.full(len(idx), 100.0),
            "Y": 100.0 + np.cumsum(np.linspace(0.1, 0.2, len(idx))),
        },
        index=idx,
    )

    filtered, dropped, _ = filter_bad_tickers(
        close=close,
        window_name="w",
        zero_ret_thresh=0.95,
        min_valid_frac=0.98,
        exempt={"SPY"},
    )

    assert "SPY" not in dropped
    assert "SPY" in filtered.columns
    assert "X" in dropped


def test_walkforward_filtering_column_intersection_sanity() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    close_hist = pd.DataFrame(
        {
            "A": np.full(len(idx), 100.0),  # stale -> dropped
            "B": 100.0 + np.arange(len(idx), dtype=float),
            "C": 120.0 + np.arange(len(idx), dtype=float),
        },
        index=idx,
    )
    filtered, dropped, _ = filter_bad_tickers(
        close=close_hist,
        window_name="wf",
        zero_ret_thresh=0.95,
        min_valid_frac=0.98,
        exempt=set(),
    )
    assert "A" in dropped

    # Mimic downstream factor score map with one factor still carrying dropped column.
    s1 = pd.DataFrame(1.0, index=idx, columns=["A", "B", "C"])
    s2 = pd.DataFrame(2.0, index=idx, columns=["B", "C"])
    allowed_cols = list(filtered.columns)
    scores = {
        "f1": s1.reindex(columns=allowed_cols),
        "f2": s2.reindex(columns=allowed_cols),
    }
    out = combine_factor_scores(scores, {"f1": 0.5, "f2": 0.5})
    assert out.columns.tolist() == allowed_cols
