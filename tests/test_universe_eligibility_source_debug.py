from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.engine.runner import _build_zero_eligible_debug_frame, _resolve_universe_eligibility


def test_score_based_eligibility_reduces_count_when_scores_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["A", "B", "C"]
    eligibility_price = pd.DataFrame(True, index=idx, columns=cols)
    scores = pd.DataFrame(
        [[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]],
        index=idx,
        columns=cols,
    )

    elig_price = _resolve_universe_eligibility(eligibility_price, scores, source="price")
    elig_score = _resolve_universe_eligibility(eligibility_price, scores, source="score")

    assert int(elig_price.loc[idx[0]].sum()) == 3
    assert int(elig_score.loc[idx[0]].sum()) == 2


def test_zero_eligible_debug_frame_only_zero_dates_and_required_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = ["A", "B"]
    rb_dates = pd.DatetimeIndex([idx[1], idx[3]])

    close = pd.DataFrame(
        [[10, 20], [10, 20], [10, 20], [10, 20]],
        index=idx,
        columns=cols,
        dtype=float,
    )
    eligibility_price = pd.DataFrame(True, index=idx, columns=cols)
    factor_scores = {
        "f1": pd.DataFrame([[1, 2], [np.nan, np.nan], [1, 2], [3, 4]], index=idx, columns=cols),
        "f2": pd.DataFrame([[1, 2], [np.nan, np.nan], [1, 2], [4, 5]], index=idx, columns=cols),
    }
    composite = pd.DataFrame([[1, 2], [np.nan, np.nan], [1, 2], [3, 4]], index=idx, columns=cols)
    eligibility_final = pd.DataFrame(
        [[True, True], [False, False], [True, True], [True, True]], index=idx, columns=cols
    )

    dbg = _build_zero_eligible_debug_frame(
        rebalance_dates=rb_dates,
        close=close,
        eligibility_price=eligibility_price,
        price_ok=eligibility_price,
        history_ok=eligibility_price,
        valid_ok=eligibility_price,
        factor_scores=factor_scores,
        composite_scores=composite,
        eligibility_final=eligibility_final,
        factor_names=["f1", "f2"],
        effective_min_history_days=252,
        valid_lookback=252,
        min_valid_frac=0.98,
        window_id=4,
    )

    assert len(dbg) == 1
    assert str(dbg.iloc[0]["date"]) == str(idx[1].date())
    required = {
        "window_id",
        "date",
        "close_valid_count",
        "eligibility_price_count",
        "price_ok_count",
        "history_ok_count",
        "valid_ok_count",
        "factor_valid_count_f1",
        "factor_valid_count_f2",
        "composite_valid_count",
        "eligibility_final_count",
        "effective_min_history_days",
        "valid_lookback",
        "min_valid_frac",
    }
    assert required.issubset(set(dbg.columns))
    assert int(dbg.iloc[0]["eligibility_final_count"]) == 0
