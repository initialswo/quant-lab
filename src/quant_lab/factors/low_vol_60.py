"""60-day low-volatility factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "low_vol_60"


def compute(close: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Return negative rolling 60-day volatility scores."""
    close = close.astype(float)
    rets = close.pct_change()
    vol = rets.rolling(lookback).std()
    scores = -vol
    return scores.astype(float)

