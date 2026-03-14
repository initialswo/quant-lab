"""Low-volatility factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "low_vol_20"


def compute(close: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Return negative rolling volatility scores."""
    close = close.astype(float)
    rets = close.pct_change()
    vol = rets.rolling(lookback).std()
    score = -vol
    return score.astype(float)
