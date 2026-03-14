"""Momentum factors."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "momentum_12_1"

def compute(
    close: pd.DataFrame,
    lookback_long: int = 252,
    lookback_short: int = 21,
) -> pd.DataFrame:
    """Compute 12-1 style momentum."""
    close = close.astype(float)
    mom = close.pct_change(lookback_long) - close.pct_change(lookback_short)
    return mom.astype(float)


def compute_momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible helper."""
    return compute(close=close, lookback_long=252, lookback_short=21)
