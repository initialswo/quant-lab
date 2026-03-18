"""5-day reversal factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "reversal_5d"


def compute(close: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Return 5-day reversal scores: -(price[t] / price[t-5] - 1)."""
    close = close.astype(float)
    scores = -close.pct_change(periods=int(lookback), fill_method=None)
    return scores.astype(float)
