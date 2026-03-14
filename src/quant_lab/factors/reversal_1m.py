"""1-month reversal factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "reversal_1m"


def compute(close: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Return 1-month reversal scores: -(price[t] / price[t-21] - 1)."""
    close = close.astype(float)
    scores = -close.pct_change(lookback)
    return scores.astype(float)

