"""5-day mean reversion factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "mean_reversion_5"


def compute(close: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Return negative recent return for mean-reversion ranking."""
    close = close.astype(float)
    scores = -(close.pct_change(lookback))
    return scores.astype(float)
