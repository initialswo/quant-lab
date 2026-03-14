"""3-1 momentum factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "momentum_3_1"


def compute(
    close: pd.DataFrame,
    lookback_long: int = 63,
    lookback_short: int = 21,
) -> pd.DataFrame:
    """Return 3-1 momentum scores: price[t-21] / price[t-63] - 1."""
    close = close.astype(float)
    scores = close.shift(lookback_short) / close.shift(lookback_long) - 1.0
    return scores.astype(float)

