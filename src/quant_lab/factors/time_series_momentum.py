"""Time-series momentum factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAME = "time_series_momentum"


def compute(
    close: pd.DataFrame,
    lookback: int = 252,
    negative_value: float = -1.0,
    zero_value: float = 0.0,
) -> pd.DataFrame:
    """Return the sign of the rolling lookback return for each asset."""
    if int(lookback) <= 0:
        raise ValueError("lookback must be > 0")

    prices = close.astype(float).sort_index()
    rolling_return = prices.pct_change(int(lookback))

    signal = pd.DataFrame(
        np.nan,
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )
    signal = signal.mask(rolling_return > 0.0, 1.0)
    signal = signal.mask(rolling_return < 0.0, float(negative_value))
    signal = signal.mask(rolling_return == 0.0, float(zero_value))
    return signal.astype(float)
