"""Risk overlays for portfolio weights."""

from __future__ import annotations

import pandas as pd


def apply_trend_filter(
    weights: pd.DataFrame,
    market_close: pd.Series,
    sma_window: int = 200,
) -> pd.DataFrame:
    """
    Gate portfolio exposure using a lagged market trend signal.

    Causal rule:
    - sma[t] = mean(market_close[t-sma_window+1 : t])
    - risk_on[t] = market_close[t] > sma[t]
    - apply risk_on.shift(1) to weights at time t
    """
    if sma_window <= 0:
        raise ValueError("sma_window must be > 0")

    market_close = pd.Series(market_close).astype(float).reindex(weights.index)
    sma = market_close.rolling(sma_window).mean()
    risk_on = (market_close > sma).shift(1)
    risk_on = risk_on.reindex(weights.index).fillna(False).astype(bool)
    assert risk_on.dtype == bool

    gated = weights.copy()
    gated.loc[~risk_on, :] = 0.0
    return gated
