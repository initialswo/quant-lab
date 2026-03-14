"""Broader liquid US equity universe builder."""

from __future__ import annotations

import pandas as pd


def build_liquid_us_universe(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
    adv_window: int = 20,
    min_history: int = 252,
) -> pd.DataFrame:
    """
    Build date x ticker boolean eligibility matrix for liquid US equities.

    Eligibility rules:
    - price >= min_price
    - rolling average dollar volume >= min_avg_dollar_volume
    - available price history count >= min_history
    """
    if int(adv_window) <= 0:
        raise ValueError("adv_window must be > 0")
    if int(min_history) <= 0:
        raise ValueError("min_history must be > 0")
    if float(min_price) < 0:
        raise ValueError("min_price must be >= 0")
    if float(min_avg_dollar_volume) < 0:
        raise ValueError("min_avg_dollar_volume must be >= 0")

    px = prices.astype(float).sort_index()
    vol = pd.DataFrame(volumes).reindex(index=px.index, columns=px.columns).astype(float)

    dollar_volume = px * vol
    adv = dollar_volume.rolling(int(adv_window), min_periods=int(adv_window)).mean()
    history = px.notna().cumsum()

    eligible = (px >= float(min_price)) & (adv >= float(min_avg_dollar_volume)) & (
        history >= int(min_history)
    )
    return eligible.reindex(index=px.index, columns=px.columns).fillna(False).astype(bool)
