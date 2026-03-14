"""Earnings yield value factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "earnings_yield"


def _get_panel(
    close: pd.DataFrame,
    fundamentals_aligned: dict[str, pd.DataFrame] | None,
    direct_panel: pd.DataFrame | None,
    key: str,
) -> pd.DataFrame:
    if direct_panel is not None:
        return direct_panel.reindex(index=close.index, columns=close.columns).astype(float)
    if fundamentals_aligned is not None and key in fundamentals_aligned:
        return fundamentals_aligned[key].reindex(index=close.index, columns=close.columns).astype(float)
    return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)


def compute(
    close: pd.DataFrame,
    fundamentals_aligned: dict[str, pd.DataFrame] | None = None,
    net_income: pd.DataFrame | None = None,
    shares_outstanding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Earnings yield = net_income / market_cap, with market_cap = close * shares_outstanding.

    Inputs must already be PIT-aligned (values become visible on/after available_date).
    Denominator <= 0 or missing inputs -> NaN.
    """
    close = close.astype(float)
    ni = _get_panel(close, fundamentals_aligned, net_income, "net_income")
    shs = _get_panel(close, fundamentals_aligned, shares_outstanding, "shares_outstanding")
    market_cap = close * shs
    out = ni / market_cap
    out = out.where(market_cap > 0.0, pd.NA)
    return out.astype(float)

