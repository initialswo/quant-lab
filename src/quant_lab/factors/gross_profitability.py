"""Gross profitability quality factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "gross_profitability"


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
    gross_profit: pd.DataFrame | None = None,
    revenue: pd.DataFrame | None = None,
    cogs: pd.DataFrame | None = None,
    total_assets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Gross profitability = gross_profit / total_assets.

    Fallback numerator when gross_profit missing: (revenue - cogs).
    Denominator <= 0 -> NaN.
    """
    close = close.astype(float)
    gp = _get_panel(close, fundamentals_aligned, gross_profit, "gross_profit")
    rev = _get_panel(close, fundamentals_aligned, revenue, "revenue")
    cgs = _get_panel(close, fundamentals_aligned, cogs, "cogs")
    assets = _get_panel(close, fundamentals_aligned, total_assets, "total_assets")

    numerator = gp.copy()
    gp_missing = numerator.isna()
    fallback_num = rev - cgs
    numerator = numerator.where(~gp_missing, fallback_num)

    scores = numerator / assets
    scores = scores.where(assets > 0.0, pd.NA).astype(float)
    return scores
