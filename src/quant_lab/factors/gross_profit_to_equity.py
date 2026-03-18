"""Equity-scaled profitability factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "gross_profit_to_equity"


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
    shareholders_equity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Gross profit to equity = gross_profit / shareholders_equity.

    Inputs must already be PIT-aligned. Missing inputs stay NaN.
    Non-positive shareholders_equity returns NaN.
    """
    close = close.astype(float)
    gp = _get_panel(close, fundamentals_aligned, gross_profit, "gross_profit")
    equity = _get_panel(close, fundamentals_aligned, shareholders_equity, "shareholders_equity")

    out = gp / equity
    out = out.where(equity > 0.0, pd.NA)
    return out.astype(float)
