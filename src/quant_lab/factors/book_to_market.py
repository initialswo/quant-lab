"""Book-to-market value factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "book_to_market"


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
    shareholders_equity: pd.DataFrame | None = None,
    shares_outstanding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Book-to-market = shareholders_equity / market_cap, with market_cap = close * shares_outstanding.

    Inputs must already be PIT-aligned (values become visible on/after available_date).
    Missing or non-positive shareholders_equity -> NaN.
    Non-positive market_cap -> NaN.
    """
    close = close.astype(float)
    book = _get_panel(close, fundamentals_aligned, shareholders_equity, "shareholders_equity")
    shares = _get_panel(close, fundamentals_aligned, shares_outstanding, "shares_outstanding")
    market_cap = close * shares
    out = book / market_cap
    out = out.where(book > 0.0, pd.NA)
    out = out.where(market_cap > 0.0, pd.NA)
    return out.astype(float)
