"""Asset growth anomaly factor."""

from __future__ import annotations

import pandas as pd

FACTOR_NAME = "asset_growth"


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
    total_assets: pd.DataFrame | None = None,
    lag_days: int = 252,
) -> pd.DataFrame:
    """
    Score low asset growth higher using PIT total_assets.

    asset_growth = assets_t / assets_t-lag - 1
    score = -asset_growth
    """
    close = close.astype(float)
    assets = _get_panel(close, fundamentals_aligned, total_assets, "total_assets")
    lagged = assets.shift(int(lag_days))
    growth = (assets / lagged) - 1.0
    growth = growth.where(lagged > 0.0, pd.NA)
    valid = assets.notna() & lagged.notna()
    return (-growth).where(valid).astype(float)
