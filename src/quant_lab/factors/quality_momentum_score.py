"""Combined quality-momentum sleeve score."""

from __future__ import annotations

import pandas as pd

from quant_lab.factors.gross_profitability import compute as compute_gross_profitability
from quant_lab.factors.momentum import compute as compute_momentum

FACTOR_NAME = "quality_momentum_score"


def compute(
    close: pd.DataFrame,
    fundamentals_aligned: dict[str, pd.DataFrame] | None = None,
    gross_profit: pd.DataFrame | None = None,
    revenue: pd.DataFrame | None = None,
    cogs: pd.DataFrame | None = None,
    total_assets: pd.DataFrame | None = None,
    lookback_long: int = 252,
    lookback_short: int = 21,
) -> pd.DataFrame:
    """Average 0-1 cross-sectional ranks of momentum and profitability."""
    momentum = compute_momentum(
        close=close,
        lookback_long=int(lookback_long),
        lookback_short=int(lookback_short),
    ).astype(float)
    profitability = compute_gross_profitability(
        close=close,
        fundamentals_aligned=fundamentals_aligned,
        gross_profit=gross_profit,
        revenue=revenue,
        cogs=cogs,
        total_assets=total_assets,
    ).astype(float)
    momentum_rank = momentum.rank(axis=1, pct=True, na_option="keep")
    profitability_rank = profitability.rank(axis=1, pct=True, na_option="keep")
    combined = 0.5 * momentum_rank + 0.5 * profitability_rank
    valid = momentum_rank.notna() & profitability_rank.notna()
    return combined.where(valid).astype(float)
