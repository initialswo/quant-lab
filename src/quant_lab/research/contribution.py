"""Ticker-level return contribution utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_ticker_contributions(
    close: pd.DataFrame,
    realized_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily additive return contributions by ticker.

    contribution[t, i] = realized_weights[t-1, i] * close_return[t, i]
    """
    px = close.astype(float).sort_index()
    w = (
        realized_weights.astype(float)
        .reindex(index=px.index, columns=px.columns)
        .ffill()
        .fillna(0.0)
    )
    ret = px.pct_change().fillna(0.0)
    contrib = w.shift(1).fillna(0.0) * ret
    if not contrib.empty:
        contrib.iloc[0] = 0.0
    return contrib.astype(float)


def summarize_contribution_concentration(
    contribution_by_ticker: pd.Series,
) -> dict[str, float]:
    """Summarize concentration using absolute contribution shares."""
    c = pd.to_numeric(contribution_by_ticker, errors="coerce").fillna(0.0).astype(float)
    abs_c = c.abs()
    total_abs = float(abs_c.sum())
    if total_abs <= 0.0:
        return {
            "top1_share_abs": float("nan"),
            "top5_share_abs": float("nan"),
            "top10_share_abs": float("nan"),
            "top20_share_abs": float("nan"),
            "herfindahl_abs": float("nan"),
            "effective_n_abs": float("nan"),
            "positive_contributors": 0.0,
            "negative_contributors": 0.0,
            "median_contribution": float("nan"),
            "mean_contribution": float("nan"),
            "std_contribution": float("nan"),
            "p90_abs_contribution": float("nan"),
            "p99_abs_contribution": float("nan"),
        }
    shares = (abs_c / total_abs).sort_values(ascending=False)
    hhi = float((shares.pow(2)).sum())
    return {
        "top1_share_abs": float(shares.iloc[:1].sum()),
        "top5_share_abs": float(shares.iloc[:5].sum()),
        "top10_share_abs": float(shares.iloc[:10].sum()),
        "top20_share_abs": float(shares.iloc[:20].sum()),
        "herfindahl_abs": hhi,
        "effective_n_abs": float(1.0 / hhi) if hhi > 0.0 else float("nan"),
        "positive_contributors": float((c > 0.0).sum()),
        "negative_contributors": float((c < 0.0).sum()),
        "median_contribution": float(c.median()),
        "mean_contribution": float(c.mean()),
        "std_contribution": float(c.std(ddof=0)),
        "p90_abs_contribution": float(abs_c.quantile(0.90)),
        "p99_abs_contribution": float(abs_c.quantile(0.99)),
    }

