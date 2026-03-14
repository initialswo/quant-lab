"""Cross-sectional factor score normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cs(scores: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    """Winsorize each date across tickers to [q(p), q(1-p)]."""
    if not 0.0 <= p < 0.5:
        raise ValueError("winsorize_cs requires 0 <= p < 0.5")

    def _winsor_row(row: pd.Series) -> pd.Series:
        finite = row.dropna()
        if finite.empty:
            return row
        lo = finite.quantile(p)
        hi = finite.quantile(1.0 - p)
        return row.clip(lower=lo, upper=hi)

    return scores.astype(float).apply(_winsor_row, axis=1)


def zscore_cs(scores: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Z-score each date across tickers, preserving NaN locations."""
    if eps <= 0:
        raise ValueError("zscore_cs requires eps > 0")

    def _zscore_row(row: pd.Series) -> pd.Series:
        out = row.copy()
        finite_mask = row.notna()
        if not finite_mask.any():
            return out
        vals = row[finite_mask].astype(float)
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=0))
        if (not np.isfinite(sigma)) or sigma < eps:
            out.loc[finite_mask] = 0.0
            return out
        out.loc[finite_mask] = (vals - mu) / (sigma + eps)
        return out

    return scores.astype(float).apply(_zscore_row, axis=1)


def zscore_cs_nanstd(scores: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Z-score each date; when cross-sectional std is ~0, output NaN for finite entries."""
    if eps <= 0:
        raise ValueError("zscore_cs_nanstd requires eps > 0")

    def _zscore_row(row: pd.Series) -> pd.Series:
        out = row.copy()
        finite_mask = row.notna()
        if not finite_mask.any():
            return out
        vals = row[finite_mask].astype(float)
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=0))
        if (not np.isfinite(sigma)) or sigma < eps:
            out.loc[finite_mask] = np.nan
            return out
        out.loc[finite_mask] = (vals - mu) / (sigma + eps)
        return out

    return scores.astype(float).apply(_zscore_row, axis=1)


def percentile_rank_cs(scores: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, preserving NaN."""
    return scores.astype(float).rank(axis=1, pct=True, na_option="keep")


def robust_preprocess_base(
    scores: pd.DataFrame,
    winsor_p: float = 0.05,
) -> pd.DataFrame:
    """Winsorize then z-score (std~0 -> NaN), without ranking."""
    w = winsorize_cs(scores.astype(float), p=float(winsor_p))
    return zscore_cs_nanstd(w).astype(float)


def preprocess_factor_scores(
    scores: pd.DataFrame,
    use_factor_normalization: bool = True,
    winsor_p: float = 0.05,
) -> pd.DataFrame:
    """
    Robust factor preprocessing pipeline:
    winsorize(5/95 by default) -> zscore (NaN if std~0) -> percentile rank.
    """
    if not bool(use_factor_normalization):
        return scores.astype(float)
    z = robust_preprocess_base(scores=scores, winsor_p=winsor_p)
    return percentile_rank_cs(z).astype(float)


def normalize_scores(
    scores: pd.DataFrame,
    method: str,
    winsor_p: float = 0.01,
) -> pd.DataFrame:
    """Normalize scores with one of: none, zscore, winsor_zscore."""
    method_norm = method.strip().lower()
    if method_norm == "none":
        return scores.astype(float)
    if method_norm == "zscore":
        return zscore_cs(scores)
    if method_norm == "winsor_zscore":
        return zscore_cs(winsorize_cs(scores, p=winsor_p))
    raise ValueError(f"Unknown normalize method '{method}'. Use none|zscore|winsor_zscore.")
