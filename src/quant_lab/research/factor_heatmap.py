"""Factor sweep heatmap utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_lab.research.factor_diagnostics import run_factor_diagnostics
from quant_lab.research.factor_returns import run_factor_return_analysis


MetricName = Literal["sharpe", "ic"]
PeriodName = Literal["year", "quarter"]


def _period_key(index: pd.DatetimeIndex, period: PeriodName) -> pd.Index:
    if period == "year":
        return pd.Index(index.year.astype(str), name="period")
    if period == "quarter":
        return pd.Index(index.to_period("Q").astype(str), name="period")
    raise ValueError("period must be one of: year, quarter")


def compute_momentum_sweep_matrix(
    close: pd.DataFrame,
    future_returns: pd.DataFrame,
    lookbacks: list[int],
    metric: MetricName = "sharpe",
    period: PeriodName = "year",
) -> pd.DataFrame:
    """
    Compute momentum sweep matrix (rows=lookback, cols=period bucket).

    - momentum score = close.pct_change(lookback)
    - metric:
      - sharpe: from factor spread return analysis
      - ic: mean Spearman IC from factor diagnostics
    """
    m = str(metric).lower()
    if m not in {"sharpe", "ic"}:
        raise ValueError("metric must be one of: sharpe, ic")
    if not lookbacks:
        raise ValueError("lookbacks cannot be empty")

    idx = pd.DatetimeIndex(close.index).intersection(pd.DatetimeIndex(future_returns.index))
    close = close.reindex(idx).astype(float)
    fwd = future_returns.reindex(idx, columns=close.columns).astype(float)
    pkey = _period_key(idx, period=period)
    periods = sorted(pd.Index(pkey.unique()).astype(str))

    out = pd.DataFrame(index=[int(x) for x in lookbacks], columns=periods, dtype=float)
    for lb in lookbacks:
        score = close.pct_change(int(lb)).astype(float)
        for p in periods:
            mask = pkey == p
            if not bool(mask.any()):
                out.loc[int(lb), p] = np.nan
                continue
            s = score.loc[mask]
            r = fwd.loc[mask]
            if m == "sharpe":
                rep = run_factor_return_analysis(s, r, quantiles=5)
                out.loc[int(lb), p] = float(rep["spread_summary"].get("sharpe", np.nan))
            else:
                rep = run_factor_diagnostics(s, r, quantiles=5, method="spearman", horizons=[1])
                out.loc[int(lb), p] = float(rep["ic_summary"].get("mean_ic", np.nan))
    out.index.name = "lookback"
    return out


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    outpath: str | Path,
    cmap: str = "RdYlGn",
) -> Path:
    """Plot and save a heatmap with red-yellow-green scale centered at zero."""
    mat = matrix.astype(float)
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * mat.shape[1]), max(3, 0.5 * mat.shape[0])))
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([str(x) for x in mat.index])
    ax.set_xlabel(str(mat.columns.name or "period"))
    ax.set_ylabel(str(mat.index.name or "parameter"))
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out

