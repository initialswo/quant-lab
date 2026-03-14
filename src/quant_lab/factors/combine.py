"""Factor score combination utilities."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from quant_lab.factors.normalize import percentile_rank_cs


def combine_factor_scores(
    scores: dict[str, pd.DataFrame],
    weights: Mapping[str, float | pd.Series],
    require_all_factors: bool = False,
) -> pd.DataFrame:
    """
    Combine factor score panels with scalar or per-date weights.

    - Per-date weights are re-normalized after removing factors that are all-NaN on each date.
    - If all factors are unusable on a date, output row is all NaN.
    """
    if not scores:
        raise ValueError("No factor scores were provided.")

    factor_names = list(scores.keys())
    base = scores[factor_names[0]]
    idx = pd.DatetimeIndex(base.index)
    cols = base.columns

    # Align all factor panels to the base shape expected by the runner.
    aligned: dict[str, pd.DataFrame] = {
        n: df.reindex(index=idx, columns=cols).astype(float) for n, df in scores.items()
    }

    # Scalar-only fast path reproduces existing deterministic behavior.
    if all(not isinstance(weights.get(n, 0.0), pd.Series) for n in factor_names):
        w = {n: float(weights.get(n, 0.0)) for n in factor_names}
        tot = float(sum(w.values()))
        if tot <= 0.0:
            raise ValueError("Scalar factor weights must sum to > 0.")
        w = {n: v / tot for n, v in w.items()}
        out = pd.DataFrame(0.0, index=idx, columns=cols, dtype=float)
        any_signal = pd.DataFrame(False, index=idx, columns=cols)
        all_signal = pd.DataFrame(True, index=idx, columns=cols)
        for n in factor_names:
            panel = aligned[n]
            out = out + float(w[n]) * panel.fillna(0.0)
            any_signal = any_signal | panel.notna()
            all_signal = all_signal & panel.notna()
        return out.where(all_signal if require_all_factors else any_signal).astype(float)

    # General path with date-varying weights.
    weight_series: dict[str, pd.Series] = {}
    for n in factor_names:
        w = weights.get(n, 0.0)
        if isinstance(w, pd.Series):
            ws = pd.Series(w, dtype=float).reindex(idx)
        else:
            ws = pd.Series(float(w), index=idx, dtype=float)
        weight_series[n] = ws

    out = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    for dt in idx:
        available = [n for n in factor_names if not aligned[n].loc[dt].isna().all()]
        if not available:
            continue

        weight_names = factor_names if require_all_factors else available
        raw = pd.Series({n: float(weight_series[n].loc[dt]) for n in weight_names}, dtype=float)
        raw = raw.clip(lower=0.0)
        tot = float(raw.sum())
        if tot <= 0.0:
            raw[:] = 1.0 / len(raw)
        else:
            raw = raw / tot

        row = pd.Series(0.0, index=cols, dtype=float)
        any_signal = pd.Series(False, index=cols)
        all_signal = pd.Series(True, index=cols)
        for n in weight_names:
            panel_row = aligned[n].loc[dt]
            row = row + float(raw[n]) * panel_row.fillna(0.0)
            any_signal = any_signal | panel_row.notna()
            all_signal = all_signal & panel_row.notna()
        out.loc[dt] = row.where(all_signal if require_all_factors else any_signal)

    return out.astype(float)


def aggregate_factor_scores(
    scores: dict[str, pd.DataFrame],
    weights: Mapping[str, float | pd.Series],
    method: str = "linear",
    require_all_factors: bool = False,
) -> pd.DataFrame:
    """Aggregate factor panels via linear or rank-based methods."""
    mode = str(method).strip().lower() or "linear"
    if mode == "linear":
        return combine_factor_scores(scores=scores, weights=weights, require_all_factors=require_all_factors)
    if mode not in {"mean_rank", "geometric_rank"}:
        raise ValueError("factor_aggregation_method must be one of: linear, mean_rank, geometric_rank")
    if not scores:
        raise ValueError("No factor scores were provided.")

    names = list(scores.keys())
    base = scores[names[0]]
    idx = pd.DatetimeIndex(base.index)
    cols = list(base.columns)
    ranked = [
        percentile_rank_cs(scores[n].reindex(index=idx, columns=cols).astype(float)).to_numpy(dtype=float)
        for n in names
    ]
    cube = np.stack(ranked, axis=2)  # T x N x F
    any_valid = np.isfinite(cube).any(axis=2)
    all_valid = np.isfinite(cube).all(axis=2)

    cnt = np.isfinite(cube).sum(axis=2).astype(float)
    if mode == "mean_rank":
        num = np.nansum(cube, axis=2)
        agg = np.divide(num, cnt, out=np.full_like(num, np.nan, dtype=float), where=cnt > 0)
    else:
        cube_pos = np.where(np.isfinite(cube), np.clip(cube, 1e-12, None), np.nan)
        log_num = np.nansum(np.log(cube_pos), axis=2)
        agg = np.exp(np.divide(log_num, cnt, out=np.full_like(log_num, np.nan, dtype=float), where=cnt > 0))

    out = pd.DataFrame(agg, index=idx, columns=cols, dtype=float)
    mask = all_valid if require_all_factors else any_valid
    return out.where(mask).astype(float)
