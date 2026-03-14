"""Cross-sectional factor orthogonalization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def orthogonalize_factor_scores_cs(
    factor_scores: dict[str, pd.DataFrame],
    factor_order: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Orthogonalize factor panels cross-sectionally per date via sequential OLS residualization.

    For each date and factor order f1, f2, ...:
      f1_ortho = f1
      f2_ortho = residual(f2 ~ 1 + f1_ortho)
      f3_ortho = residual(f3 ~ 1 + f1_ortho + f2_ortho)
      ...
    """
    if not factor_scores:
        return {}

    names = list(factor_order) if factor_order is not None else list(factor_scores.keys())
    if set(names) != set(factor_scores.keys()):
        raise ValueError("factor_order must contain exactly the same factor names as factor_scores.")

    base = factor_scores[names[0]]
    idx = pd.DatetimeIndex(base.index)
    cols = list(base.columns)
    aligned = {n: factor_scores[n].reindex(index=idx, columns=cols).astype(float) for n in names}
    out = {n: pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float) for n in names}

    for dt in idx:
        for i, name in enumerate(names):
            y = aligned[name].loc[dt].astype(float)
            y_valid = y.notna()
            if not bool(y_valid.any()):
                continue
            if i == 0:
                out[name].loc[dt, y_valid] = y.loc[y_valid]
                continue

            prev_names = names[:i]
            prev_df = pd.DataFrame({p: out[p].loc[dt].astype(float) for p in prev_names})
            regress_mask = y_valid.copy()
            for p in prev_names:
                regress_mask = regress_mask & prev_df[p].notna()
            k = len(prev_names) + 1  # predictors + intercept
            n = int(regress_mask.sum())
            if n <= k:
                out[name].loc[dt, y_valid] = y.loc[y_valid]
                continue

            yv = y.loc[regress_mask].to_numpy(dtype=float)
            Xm = prev_df.loc[regress_mask, prev_names].to_numpy(dtype=float)
            Xd = np.column_stack([np.ones(n, dtype=float), Xm])
            beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
            resid = yv - Xd @ beta
            out[name].loc[dt, regress_mask] = resid

            # Preserve valid values where regression inputs are unavailable.
            not_regressed = y_valid & (~regress_mask)
            if bool(not_regressed.any()):
                out[name].loc[dt, not_regressed] = y.loc[not_regressed]

    return {n: out[n].astype(float) for n in names}


def maybe_orthogonalize_factor_scores(
    factor_scores: dict[str, pd.DataFrame],
    enabled: bool = False,
    factor_order: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Optionally apply cross-sectional factor orthogonalization."""
    if not bool(enabled):
        return {k: v.astype(float) for k, v in factor_scores.items()}
    return orthogonalize_factor_scores_cs(factor_scores=factor_scores, factor_order=factor_order)

