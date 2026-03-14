"""Cross-sectional factor neutralization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_scores_cs(
    scores: pd.DataFrame,
    sector_by_ticker: dict[str, str] | None = None,
    log_market_cap_by_ticker: dict[str, float] | None = None,
    log_market_cap_exposure: pd.DataFrame | None = None,
    beta_exposure: pd.DataFrame | None = None,
    use_beta_neutralization: bool = False,
    use_sector_neutralization: bool = True,
    use_size_neutralization: bool = True,
) -> pd.DataFrame:
    """
    Neutralize scores cross-sectionally per date using OLS residuals.

    Model per date:
      score = b0 + b_beta * beta + sector_dummies + b_size * log_market_cap + error
    """
    out = scores.astype(float).copy()
    cols = list(out.columns)
    sector_map = {str(k): str(v) for k, v in (sector_by_ticker or {}).items()}
    size_map = {str(k): float(v) for k, v in (log_market_cap_by_ticker or {}).items()}

    sector_series = pd.Series({c: sector_map.get(c, "UNKNOWN") for c in cols}, dtype=object)
    size_series = pd.Series({c: size_map.get(c, np.nan) for c in cols}, dtype=float)

    for dt in out.index:
        y = out.loc[dt].astype(float)
        valid = y.notna()
        if not bool(valid.any()):
            continue

        X_parts: list[pd.DataFrame] = []
        idx = y.index[valid]
        if bool(use_beta_neutralization) and beta_exposure is not None:
            beta_row = (
                beta_exposure.reindex(index=[dt], columns=out.columns)
                .iloc[0]
                .astype(float)
                .rename("beta")
            )
            if beta_row.notna().any():
                X_parts.append(beta_row.loc[idx].to_frame("beta"))
        if bool(use_sector_neutralization):
            sec = pd.get_dummies(sector_series.loc[idx], drop_first=True, dtype=float)
            if not sec.empty:
                X_parts.append(sec)
        if bool(use_size_neutralization):
            if log_market_cap_exposure is not None:
                size = (
                    log_market_cap_exposure.reindex(index=[dt], columns=out.columns)
                    .iloc[0]
                    .astype(float)
                    .loc[idx]
                )
            else:
                size = size_series.loc[idx]
            if size.notna().any():
                X_parts.append(size.to_frame("log_market_cap"))

        if X_parts:
            X = pd.concat(X_parts, axis=1).astype(float)
            xy = pd.concat([y.loc[idx].rename("y"), X], axis=1).dropna()
            if xy.empty:
                out.loc[dt, valid] = np.nan
                continue
            yv = xy["y"].to_numpy(dtype=float)
            Xm = xy.drop(columns=["y"]).to_numpy(dtype=float)
            Xd = np.column_stack([np.ones(len(yv), dtype=float), Xm])
            if Xd.shape[0] <= Xd.shape[1]:
                # Insufficient observations for this date to estimate the model robustly.
                out.loc[dt, xy.index] = np.nan
                continue
            beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
            resid = yv - Xd @ beta
            out.loc[dt, xy.index] = resid
            dropped = idx.difference(xy.index)
            if len(dropped) > 0:
                out.loc[dt, dropped] = np.nan
        else:
            # No usable covariates for this date -> leave values unchanged.
            out.loc[dt, idx] = y.loc[idx].astype(float)
    return out.astype(float)
