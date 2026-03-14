"""Reusable cross-sectional factor diagnostics with point-in-time-safe forward returns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_lab.strategies.topn import rebalance_mask

DEFAULT_SUBPERIODS: list[tuple[str, str, str]] = [
    ("2005-2009", "2005-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2024", "2020-01-01", "2024-12-31"),
]

def compute_forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute close-to-close forward return at horizon h: close[t+h]/close[t]-1."""
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be > 0")
    px = pd.DataFrame(close).astype(float).sort_index()
    return (px.shift(-h) / px) - 1.0


def _align_inputs(
    factor_scores: pd.DataFrame,
    close: pd.DataFrame,
    eligibility_mask: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    idx = pd.DatetimeIndex(factor_scores.index).intersection(pd.DatetimeIndex(close.index))
    cols = factor_scores.columns.intersection(close.columns)
    scores = factor_scores.reindex(index=idx, columns=cols).astype(float)
    px = close.reindex(index=idx, columns=cols).astype(float)
    elig: pd.DataFrame | None = None
    if eligibility_mask is not None:
        elig = eligibility_mask.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    return scores, px, elig


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")
    xv = xr.to_numpy(dtype=float)
    yv = yr.to_numpy(dtype=float)
    if len(xv) < 2:
        return float("nan")
    xstd = float(np.std(xv))
    ystd = float(np.std(yv))
    if (not np.isfinite(xstd)) or (not np.isfinite(ystd)) or xstd <= 0.0 or ystd <= 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _period_mask(index: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    return pd.Series((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)), index=idx)


def _summary_stats(values: pd.Series) -> dict[str, float]:
    s = pd.Series(values).dropna().astype(float)
    if s.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def summarize_coverage(
    coverage_by_date: pd.DataFrame,
    subperiods: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Summarize valid-name counts and coverage fractions overall and by subperiod."""
    sp = subperiods or DEFAULT_SUBPERIODS
    c = pd.DataFrame(coverage_by_date).copy()
    if c.empty:
        empty = {
            "overall": {
                "valid_names": _summary_stats(pd.Series(dtype=float)),
                "coverage_fraction": _summary_stats(pd.Series(dtype=float)),
            },
            "by_subperiod": {},
        }
        return empty

    out: dict[str, Any] = {
        "overall": {
            "valid_names": _summary_stats(c["valid_names"]),
            "coverage_fraction": _summary_stats(c["coverage_fraction"]),
        },
        "by_subperiod": {},
    }
    for label, start, end in sp:
        mask = _period_mask(pd.DatetimeIndex(c.index), start, end)
        part = c.loc[mask]
        out["by_subperiod"][label] = {
            "valid_names": _summary_stats(part["valid_names"] if not part.empty else pd.Series(dtype=float)),
            "coverage_fraction": _summary_stats(
                part["coverage_fraction"] if not part.empty else pd.Series(dtype=float)
            ),
        }
    return out


def compute_coverage_by_date(
    factor_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligibility_mask: pd.DataFrame | None = None,
    signal_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Build per-date coverage table using same-date signal and future return availability."""
    idx = pd.DatetimeIndex(factor_scores.index).intersection(pd.DatetimeIndex(forward_returns.index))
    cols = factor_scores.columns.intersection(forward_returns.columns)
    f = factor_scores.reindex(index=idx, columns=cols).astype(float)
    r = forward_returns.reindex(index=idx, columns=cols).astype(float)

    if eligibility_mask is not None:
        e = eligibility_mask.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    else:
        e = pd.DataFrame(True, index=idx, columns=cols)

    valid = f.notna() & r.notna() & e
    valid_names = valid.sum(axis=1).astype(int)
    eligible_names = e.sum(axis=1).astype(int)
    eligible_nonzero = eligible_names.replace(0, np.nan)
    coverage = (valid_names / eligible_nonzero).astype(float)

    out = pd.DataFrame(
        {
            "valid_names": valid_names,
            "eligible_names": eligible_names,
            "coverage_fraction": coverage,
        },
        index=idx,
    )
    if signal_mask is not None:
        sm = pd.Series(signal_mask).reindex(idx).fillna(False).astype(bool)
        out = out.loc[sm]
    return out.sort_index()


def compute_ic_by_date(
    factor_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligibility_mask: pd.DataFrame | None = None,
    signal_mask: pd.Series | None = None,
    min_obs: int = 5,
) -> pd.DataFrame:
    """Compute per-date Spearman rank IC using same-date signal vs future returns."""
    idx = pd.DatetimeIndex(factor_scores.index).intersection(pd.DatetimeIndex(forward_returns.index))
    cols = factor_scores.columns.intersection(forward_returns.columns)
    f = factor_scores.reindex(index=idx, columns=cols).astype(float)
    r = forward_returns.reindex(index=idx, columns=cols).astype(float)
    if eligibility_mask is not None:
        e = eligibility_mask.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    else:
        e = pd.DataFrame(True, index=idx, columns=cols)

    sm = pd.Series(signal_mask).reindex(idx).fillna(False).astype(bool) if signal_mask is not None else None
    rows: list[dict[str, float]] = []
    for dt in idx:
        if sm is not None and not bool(sm.loc[dt]):
            continue
        s = f.loc[dt]
        y = r.loc[dt]
        mask = s.notna() & y.notna() & e.loc[dt]
        n = int(mask.sum())
        ic = float("nan")
        if n >= int(min_obs):
            ic = _spearman_corr(s[mask], y[mask])
        rows.append({"Date": dt, "IC": ic, "N": float(n)})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["IC", "N"], index=pd.DatetimeIndex([]))
    out = out.set_index("Date").sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out


def summarize_ic(
    ic_by_date: pd.DataFrame,
    subperiods: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    sp = subperiods or DEFAULT_SUBPERIODS
    ic = pd.DataFrame(ic_by_date).copy()

    def _one(s: pd.Series) -> dict[str, float]:
        vals = pd.Series(s).dropna().astype(float)
        n = int(vals.shape[0])
        if n == 0:
            return {
                "mean_ic": float("nan"),
                "std_ic": float("nan"),
                "ic_ir": float("nan"),
                "ic_hit_rate": float("nan"),
                "count": 0.0,
            }
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else float("nan")
        ir = float(mean / std) if n > 1 and np.isfinite(std) and std > 0.0 else float("nan")
        return {
            "mean_ic": mean,
            "std_ic": std,
            "ic_ir": ir,
            "ic_hit_rate": float((vals > 0.0).mean()),
            "count": float(n),
        }

    out: dict[str, Any] = {"overall": _one(ic["IC"]) if "IC" in ic.columns else _one(pd.Series(dtype=float)), "by_subperiod": {}}
    for label, start, end in sp:
        if ic.empty:
            part = pd.Series(dtype=float)
        else:
            mask = _period_mask(pd.DatetimeIndex(ic.index), start, end)
            part = ic.loc[mask, "IC"] if "IC" in ic.columns else pd.Series(dtype=float)
        out["by_subperiod"][label] = _one(part)
    return out


def compute_quantile_returns_by_date(
    factor_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantiles: int = 5,
    eligibility_mask: pd.DataFrame | None = None,
    signal_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute per-date quantile forward returns and top-bottom spread."""
    q = int(quantiles)
    if q < 2:
        raise ValueError("quantiles must be >= 2")

    idx = pd.DatetimeIndex(factor_scores.index).intersection(pd.DatetimeIndex(forward_returns.index))
    cols = factor_scores.columns.intersection(forward_returns.columns)
    f = factor_scores.reindex(index=idx, columns=cols).astype(float)
    r = forward_returns.reindex(index=idx, columns=cols).astype(float)
    if eligibility_mask is not None:
        e = eligibility_mask.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    else:
        e = pd.DataFrame(True, index=idx, columns=cols)

    labels = list(range(1, q + 1))
    sm = pd.Series(signal_mask).reindex(idx).fillna(False).astype(bool) if signal_mask is not None else None
    rows: list[dict[str, float]] = []
    for dt in idx:
        if sm is not None and not bool(sm.loc[dt]):
            continue
        s = f.loc[dt]
        y = r.loc[dt]
        mask = s.notna() & y.notna() & e.loc[dt]
        n = int(mask.sum())
        row: dict[str, float] = {f"Q{k}": float("nan") for k in labels}
        row["N"] = float(n)
        row["Spread_QTop_Q1"] = float("nan")
        if n >= q:
            s_m = s[mask]
            y_m = y[mask]
            try:
                bins = pd.qcut(s_m.rank(method="first"), q=q, labels=labels)
            except ValueError:
                bins = None
            if bins is not None:
                q_ret = y_m.groupby(bins).mean()
                for k in labels:
                    row[f"Q{k}"] = float(q_ret.get(k, np.nan))
                row["Spread_QTop_Q1"] = float(row[f"Q{q}"] - row["Q1"])
        row["Date"] = dt
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        columns = [f"Q{k}" for k in labels] + ["Spread_QTop_Q1", "N"]
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]))
    out = out.set_index("Date").sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out


def summarize_quantiles(
    quantile_returns_by_date: pd.DataFrame,
    quantiles: int = 5,
    subperiods: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Summarize quantile means and top-bottom spread behavior overall and by subperiod."""
    sp = subperiods or DEFAULT_SUBPERIODS
    q = int(quantiles)
    qr = pd.DataFrame(quantile_returns_by_date).copy()

    def _one(frame: pd.DataFrame, period_label: str) -> dict[str, float | str]:
        row: dict[str, float | str] = {"Period": period_label}
        for k in range(1, q + 1):
            col = f"Q{k}"
            row[col] = float(frame[col].mean()) if col in frame.columns and not frame.empty else float("nan")
        spread = frame["Spread_QTop_Q1"] if "Spread_QTop_Q1" in frame.columns else pd.Series(dtype=float)
        spread = pd.Series(spread).dropna().astype(float)
        n = int(spread.shape[0])
        mean = float(spread.mean()) if n > 0 else float("nan")
        std = float(spread.std(ddof=1)) if n > 1 else float("nan")
        sharpe_like = float(mean / std) if n > 1 and np.isfinite(std) and std > 0.0 else float("nan")
        hit = float((spread > 0.0).mean()) if n > 0 else float("nan")
        row["SpreadMean"] = mean
        row["SpreadStd"] = std
        row["SpreadSharpeLike"] = sharpe_like
        row["SpreadHitRate"] = hit
        row["Count"] = float(n)
        return row

    rows: list[dict[str, float | str]] = [_one(qr, "overall")]
    for label, start, end in sp:
        if qr.empty:
            part = qr
        else:
            mask = _period_mask(pd.DatetimeIndex(qr.index), start, end)
            part = qr.loc[mask]
        rows.append(_one(part, label))
    return pd.DataFrame(rows)


def compute_factor_correlation_summary(
    target_factor: pd.DataFrame,
    peer_factors: dict[str, pd.DataFrame],
    eligibility_mask: pd.DataFrame | None = None,
    signal_mask: pd.Series | None = None,
    min_overlap: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-sectional Spearman correlations per date vs peer factors."""
    if not peer_factors:
        return (
            pd.DataFrame(columns=["Date", "Peer", "Correlation", "N"]),
            pd.DataFrame(columns=["Peer", "AverageCorrelation", "MedianCorrelation", "Count"]),
        )

    idx = pd.DatetimeIndex(target_factor.index)
    cols = target_factor.columns
    tgt = target_factor.astype(float)
    if eligibility_mask is not None:
        elig = eligibility_mask.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    else:
        elig = pd.DataFrame(True, index=idx, columns=cols)

    by_date_rows: list[dict[str, float | str | pd.Timestamp]] = []
    summary_rows: list[dict[str, float | str]] = []

    sm = pd.Series(signal_mask).reindex(idx).fillna(False).astype(bool) if signal_mask is not None else None

    for peer_name, peer_df in peer_factors.items():
        p = peer_df.reindex(index=idx, columns=cols).astype(float)
        vals: list[float] = []
        for dt in idx:
            if sm is not None and not bool(sm.loc[dt]):
                continue
            a = tgt.loc[dt]
            b = p.loc[dt]
            mask = a.notna() & b.notna() & elig.loc[dt]
            n = int(mask.sum())
            corr = float("nan")
            if n >= int(min_overlap):
                corr = _spearman_corr(a[mask], b[mask])
                if np.isfinite(corr):
                    vals.append(float(corr))
            by_date_rows.append(
                {"Date": dt, "Peer": peer_name, "Correlation": float(corr), "N": float(n)}
            )
        s = pd.Series(vals, dtype=float)
        summary_rows.append(
            {
                "Peer": peer_name,
                "AverageCorrelation": float(s.mean()) if not s.empty else float("nan"),
                "MedianCorrelation": float(s.median()) if not s.empty else float("nan"),
                "Count": float(s.shape[0]),
            }
        )

    by_date = pd.DataFrame(by_date_rows)
    if not by_date.empty:
        by_date = by_date.sort_values(["Date", "Peer"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values("Peer").reset_index(drop=True)
    return by_date, summary


def run_cross_sectional_factor_diagnostics(
    factor_scores: pd.DataFrame,
    close: pd.DataFrame,
    eligibility_mask: pd.DataFrame | None = None,
    rebalance: str = "monthly",
    quantiles: int = 5,
    horizon: int = 21,
    subperiods: list[tuple[str, str, str]] | None = None,
    peer_factors: dict[str, pd.DataFrame] | None = None,
    min_obs_ic: int = 5,
    min_overlap_corr: int = 20,
) -> dict[str, Any]:
    """
    Run reusable diagnostics using PIT-safe construction.

    Signal definition: factor_scores at date t.
    Forward return definition: close[t+h] / close[t] - 1.
    """
    scores, px, elig = _align_inputs(
        factor_scores=factor_scores,
        close=close,
        eligibility_mask=eligibility_mask,
    )
    signal_dates = rebalance_mask(pd.DatetimeIndex(scores.index), rebalance=str(rebalance))
    fwd = compute_forward_returns(px, horizon=int(horizon))

    coverage_by_date = compute_coverage_by_date(
        factor_scores=scores,
        forward_returns=fwd,
        eligibility_mask=elig,
        signal_mask=signal_dates,
    )
    coverage_summary = summarize_coverage(coverage_by_date, subperiods=subperiods)

    ic_by_date = compute_ic_by_date(
        factor_scores=scores,
        forward_returns=fwd,
        eligibility_mask=elig,
        signal_mask=signal_dates,
        min_obs=int(min_obs_ic),
    )
    ic_summary = summarize_ic(ic_by_date, subperiods=subperiods)

    quantile_by_date = compute_quantile_returns_by_date(
        factor_scores=scores,
        forward_returns=fwd,
        quantiles=int(quantiles),
        eligibility_mask=elig,
        signal_mask=signal_dates,
    )
    quantile_summary = summarize_quantiles(
        quantile_returns_by_date=quantile_by_date,
        quantiles=int(quantiles),
        subperiods=subperiods,
    )

    corr_by_date, corr_summary = compute_factor_correlation_summary(
        target_factor=scores,
        peer_factors=peer_factors or {},
        eligibility_mask=elig,
        signal_mask=signal_dates,
        min_overlap=int(min_overlap_corr),
    )

    return {
        "config": {
            "rebalance": str(rebalance),
            "quantiles": int(quantiles),
            "horizon": int(horizon),
            "min_obs_ic": int(min_obs_ic),
            "min_overlap_corr": int(min_overlap_corr),
        },
        "forward_returns": fwd,
        "coverage_by_date": coverage_by_date,
        "coverage_summary": coverage_summary,
        "ic_by_date": ic_by_date,
        "ic_summary": ic_summary,
        "quantile_returns_by_date": quantile_by_date,
        "quantile_summary": quantile_summary,
        "factor_correlation_by_date": corr_by_date,
        "factor_correlation_summary": corr_summary,
    }
