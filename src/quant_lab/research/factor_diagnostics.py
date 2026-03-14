"""Factor diagnostics for cross-sectional signal research."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _cross_section_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    xv = pd.Series(x).astype(float)
    yv = pd.Series(y).astype(float)
    if method == "spearman":
        xv = xv.rank(method="average")
        yv = yv.rank(method="average")
    x_arr = xv.to_numpy(dtype=float)
    y_arr = yv.to_numpy(dtype=float)
    x_std = float(np.std(x_arr))
    y_std = float(np.std(y_arr))
    if not np.isfinite(x_std) or not np.isfinite(y_std) or x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _align_panels(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.DatetimeIndex(factor_scores.index).intersection(pd.DatetimeIndex(future_returns.index))
    cols = factor_scores.columns.intersection(future_returns.columns)
    f = factor_scores.reindex(index=idx, columns=cols).astype(float)
    r = future_returns.reindex(index=idx, columns=cols).astype(float)
    return f, r


def _compound_forward_returns(one_period_fwd: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if horizon == 1:
        return one_period_fwd.copy()
    out = pd.DataFrame(1.0, index=one_period_fwd.index, columns=one_period_fwd.columns, dtype=float)
    for k in range(horizon):
        out = out * (1.0 + one_period_fwd.shift(-k))
    return out - 1.0


def compute_ic_by_date(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    method: str = "spearman",
    min_obs: int = 3,
) -> pd.Series:
    f, r = _align_panels(factor_scores, future_returns)
    m = str(method).lower()
    if m not in {"spearman", "pearson"}:
        raise ValueError("method must be one of: spearman, pearson")

    vals: list[float] = []
    dates: list[pd.Timestamp] = []
    for dt in f.index:
        s = f.loc[dt]
        y = r.loc[dt]
        mask = s.notna() & y.notna()
        n = int(mask.sum())
        if n < int(min_obs):
            vals.append(np.nan)
            dates.append(dt)
            continue
        ic = _cross_section_corr(s[mask], y[mask], method=m)
        vals.append(float(ic) if ic is not None else np.nan)
        dates.append(dt)
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name="IC")


def summarize_ic(ic_by_date: pd.Series) -> dict[str, float]:
    s = pd.Series(ic_by_date).dropna().astype(float)
    n = int(len(s))
    if n == 0:
        return {
            "mean_ic": float("nan"),
            "std_ic": float("nan"),
            "ic_tstat": float("nan"),
            "ic_hit_rate": float("nan"),
            "n_obs": 0.0,
        }
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and np.isfinite(std) and std > 0.0:
        tstat = float(mean / (std / np.sqrt(n)))
    else:
        tstat = float("nan")
    hit = float((s > 0.0).mean())
    return {
        "mean_ic": mean,
        "std_ic": std,
        "ic_tstat": tstat,
        "ic_hit_rate": hit,
        "n_obs": float(n),
    }


def compute_quantile_returns(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    quantiles: int = 5,
    min_obs: int | None = None,
) -> dict[str, Any]:
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    min_required = int(min_obs) if min_obs is not None else int(quantiles)
    f, r = _align_panels(factor_scores, future_returns)

    q_labels = list(range(1, quantiles + 1))
    rows: list[dict[str, float]] = []
    row_idx: list[pd.Timestamp] = []

    for dt in f.index:
        s = f.loc[dt]
        y = r.loc[dt]
        mask = s.notna() & y.notna()
        n = int(mask.sum())
        if n < max(min_required, quantiles):
            continue
        s_m = s[mask]
        y_m = y[mask]
        try:
            bins = pd.qcut(s_m.rank(method="first"), q=quantiles, labels=q_labels)
        except ValueError:
            # Too many ties / insufficient unique values.
            continue
        q_ret = y_m.groupby(bins).mean()
        row = {f"Q{q}": float(q_ret.get(q, np.nan)) for q in q_labels}
        rows.append(row)
        row_idx.append(dt)

    by_date = pd.DataFrame(rows, index=pd.DatetimeIndex(row_idx)).sort_index()
    if by_date.empty:
        summary = {f"Q{q}": float("nan") for q in q_labels}
        spread = float("nan")
    else:
        avg = by_date.mean(axis=0)
        summary = {k: float(v) for k, v in avg.to_dict().items()}
        spread = float(summary.get(f"Q{quantiles}", np.nan) - summary.get("Q1", np.nan))

    return {
        "quantile_returns_by_date": by_date,
        "quantile_return_summary": summary,
        "top_minus_bottom_spread": spread,
    }


def compute_decay_summary(
    factor_scores: pd.DataFrame,
    one_period_future_returns: pd.DataFrame,
    horizons: list[int],
    method: str = "spearman",
) -> dict[int, float]:
    out: dict[int, float] = {}
    for h in horizons:
        fwd_h = _compound_forward_returns(one_period_future_returns, int(h))
        ic_h = compute_ic_by_date(factor_scores, fwd_h, method=method)
        out[int(h)] = float(pd.Series(ic_h).mean(skipna=True))
    return out


def compute_coverage_summary(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    min_obs_for_ic: int = 3,
) -> dict[str, Any]:
    f, r = _align_panels(factor_scores, future_returns)
    factor_valid = f.notna().sum(axis=1).astype(int)
    returns_valid = r.notna().sum(axis=1).astype(int)
    used = (f.notna() & r.notna()).sum(axis=1).astype(int)
    return {
        "factor_valid_count_by_date": factor_valid,
        "future_return_valid_count_by_date": returns_valid,
        "used_in_ic_count_by_date": used,
        "factor_valid_median": float(factor_valid.median()) if not factor_valid.empty else float("nan"),
        "future_return_valid_median": float(returns_valid.median()) if not returns_valid.empty else float("nan"),
        "used_in_ic_median": float(used.median()) if not used.empty else float("nan"),
        "min_obs_for_ic": int(min_obs_for_ic),
    }


def run_factor_diagnostics(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    quantiles: int = 5,
    method: str = "spearman",
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    h = [1, 5, 21, 63] if horizons is None else [int(x) for x in horizons]
    h = sorted({x for x in h if int(x) > 0})
    if not h:
        raise ValueError("horizons must contain at least one positive integer")

    ic_by_date = compute_ic_by_date(factor_scores, future_returns, method=method)
    ic_summary = summarize_ic(ic_by_date)
    q = compute_quantile_returns(
        factor_scores=factor_scores,
        future_returns=future_returns,
        quantiles=quantiles,
    )
    decay = compute_decay_summary(
        factor_scores=factor_scores,
        one_period_future_returns=future_returns,
        horizons=h,
        method=method,
    )
    coverage = compute_coverage_summary(factor_scores, future_returns)

    return {
        "ic_by_date": ic_by_date,
        "ic_summary": ic_summary,
        "quantile_returns_by_date": q["quantile_returns_by_date"],
        "quantile_return_summary": q["quantile_return_summary"],
        "top_minus_bottom_spread": q["top_minus_bottom_spread"],
        "decay_summary": decay,
        "coverage_summary": coverage,
        "method": str(method).lower(),
        "quantiles": int(quantiles),
        "horizons": h,
    }


def print_factor_diagnostics(report: dict[str, Any], factor_name: str = "") -> None:
    ic = report.get("ic_summary", {}) or {}
    q = report.get("quantile_return_summary", {}) or {}
    decay = report.get("decay_summary", {}) or {}
    cov = report.get("coverage_summary", {}) or {}
    quantiles = int(report.get("quantiles", 5))
    method = str(report.get("method", "spearman"))

    print("FACTOR DIAGNOSTICS")
    print("------------------")
    if factor_name:
        print(f"Factor: {factor_name}")
    print(f"Method: {method}")
    print("")
    print("IC SUMMARY")
    print(f"Mean IC: {ic.get('mean_ic')}")
    print(f"Std IC: {ic.get('std_ic')}")
    print(f"IC t-stat: {ic.get('ic_tstat')}")
    print(f"IC Hit Rate: {ic.get('ic_hit_rate')}")
    print("")
    print("QUANTILE RETURNS")
    for qn in range(1, quantiles + 1):
        print(f"Q{qn}: {q.get(f'Q{qn}')}")
    print(f"Q{quantiles}-Q1 Spread: {report.get('top_minus_bottom_spread')}")
    print("")
    print("DECAY")
    for hz in sorted(decay.keys()):
        print(f"{hz}: {decay[hz]}")
    print("")
    print("COVERAGE")
    print(f"Median names per date: {cov.get('used_in_ic_median')}")
