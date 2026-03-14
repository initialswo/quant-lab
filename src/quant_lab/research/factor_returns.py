"""Factor return series diagnostics for research workflows."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _align_panels(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.DatetimeIndex(factor_scores.index).intersection(pd.DatetimeIndex(future_returns.index))
    cols = factor_scores.columns.intersection(future_returns.columns)
    scores = factor_scores.reindex(index=idx, columns=cols).astype(float)
    rets = future_returns.reindex(index=idx, columns=cols).astype(float)
    return scores, rets


def _max_drawdown_pnl(spread_returns: pd.Series) -> float:
    """Max drawdown on cumulative spread PnL (additive long-short convention)."""
    s = pd.Series(spread_returns).dropna().astype(float)
    if s.empty:
        return float("nan")
    pnl = s.cumsum()
    dd = pnl - pnl.cummax()
    return float(dd.min())


def run_factor_return_analysis(
    factor_scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    quantiles: int = 5,
    rolling_window: int = 52,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Build quantile and spread return series from factor scores and forward returns."""
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    if rolling_window <= 1:
        raise ValueError("rolling_window must be > 1")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")

    scores, fwd = _align_panels(factor_scores, future_returns)
    q_labels = list(range(1, quantiles + 1))
    q_rows: list[dict[str, float]] = []
    row_idx: list[pd.Timestamp] = []
    used_counts: list[int] = []

    for dt in scores.index:
        s = scores.loc[dt]
        r = fwd.loc[dt]
        mask = s.notna() & r.notna()
        n = int(mask.sum())
        used_counts.append(n)
        row_idx.append(dt)
        if n < quantiles:
            q_rows.append({f"Q{q}": float("nan") for q in q_labels})
            continue
        s_m = s[mask]
        r_m = r[mask]
        try:
            bins = pd.qcut(s_m.rank(method="first"), q=quantiles, labels=q_labels)
        except ValueError:
            q_rows.append({f"Q{q}": float("nan") for q in q_labels})
            continue
        q_ret = r_m.groupby(bins).mean()
        q_rows.append({f"Q{q}": float(q_ret.get(q, np.nan)) for q in q_labels})

    quantile_by_date = pd.DataFrame(q_rows, index=pd.DatetimeIndex(row_idx)).sort_index()
    quantile_mean = quantile_by_date.mean(axis=0, skipna=True)
    quantile_mean_dict = {k: float(v) for k, v in quantile_mean.to_dict().items()}
    spread = (quantile_by_date[f"Q{quantiles}"] - quantile_by_date["Q1"]).rename("Spread")
    spread_valid = spread.dropna()

    if spread_valid.empty:
        spread_summary = {
            "mean_period_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "sharpe": float("nan"),
            "cumulative_return": float("nan"),  # Backward-compatible alias of cumulative_pnl.
            "cumulative_pnl": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "n_obs": 0.0,
            "periods_per_year": float(periods_per_year),
            "return_convention": "long_short_spread_pnl",
            "max_drawdown_unit": "pnl",
        }
    else:
        n = int(len(spread_valid))
        mean_period = float(spread_valid.mean())
        ann_ret = float(mean_period * float(periods_per_year))
        ann_vol = float(spread_valid.std(ddof=0) * np.sqrt(float(periods_per_year)))
        cumulative_pnl = float(spread_valid.cumsum().iloc[-1])
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0.0 else float("nan")
        spread_summary = {
            "mean_period_return": mean_period,
            "annualized_return": ann_ret,
            "annualized_vol": ann_vol,
            "sharpe": sharpe,
            "cumulative_return": cumulative_pnl,  # Backward-compatible alias.
            "cumulative_pnl": cumulative_pnl,
            "max_drawdown": _max_drawdown_pnl(spread_valid),
            "hit_rate": float((spread_valid > 0.0).mean()),
            "n_obs": float(n),
            "periods_per_year": float(periods_per_year),
            "return_convention": "long_short_spread_pnl",
            "max_drawdown_unit": "pnl",
        }

    rolling_mean = spread.rolling(int(rolling_window)).mean()
    rolling_vol = spread.rolling(int(rolling_window)).std(ddof=0) * np.sqrt(float(periods_per_year))
    rolling_sharpe = rolling_mean / rolling_vol.replace(0.0, np.nan)
    used = pd.Series(used_counts, index=pd.DatetimeIndex(row_idx), name="assets_used").astype(int)
    coverage_summary = {
        "median_assets_used": float(used.median()) if not used.empty else float("nan"),
        "min_assets_used": int(used.min()) if not used.empty else 0,
        "max_assets_used": int(used.max()) if not used.empty else 0,
        "valid_dates": int(spread_valid.shape[0]),
        "assets_used_by_date": used,
    }

    return {
        "quantile_returns_by_date": quantile_by_date,
        "quantile_mean_returns": quantile_mean_dict,
        "spread_returns_by_date": spread,
        "spread_summary": spread_summary,
        "coverage_summary": coverage_summary,
        "rolling_diagnostics": {
            "rolling_mean": rolling_mean,
            "rolling_vol": rolling_vol,
            "rolling_sharpe": rolling_sharpe,
            "window": int(rolling_window),
        },
        "top_minus_bottom_spread_mean": float(quantile_mean_dict.get(f"Q{quantiles}", np.nan) - quantile_mean_dict.get("Q1", np.nan)),
        "quantiles": int(quantiles),
    }


def run_factor_return_correlation(spread_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Return correlation matrix across factor spread return series."""
    if not spread_series:
        raise ValueError("spread_series cannot be empty")
    panel = pd.concat({k: pd.Series(v, dtype=float) for k, v in spread_series.items()}, axis=1)
    return panel.corr()


def run_factor_seasonality(
    spread_returns: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compute month-of-year seasonality stats for a spread return series."""
    s = pd.Series(spread_returns).dropna().astype(float)
    if s.empty:
        return pd.DataFrame(
            columns=["month", "mean_return", "volatility", "sharpe", "hit_rate", "n_obs"]
        )
    month = pd.DatetimeIndex(s.index).month
    rows: list[dict[str, float | int]] = []
    for m in range(1, 13):
        vals = s.loc[month == m]
        if vals.empty:
            rows.append(
                {
                    "month": m,
                    "mean_return": float("nan"),
                    "volatility": float("nan"),
                    "sharpe": float("nan"),
                    "hit_rate": float("nan"),
                    "n_obs": 0,
                }
            )
            continue
        mean = float(vals.mean())
        vol = float(vals.std(ddof=0))
        ann_ret = mean * float(periods_per_year)
        ann_vol = vol * float(np.sqrt(periods_per_year))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0.0 else float("nan")
        rows.append(
            {
                "month": m,
                "mean_return": mean,
                "volatility": vol,
                "sharpe": sharpe,
                "hit_rate": float((vals > 0.0).mean()),
                "n_obs": int(vals.shape[0]),
            }
        )
    out = pd.DataFrame(rows).set_index("month")
    return out


def plot_factor_seasonality(
    seasonality: pd.DataFrame,
    outpath: str,
    title: str = "Factor Seasonality: Mean Spread Return by Month",
) -> str:
    """Plot mean spread return by calendar month and save to file."""
    s = pd.DataFrame(seasonality).copy()
    if "mean_return" not in s.columns:
        raise ValueError("seasonality must contain 'mean_return' column")
    x = s.index.to_list()
    y = s["mean_return"].astype(float).to_numpy()
    colors = ["#2ca02c" if (np.isfinite(v) and v >= 0.0) else "#d62728" for v in y]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, y, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Spread Return")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


def print_factor_seasonality(seasonality: pd.DataFrame, factor_name: str = "") -> None:
    """Print month-of-year seasonality table for a factor spread series."""
    print("FACTOR SEASONALITY")
    print("------------------")
    if factor_name:
        print(f"Factor: {factor_name}")
    if seasonality.empty:
        print("No seasonality observations available.")
        return
    printable = seasonality.copy()
    printable = printable[["mean_return", "volatility", "sharpe", "hit_rate", "n_obs"]]
    print(printable.round(6).to_string())


def print_factor_return_analysis(report: dict[str, Any], factor_name: str = "") -> None:
    """Print concise terminal summary for factor return analysis."""
    spread = report.get("spread_summary", {}) or {}
    q = report.get("quantile_mean_returns", {}) or {}
    quantiles = int(report.get("quantiles", 5))
    cov = report.get("coverage_summary", {}) or {}

    print("FACTOR RETURN ANALYSIS")
    print("----------------------")
    if factor_name:
        print(f"Factor: {factor_name}")
    print("")
    print("SPREAD SUMMARY")
    print(f"Annualized Return: {spread.get('annualized_return')}")
    print(f"Annualized Vol: {spread.get('annualized_vol')}")
    print(f"Sharpe: {spread.get('sharpe')}")
    print(f"Cumulative PnL: {spread.get('cumulative_pnl')}")
    print(f"MaxDD (PnL): {spread.get('max_drawdown')}")
    print(f"Hit Rate: {spread.get('hit_rate')}")
    print("")
    print("QUANTILE MEAN RETURNS")
    for qn in range(1, quantiles + 1):
        print(f"Q{qn}: {q.get(f'Q{qn}')}")
    print(f"Q{quantiles}-Q1 Spread: {report.get('top_minus_bottom_spread_mean')}")
    print("")
    print("COVERAGE")
    print(f"Median names per date: {cov.get('median_assets_used')}")
