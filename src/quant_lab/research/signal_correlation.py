"""Cross-sectional signal correlation diagnostics."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _pair_corr(x: pd.Series, y: pd.Series, method: str) -> float:
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


def run_signal_correlation(
    signal_panels: dict[str, pd.DataFrame],
    method: str = "spearman",
) -> dict[str, Any]:
    """Compute pairwise cross-sectional signal correlations."""
    m = str(method).lower()
    if m not in {"spearman", "pearson"}:
        raise ValueError("method must be one of: spearman, pearson")
    names = [str(k) for k in signal_panels.keys()]
    if len(names) < 2:
        raise ValueError("signal_panels must include at least two signals")

    pair_series: dict[str, pd.Series] = {}
    coverage_rows: list[dict[str, Any]] = []
    avg = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for n in names:
        avg.loc[n, n] = 1.0

    for a, b in combinations(names, 2):
        pa = signal_panels[a].astype(float)
        pb = signal_panels[b].astype(float)
        idx = pd.DatetimeIndex(pa.index).intersection(pd.DatetimeIndex(pb.index))
        cols = pa.columns.intersection(pb.columns)
        pa = pa.reindex(index=idx, columns=cols)
        pb = pb.reindex(index=idx, columns=cols)

        corr_vals: list[float] = []
        overlap_vals: list[int] = []
        for dt in idx:
            xa = pa.loc[dt]
            xb = pb.loc[dt]
            mask = xa.notna() & xb.notna()
            n = int(mask.sum())
            overlap_vals.append(n)
            if n < 3:
                corr_vals.append(np.nan)
                continue
            corr_vals.append(_pair_corr(xa[mask], xb[mask], method=m))

        corr_s = pd.Series(corr_vals, index=idx, name=f"{a}|{b}", dtype=float)
        overlap_s = pd.Series(overlap_vals, index=idx, name=f"{a}|{b}", dtype=int)
        pair_key = f"{a}|{b}"
        pair_series[pair_key] = corr_s

        mean_corr = float(corr_s.mean(skipna=True))
        avg.loc[a, b] = mean_corr
        avg.loc[b, a] = mean_corr
        coverage_rows.append(
            {
                "signal_a": a,
                "signal_b": b,
                "median_overlap": float(overlap_s.median()) if not overlap_s.empty else float("nan"),
                "valid_dates": int(corr_s.notna().sum()),
                "mean_corr": mean_corr,
            }
        )

    coverage = pd.DataFrame(coverage_rows).sort_values(["signal_a", "signal_b"]).reset_index(drop=True)
    ranking = coverage.dropna(subset=["mean_corr"]).copy()
    ranking["abs_corr"] = ranking["mean_corr"].abs()
    most = (
        ranking.sort_values("abs_corr", ascending=False)
        .head(5)[["signal_a", "signal_b", "mean_corr"]]
        .to_dict(orient="records")
    )
    least = (
        ranking.sort_values("abs_corr", ascending=True)
        .head(5)[["signal_a", "signal_b", "mean_corr"]]
        .to_dict(orient="records")
    )

    return {
        "method": m,
        "pairwise_correlation_by_date": pair_series,
        "average_correlation_matrix": avg,
        "coverage_summary": coverage,
        "most_correlated_pairs": most,
        "least_correlated_pairs": least,
    }


def print_signal_correlation(report: dict[str, Any]) -> None:
    """Print concise terminal summary for signal correlation diagnostics."""
    method = str(report.get("method", "spearman"))
    avg = report.get("average_correlation_matrix")
    most = report.get("most_correlated_pairs", []) or []
    least = report.get("least_correlated_pairs", []) or []

    print("SIGNAL CORRELATION")
    print("------------------")
    print(f"Method: {method}")
    print("")
    print("AVERAGE CORRELATION MATRIX")
    if isinstance(avg, pd.DataFrame):
        rounded = avg.copy().astype(float).round(4)
        print(rounded.to_string())
    else:
        print("-")
    print("")
    print("MOST CORRELATED PAIRS")
    if most:
        for row in most:
            print(f"{row['signal_a']} / {row['signal_b']}: {row['mean_corr']:.4f}")
    else:
        print("-")
    print("")
    print("LEAST CORRELATED PAIRS")
    if least:
        for row in least:
            print(f"{row['signal_a']} / {row['signal_b']}: {row['mean_corr']:.4f}")
    else:
        print("-")
