"""Diversification test: long-only benchmark vs Long/Short Equity v1."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.engine.metrics import compute_metrics


def _find_benchmark_v11_equity_curve() -> Path:
    rank_csv = Path("results/rank_aggregation_test/rank_aggregation_comparison.csv")
    if rank_csv.exists():
        df = pd.read_csv(rank_csv)
        hit = df.loc[df["FactorAggregationMethod"].astype(str).str.lower() == "geometric_rank"]
        if not hit.empty:
            outdir = str(hit.iloc[0]["Outdir"])
            p = Path(outdir) / "equity_curve.csv"
            if p.exists():
                return p

    candidates: list[tuple[float, Path]] = []
    for s in Path("results").glob("*/summary.json"):
        try:
            d = json.loads(s.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(d.get("FactorNames", "")) != "momentum_12_1;reversal_1m;low_vol_20;gross_profitability":
            continue
        if str(d.get("FactorAggregationMethod", "")).lower() != "geometric_rank":
            continue
        if str(d.get("PortfolioMode", "composite")) != "composite":
            continue
        if bool(d.get("DynamicFactorWeights", False)) is not True:
            continue
        if int(d.get("RankBuffer", -1)) != 20:
            continue
        if float(d.get("TargetVol", 0.0)) != 0.14:
            continue
        if float(d.get("BearExposureScale", 0.0)) != 1.0:
            continue
        if float(d.get("CostsBps", 10.0)) != 10.0:
            continue
        if int(d.get("ExecutionDelayDays", 0)) != 0:
            continue
        if str(d.get("Start", "")) != "2005-01-01" or str(d.get("End", "")) != "2024-12-31":
            continue
        p = s.parent / "equity_curve.csv"
        if p.exists():
            candidates.append((s.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError("Could not locate Benchmark v1.1 geometric_rank equity_curve.csv")
    candidates.sort()
    return candidates[-1][1]


def _find_long_short_best_returns() -> Path:
    latest = Path("results/long_short_equity_benchmark/best_variant_returns.csv")
    if latest.exists():
        return latest
    cands = sorted(Path("results/long_short_equity_benchmark").glob("*/best_variant_returns.csv"))
    if not cands:
        raise FileNotFoundError("Could not locate long_short best_variant_returns.csv")
    return cands[-1]


def _drawdown(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.astype(float).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def main() -> None:
    benchmark_path = _find_benchmark_v11_equity_curve()
    ls_path = _find_long_short_best_returns()

    benchmark = (
        pd.read_csv(benchmark_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "benchmark_return"})
        .set_index("date")
        .sort_index()
    )
    ls = (
        pd.read_csv(ls_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "long_short_return"})
        .set_index("date")
        .sort_index()
    )
    aligned = benchmark.join(ls, how="inner").dropna().copy()
    if aligned.empty:
        raise RuntimeError("No overlapping dates between benchmark and long/short returns.")

    corr = float(aligned["benchmark_return"].corr(aligned["long_short_return"]))
    rolling_corr = aligned["benchmark_return"].rolling(252).corr(aligned["long_short_return"])

    dd_bench = _drawdown(aligned["benchmark_return"])
    dd_ls = _drawdown(aligned["long_short_return"])
    in_dd_bench = dd_bench < 0.0
    in_dd_ls = dd_ls < 0.0
    both = in_dd_bench & in_dd_ls
    either = in_dd_bench | in_dd_ls
    dd_overlap_all = float(both.mean())
    dd_overlap_cond = float(both.sum() / either.sum()) if either.any() else float("nan")

    blends = {
        "100% Benchmark": (1.0, 0.0),
        "100% Long/Short": (0.0, 1.0),
        "80/20": (0.8, 0.2),
        "70/30": (0.7, 0.3),
        "60/40": (0.6, 0.4),
    }

    out = aligned.copy()
    rows: list[dict[str, float | str]] = []
    for name, (wb, wl) in blends.items():
        col = f"portfolio_{name.replace('%', 'pct').replace('/', '_').replace(' ', '').lower()}"
        out[col] = wb * out["benchmark_return"] + wl * out["long_short_return"]
        m = compute_metrics(out[col])
        rows.append(
            {
                "Portfolio": name,
                "WeightBenchmark": wb,
                "WeightLongShort": wl,
                "CAGR": float(m["CAGR"]),
                "Vol": float(m["Vol"]),
                "Sharpe": float(m["Sharpe"]),
                "MaxDD": float(m["MaxDD"]),
            }
        )

    corr_df = pd.DataFrame(
        [
            {
                "full_period_corr": corr,
                "rolling_corr_252_mean": float(rolling_corr.mean()),
                "rolling_corr_252_median": float(rolling_corr.median()),
                "rolling_corr_252_min": float(rolling_corr.min()),
                "rolling_corr_252_max": float(rolling_corr.max()),
                "drawdown_overlap_all_days": dd_overlap_all,
                "drawdown_overlap_conditional": dd_overlap_cond,
                "benchmark_returns_path": str(benchmark_path),
                "long_short_returns_path": str(ls_path),
            }
        ]
    )
    portfolio_df = pd.DataFrame(rows)
    out_with_diag = out.assign(
        drawdown_benchmark=dd_bench,
        drawdown_long_short=dd_ls,
        rolling_corr_252=rolling_corr,
        in_drawdown_benchmark=in_dd_bench.astype(int),
        in_drawdown_long_short=in_dd_ls.astype(int),
        in_drawdown_both=both.astype(int),
    ).reset_index()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path("results/long_short_strategy_combination_test")
    outdir = root / ts
    outdir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    combined_path = outdir / "combined_returns.csv"
    corr_path = outdir / "strategy_correlation.csv"
    summary_path = outdir / "portfolio_summary.csv"

    out_with_diag.to_csv(combined_path, index=False, float_format="%.10g")
    corr_df.to_csv(corr_path, index=False, float_format="%.10g")
    portfolio_df.to_csv(summary_path, index=False, float_format="%.10g")

    payload = {
        "benchmark_returns_path": str(benchmark_path),
        "long_short_returns_path": str(ls_path),
        "correlation": corr_df.to_dict(orient="records")[0],
        "portfolio_summary": rows,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_with_diag.to_csv(root / "combined_returns.csv", index=False, float_format="%.10g")
    corr_df.to_csv(root / "strategy_correlation.csv", index=False, float_format="%.10g")
    portfolio_df.to_csv(root / "portfolio_summary.csv", index=False, float_format="%.10g")
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("LONG/SHORT STRATEGY COMBINATION TEST")
    print("------------------------------------")
    print(corr_df.to_string(index=False))
    print("")
    print(portfolio_df.to_string(index=False))
    print("")
    print(f"Saved: {combined_path}")
    print(f"Saved: {corr_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
