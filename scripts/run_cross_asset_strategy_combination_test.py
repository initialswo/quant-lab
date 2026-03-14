"""Diversification analysis: Benchmark v1.1 vs cross-asset trend benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.metrics import compute_metrics


def _find_benchmark_v11_equity_curve() -> Path:
    """Locate the canonical Benchmark v1.1 run (geometric_rank lead baseline)."""
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
        if float(d.get("SlippageBps", 0.0)) != 0.0:
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


def _find_cross_asset_returns() -> Path:
    root = Path("results/cross_asset_trend_benchmark/cross_asset_trend_returns.csv")
    if root.exists():
        return root
    cands = sorted(Path("results/cross_asset_trend_benchmark").glob("*/cross_asset_trend_returns.csv"))
    if not cands:
        raise FileNotFoundError("Could not locate cross_asset_trend_returns.csv")
    return cands[-1]


def _drawdown(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.astype(float).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def main() -> None:
    factor_path = _find_benchmark_v11_equity_curve()
    cross_path = _find_cross_asset_returns()

    factor = (
        pd.read_csv(factor_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "factor_return"})
        .set_index("date")
        .sort_index()
    )
    cross = (
        pd.read_csv(cross_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "cross_asset_return"})
        .set_index("date")
        .sort_index()
    )
    aligned = factor.join(cross, how="inner").dropna().copy()

    corr = float(aligned["factor_return"].corr(aligned["cross_asset_return"]))
    roll = aligned["factor_return"].rolling(252).corr(aligned["cross_asset_return"])

    dd_factor = _drawdown(aligned["factor_return"])
    dd_cross = _drawdown(aligned["cross_asset_return"])
    in_dd_factor = dd_factor < 0.0
    in_dd_cross = dd_cross < 0.0
    both = in_dd_factor & in_dd_cross
    either = in_dd_factor | in_dd_cross
    dd_overlap_all = float(both.mean())
    dd_overlap_cond = float(both.sum() / either.sum()) if either.any() else float("nan")

    blends = {
        "100% Factor": (1.0, 0.0),
        "100% CrossAssetTrend": (0.0, 1.0),
        "90/10": (0.9, 0.1),
        "80/20": (0.8, 0.2),
        "70/30": (0.7, 0.3),
    }
    rows: list[dict[str, float | str]] = []
    out = aligned.copy()
    for name, (wf, wc) in blends.items():
        col = f"portfolio_{name.replace('%', 'pct').replace('/', '_').replace(' ', '')}"
        out[col] = wf * out["factor_return"] + wc * out["cross_asset_return"]
        m = compute_metrics(out[col])
        rows.append(
            {
                "Portfolio": name,
                "CAGR": float(m["CAGR"]),
                "Vol": float(m["Vol"]),
                "Sharpe": float(m["Sharpe"]),
                "MaxDD": float(m["MaxDD"]),
            }
        )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results/cross_asset_strategy_combination_test") / ts
    outdir.mkdir(parents=True, exist_ok=True)

    combined_path = outdir / "combined_returns.csv"
    corr_path = outdir / "strategy_correlation.csv"
    summary_path = outdir / "portfolio_summary.csv"
    json_path = outdir / "combination_summary.json"

    out.assign(
        drawdown_factor=dd_factor,
        drawdown_cross_asset=dd_cross,
        rolling_corr_252=roll,
        in_drawdown_factor=in_dd_factor.astype(int),
        in_drawdown_cross_asset=in_dd_cross.astype(int),
        in_drawdown_both=both.astype(int),
    ).reset_index().to_csv(combined_path, index=False, float_format="%.10g")

    corr_df = pd.DataFrame(
        [
            {
                "full_period_corr": corr,
                "rolling_corr_252_mean": float(roll.mean()),
                "rolling_corr_252_median": float(roll.median()),
                "rolling_corr_252_min": float(roll.min()),
                "rolling_corr_252_max": float(roll.max()),
                "drawdown_overlap_all_days": dd_overlap_all,
                "drawdown_overlap_conditional": dd_overlap_cond,
                "factor_returns_path": str(factor_path),
                "cross_asset_returns_path": str(cross_path),
            }
        ]
    )
    corr_df.to_csv(corr_path, index=False, float_format="%.10g")

    sum_df = pd.DataFrame(rows)
    sum_df.to_csv(summary_path, index=False, float_format="%.10g")

    payload = {
        "factor_returns_path": str(factor_path),
        "cross_asset_returns_path": str(cross_path),
        "correlation": corr_df.to_dict(orient="records")[0],
        "portfolio_summary": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    latest_root = Path("results/cross_asset_strategy_combination_test")
    out.assign(
        drawdown_factor=dd_factor,
        drawdown_cross_asset=dd_cross,
        rolling_corr_252=roll,
        in_drawdown_factor=in_dd_factor.astype(int),
        in_drawdown_cross_asset=in_dd_cross.astype(int),
        in_drawdown_both=both.astype(int),
    ).reset_index().to_csv(latest_root / "combined_returns.csv", index=False, float_format="%.10g")
    corr_df.to_csv(latest_root / "strategy_correlation.csv", index=False, float_format="%.10g")
    sum_df.to_csv(latest_root / "portfolio_summary.csv", index=False, float_format="%.10g")
    (latest_root / "combination_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("CROSS-ASSET STRATEGY COMBINATION TEST")
    print("-------------------------------------")
    print(corr_df.to_string(index=False))
    print("")
    print(sum_df.to_string(index=False))
    print("")
    print(f"Saved: {combined_path}")
    print(f"Saved: {corr_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
