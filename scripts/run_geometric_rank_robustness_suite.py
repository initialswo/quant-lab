"""Robustness suite for geometric-rank lead variant with linear-benchmark comparison."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest


SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]
WF_WINDOWS: list[tuple[str, str, str, str]] = [
    ("2005-01-01", "2012-12-31", "2013-01-01", "2014-12-31"),
    ("2005-01-01", "2014-12-31", "2015-01-01", "2016-12-31"),
    ("2005-01-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("2005-01-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("2005-01-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("2005-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),
]
COSTS_BPS = [5, 10, 20, 35, 50]
SLIPPAGE_BPS = [0, 5, 10, 20]
DELAYS = [0, 1, 3, 5]

BASE_CONFIG: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "universe": "sp500",
    "max_tickers": 2000,
    "top_n": 50,
    "rank_buffer": 20,
    "rebalance": "monthly",
    "costs_bps": 10.0,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_weights": [0.43, 0.27, 0.15, 0.15],
    "dynamic_factor_weights": True,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "regime_bull_weights": "momentum_12_1:0.48,reversal_1m:0.22,low_vol_20:0.10,gross_profitability:0.20",
    "regime_bear_weights": "momentum_12_1:0.28,reversal_1m:0.22,low_vol_20:0.30,gross_profitability:0.20",
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "fundamentals_path": "data/fundamentals/fundamentals_fmp.parquet",
    "fundamentals_fallback_lag_days": 60,
    "save_artifacts": True,
}


def _stability_from_outdir(outdir: str) -> tuple[float, float, float, float, float]:
    eq = pd.read_csv(Path(outdir) / "equity_curve.csv", parse_dates=["date"]).set_index("date").sort_index()
    r = pd.to_numeric(eq["returns"], errors="coerce").fillna(0.0)
    vals: list[float] = []
    for s, e in SUBPERIODS:
        rr = r.loc[(r.index >= pd.Timestamp(s)) & (r.index <= pd.Timestamp(e))]
        vals.append(float(compute_metrics(rr).get("Sharpe", float("nan"))))
    s = pd.Series(vals, dtype=float)
    mean_sub = float(s.mean())
    std_sub = float(s.std(ddof=0))
    min_sub = float(s.min())
    max_sub = float(s.max())
    stability = float(mean_sub - 0.5 * std_sub)
    return mean_sub, std_sub, min_sub, max_sub, stability


def _run_walkforward(
    method: str,
    outdir: Path,
    run_cache: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for train_start, train_end, test_start, test_end in WF_WINDOWS:
        cfg = dict(BASE_CONFIG)
        cfg["factor_aggregation_method"] = method
        cfg["start"] = "2005-01-01"
        cfg["end"] = test_end
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        eq = pd.read_csv(Path(run_outdir) / "equity_curve.csv", parse_dates=["date"]).set_index("date").sort_index()
        test_ret = eq.loc[(eq.index >= pd.Timestamp(test_start)) & (eq.index <= pd.Timestamp(test_end)), "returns"]
        m = compute_metrics(test_ret)
        rows.append(
            {
                "Method": method,
                "TrainStart": train_start,
                "TrainEnd": train_end,
                "TestStart": test_start,
                "TestEnd": test_end,
                "CAGR": float(m.get("CAGR", float("nan"))),
                "Vol": float(m.get("Vol", float("nan"))),
                "Sharpe": float(m.get("Sharpe", float("nan"))),
                "MaxDD": float(m.get("MaxDD", float("nan"))),
                "RunOutdir": str(run_outdir),
                "FullRunSharpe": float(summary.get("Sharpe", float("nan"))),
            }
        )
    df = pd.DataFrame(rows)
    sharpe = pd.to_numeric(df["Sharpe"], errors="coerce")
    agg = {
        "avg_oos_sharpe": float(sharpe.mean()),
        "median_oos_sharpe": float(sharpe.median()),
        "worst_window_sharpe": float(sharpe.min()),
        "best_window_sharpe": float(sharpe.max()),
    }
    df.to_csv(outdir / f"walkforward_{method}.csv", index=False, float_format="%.10g")
    return df, agg


def _run_delay_robustness(
    method: str,
    outdir: Path,
    run_cache: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for d in DELAYS:
        cfg = dict(BASE_CONFIG)
        cfg["factor_aggregation_method"] = method
        cfg["execution_delay_days"] = int(d)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        mean_sub, std_sub, min_sub, max_sub, stability = _stability_from_outdir(run_outdir)
        rows.append(
            {
                "Method": method,
                "ExecutionDelayDays": int(d),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "MeanSubSharpe": mean_sub,
                "StdSubSharpe": std_sub,
                "MinSubSharpe": min_sub,
                "MaxSubSharpe": max_sub,
                "StabilityScore": stability,
                "Outdir": str(run_outdir),
            }
        )
    df = pd.DataFrame(rows).sort_values("ExecutionDelayDays").reset_index(drop=True)
    df.to_csv(outdir / f"delay_robustness_{method}.csv", index=False, float_format="%.10g")
    return df


def _run_cost_robustness_geometric(
    outdir: Path,
    run_cache: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combos = [(c, s) for c in COSTS_BPS for s in SLIPPAGE_BPS]
    total = len(combos)
    for i, (c, s) in enumerate(combos, start=1):
        cfg = dict(BASE_CONFIG)
        cfg["factor_aggregation_method"] = "geometric_rank"
        cfg["costs_bps"] = float(c)
        cfg["slippage_bps"] = float(s)
        cfg["slippage_vol_mult"] = 0.0
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        mean_sub, std_sub, min_sub, max_sub, stability = _stability_from_outdir(run_outdir)
        rows.append(
            {
                "CostsBps": int(c),
                "SlippageBps": int(s),
                "BaseReference": bool(int(c) == 10 and int(s) == 0),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "MeanSubSharpe": mean_sub,
                "StdSubSharpe": std_sub,
                "MinSubSharpe": min_sub,
                "MaxSubSharpe": max_sub,
                "StabilityScore": stability,
                "Outdir": str(run_outdir),
            }
        )
        print(
            f"[cost {i}/{total}] c={c} s={s} Sharpe={rows[-1]['Sharpe']:.4f} "
            f"CAGR={rows[-1]['CAGR']:.4f} MaxDD={rows[-1]['MaxDD']:.4f}"
        )
    df = pd.DataFrame(rows).sort_values(["CostsBps", "SlippageBps"]).reset_index(drop=True)
    df.to_csv(outdir / "cost_robustness_geometric_full.csv", index=False, float_format="%.10g")
    return df


def _latest_linear_cost_table() -> pd.DataFrame | None:
    cands = sorted(Path("results").glob("cost_robustness_lead_sweep_*/cost_robustness_full.csv"))
    if not cands:
        return None
    return pd.read_csv(cands[-1])


def main() -> None:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"geometric_rank_robustness_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    run_cache: dict[str, Any] = {}

    # Full-period metrics for linear and geometric.
    full_rows: list[dict[str, Any]] = []
    for method in ["linear", "geometric_rank"]:
        cfg = dict(BASE_CONFIG)
        cfg["factor_aggregation_method"] = method
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        _, _, _, _, stability = _stability_from_outdir(run_outdir)
        full_rows.append(
            {
                "Method": method,
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "StabilityScore": float(stability),
                "Outdir": str(run_outdir),
            }
        )
    full_df = pd.DataFrame(full_rows)
    full_df.to_csv(outdir / "full_period_linear_vs_geometric.csv", index=False, float_format="%.10g")

    # Walk-forward OOS: run both for direct comparison.
    wf_geo_df, wf_geo_agg = _run_walkforward("geometric_rank", outdir=outdir, run_cache=run_cache)
    wf_lin_df, wf_lin_agg = _run_walkforward("linear", outdir=outdir, run_cache=run_cache)

    # Delay robustness: run both for direct comparison.
    delay_geo_df = _run_delay_robustness("geometric_rank", outdir=outdir, run_cache=run_cache)
    delay_lin_df = _run_delay_robustness("linear", outdir=outdir, run_cache=run_cache)

    # Cost robustness: full sweep for geometric, compare key points with latest linear table.
    cost_geo_df = _run_cost_robustness_geometric(outdir=outdir, run_cache=run_cache)
    cost_lin_df = _latest_linear_cost_table()

    key_points = pd.DataFrame(
        [
            {"label": "base_10_0", "CostsBps": 10, "SlippageBps": 0},
            {"label": "moderate_20_10", "CostsBps": 20, "SlippageBps": 10},
            {"label": "high_50_20", "CostsBps": 50, "SlippageBps": 20},
        ]
    )
    kp_geo = key_points.merge(cost_geo_df, on=["CostsBps", "SlippageBps"], how="left")
    kp_geo["Method"] = "geometric_rank"
    kp_lin = None
    if cost_lin_df is not None:
        kp_lin = key_points.merge(cost_lin_df, on=["CostsBps", "SlippageBps"], how="left")
        kp_lin["Method"] = "linear"
    kp = pd.concat([kp_geo, kp_lin], ignore_index=True) if kp_lin is not None else kp_geo
    kp.to_csv(outdir / "cost_robustness_key_points_comparison.csv", index=False, float_format="%.10g")

    summary_payload = {
        "config": BASE_CONFIG,
        "full_period_linear_vs_geometric": full_rows,
        "walkforward_geometric": wf_geo_agg,
        "walkforward_linear": wf_lin_agg,
        "delay_geometric": delay_geo_df.to_dict(orient="records"),
        "delay_linear": delay_lin_df.to_dict(orient="records"),
        "cost_geometric_key_points": kp_geo.to_dict(orient="records"),
        "cost_linear_key_points": kp_lin.to_dict(orient="records") if kp_lin is not None else None,
        "artifacts": {
            "walkforward_geometric": str(outdir / "walkforward_geometric_rank.csv"),
            "walkforward_linear": str(outdir / "walkforward_linear.csv"),
            "delay_geometric": str(outdir / "delay_robustness_geometric_rank.csv"),
            "delay_linear": str(outdir / "delay_robustness_linear.csv"),
            "cost_geometric_full": str(outdir / "cost_robustness_geometric_full.csv"),
            "cost_key_comparison": str(outdir / "cost_robustness_key_points_comparison.csv"),
            "full_period": str(outdir / "full_period_linear_vs_geometric.csv"),
        },
    }
    (outdir / "robustness_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("GEOMETRIC RANK ROBUSTNESS SUMMARY")
    print("---------------------------------")
    print("\nFull-period linear vs geometric:")
    print(full_df.to_string(index=False))
    print("\nWalkforward OOS aggregate:")
    print(pd.DataFrame([{"Method": "linear", **wf_lin_agg}, {"Method": "geometric_rank", **wf_geo_agg}]).to_string(index=False))
    print("\nDelay robustness (geometric):")
    print(delay_geo_df[["ExecutionDelayDays", "CAGR", "Vol", "Sharpe", "MaxDD", "StabilityScore"]].to_string(index=False))
    print("\nCost robustness key points:")
    print(kp[["Method", "label", "CostsBps", "SlippageBps", "CAGR", "Vol", "Sharpe", "MaxDD", "StabilityScore"]].to_string(index=False))
    print(f"\nSaved results directory: {outdir}")


if __name__ == "__main__":
    main()
