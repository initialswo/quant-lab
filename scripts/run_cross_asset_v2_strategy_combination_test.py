"""Diversification analysis: Benchmark v1.1 vs Cross-Asset Trend v2 best variant."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market, compute_metrics
from quant_lab.engine.runner import run_backtest
from quant_lab.strategies.topn import rebalance_mask


OUTDIR = Path("results/cross_asset_v2_strategy_combination_test")
VOL_TARGET_OUTDIR = Path("results/portfolio_vol_target_test")
ASSETS = [
    "SPY",
    "TLT",
    "GLD",
    "DBC",
    "VNQ",
    "SHY",
    "EFA",
    "EEM",
    "IEF",
    "LQD",
    "TIP",
    "UUP",
    "FXE",
]

BENCHMARK_V11: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "universe": "sp500",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
    "factor_weights": None,
    "dynamic_factor_weights": True,
    "factor_aggregation_method": "geometric_rank",
    "top_n": 50,
    "rank_buffer": 20,
    "rebalance": "monthly",
    "execution_delay_days": 0,
    "target_vol": 0.14,
    "port_vol_lookback": 20,
    "max_leverage": 1.5,
    "regime_filter": False,
    "regime_benchmark": "SPY",
    "bear_exposure_scale": 1.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 5_000_000.0,
    "liquidity_lookback": 20,
    "costs_bps": 10.0,
    "save_artifacts": True,
}

CROSS_ASSET_V2: dict[str, Any] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
    "lookback": 126,
    "signal_type": "dual",
    "top_n": 4,
    "weighting": "inverse_vol",
    "rebalance": "monthly",
    "signal_lag_days": 1,
    "skip_days": 21,
}


def _load_close_panel(root: str = "data/cross_asset") -> pd.DataFrame:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"cross-asset directory missing: {root}")
    panel: dict[str, pd.Series] = {}
    for sym in ASSETS:
        p = base / f"{sym}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing cross-asset parquet: {p}")
        df = pd.read_parquet(p, columns=["date", "close"]).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        panel[sym] = pd.Series(df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=sym)
    close = pd.concat(panel.values(), axis=1, join="outer").sort_index()
    close.columns = list(panel.keys())
    close = close.loc[
        (close.index >= pd.Timestamp(CROSS_ASSET_V2["start"])) & (close.index <= pd.Timestamp(CROSS_ASSET_V2["end"]))
    ]
    return close.astype(float)


def _inverse_vol_weights(vol_row: pd.Series, selected: list[str]) -> pd.Series:
    out = pd.Series(0.0, index=vol_row.index, dtype=float)
    if not selected:
        return out
    vol_sel = pd.to_numeric(vol_row.reindex(selected), errors="coerce")
    valid = vol_sel[(vol_sel > 0.0) & vol_sel.notna()]
    if valid.empty:
        out.loc[selected] = 1.0 / float(len(selected))
        return out
    inv = 1.0 / valid
    inv = inv / float(inv.sum())
    out.loc[inv.index] = inv.to_numpy(dtype=float)
    return out


def _compute_cross_asset_v2_returns() -> tuple[pd.Series, pd.DataFrame]:
    close = _load_close_panel()
    lookback = int(CROSS_ASSET_V2["lookback"])
    skip_days = int(CROSS_ASSET_V2["skip_days"])
    signal_lag = int(CROSS_ASSET_V2["signal_lag_days"])
    top_n = int(CROSS_ASSET_V2["top_n"])
    rebalance = str(CROSS_ASSET_V2["rebalance"])

    signal = close.shift(skip_days) / close.shift(lookback) - 1.0
    signal_lagged = signal.shift(signal_lag)
    vol20 = close.pct_change().rolling(20).std(ddof=0).shift(1)

    weights_rebal = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    rb = rebalance_mask(pd.DatetimeIndex(close.index), rebalance)
    current = pd.Series(0.0, index=close.columns, dtype=float)
    for dt in close.index:
        if bool(rb.loc[dt]):
            row = signal_lagged.loc[dt].dropna().sort_values(ascending=False)
            selected = row.index.tolist()[:top_n]
            current = _inverse_vol_weights(vol_row=vol20.loc[dt], selected=selected)
        weights_rebal.loc[dt] = current

    rb_dates = pd.DatetimeIndex(weights_rebal.index[rb])
    _, daily_ret, weights_daily = compute_daily_mark_to_market(
        close=close,
        weights_rebal=weights_rebal,
        rebalance_dates=rb_dates,
        costs_bps=0.0,
        slippage_bps=0.0,
    )
    return daily_ret.rename("cross_asset_v2_return"), weights_daily


def _drawdown(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.astype(float).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    VOL_TARGET_OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Running Benchmark v1.1...")
    _, bench_outdir = run_backtest(**BENCHMARK_V11)
    bench_path = Path(bench_outdir) / "equity_curve.csv"
    if not bench_path.exists():
        raise FileNotFoundError(f"Missing benchmark equity curve: {bench_path}")
    bench_ret = (
        pd.read_csv(bench_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "benchmark_return"})
        .set_index("date")
        .sort_index()["benchmark_return"]
    )

    print("Computing Cross-Asset Trend v2 returns...")
    cross_ret, cross_weights = _compute_cross_asset_v2_returns()

    aligned = pd.concat([bench_ret, cross_ret], axis=1, join="inner").dropna()
    corr = float(aligned["benchmark_return"].corr(aligned["cross_asset_v2_return"]))
    rolling_corr = aligned["benchmark_return"].rolling(252).corr(aligned["cross_asset_v2_return"])

    dd_bench = _drawdown(aligned["benchmark_return"])
    dd_cross = _drawdown(aligned["cross_asset_v2_return"])
    in_dd_bench = dd_bench < 0.0
    in_dd_cross = dd_cross < 0.0
    both = in_dd_bench & in_dd_cross
    either = in_dd_bench | in_dd_cross
    dd_overlap_all = float(both.mean())
    dd_overlap_cond = float(both.sum() / either.sum()) if either.any() else float("nan")

    blends = {
        "100% Benchmark": (1.0, 0.0),
        "90/10": (0.9, 0.1),
        "80/20": (0.8, 0.2),
        "70/30": (0.7, 0.3),
        "60/40": (0.6, 0.4),
        "100% CrossAsset": (0.0, 1.0),
    }
    combined = aligned.copy()
    rows: list[dict[str, float | str]] = []
    for name, (wb, wc) in blends.items():
        col = f"portfolio_{name.replace('%', 'pct').replace('/', '_').replace(' ', '').lower()}"
        combined[col] = wb * combined["benchmark_return"] + wc * combined["cross_asset_v2_return"]
        m = compute_metrics(combined[col])
        rows.append(
            {
                "Portfolio": name,
                "WeightBenchmark": wb,
                "WeightCrossAssetV2": wc,
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
                "benchmark_returns_path": str(bench_path),
                "cross_asset_v2_params": json.dumps(CROSS_ASSET_V2, sort_keys=True),
            }
        ]
    )
    portfolio_df = pd.DataFrame(rows)

    rolling_corr_plot = OUTDIR / "rolling_correlation.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_corr.index, rolling_corr.values, label="Rolling 252D Correlation", color="tab:blue", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Rolling 252-Day Correlation: Benchmark vs Cross-Asset Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(rolling_corr_plot, dpi=150)
    plt.close(fig)

    cumulative_plot = OUTDIR / "cumulative_returns_comparison.png"
    benchmark_equity = (1.0 + aligned["benchmark_return"].astype(float).fillna(0.0)).cumprod()
    cross_equity = (1.0 + aligned["cross_asset_v2_return"].astype(float).fillna(0.0)).cumprod()
    blend_70_30 = combined["portfolio_70_30"].astype(float).fillna(0.0)
    blend_equity = (1.0 + blend_70_30).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(benchmark_equity.index, benchmark_equity.values, label="Benchmark", linewidth=1.7)
    ax.plot(cross_equity.index, cross_equity.values, label="CrossAsset", linewidth=1.7)
    ax.plot(blend_equity.index, blend_equity.values, label="70/30", linewidth=1.7)
    ax.set_title("Cumulative Returns: Benchmark vs Cross-Asset vs 70/30")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(cumulative_plot, dpi=150)
    plt.close(fig)

    target_vol = 0.12
    static_70_30 = combined["portfolio_70_30"].astype(float)
    realized_vol_63 = static_70_30.rolling(63).std() * math.sqrt(252.0)
    leverage_raw = target_vol / realized_vol_63.replace(0.0, pd.NA)
    leverage_capped = pd.to_numeric(leverage_raw, errors="coerce").clip(lower=0.25, upper=2.0)
    leverage_lag1 = leverage_capped.shift(1).fillna(1.0)
    vol_target_70_30 = static_70_30 * leverage_lag1

    static_metrics = compute_metrics(static_70_30)
    vol_target_metrics = compute_metrics(vol_target_70_30)
    vol_target_summary_df = pd.DataFrame(
        [
            {
                "Portfolio": "Static 70/30",
                "TargetVol": "",
                "CAGR": float(static_metrics["CAGR"]),
                "Vol": float(static_metrics["Vol"]),
                "Sharpe": float(static_metrics["Sharpe"]),
                "MaxDD": float(static_metrics["MaxDD"]),
            },
            {
                "Portfolio": "VolTargeted 70/30",
                "TargetVol": target_vol,
                "CAGR": float(vol_target_metrics["CAGR"]),
                "Vol": float(vol_target_metrics["Vol"]),
                "Sharpe": float(vol_target_metrics["Sharpe"]),
                "MaxDD": float(vol_target_metrics["MaxDD"]),
            },
        ]
    )

    vol_target_returns = pd.DataFrame(
        {
            "date": combined.index,
            "static_70_30_return": static_70_30.to_numpy(dtype=float),
            "realized_vol_63": realized_vol_63.to_numpy(dtype=float),
            "leverage_raw": pd.to_numeric(leverage_raw, errors="coerce").to_numpy(dtype=float),
            "leverage_capped": leverage_capped.to_numpy(dtype=float),
            "leverage_lag1": leverage_lag1.to_numpy(dtype=float),
            "vol_target_70_30_return": vol_target_70_30.to_numpy(dtype=float),
            "static_70_30_equity": (1.0 + static_70_30.fillna(0.0)).cumprod().to_numpy(dtype=float),
            "vol_target_70_30_equity": (1.0 + vol_target_70_30.fillna(0.0)).cumprod().to_numpy(dtype=float),
        }
    )

    vol_target_returns.to_csv(VOL_TARGET_OUTDIR / "vol_target_returns.csv", index=False, float_format="%.10g")
    vol_target_summary_df.to_csv(VOL_TARGET_OUTDIR / "portfolio_summary.csv", index=False, float_format="%.10g")
    vol_target_payload = {
        "target_vol": target_vol,
        "realized_vol_window_days": 63,
        "leverage_cap_min": 0.25,
        "leverage_cap_max": 2.0,
        "lag_rule": "scaled_returns = static_returns * leverage.shift(1)",
        "metrics": vol_target_summary_df.to_dict(orient="records"),
        "artifacts": {
            "returns": str(VOL_TARGET_OUTDIR / "vol_target_returns.csv"),
            "summary": str(VOL_TARGET_OUTDIR / "portfolio_summary.csv"),
            "json_summary": str(VOL_TARGET_OUTDIR / "vol_target_summary.json"),
            "plot": str(VOL_TARGET_OUTDIR / "cumulative_returns_comparison.png"),
        },
    }
    (VOL_TARGET_OUTDIR / "vol_target_summary.json").write_text(json.dumps(vol_target_payload, indent=2), encoding="utf-8")

    vt_plot = VOL_TARGET_OUTDIR / "cumulative_returns_comparison.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        vol_target_returns["date"],
        vol_target_returns["static_70_30_equity"],
        label="Static 70/30",
        linewidth=1.8,
    )
    ax.plot(
        vol_target_returns["date"],
        vol_target_returns["vol_target_70_30_equity"],
        label="Vol-targeted 70/30",
        linewidth=1.8,
    )
    ax.set_title("Cumulative Returns: Static vs Vol-Targeted 70/30")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(vt_plot, dpi=150)
    plt.close(fig)

    combined.assign(
        drawdown_benchmark=dd_bench,
        drawdown_cross_asset_v2=dd_cross,
        rolling_corr_252=rolling_corr,
        in_drawdown_benchmark=in_dd_bench.astype(int),
        in_drawdown_cross_asset_v2=in_dd_cross.astype(int),
        in_drawdown_both=both.astype(int),
    ).reset_index().to_csv(OUTDIR / "combined_returns.csv", index=False, float_format="%.10g")
    corr_df.to_csv(OUTDIR / "strategy_correlation.csv", index=False, float_format="%.10g")
    portfolio_df.to_csv(OUTDIR / "portfolio_summary.csv", index=False, float_format="%.10g")

    payload = {
        "benchmark_config": BENCHMARK_V11,
        "cross_asset_v2_config": CROSS_ASSET_V2,
        "correlation": corr_df.to_dict(orient="records")[0],
        "portfolio_summary": rows,
        "cross_asset_weights_shape": list(cross_weights.shape),
    }
    (OUTDIR / "combination_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("CROSS-ASSET V2 STRATEGY COMBINATION TEST")
    print("----------------------------------------")
    print(corr_df.to_string(index=False))
    print("")
    print(portfolio_df.to_string(index=False))
    print("")
    print(f"Saved: {OUTDIR / 'combined_returns.csv'}")
    print(f"Saved: {OUTDIR / 'strategy_correlation.csv'}")
    print(f"Saved: {OUTDIR / 'portfolio_summary.csv'}")
    print(f"Saved: {OUTDIR / 'combination_summary.json'}")
    print(f"Saved: {rolling_corr_plot}")
    print(f"Saved: {cumulative_plot}")
    print("")
    print("PORTFOLIO VOL TARGET TEST")
    print("-------------------------")
    print(vol_target_summary_df.to_string(index=False))
    print(f"Saved: {VOL_TARGET_OUTDIR / 'vol_target_returns.csv'}")
    print(f"Saved: {VOL_TARGET_OUTDIR / 'portfolio_summary.csv'}")
    print(f"Saved: {VOL_TARGET_OUTDIR / 'vol_target_summary.json'}")
    print(f"Saved: {vt_plot}")


if __name__ == "__main__":
    main()
