"""Research-only cross-asset trend benchmark (independent Strategy 2 track)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market, compute_metrics
from quant_lab.research.cross_asset_trend import (
    PREFERRED_ASSET_CANDIDATES,
    annual_turnover,
    build_cross_asset_trend_weights,
    compute_trend_signal_12_1,
)
from quant_lab.strategies.topn import rebalance_mask


def _load_cross_asset_close_panel(root: str = "data/cross_asset") -> pd.DataFrame:
    r = Path(root)
    if not r.exists():
        raise FileNotFoundError(f"cross-asset directory missing: {root}")
    close_map: dict[str, pd.Series] = {}
    for sym in PREFERRED_ASSET_CANDIDATES:
        p = r / f"{sym}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing cross-asset parquet: {p}")
        df = pd.read_parquet(p, columns=["date", "close"]).copy()
        if df.empty:
            raise ValueError(f"empty cross-asset dataset: {p}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        if df.empty:
            raise ValueError(f"no valid rows after parsing: {p}")
        close_map[sym] = pd.Series(df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=sym)
    close = pd.concat(close_map.values(), axis=1, join="outer").sort_index()
    close.columns = list(close_map.keys())
    return close.astype(float)


def main() -> None:
    start = "2005-01-01"
    end = "2024-12-31"
    rebalance = "monthly"
    skip_days = 21
    lookback_days = 252
    lag_days = 1

    close = _load_cross_asset_close_panel(root="data/cross_asset")
    close = close.loc[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
    if close.empty:
        raise ValueError("No usable close panel for selected assets.")

    signal = compute_trend_signal_12_1(close=close, skip_days=skip_days, lookback_days=lookback_days)
    weights = build_cross_asset_trend_weights(
        trend_signal=signal,
        rebalance=rebalance,
        lag_days=lag_days,
    )

    rb = rebalance_mask(pd.DatetimeIndex(weights.index), rebalance)
    rb_dates = pd.DatetimeIndex(weights.index[rb])
    equity, daily_ret, weights_daily = compute_daily_mark_to_market(
        close=close,
        weights_rebal=weights,
        rebalance_dates=rb_dates,
        costs_bps=0.0,
        slippage_bps=0.0,
    )
    m = compute_metrics(daily_ret)
    ann_turn = annual_turnover(weights_daily, rebalance=rebalance)
    avg_held = float((weights_daily > 0.0).sum(axis=1).mean())

    out_returns = pd.DataFrame(
        {
            "date": equity.index,
            "equity": equity.to_numpy(dtype=float),
            "returns": daily_ret.to_numpy(dtype=float),
            "assets_held": (weights_daily > 0.0).sum(axis=1).to_numpy(dtype=int),
        }
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / "cross_asset_trend_benchmark" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    returns_path = outdir / "cross_asset_trend_returns.csv"
    weights_path = outdir / "cross_asset_trend_weights.csv"
    signals_path = outdir / "cross_asset_trend_signals.csv"
    summary_path = outdir / "cross_asset_trend_summary.json"

    out_returns.to_csv(returns_path, index=False, float_format="%.10g")
    weights_daily.reset_index().rename(columns={"index": "date"}).to_csv(
        weights_path, index=False, float_format="%.10g"
    )
    signal.reset_index().rename(columns={"index": "date"}).to_csv(
        signals_path, index=False, float_format="%.10g"
    )

    summary = {
        "start": start,
        "end": end,
        "rebalance": rebalance,
        "preferred_assets": PREFERRED_ASSET_CANDIDATES,
        "available_asset_mapping": {sym: sym for sym in PREFERRED_ASSET_CANDIDATES},
        "assets_used": list(close.columns),
        "signal_formula": f"close[t-{skip_days}] / close[t-{lookback_days}] - 1",
        "signal_lag_days": int(lag_days),
        "selection_rule": "hold assets with lagged trend_signal > 0; equal weight among selected",
        "fallback_rule": "zero exposure when no asset has positive trend",
        "cagr": float(m.get("CAGR", float("nan"))),
        "vol": float(m.get("Vol", float("nan"))),
        "sharpe": float(m.get("Sharpe", float("nan"))),
        "maxdd": float(m.get("MaxDD", float("nan"))),
        "annual_turnover": float(ann_turn),
        "avg_assets_held": float(avg_held),
        "artifacts": {
            "returns": str(returns_path),
            "weights": str(weights_path),
            "signals": str(signals_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_root = Path("results") / "cross_asset_trend_benchmark"
    out_returns.to_csv(latest_root / "cross_asset_trend_returns.csv", index=False, float_format="%.10g")
    weights_daily.reset_index().rename(columns={"index": "date"}).to_csv(
        latest_root / "cross_asset_trend_weights.csv", index=False, float_format="%.10g"
    )
    signal.reset_index().rename(columns={"index": "date"}).to_csv(
        latest_root / "cross_asset_trend_signals.csv", index=False, float_format="%.10g"
    )
    (latest_root / "cross_asset_trend_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("CROSS-ASSET TREND BENCHMARK")
    print("---------------------------")
    print(f"Assets used: {', '.join(close.columns)}")
    print(f"CAGR: {summary['cagr']:.6f}")
    print(f"Vol: {summary['vol']:.6f}")
    print(f"Sharpe: {summary['sharpe']:.6f}")
    print(f"MaxDD: {summary['maxdd']:.6f}")
    print(f"AnnualTurnover: {summary['annual_turnover']:.6f}")
    print(f"AvgAssetsHeld: {summary['avg_assets_held']:.6f}")
    print(f"Saved: {returns_path}")
    print(f"Saved: {weights_path}")
    print(f"Saved: {signals_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
