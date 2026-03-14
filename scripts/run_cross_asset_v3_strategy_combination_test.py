"""Diversification test: Benchmark sleeve vs Cross-Asset Trend v3 best variant."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_daily_mark_to_market, compute_metrics
from quant_lab.strategies.topn import rebalance_mask


OUTDIR = Path("results/cross_asset_v3_strategy_combination_test")
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
START = "2005-01-01"
END = "2024-12-31"
SIGNAL_SKIP_DAYS = 21
SIGNAL_LAG_DAYS = 1


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
        if str(d.get("Start", "")) != START or str(d.get("End", "")) != END:
            continue
        p = s.parent / "equity_curve.csv"
        if p.exists():
            candidates.append((s.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError("Could not locate Benchmark v1.1 geometric_rank equity_curve.csv")
    candidates.sort()
    return candidates[-1][1]


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
    close = close.loc[(close.index >= pd.Timestamp(START)) & (close.index <= pd.Timestamp(END))]
    return close.astype(float)


def _inverse_vol_weights(vol_row: pd.Series, selected: list[str], budget: float) -> pd.Series:
    out = pd.Series(0.0, index=vol_row.index, dtype=float)
    if not selected or float(budget) <= 0.0:
        return out
    vol_sel = pd.to_numeric(vol_row.reindex(selected), errors="coerce")
    valid = vol_sel[(vol_sel > 0.0) & vol_sel.notna()]
    if valid.empty:
        out.loc[selected] = float(budget) / float(len(selected))
        return out
    inv = 1.0 / valid
    inv = inv / float(inv.sum())
    out.loc[inv.index] = inv.to_numpy(dtype=float) * float(budget)
    return out


def _raw_signal(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return close.shift(SIGNAL_SKIP_DAYS) / close.shift(int(lookback)) - 1.0


def _score_signal(raw_signal: pd.DataFrame, signal_type: str) -> pd.DataFrame:
    mode = str(signal_type).lower().strip()
    if mode == "relative_only":
        return raw_signal.astype(float)
    if mode == "dual":
        if "SHY" not in raw_signal.columns:
            raise ValueError("SHY required for dual signal.")
        return raw_signal.sub(raw_signal["SHY"], axis=0).astype(float)
    raise ValueError(f"unknown signal_type: {signal_type}")


def _build_v3_weights(
    close: pd.DataFrame,
    lookback: int,
    signal_type: str,
    top_n: int,
    weighting: str,
    rebalance: str,
    absolute_momentum_filter: bool,
) -> pd.DataFrame:
    raw = _raw_signal(close=close, lookback=int(lookback)).astype(float)
    score = _score_signal(raw_signal=raw, signal_type=signal_type).astype(float)
    vol20 = close.pct_change().rolling(20).std(ddof=0).shift(1)
    raw_lag = raw.shift(SIGNAL_LAG_DAYS)
    score_lag = score.shift(SIGNAL_LAG_DAYS)

    rb = rebalance_mask(pd.DatetimeIndex(close.index), str(rebalance))
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    current = pd.Series(0.0, index=close.columns, dtype=float)
    for dt in close.index:
        if bool(rb.loc[dt]):
            ranked = score_lag.loc[dt].dropna().sort_values(ascending=False)
            selected = ranked.index.tolist()[: int(top_n)]
            if bool(absolute_momentum_filter):
                abs_row = raw_lag.loc[dt]
                selected = [a for a in selected if pd.notna(abs_row.get(a)) and float(abs_row.get(a)) > 0.0]
            missing_slots = max(0, int(top_n) - len(selected))
            shy_floor = float(missing_slots) / float(top_n)

            nxt = pd.Series(0.0, index=close.columns, dtype=float)
            if str(weighting).lower() == "equal":
                if selected:
                    nxt.loc[selected] += 1.0 / float(top_n)
                if missing_slots > 0:
                    nxt.loc["SHY"] += shy_floor
            elif str(weighting).lower() == "inverse_vol":
                remaining = 1.0 - shy_floor
                nxt = _inverse_vol_weights(vol_row=vol20.loc[dt], selected=selected, budget=remaining)
                if missing_slots > 0:
                    nxt.loc["SHY"] += shy_floor
            else:
                raise ValueError(f"unknown weighting: {weighting}")
            current = nxt
        w.loc[dt] = current
    return w.astype(float)


def _apply_vol_target(daily_ret: pd.Series, vol_target: float | None) -> tuple[pd.Series, pd.Series]:
    r = daily_ret.astype(float)
    if vol_target is None:
        lev = pd.Series(1.0, index=r.index, dtype=float)
        return r, lev
    rv63 = r.rolling(63).std(ddof=0) * (252.0**0.5)
    lev_raw = float(vol_target) / rv63.replace(0.0, pd.NA)
    lev = pd.to_numeric(lev_raw, errors="coerce").clip(lower=0.25, upper=2.0).shift(1).fillna(1.0)
    return (r * lev).astype(float), lev.astype(float)


def _find_latest_v3_best_variant() -> dict[str, Any]:
    latest_grid = Path("results/cross_asset_trend_v3_grid_latest.csv")
    if not latest_grid.exists():
        raise FileNotFoundError("Missing results/cross_asset_trend_v3_grid_latest.csv")
    grid = pd.read_csv(latest_grid)
    if grid.empty:
        raise ValueError("v3 latest grid CSV is empty.")
    row = grid.sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").iloc[0]
    out: dict[str, Any] = {}
    for k, v in dict(row).items():
        if hasattr(v, "item"):
            try:
                out[str(k)] = v.item()
                continue
            except Exception:
                pass
        out[str(k)] = v
    return out


def _drawdown(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.astype(float).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    benchmark_path = _find_benchmark_v11_equity_curve()
    benchmark = (
        pd.read_csv(benchmark_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "benchmark_return"})
        .set_index("date")
        .sort_index()["benchmark_return"]
    )

    best_variant = _find_latest_v3_best_variant()
    close = _load_close_panel(root="data/cross_asset")
    vol_target_val = None if pd.isna(best_variant.get("vol_target")) else float(best_variant["vol_target"])
    weights = _build_v3_weights(
        close=close,
        lookback=int(best_variant["lookback"]),
        signal_type=str(best_variant["signal_type"]),
        top_n=int(best_variant["top_n"]),
        weighting=str(best_variant["weighting"]),
        rebalance=str(best_variant["rebalance"]),
        absolute_momentum_filter=bool(best_variant["absolute_momentum_filter"]),
    )
    rb = rebalance_mask(pd.DatetimeIndex(weights.index), str(best_variant["rebalance"]))
    rb_dates = pd.DatetimeIndex(weights.index[rb])
    _, daily_ret, _ = compute_daily_mark_to_market(
        close=close,
        weights_rebal=weights,
        rebalance_dates=rb_dates,
        costs_bps=0.0,
        slippage_bps=0.0,
    )
    v3_ret, lev = _apply_vol_target(daily_ret, vol_target=vol_target_val)
    v3_ret = v3_ret.rename("cross_asset_v3_return")

    aligned = pd.concat([benchmark, v3_ret], axis=1, join="inner").dropna()
    corr = float(aligned["benchmark_return"].corr(aligned["cross_asset_v3_return"]))
    rolling_corr = aligned["benchmark_return"].rolling(252).corr(aligned["cross_asset_v3_return"])

    dd_bench = _drawdown(aligned["benchmark_return"])
    dd_v3 = _drawdown(aligned["cross_asset_v3_return"])
    in_dd_bench = dd_bench < 0.0
    in_dd_v3 = dd_v3 < 0.0
    both = in_dd_bench & in_dd_v3
    either = in_dd_bench | in_dd_v3
    dd_overlap_all = float(both.mean())
    dd_overlap_cond = float(both.sum() / either.sum()) if either.any() else float("nan")

    blends = {
        "100% Benchmark": (1.0, 0.0),
        "100% CrossAssetV3": (0.0, 1.0),
        "80/20": (0.8, 0.2),
        "70/30": (0.7, 0.3),
        "60/40": (0.6, 0.4),
    }
    combined = aligned.copy()
    rows: list[dict[str, float | str]] = []
    for name, (wb, wc) in blends.items():
        col = f"portfolio_{name.replace('%', 'pct').replace('/', '_').replace(' ', '').lower()}"
        combined[col] = wb * combined["benchmark_return"] + wc * combined["cross_asset_v3_return"]
        m = compute_metrics(combined[col])
        rows.append(
            {
                "Portfolio": name,
                "WeightBenchmark": wb,
                "WeightCrossAssetV3": wc,
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
                "cross_asset_v3_variant": json.dumps(best_variant, sort_keys=True),
                "cross_asset_v3_avg_leverage": float(lev.mean()),
            }
        ]
    )
    portfolio_df = pd.DataFrame(rows)

    out_combined = combined.assign(
        drawdown_benchmark=dd_bench,
        drawdown_cross_asset_v3=dd_v3,
        rolling_corr_252=rolling_corr,
        in_drawdown_benchmark=in_dd_bench.astype(int),
        in_drawdown_cross_asset_v3=in_dd_v3.astype(int),
        in_drawdown_both=both.astype(int),
    ).reset_index()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = OUTDIR / ts
    outdir.mkdir(parents=True, exist_ok=True)

    combined_path = outdir / "combined_returns.csv"
    corr_path = outdir / "strategy_correlation.csv"
    portfolio_path = outdir / "portfolio_summary.csv"

    out_combined.to_csv(combined_path, index=False, float_format="%.10g")
    corr_df.to_csv(corr_path, index=False, float_format="%.10g")
    portfolio_df.to_csv(portfolio_path, index=False, float_format="%.10g")

    out_combined.to_csv(OUTDIR / "combined_returns.csv", index=False, float_format="%.10g")
    corr_df.to_csv(OUTDIR / "strategy_correlation.csv", index=False, float_format="%.10g")
    portfolio_df.to_csv(OUTDIR / "portfolio_summary.csv", index=False, float_format="%.10g")

    print("CROSS-ASSET V3 STRATEGY COMBINATION TEST")
    print("----------------------------------------")
    print(corr_df.to_string(index=False))
    print("")
    print(portfolio_df.to_string(index=False))
    print("")
    print(f"Saved: {combined_path}")
    print(f"Saved: {corr_path}")
    print(f"Saved: {portfolio_path}")


if __name__ == "__main__":
    main()
