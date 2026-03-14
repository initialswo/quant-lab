"""Universe expansion comparison for the current lead strategy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest
from quant_lab.strategies.topn import rebalance_mask


START = "2005-01-01"
END = "2024-12-31"
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

RESULTS_ROOT = Path("results") / "universe_expansion_test"
INPUTS_ROOT = RESULTS_ROOT / "_inputs"

BASE_CONFIG: dict[str, Any] = {
    "start": START,
    "end": END,
    "max_tickers": 4000,
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


def _load_price_panels(start: str, end: str, data_cache_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_path = Path(data_cache_dir) / "daily_ohlcv.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing price parquet: {daily_path}")
    start_ts = pd.Timestamp(start) - pd.Timedelta(days=120)
    end_ts = pd.Timestamp(end)
    frame = pd.read_parquet(daily_path, columns=["date", "ticker", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].copy()
    frame["ticker"] = (
        frame["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    close = frame.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    volume = frame.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").sort_index()
    close = close.astype(float)
    volume = volume.astype(float)
    close = close.loc[(close.index >= start_ts) & (close.index <= end_ts)]
    volume = volume.reindex(index=close.index, columns=close.columns)
    return close, volume


def _build_liquid_topk_membership_csv(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    start: str,
    end: str,
    top_k: int,
    min_price: float,
    min_avg_dollar_volume: float,
    liquidity_lookback: int,
    rebalance: str,
    out_path: Path,
) -> str:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    rb = rebalance_mask(pd.DatetimeIndex(close.index), rebalance)
    rb_dates = pd.DatetimeIndex(close.index[rb])
    rb_dates = rb_dates[(rb_dates >= pd.Timestamp(start)) & (rb_dates <= pd.Timestamp(end))]
    if rb_dates.empty:
        raise ValueError("No rebalance dates available to build proxy membership.")

    dollar_vol = close * volume
    adv = dollar_vol.rolling(int(liquidity_lookback), min_periods=int(liquidity_lookback)).mean().shift(1)
    elig_price = close.ge(float(min_price))
    elig_adv = adv.ge(float(min_avg_dollar_volume))
    elig = (elig_price & elig_adv).fillna(False)

    membership = pd.DataFrame(False, index=rb_dates, columns=close.columns, dtype=bool)
    cols = list(close.columns)
    for dt in rb_dates:
        adv_row = adv.loc[dt]
        ok = elig.loc[dt]
        candidates = adv_row[ok & adv_row.notna()]
        if candidates.empty:
            continue
        candidates = candidates.sort_values(ascending=False, kind="mergesort")
        top = set(candidates.head(int(top_k)).index.astype(str))
        row_values = [c in top for c in cols]
        membership.loc[dt, cols] = row_values

    wide = membership.copy()
    wide.insert(0, "date", wide.index.strftime("%Y-%m-%d"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    return str(out_path)


def _stable_score_from_equity_curve(outdir: str) -> tuple[float, float, float, float, float]:
    eq = pd.read_csv(Path(outdir) / "equity_curve.csv", parse_dates=["date"]).set_index("date").sort_index()
    rets = pd.to_numeric(eq["returns"], errors="coerce").fillna(0.0)
    sharpes: list[float] = []
    for s, e in SUBPERIODS:
        rr = rets.loc[(rets.index >= pd.Timestamp(s)) & (rets.index <= pd.Timestamp(e))]
        m = compute_metrics(rr)
        sharpes.append(float(m.get("Sharpe", float("nan"))))
    ser = pd.Series(sharpes, dtype=float)
    mean_sub = float(ser.mean())
    std_sub = float(ser.std(ddof=0))
    min_sub = float(ser.min())
    max_sub = float(ser.max())
    stability = float(mean_sub - 0.5 * std_sub)
    return mean_sub, std_sub, min_sub, max_sub, stability


def _average_universe_size(outdir: str) -> float:
    p = Path(outdir) / "universe_eligibility_summary.csv"
    if not p.exists():
        return float("nan")
    df = pd.read_csv(p)
    if "eligible_count" not in df.columns or df.empty:
        return float("nan")
    return float(pd.to_numeric(df["eligible_count"], errors="coerce").dropna().mean())


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    INPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    close, volume = _load_price_panels(start=START, end=END, data_cache_dir=str(BASE_CONFIG["data_cache_dir"]))

    sp1000_path = Path("data/universe/sp1000_historical_membership.csv")
    sp1500_path = Path("data/universe/sp1500_historical_membership.csv")

    sp1000_source = "exact"
    sp1500_source = "exact"

    if not sp1000_path.exists():
        sp1000_path = INPUTS_ROOT / "sp1000_liquidity_proxy_membership.csv"
        _build_liquid_topk_membership_csv(
            close=close,
            volume=volume,
            start=START,
            end=END,
            top_k=1000,
            min_price=float(BASE_CONFIG["min_price"]),
            min_avg_dollar_volume=float(BASE_CONFIG["min_avg_dollar_volume"]),
            liquidity_lookback=int(BASE_CONFIG["liquidity_lookback"]),
            rebalance=str(BASE_CONFIG["rebalance"]),
            out_path=sp1000_path,
        )
        sp1000_source = "liquidity_proxy"

    if not sp1500_path.exists():
        sp1500_path = INPUTS_ROOT / "sp1500_liquidity_proxy_membership.csv"
        _build_liquid_topk_membership_csv(
            close=close,
            volume=volume,
            start=START,
            end=END,
            top_k=1500,
            min_price=float(BASE_CONFIG["min_price"]),
            min_avg_dollar_volume=float(BASE_CONFIG["min_avg_dollar_volume"]),
            liquidity_lookback=int(BASE_CONFIG["liquidity_lookback"]),
            rebalance=str(BASE_CONFIG["rebalance"]),
            out_path=sp1500_path,
        )
        sp1500_source = "liquidity_proxy"

    variants = [
        {
            "UniverseVariant": "SP500",
            "universe": "sp500",
            "historical_membership_path": "data/universe/sp500_historical_membership.csv",
            "UniverseSource": "historical_membership",
        },
        {
            "UniverseVariant": "SP1000",
            "universe": "all",
            "historical_membership_path": str(sp1000_path),
            "UniverseSource": sp1000_source,
        },
        {
            "UniverseVariant": "SP1500",
            "universe": "all",
            "historical_membership_path": str(sp1500_path),
            "UniverseSource": sp1500_source,
        },
        {
            "UniverseVariant": "AllLiquidUS",
            "universe": "all",
            "historical_membership_path": "",
            "UniverseSource": "all_us_no_membership",
        },
    ]

    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for i, variant in enumerate(variants, start=1):
        cfg = dict(BASE_CONFIG)
        cfg.update(
            {
                "universe": str(variant["universe"]),
                "historical_membership_path": str(variant["historical_membership_path"]),
            }
        )
        print(
            f"[{i}/{len(variants)}] Running {variant['UniverseVariant']} "
            f"(seed={variant['universe']}, source={variant['UniverseSource']})"
        )
        summary, outdir = run_backtest(**cfg, run_cache=run_cache)
        mean_sub, std_sub, min_sub, max_sub, stability = _stable_score_from_equity_curve(outdir=outdir)
        avg_u = _average_universe_size(outdir=outdir)
        rows.append(
            {
                "UniverseVariant": str(variant["UniverseVariant"]),
                "UniverseSeed": str(variant["universe"]),
                "UniverseSource": str(variant["UniverseSource"]),
                "MembershipPath": str(variant["historical_membership_path"]),
                "CAGR": float(summary.get("CAGR", float("nan"))),
                "Vol": float(summary.get("Vol", float("nan"))),
                "Sharpe": float(summary.get("Sharpe", float("nan"))),
                "MaxDD": float(summary.get("MaxDD", float("nan"))),
                "MeanSubSharpe": mean_sub,
                "StdSubSharpe": std_sub,
                "MinSubSharpe": min_sub,
                "MaxSubSharpe": max_sub,
                "StabilityScore": stability,
                "AverageUniverseSize": avg_u,
                "Outdir": str(outdir),
            }
        )
        print(
            f"    Sharpe={rows[-1]['Sharpe']:.4f} CAGR={rows[-1]['CAGR']:.4f} "
            f"MaxDD={rows[-1]['MaxDD']:.4f} AvgUniverse={rows[-1]['AverageUniverseSize']:.1f}"
        )

    df = pd.DataFrame(rows)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_ROOT / f"universe_expansion_comparison_{ts}.csv"
    out_json = RESULTS_ROOT / f"universe_expansion_summary_{ts}.json"
    latest_csv = RESULTS_ROOT / "universe_expansion_comparison_latest.csv"
    latest_json = RESULTS_ROOT / "universe_expansion_summary_latest.json"

    df.to_csv(out_csv, index=False, float_format="%.10g")
    df.to_csv(latest_csv, index=False, float_format="%.10g")
    payload = {
        "base_config": BASE_CONFIG,
        "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
        "variants": variants,
        "results": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nUNIVERSE EXPANSION COMPARISON")
    print("-----------------------------")
    cols = [
        "UniverseVariant",
        "UniverseSource",
        "AverageUniverseSize",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "StabilityScore",
    ]
    print(df[cols].to_string(index=False))
    print("")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    print(f"Saved: {latest_csv}")
    print(f"Saved: {latest_json}")


if __name__ == "__main__":
    main()
