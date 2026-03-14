"""Analyze market-cap and liquidity exposure for the latest Top-N sweep run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.engine import runner


RESULTS_ROOT = Path("results") / "topn_sweep"
DATA_CACHE_DIR = "data/equities"
DATA_SOURCE = "parquet"
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60
OUTPUT_NAME = "liquidity_exposure.csv"



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--fundamentals_path", default=FUNDAMENTALS_PATH)
    return parser.parse_args()



def _latest_run_dir(results_root: Path) -> Path:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    candidates = [p for p in results_root.iterdir() if p.is_dir() and p.name != "latest"]
    if not candidates:
        raise FileNotFoundError(f"No timestamped run directories found under: {results_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]



def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))



def _load_results(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "topn_sweep_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results CSV: {path}")
    return pd.read_csv(path)



def _run_backtest_dir(manifest: dict[str, Any], top_n: int) -> Path:
    for run in manifest.get("runs", []) or []:
        if int(run.get("top_n", -1)) == int(top_n):
            outdir = str(run.get("backtest_outdir", "")).strip()
            if outdir:
                return Path(outdir)
    raise FileNotFoundError(f"No backtest_outdir found in manifest for top_n={top_n}")



def _load_panels(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = runner._load_universe_seed_tickers(
        universe="liquid_us",
        max_tickers=2000,
        data_cache_dir=DATA_CACHE_DIR,
    )
    ohlcv_map, _ = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=DATA_CACHE_DIR,
        data_source=DATA_SOURCE,
        refresh=False,
        bulk_prepare=False,
    )
    close_cols, used_tickers, missing_tickers, rejected_tickers, _ = runner._collect_close_series(
        ohlcv_map=ohlcv_map,
        requested_tickers=tickers,
    )
    if not close_cols:
        raise ValueError(
            "No valid OHLCV data found. "
            f"missing={len(missing_tickers)} rejected={len(rejected_tickers)}"
        )
    close = pd.concat(close_cols, axis=1, join="outer")
    close = runner._prepare_close_panel(close_raw=close, price_fill_mode="ffill").astype(float)
    volume = runner._collect_numeric_panel(
        ohlcv_map=ohlcv_map,
        requested_tickers=used_tickers,
        field="Volume",
    ).reindex(index=close.index, columns=close.columns).astype(float)
    return close, volume



def _market_cap_and_adv(close: pd.DataFrame, volume: pd.DataFrame, fundamentals_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fundamentals = load_fundamentals_file(
        path=str(fundamentals_path),
        fallback_lag_days=int(FUNDAMENTALS_FALLBACK_LAG_DAYS),
    )
    aligned = align_fundamentals_to_daily_panel(
        fundamentals=fundamentals,
        price_index=pd.DatetimeIndex(close.index),
        price_columns=close.columns,
        value_columns=["shares_outstanding"],
    )
    shares = aligned.get("shares_outstanding")
    if not isinstance(shares, pd.DataFrame):
        raise ValueError("Aligned fundamentals did not produce a shares_outstanding panel.")
    market_cap = (close.astype(float) * shares.reindex(index=close.index, columns=close.columns).astype(float)).astype(float)
    market_cap = market_cap.where(market_cap > 0.0)
    adv20 = (close.astype(float) * volume.astype(float)).rolling(20, min_periods=20).mean().astype(float)
    adv20 = adv20.where(adv20 > 0.0)
    return market_cap, adv20



def _read_holdings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing holdings artifact: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index().astype(float)



def _read_score_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing composite score snapshot: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index().astype(float)



def _row_stats(values: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if v.empty:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.median())



def _portfolio_and_universe_rows(
    strategy: str,
    holdings: pd.DataFrame,
    score_snapshot: pd.DataFrame,
    market_cap: pd.DataFrame,
    adv20: pd.DataFrame,
) -> list[dict[str, Any]]:
    rb_dates = pd.DatetimeIndex(score_snapshot.index)
    rows: list[dict[str, Any]] = []
    for dt in rb_dates:
        if dt not in holdings.index or dt not in market_cap.index or dt not in adv20.index:
            continue
        weights = holdings.loc[dt].astype(float)
        held = weights[weights > 0.0].index
        eligible = score_snapshot.loc[dt].dropna().index
        if len(held) == 0 or len(eligible) == 0:
            continue

        portfolio_mc = market_cap.loc[dt, held]
        portfolio_adv = adv20.loc[dt, held]
        universe_mc = market_cap.loc[dt, eligible]
        universe_adv = adv20.loc[dt, eligible]

        portfolio_avg_market_cap, portfolio_median_market_cap = _row_stats(portfolio_mc)
        portfolio_avg_dollar_volume, portfolio_median_dollar_volume = _row_stats(portfolio_adv)
        universe_avg_market_cap, _ = _row_stats(universe_mc)
        universe_avg_dollar_volume, _ = _row_stats(universe_adv)

        market_cap_ratio = (
            portfolio_avg_market_cap / universe_avg_market_cap
            if pd.notna(portfolio_avg_market_cap) and pd.notna(universe_avg_market_cap) and universe_avg_market_cap > 0.0
            else float("nan")
        )
        volume_ratio = (
            portfolio_avg_dollar_volume / universe_avg_dollar_volume
            if pd.notna(portfolio_avg_dollar_volume) and pd.notna(universe_avg_dollar_volume) and universe_avg_dollar_volume > 0.0
            else float("nan")
        )
        rows.append(
            {
                "strategy": str(strategy),
                "date": pd.Timestamp(dt),
                "portfolio_avg_market_cap": float(portfolio_avg_market_cap),
                "portfolio_median_market_cap": float(portfolio_median_market_cap),
                "portfolio_avg_dollar_volume": float(portfolio_avg_dollar_volume),
                "portfolio_median_dollar_volume": float(portfolio_median_dollar_volume),
                "universe_avg_market_cap": float(universe_avg_market_cap),
                "universe_avg_dollar_volume": float(universe_avg_dollar_volume),
                "market_cap_ratio": float(market_cap_ratio),
                "volume_ratio": float(volume_ratio),
            }
        )
    return rows



def _format_number(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4g}"



def main() -> None:
    args = _parse_args()
    results_root = Path(str(args.results_root)).expanduser()
    run_dir = Path(str(args.run_dir)).expanduser() if str(args.run_dir).strip() else _latest_run_dir(results_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifest = _load_manifest(run_dir)
    results_df = _load_results(run_dir)
    start = str(manifest.get("start") or results_df["start"].iloc[0])
    end = str(manifest.get("end") or results_df["end"].iloc[0])
    close, volume = _load_panels(start=start, end=end)
    market_cap, adv20 = _market_cap_and_adv(close=close, volume=volume, fundamentals_path=str(args.fundamentals_path))

    rows: list[dict[str, Any]] = []
    for result in results_df.to_dict(orient="records"):
        top_n = int(result["top_n"])
        strategy = f"Top{top_n}"
        backtest_dir = _run_backtest_dir(manifest=manifest, top_n=top_n)
        holdings = _read_holdings(backtest_dir / "holdings.csv")
        score_snapshot = _read_score_snapshot(backtest_dir / "composite_scores_snapshot.csv")
        rows.extend(
            _portfolio_and_universe_rows(
                strategy=strategy,
                holdings=holdings,
                score_snapshot=score_snapshot,
                market_cap=market_cap,
                adv20=adv20,
            )
        )

    exposure_df = pd.DataFrame(
        rows,
        columns=[
            "strategy",
            "date",
            "portfolio_avg_market_cap",
            "portfolio_median_market_cap",
            "portfolio_avg_dollar_volume",
            "portfolio_median_dollar_volume",
            "universe_avg_market_cap",
            "universe_avg_dollar_volume",
            "market_cap_ratio",
            "volume_ratio",
        ],
    )
    exposure_df = exposure_df.sort_values(["strategy", "date"], kind="mergesort").reset_index(drop=True)

    out_path = run_dir / OUTPUT_NAME
    exposure_df.to_csv(out_path, index=False, float_format="%.10g")

    print("LIQUIDITY EXPOSURE")
    print("------------------")
    for strategy, part in exposure_df.groupby("strategy", sort=True):
        print("")
        print(f"Strategy: {strategy}")
        print(f"Portfolio Avg Market Cap: {_format_number(float(part['portfolio_avg_market_cap'].mean()))}")
        print(f"Universe Avg Market Cap: {_format_number(float(part['universe_avg_market_cap'].mean()))}")
        print("")
        print(f"Portfolio Avg Dollar Volume: {_format_number(float(part['portfolio_avg_dollar_volume'].mean()))}")
        print(f"Universe Avg Dollar Volume: {_format_number(float(part['universe_avg_dollar_volume'].mean()))}")
        print("")
        print(f"Market Cap Ratio: {_format_number(float(part['market_cap_ratio'].mean()))}")
        print(f"Volume Ratio: {_format_number(float(part['volume_ratio'].mean()))}")
    print("")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
