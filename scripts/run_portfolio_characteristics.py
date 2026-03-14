"""Measure average portfolio characteristics for selected strategy variants."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_composite_vs_sleeves as composite
from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.engine import runner
from quant_lab.engine.runner import run_backtest
from quant_lab.factors.registry import compute_factors
from quant_lab.strategies.topn import rebalance_mask


RESULTS_ROOT = Path("results") / "portfolio_characteristics"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 2000

STRATEGIES: list[dict[str, Any]] = [
    {
        "strategy_name": "benchmark_4factor",
        "portfolio_mode": "composite",
        "factor_names": [
            "momentum_12_1",
            "reversal_1m",
            "low_vol_20",
            "gross_profitability",
        ],
        "factor_weights": [0.25, 0.25, 0.25, 0.25],
    },
    {
        "strategy_name": "rev_tilt_2",
        "portfolio_mode": "multi_sleeve",
        "factor_names": [
            "momentum_12_1",
            "reversal_1m",
            "low_vol_20",
            "gross_profitability",
        ],
        "factor_weights": [0.25, 0.25, 0.25, 0.25],
        "multi_sleeve_config": {
            "sleeves": [
                {
                    "name": "momentum",
                    "factors": ["momentum_12_1"],
                    "factor_weights": [1.0],
                    "allocation": 0.15,
                    "top_n": 50,
                },
                {
                    "name": "reversal",
                    "factors": ["reversal_1m"],
                    "factor_weights": [1.0],
                    "allocation": 0.40,
                    "top_n": 50,
                },
                {
                    "name": "low_vol",
                    "factors": ["low_vol_20"],
                    "factor_weights": [1.0],
                    "allocation": 0.10,
                    "top_n": 50,
                },
                {
                    "name": "profitability",
                    "factors": ["gross_profitability"],
                    "factor_weights": [1.0],
                    "allocation": 0.35,
                    "top_n": 50,
                },
            ]
        },
    },
    {
        "strategy_name": "baseline_reversal",
        "portfolio_mode": "composite",
        "factor_names": ["reversal_1m"],
        "factor_weights": [1.0],
    },
]


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "top_n": composite.TOP_N,
        "rebalance": composite.REBALANCE,
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": MAX_TICKERS,
        "data_source": composite.DATA_SOURCE,
        "data_cache_dir": composite.DATA_CACHE_DIR,
        "fundamentals_path": composite.FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": composite.FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "save_artifacts": True,
    }


def _strategy_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_base_config())
    cfg["portfolio_mode"] = str(spec["portfolio_mode"])
    cfg["factor_name"] = list(spec["factor_names"])
    cfg["factor_names"] = list(spec["factor_names"])
    cfg["factor_weights"] = list(spec["factor_weights"])
    if "multi_sleeve_config" in spec:
        cfg["multi_sleeve_config"] = spec["multi_sleeve_config"]
    return cfg


def _load_panels(start: str, end: str, max_tickers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = runner._load_universe_seed_tickers(
        universe="liquid_us",
        max_tickers=int(max_tickers),
        data_cache_dir=composite.DATA_CACHE_DIR,
    )
    ohlcv_map, _ = fetch_ohlcv_with_summary(
        tickers=tickers,
        start=str(start),
        end=str(end),
        cache_dir=composite.DATA_CACHE_DIR,
        data_source=composite.DATA_SOURCE,
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


def _build_characteristic_panels(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    factor_names = ["gross_profitability", "momentum_12_1", "reversal_1m"]
    factor_params = runner._augment_factor_params_with_fundamentals(
        factor_names=factor_names,
        factor_params_map={},
        close=close,
        fundamentals_path=composite.FUNDAMENTALS_PATH,
        fundamentals_fallback_lag_days=composite.FUNDAMENTALS_FALLBACK_LAG_DAYS,
    )
    scores = compute_factors(
        factor_names=factor_names,
        close=close,
        factor_params=factor_params,
    )
    vol20 = close.pct_change().rolling(20).std(ddof=0).astype(float)
    adv20 = (close * volume).rolling(20, min_periods=20).mean().astype(float)

    shares = None
    gp_params = factor_params.get("gross_profitability", {})
    fundamentals_aligned = gp_params.get("fundamentals_aligned") if isinstance(gp_params, dict) else None
    if isinstance(fundamentals_aligned, dict):
        shares = fundamentals_aligned.get("shares_outstanding")
    if isinstance(shares, pd.DataFrame):
        market_cap = close.astype(float) * shares.reindex(index=close.index, columns=close.columns).astype(float)
        market_cap = market_cap.where(market_cap > 0.0)
    else:
        market_cap = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)

    return {
        "market_cap": market_cap.astype(float),
        "gross_profitability": scores["gross_profitability"].astype(float),
        "momentum_12_1": scores["momentum_12_1"].astype(float),
        "reversal_1m": scores["reversal_1m"].astype(float),
        "volatility_20": vol20,
        "avg_dollar_volume": adv20,
    }


def _read_holdings(path: Path) -> pd.DataFrame:
    holdings_path = path / "holdings.csv"
    if not holdings_path.exists():
        raise FileNotFoundError(f"Missing holdings artifact: {holdings_path}")
    return pd.read_csv(holdings_path, index_col=0, parse_dates=True).sort_index().astype(float)


def _weighted_panel_average(weights_row: pd.Series, panel_row: pd.Series) -> float:
    aligned = pd.concat([weights_row.astype(float), panel_row.astype(float)], axis=1, join="inner")
    aligned.columns = ["w", "x"]
    aligned = aligned.loc[aligned["w"] > 0.0]
    aligned = aligned.dropna()
    if aligned.empty:
        return float("nan")
    total = float(aligned["w"].sum())
    if total <= 0.0:
        return float("nan")
    return float((aligned["w"] * aligned["x"]).sum() / total)


def _summarize_strategy(
    strategy_name: str,
    holdings: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    sector_by_ticker: dict[str, str] | None,
) -> tuple[dict[str, Any], pd.Series]:
    rb = rebalance_mask(pd.DatetimeIndex(holdings.index), composite.REBALANCE)
    rb_holdings = holdings.loc[rb].copy()
    rows: list[dict[str, float]] = []
    sector_rows: list[pd.Series] = []
    for dt in rb_holdings.index:
        weights_row = rb_holdings.loc[dt].astype(float)
        if float(weights_row.sum()) <= 0.0:
            continue
        row = {"date": dt}
        for name, panel in panels.items():
            if dt not in panel.index:
                row[name] = float("nan")
                continue
            row[name] = _weighted_panel_average(weights_row, panel.loc[dt])
        rows.append(row)
        if sector_by_ticker is not None:
            sectors = pd.Series(
                {
                    ticker: str(sector_by_ticker.get(ticker, "UNKNOWN"))
                    for ticker in weights_row.index[weights_row > 0.0]
                }
            )
            if not sectors.empty:
                sec_w = weights_row.loc[sectors.index].groupby(sectors).sum()
                sector_rows.append(sec_w.astype(float))
    char_df = pd.DataFrame(rows)
    summary = {
        "Strategy": strategy_name,
        "AvgMktCap": float(char_df["market_cap"].mean()) if "market_cap" in char_df else float("nan"),
        "Profitability": float(char_df["gross_profitability"].mean()) if "gross_profitability" in char_df else float("nan"),
        "Momentum": float(char_df["momentum_12_1"].mean()) if "momentum_12_1" in char_df else float("nan"),
        "Reversal": float(char_df["reversal_1m"].mean()) if "reversal_1m" in char_df else float("nan"),
        "Volatility": float(char_df["volatility_20"].mean()) if "volatility_20" in char_df else float("nan"),
        "AvgDollarVolume": float(char_df["avg_dollar_volume"].mean()) if "avg_dollar_volume" in char_df else float("nan"),
    }
    if sector_rows:
        sector_df = pd.DataFrame(sector_rows).fillna(0.0)
        avg_sector = sector_df.mean(axis=0).sort_index()
    else:
        avg_sector = pd.Series(dtype=float)
    return summary, avg_sector


def main() -> None:
    global START, END, MAX_TICKERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    START = str(args.start)
    END = str(args.end)
    MAX_TICKERS = int(args.max_tickers)

    results_root = Path(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = results_root / ts
    outdir.mkdir(parents=True, exist_ok=True)

    close, volume = _load_panels(start=START, end=END, max_tickers=MAX_TICKERS)
    panels = _build_characteristic_panels(close=close, volume=volume)
    sector_by_ticker = runner._load_sector_map("", list(close.columns))

    run_cache: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    sector_series: dict[str, pd.Series] = {}
    manifest_runs: list[dict[str, Any]] = []
    for spec in STRATEGIES:
        strategy_name = str(spec["strategy_name"])
        print(f"Running {strategy_name}...")
        cfg = _strategy_config(spec)
        summary, run_outdir = run_backtest(**cfg, run_cache=run_cache)
        holdings = _read_holdings(Path(run_outdir))
        row, avg_sector = _summarize_strategy(
            strategy_name=strategy_name,
            holdings=holdings,
            panels=panels,
            sector_by_ticker=sector_by_ticker,
        )
        summary_rows.append(row)
        sector_series[strategy_name] = avg_sector
        manifest_runs.append(
            {
                "strategy_name": strategy_name,
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
                "run_config": cfg,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    order = ["benchmark_4factor", "rev_tilt_2", "baseline_reversal"]
    summary_df["_order"] = summary_df["Strategy"].map({name: i for i, name in enumerate(order)})
    summary_df = summary_df.sort_values(["_order"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    sector_df = pd.DataFrame(sector_series).T.fillna(0.0)
    sector_df.index.name = "Strategy"
    sector_df = sector_df.reset_index()

    summary_path = outdir / "portfolio_characteristics_summary.csv"
    sector_path = outdir / "sector_weights.csv"
    manifest_path = outdir / "manifest.json"
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    sector_df.to_csv(sector_path, index=False, float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": ts,
                "results_dir": str(outdir),
                "date_range": {"start": START, "end": END},
                "universe": "liquid_us",
                "universe_mode": "dynamic",
                "rebalance": composite.REBALANCE,
                "top_n": composite.TOP_N,
                "weighting": composite.WEIGHTING,
                "costs_bps": composite.COSTS_BPS,
                "max_tickers": MAX_TICKERS,
                "strategies": [spec["strategy_name"] for spec in STRATEGIES],
                "characteristics": list(panels.keys()),
                "sector_map_available": sector_by_ticker is not None,
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "portfolio_characteristics_summary.csv": summary_path,
            "sector_weights.csv": sector_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = summary_df[["Strategy", "AvgMktCap", "Profitability", "Momentum", "Volatility"]].copy()
    for col in ["AvgMktCap", "Profitability", "Momentum", "Volatility"]:
        table[col] = table[col].map(composite._format_float)
    print("")
    print("PORTFOLIO CHARACTERISTICS")
    print("-------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    print(f"Saved: {summary_path}")
    print(f"Saved: {sector_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
