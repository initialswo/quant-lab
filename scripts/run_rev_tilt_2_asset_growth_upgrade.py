"""Compare rev_tilt_2 against an asset-growth sleeve upgrade."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_asset_growth_sleeve as asset_growth_sleeve
import run_composite_vs_sleeves as composite
from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


RESULTS_ROOT = Path("results") / "rev_tilt_2_asset_growth_upgrade"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 300

SLEEVE_SPECS: list[dict[str, Any]] = [
    {
        "strategy_name": "sleeve_reversal",
        "factor_name": "reversal_1m",
        "factor_params": {},
        "notes": "Standalone reversal sleeve.",
    },
    {
        "strategy_name": "sleeve_gross_profitability",
        "factor_name": "gross_profitability",
        "factor_params": {},
        "notes": "Standalone profitability sleeve.",
    },
    {
        "strategy_name": "sleeve_momentum",
        "factor_name": "momentum_12_1",
        "factor_params": {},
        "notes": "Standalone momentum sleeve.",
    },
    {
        "strategy_name": "sleeve_low_vol",
        "factor_name": "low_vol_20",
        "factor_params": {},
        "notes": "20-day volatility sleeve using lower-vol preference.",
    },
    {
        "strategy_name": "sleeve_asset_growth",
        "factor_name": "asset_growth",
        "factor_params": {},
        "notes": "Low asset growth sleeve.",
    },
]

ORIGINAL_WEIGHTS: dict[str, float] = {
    "sleeve_reversal": 0.40,
    "sleeve_gross_profitability": 0.30,
    "sleeve_momentum": 0.20,
    "sleeve_low_vol": 0.10,
}
UPGRADE_WEIGHTS: dict[str, float] = {
    "sleeve_reversal": 0.35,
    "sleeve_gross_profitability": 0.25,
    "sleeve_momentum": 0.15,
    "sleeve_low_vol": 0.10,
    "sleeve_asset_growth": 0.15,
}


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
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


def _run_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_base_config())
    cfg["factor_name"] = [str(spec["factor_name"])]
    cfg["factor_names"] = [str(spec["factor_name"])]
    cfg["factor_weights"] = [1.0]
    if spec["factor_params"]:
        cfg["factor_params"] = {str(spec["factor_name"]): dict(spec["factor_params"])}
    return cfg


def _run_sleeves(asset_growth_params: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    for raw_spec in SLEEVE_SPECS:
        spec = dict(raw_spec)
        strategy_name = str(spec["strategy_name"])
        if strategy_name == "sleeve_asset_growth":
            spec["factor_params"] = asset_growth_params
        print(f"Running {strategy_name}...")
        summary, run_outdir = run_backtest(**_run_config(spec), run_cache=run_cache)
        payload = composite._strategy_payload(name=strategy_name, summary=summary, outdir=run_outdir)
        payloads.append(payload)
        manifest_runs.append(
            {
                "strategy_name": strategy_name,
                "factor_name": str(spec["factor_name"]),
                "notes": str(spec["notes"]),
                "annual_turnover_from_summary": extract_annual_turnover(summary=summary, outdir=run_outdir),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
            }
        )
    return payloads, manifest_runs


def _combine_portfolios(sleeve_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(p["name"]): p for p in sleeve_payloads}
    original = composite.weighted_sleeve_portfolio(
        sleeve_payloads=[by_name[name] for name in ORIGINAL_WEIGHTS],
        weights=[ORIGINAL_WEIGHTS[name] for name in ORIGINAL_WEIGHTS],
        name="rev_tilt_2_original",
    )
    upgrade = composite.weighted_sleeve_portfolio(
        sleeve_payloads=[by_name[name] for name in UPGRADE_WEIGHTS],
        weights=[UPGRADE_WEIGHTS[name] for name in UPGRADE_WEIGHTS],
        name="rev_tilt_2_asset_growth",
    )
    return [original, upgrade]


def _build_manifest(
    outdir: Path,
    sleeve_manifest: list[dict[str, Any]],
    start: str,
    end: str,
    max_tickers: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": start, "end": end},
        "universe": composite.UNIVERSE,
        "universe_mode": composite.UNIVERSE_MODE,
        "top_n": composite.TOP_N,
        "rebalance": composite.REBALANCE,
        "weighting": composite.WEIGHTING,
        "costs_bps": composite.COSTS_BPS,
        "max_tickers": max_tickers,
        "strategy_definitions": {
            "rev_tilt_2_original": ORIGINAL_WEIGHTS,
            "rev_tilt_2_asset_growth": UPGRADE_WEIGHTS,
        },
        "notes": [
            "Portfolios are return-level combinations of single-factor sleeves.",
            "Requested volatility_20 ascending sleeve is implemented with existing factor low_vol_20.",
            "asset_growth uses PIT-aligned total_assets via the standalone asset growth sleeve helper.",
            "All sleeve runs reuse run_backtest with the default causal execution behavior.",
        ],
        "sleeve_runs": sleeve_manifest,
    }


def _print_interpretation(metrics_df: pd.DataFrame) -> None:
    by_name = metrics_df.set_index("Strategy")
    original = by_name.loc["rev_tilt_2_original"]
    upgrade = by_name.loc["rev_tilt_2_asset_growth"]
    sharpe_note = (
        f"Asset-growth upgrade improved Sharpe ({upgrade['Sharpe']:.4f} vs {original['Sharpe']:.4f})."
        if float(upgrade["Sharpe"]) > float(original["Sharpe"])
        else f"Asset-growth upgrade did not improve Sharpe ({upgrade['Sharpe']:.4f} vs {original['Sharpe']:.4f})."
    )
    dd_note = (
        f"Drawdown improved ({upgrade['MaxDD']:.4f} vs {original['MaxDD']:.4f})."
        if float(upgrade["MaxDD"]) > float(original["MaxDD"])
        else f"Drawdown did not improve ({upgrade['MaxDD']:.4f} vs {original['MaxDD']:.4f})."
    )
    turnover_delta = float(upgrade["Turnover"] - original["Turnover"])
    print("INTERPRETATION")
    print("--------------")
    print(sharpe_note)
    print(dd_note)
    print(f"Turnover delta: {turnover_delta:+.4f} annualized.")


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

    asset_growth_params = asset_growth_sleeve._build_asset_growth_params(
        start=START,
        end=END,
        max_tickers=MAX_TICKERS,
    )
    sleeve_payloads, sleeve_manifest = _run_sleeves(asset_growth_params=asset_growth_params)
    portfolio_payloads = _combine_portfolios(sleeve_payloads=sleeve_payloads)

    metrics_df = pd.DataFrame(
        [
            composite._portfolio_metrics(
                name=payload["name"],
                daily_return=payload["daily_return"],
                turnover=payload["daily_turnover"],
            )
            for payload in portfolio_payloads
        ]
    )
    metrics_df = (
        metrics_df.set_index("Strategy")
        .loc[["rev_tilt_2_original", "rev_tilt_2_asset_growth"]]
        .reset_index()
    )
    daily_returns_df = pd.concat([p["daily_return"] for p in portfolio_payloads], axis=1).sort_index().fillna(0.0)

    results_path = outdir / "asset_growth_upgrade_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    manifest_path = outdir / "manifest.json"

    metrics_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                sleeve_manifest=sleeve_manifest,
                start=START,
                end=END,
                max_tickers=MAX_TICKERS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "asset_growth_upgrade_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    table = metrics_df[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)
    print("REV_TILT_2 ASSET GROWTH UPGRADE TEST")
    print("------------------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    _print_interpretation(metrics_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
