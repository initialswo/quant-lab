"""Run a leave-one-factor-out study on the six-factor benchmark portfolio."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_asset_growth_sleeve as asset_growth_sleeve
import run_book_to_market_sleeve as book_to_market_sleeve
import run_composite_vs_sleeves as composite
from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover


RESULTS_ROOT = Path("results") / "factor_leave_one_out"
START = "2000-01-01"
END = "2024-12-31"
MAX_TICKERS = 300

SLEEVE_SPECS: list[dict[str, Any]] = [
    {
        "strategy_name": "sleeve_momentum",
        "factor_name": "momentum_12_1",
        "factor_params": {},
        "notes": "Standalone momentum sleeve.",
    },
    {
        "strategy_name": "sleeve_reversal",
        "factor_name": "reversal_1m",
        "factor_params": {},
        "notes": "Standalone reversal sleeve.",
    },
    {
        "strategy_name": "sleeve_low_vol",
        "factor_name": "low_vol_20",
        "factor_params": {},
        "notes": "Standalone low-vol sleeve.",
    },
    {
        "strategy_name": "sleeve_gross_profitability",
        "factor_name": "gross_profitability",
        "factor_params": {},
        "notes": "Standalone profitability sleeve.",
    },
    {
        "strategy_name": "sleeve_book_to_market",
        "factor_name": "book_to_market",
        "factor_params": {},
        "notes": "Classic value sleeve using PIT shareholders_equity divided by market_cap.",
    },
    {
        "strategy_name": "sleeve_asset_growth",
        "factor_name": "asset_growth",
        "factor_params": {},
        "notes": "Low asset growth sleeve using PIT total_assets.",
    },
]

PORTFOLIO_DEFINITIONS: dict[str, list[str]] = {
    "factor_benchmark_equal_weight": [
        "sleeve_momentum",
        "sleeve_reversal",
        "sleeve_low_vol",
        "sleeve_gross_profitability",
        "sleeve_book_to_market",
        "sleeve_asset_growth",
    ],
    "no_momentum": [
        "sleeve_reversal",
        "sleeve_low_vol",
        "sleeve_gross_profitability",
        "sleeve_book_to_market",
        "sleeve_asset_growth",
    ],
    "no_reversal": [
        "sleeve_momentum",
        "sleeve_low_vol",
        "sleeve_gross_profitability",
        "sleeve_book_to_market",
        "sleeve_asset_growth",
    ],
    "no_low_vol": [
        "sleeve_momentum",
        "sleeve_reversal",
        "sleeve_gross_profitability",
        "sleeve_book_to_market",
        "sleeve_asset_growth",
    ],
    "no_profitability": [
        "sleeve_momentum",
        "sleeve_reversal",
        "sleeve_low_vol",
        "sleeve_book_to_market",
        "sleeve_asset_growth",
    ],
    "no_value": [
        "sleeve_momentum",
        "sleeve_reversal",
        "sleeve_low_vol",
        "sleeve_gross_profitability",
        "sleeve_asset_growth",
    ],
    "no_asset_growth": [
        "sleeve_momentum",
        "sleeve_reversal",
        "sleeve_low_vol",
        "sleeve_gross_profitability",
        "sleeve_book_to_market",
    ],
}

DISPLAY_ORDER: list[str] = [
    "factor_benchmark_equal_weight",
    "no_momentum",
    "no_reversal",
    "no_low_vol",
    "no_profitability",
    "no_value",
    "no_asset_growth",
]


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "top_n": 50,
        "rebalance": "weekly",
        "weighting": "equal",
        "costs_bps": 10.0,
        "max_tickers": MAX_TICKERS,
        "data_source": composite.DATA_SOURCE,
        "data_cache_dir": composite.DATA_CACHE_DIR,
        "fundamentals_path": composite.FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": composite.FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "save_artifacts": True,
    }


def _run_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_base_config())
    factor_name = str(spec["factor_name"])
    cfg["factor_name"] = [factor_name]
    cfg["factor_names"] = [factor_name]
    cfg["factor_weights"] = [1.0]
    if spec["factor_params"]:
        cfg["factor_params"] = {factor_name: dict(spec["factor_params"])}
    return cfg


def _run_sleeves(
    book_to_market_params: dict[str, Any],
    asset_growth_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_cache: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    for raw_spec in SLEEVE_SPECS:
        spec = dict(raw_spec)
        strategy_name = str(spec["strategy_name"])
        if strategy_name == "sleeve_book_to_market":
            spec["factor_params"] = book_to_market_params
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
    portfolios: list[dict[str, Any]] = []
    for strategy_name, sleeve_names in PORTFOLIO_DEFINITIONS.items():
        weight = 1.0 / float(len(sleeve_names))
        portfolios.append(
            composite.weighted_sleeve_portfolio(
                sleeve_payloads=[by_name[name] for name in sleeve_names],
                weights=[weight] * len(sleeve_names),
                name=strategy_name,
            )
        )
    return portfolios


def _build_manifest(
    outdir: Path,
    sleeve_manifest: list[dict[str, Any]],
    max_tickers: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": outdir.name,
        "results_dir": str(outdir),
        "date_range": {"start": START, "end": END},
        "universe": "liquid_us",
        "universe_mode": "dynamic",
        "top_n": 50,
        "rebalance": "weekly",
        "weighting": "equal",
        "costs_bps": 10.0,
        "max_tickers": max_tickers,
        "sleeves": [str(spec["strategy_name"]) for spec in SLEEVE_SPECS],
        "portfolio_definitions": {
            name: {
                "sleeves": sleeves,
                "weights": {sleeve: float(1.0 / len(sleeves)) for sleeve in sleeves},
            }
            for name, sleeves in PORTFOLIO_DEFINITIONS.items()
        },
        "notes": [
            "This is a leave-one-factor-out benchmark study.",
            "Portfolios are equal-weight return-level combinations of standalone sleeve daily returns.",
            "book_to_market and asset_growth receive PIT-aligned fundamentals explicitly through helper builders.",
        ],
        "sleeve_runs": sleeve_manifest,
    }


def _print_interpretation(metrics_df: pd.DataFrame) -> None:
    by_name = metrics_df.set_index("Strategy")
    baseline = by_name.loc["factor_benchmark_equal_weight"]
    omissions = metrics_df.loc[metrics_df["Strategy"].ne("factor_benchmark_equal_weight")].copy()
    omissions["sharpe_delta"] = omissions["Sharpe"] - float(baseline["Sharpe"])
    omissions["cagr_delta"] = omissions["CAGR"] - float(baseline["CAGR"])
    omissions["maxdd_delta"] = omissions["MaxDD"] - float(baseline["MaxDD"])

    worst_sharpe = omissions.sort_values("sharpe_delta", ascending=True, kind="mergesort").iloc[0]
    best_sharpe = omissions.sort_values("sharpe_delta", ascending=False, kind="mergesort").iloc[0]
    worst_cagr = omissions.sort_values("cagr_delta", ascending=True, kind="mergesort").iloc[0]
    worst_dd = omissions.sort_values("maxdd_delta", ascending=True, kind="mergesort").iloc[0]

    improved = float(best_sharpe["sharpe_delta"]) > 0.0

    print("INTERPRETATION")
    print("--------------")
    print(
        f"Omission hurting Sharpe most: {worst_sharpe['Strategy']} "
        f"({float(worst_sharpe['Sharpe']):.4f} vs {float(baseline['Sharpe']):.4f})."
    )
    if improved:
        print(
            f"Removing a factor did improve Sharpe: {best_sharpe['Strategy']} "
            f"({float(best_sharpe['Sharpe']):.4f} vs {float(baseline['Sharpe']):.4f})."
        )
    else:
        print(f"No leave-one-out portfolio improved Sharpe versus the full benchmark ({float(baseline['Sharpe']):.4f}).")
    print(
        f"Most important for return appears to be the omitted factor in {worst_cagr['Strategy']} "
        f"(CAGR {float(worst_cagr['CAGR']):.4f} vs {float(baseline['CAGR']):.4f})."
    )
    print(
        f"Most important for diversification / drawdown appears to be the omitted factor in {worst_dd['Strategy']} "
        f"(MaxDD {float(worst_dd['MaxDD']):.4f} vs {float(baseline['MaxDD']):.4f})."
    )


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

    book_to_market_params = book_to_market_sleeve._build_book_to_market_params(
        start=START,
        end=END,
        max_tickers=MAX_TICKERS,
    )
    asset_growth_params = asset_growth_sleeve._build_asset_growth_params(
        start=START,
        end=END,
        max_tickers=MAX_TICKERS,
    )
    sleeve_payloads, sleeve_manifest = _run_sleeves(
        book_to_market_params=book_to_market_params,
        asset_growth_params=asset_growth_params,
    )
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
    ).sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    daily_returns_df = pd.concat([p["daily_return"] for p in portfolio_payloads], axis=1).sort_index().fillna(0.0)

    results_path = outdir / "leave_one_out_results.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    manifest_path = outdir / "manifest.json"

    metrics_df.to_csv(results_path, index=False, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                outdir=outdir,
                sleeve_manifest=sleeve_manifest,
                max_tickers=MAX_TICKERS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    composite._copy_latest(
        files={
            "leave_one_out_results.csv": results_path,
            "daily_returns.csv": daily_returns_path,
            "manifest.json": manifest_path,
        },
        latest_root=results_root / "latest",
    )

    ordered_display = (
        metrics_df.set_index("Strategy")
        .reindex(DISPLAY_ORDER)
        .dropna(how="all")
        .reset_index()
        .sort_values(["Sharpe", "CAGR"], ascending=[False, False], kind="mergesort")
        .reset_index(drop=True)
    )
    table = ordered_display[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]].copy()
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover"]:
        table[col] = table[col].map(composite._format_float)

    print("FACTOR LEAVE-ONE-OUT TEST")
    print("-------------------------")
    print("")
    print(table.to_string(index=False))
    print("")
    _print_interpretation(metrics_df)
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
