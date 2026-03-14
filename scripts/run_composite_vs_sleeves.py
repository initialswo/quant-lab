"""Compare the composite benchmark against independent equal-weight factor sleeves."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.engine.metrics import compute_metrics
from quant_lab.engine.runner import run_backtest
from quant_lab.research.sweep_metrics import extract_annual_turnover
from quant_lab.strategies.topn import rebalance_mask


RESULTS_ROOT = Path("results") / "composite_vs_sleeves"
DOC_LOG_PATH = Path("docs/experiment_log.md")
START = "2000-01-01"
END = "2024-12-31"
UNIVERSE = "liquid_us"
UNIVERSE_MODE = "dynamic"
REBALANCE = "weekly"
TOP_N = 50
WEIGHTING = "equal"
COSTS_BPS = 10.0
MAX_TICKERS = 2000
DATA_SOURCE = "parquet"
DATA_CACHE_DIR = "data/equities"
FUNDAMENTALS_PATH = "data/fundamentals/fundamentals_fmp.parquet"
FUNDAMENTALS_FALLBACK_LAG_DAYS = 60

FACTORS: list[str] = [
    "momentum_12_1",
    "reversal_1m",
    "low_vol_20",
    "gross_profitability",
]

STRATEGY_SPECS: list[dict[str, Any]] = [
    {
        "strategy": "composite_benchmark",
        "factor_names": FACTORS,
        "factor_weights": [0.25, 0.25, 0.25, 0.25],
        "notes": "Composite benchmark with normalized linear factor aggregation.",
    },
    {
        "strategy": "sleeve_momentum",
        "factor_names": ["momentum_12_1"],
        "factor_weights": [1.0],
        "notes": "Single-factor sleeve.",
    },
    {
        "strategy": "sleeve_reversal",
        "factor_names": ["reversal_1m"],
        "factor_weights": [1.0],
        "notes": "Single-factor sleeve.",
    },
    {
        "strategy": "sleeve_low_vol",
        "factor_names": ["low_vol_20"],
        "factor_weights": [1.0],
        "notes": "Single-factor sleeve.",
    },
    {
        "strategy": "sleeve_gross_profitability",
        "factor_names": ["gross_profitability"],
        "factor_weights": [1.0],
        "notes": "Single-factor sleeve.",
    },
]

SLEEVE_NAMES: list[str] = [
    "sleeve_momentum",
    "sleeve_reversal",
    "sleeve_low_vol",
    "sleeve_gross_profitability",
]
SLEEVE_SPECS: list[dict[str, Any]] = [
    spec for spec in STRATEGY_SPECS if str(spec["strategy"]) in set(SLEEVE_NAMES)
]


def _base_config() -> dict[str, Any]:
    return {
        "start": START,
        "end": END,
        "universe": UNIVERSE,
        "universe_mode": UNIVERSE_MODE,
        "top_n": TOP_N,
        "rebalance": REBALANCE,
        "weighting": WEIGHTING,
        "costs_bps": COSTS_BPS,
        "max_tickers": MAX_TICKERS,
        "data_source": DATA_SOURCE,
        "data_cache_dir": DATA_CACHE_DIR,
        "fundamentals_path": FUNDAMENTALS_PATH,
        "fundamentals_fallback_lag_days": FUNDAMENTALS_FALLBACK_LAG_DAYS,
        "save_artifacts": True,
    }


def _run_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_base_config())
    cfg["factor_name"] = list(spec["factor_names"])
    cfg["factor_names"] = list(spec["factor_names"])
    cfg["factor_weights"] = list(spec["factor_weights"])
    return cfg


def _read_equity_frame(outdir: Path) -> pd.DataFrame:
    path = outdir / "equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity artifact: {path}")
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return frame


def _read_holdings_frame(outdir: Path) -> pd.DataFrame:
    path = outdir / "holdings.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing holdings artifact: {path}")
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return frame


def _strategy_payload(
    name: str,
    summary: dict[str, Any],
    outdir: str,
) -> dict[str, Any]:
    outdir_path = Path(outdir)
    equity = _read_equity_frame(outdir_path)
    holdings = _read_holdings_frame(outdir_path)
    daily_return = pd.to_numeric(equity["DailyReturn"], errors="coerce").fillna(0.0).rename(name)
    turnover = (
        pd.to_numeric(equity["Turnover"], errors="coerce").fillna(0.0)
        if "Turnover" in equity.columns
        else pd.Series(0.0, index=equity.index, name="Turnover")
    )
    return {
        "name": name,
        "summary": summary,
        "outdir": outdir_path,
        "equity": equity,
        "holdings": holdings,
        "daily_return": daily_return,
        "daily_turnover": turnover.rename(name),
    }


def _portfolio_metrics(
    name: str,
    daily_return: pd.Series,
    turnover: pd.Series,
    annual_turnover_override: float | None = None,
) -> dict[str, Any]:
    r = pd.to_numeric(daily_return, errors="coerce").fillna(0.0).astype(float)
    metrics = compute_metrics(r)
    non_na = int(r.notna().sum())
    hit_rate = float((r > 0.0).mean()) if non_na > 0 else float("nan")
    total_return = float((1.0 + r).prod() - 1.0) if non_na > 0 else float("nan")
    annual_turnover = (
        float(annual_turnover_override)
        if annual_turnover_override is not None and pd.notna(annual_turnover_override)
        else float(pd.to_numeric(turnover, errors="coerce").fillna(0.0).mean() * 252.0)
    )
    return {
        "Strategy": name,
        "CAGR": float(metrics.get("CAGR", float("nan"))),
        "Vol": float(metrics.get("Vol", float("nan"))),
        "Sharpe": float(metrics.get("Sharpe", float("nan"))),
        "MaxDD": float(metrics.get("MaxDD", float("nan"))),
        "Turnover": annual_turnover,
        "HitRate": hit_rate,
        "TotalReturn": total_return,
    }


def _combined_sleeve_payload(sleeve_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return weighted_sleeve_portfolio(
        sleeve_payloads=sleeve_payloads,
        weights=np.full(len(sleeve_payloads), 1.0 / len(sleeve_payloads), dtype=float),
        name="sleeve_combined_equal_weight",
    )


def weighted_sleeve_portfolio(
    sleeve_payloads: list[dict[str, Any]],
    weights: list[float] | np.ndarray,
    name: str,
) -> dict[str, Any]:
    ret_df = pd.concat([p["daily_return"] for p in sleeve_payloads], axis=1).sort_index().fillna(0.0)
    turnover_df = (
        pd.concat([p["daily_turnover"] for p in sleeve_payloads], axis=1).sort_index().fillna(0.0)
    )
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(sleeve_payloads):
        raise ValueError("weights length must match sleeve_payloads length")
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to > 0")
    w = w / total
    combined_ret = ret_df.mul(w, axis=1).sum(axis=1).rename(name)
    combined_turnover = turnover_df.mul(w, axis=1).sum(axis=1).rename(name)
    equity = (1.0 + combined_ret).cumprod().rename("Equity")
    frame = pd.DataFrame(
        {
            "Equity": equity,
            "DailyReturn": combined_ret,
            "Turnover": combined_turnover,
        }
    )
    return {
        "name": name,
        "summary": {},
        "outdir": None,
        "equity": frame,
        "holdings": None,
        "daily_return": combined_ret,
        "daily_turnover": combined_turnover,
    }


def run_sleeve_backtests(run_cache: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache = {} if run_cache is None else run_cache
    payloads: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    for spec in SLEEVE_SPECS:
        cfg = _run_config(spec)
        strategy = str(spec["strategy"])
        print(f"Running {strategy}...")
        summary, run_outdir = run_backtest(**cfg, run_cache=cache)
        payload = _strategy_payload(name=strategy, summary=summary, outdir=run_outdir)
        payloads.append(payload)
        run_manifest.append(
            {
                "strategy": strategy,
                "factor_names": list(spec["factor_names"]),
                "factor_weights": list(spec["factor_weights"]),
                "notes": str(spec["notes"]),
                "backtest_outdir": str(run_outdir),
                "summary_path": str(Path(run_outdir) / "summary.json"),
                "annual_turnover_from_summary": extract_annual_turnover(summary=summary, outdir=run_outdir),
            }
        )
    return payloads, run_manifest


def _average_overlap_matrix(holdings_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ref_index = next(iter(holdings_map.values())).index
    rb = rebalance_mask(pd.DatetimeIndex(ref_index), REBALANCE)
    rb_dates = pd.DatetimeIndex(ref_index[rb]).sort_values()
    names = list(holdings_map.keys())
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for left in names:
        h_left = holdings_map[left].reindex(index=rb_dates).fillna(0.0)
        for right in names:
            h_right = holdings_map[right].reindex(index=rb_dates).fillna(0.0)
            vals: list[float] = []
            for dt in rb_dates:
                left_set = set(h_left.columns[h_left.loc[dt].to_numpy(dtype=float) > 0.0])
                right_set = set(h_right.columns[h_right.loc[dt].to_numpy(dtype=float) > 0.0])
                vals.append(len(left_set.intersection(right_set)) / float(TOP_N))
            matrix.loc[left, right] = float(np.mean(vals)) if vals else float("nan")
    return matrix


def _holdings_summary(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        holdings = payload.get("holdings")
        if holdings is None:
            continue
        counts = (holdings.fillna(0.0).astype(float) > 0.0).sum(axis=1)
        rb = rebalance_mask(pd.DatetimeIndex(holdings.index), REBALANCE)
        rb_counts = counts.loc[rb]
        rows.append(
            {
                "Strategy": payload["name"],
                "HoldingsAvg": float(counts.mean()) if not counts.empty else float("nan"),
                "HoldingsMin": int(counts.min()) if not counts.empty else 0,
                "HoldingsMax": int(counts.max()) if not counts.empty else 0,
                "HoldingsOnRebalanceAvg": float(rb_counts.mean()) if not rb_counts.empty else float("nan"),
                "HoldingsOnRebalanceMin": int(rb_counts.min()) if not rb_counts.empty else 0,
                "HoldingsOnRebalanceMax": int(rb_counts.max()) if not rb_counts.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _relative_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    by_name = metrics_df.set_index("Strategy")
    comp = by_name.loc["composite_benchmark"]
    sleeves = by_name.loc["sleeve_combined_equal_weight"]
    return pd.DataFrame(
        [
            {
                "lhs": "composite_benchmark",
                "rhs": "sleeve_combined_equal_weight",
                "delta_CAGR": float(comp["CAGR"] - sleeves["CAGR"]),
                "delta_Sharpe": float(comp["Sharpe"] - sleeves["Sharpe"]),
                "delta_MaxDD": float(comp["MaxDD"] - sleeves["MaxDD"]),
                "delta_Turnover": float(comp["Turnover"] - sleeves["Turnover"]),
            }
        ]
    )


def _copy_latest(files: dict[str, Path], latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, latest_root / name)


def _format_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _format_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(_format_float))


def _print_summary(
    metrics_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> None:
    ordered = [
        "composite_benchmark",
        "sleeve_combined_equal_weight",
        "sleeve_momentum",
        "sleeve_reversal",
        "sleeve_low_vol",
        "sleeve_gross_profitability",
    ]
    metric_cols = ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Turnover", "HitRate", "TotalReturn"]
    table = metrics_df.set_index("Strategy").loc[ordered].reset_index()[metric_cols].copy()
    for col in metric_cols[1:]:
        table[col] = table[col].map(_format_float)
    print("COMPOSITE VS SLEEVES")
    print("--------------------")
    print(table.to_string(index=False))
    print("")
    print("SLEEVE RETURN CORRELATIONS")
    print("-------------------------")
    print(_format_df(corr_df).to_string())
    print("")
    print("SLEEVE HOLDINGS OVERLAP")
    print("----------------------")
    print(_format_df(overlap_df).to_string())
    print("")

    comp = metrics_df.set_index("Strategy").loc["composite_benchmark"]
    sleeves = metrics_df.set_index("Strategy").loc["sleeve_combined_equal_weight"]
    corr_offdiag = corr_df.where(~np.eye(len(corr_df), dtype=bool)).stack()
    overlap_offdiag = overlap_df.where(~np.eye(len(overlap_df), dtype=bool)).stack()
    sharpe_cmp = "beat" if comp["Sharpe"] > sleeves["Sharpe"] else "lagged"
    dd_cmp = "reduced" if sleeves["MaxDD"] > comp["MaxDD"] else "did not reduce"
    corr_note = (
        "low enough to justify allocator research"
        if not corr_offdiag.empty and float(corr_offdiag.mean()) < 0.6
        else "not especially low"
    )
    overlap_note = (
        "meaningful diversification"
        if not overlap_offdiag.empty and float(overlap_offdiag.mean()) < 0.5
        else "substantial name overlap"
    )
    print("INTERPRETATION")
    print("--------------")
    print(
        f"Composite {sharpe_cmp} sleeve_combined_equal_weight on Sharpe "
        f"({comp['Sharpe']:.4f} vs {sleeves['Sharpe']:.4f})."
    )
    print(
        f"Sleeve combination {dd_cmp} drawdown versus composite "
        f"({sleeves['MaxDD']:.4f} vs {comp['MaxDD']:.4f})."
    )
    print(f"Sleeve return correlations appear {corr_note}.")
    print(f"Holdings overlap suggests {overlap_note}.")


def _append_experiment_log(
    outdir: Path,
    metrics_df: pd.DataFrame,
    relative_df: pd.DataFrame,
) -> None:
    by_name = metrics_df.set_index("Strategy")
    comp = by_name.loc["composite_benchmark"]
    sleeves = by_name.loc["sleeve_combined_equal_weight"]
    delta = relative_df.iloc[0]
    entry = f"""

## {datetime.now(timezone.utc).strftime("%Y-%m-%d")} — Composite vs Independent Factor Sleeves

### Objective
Compare the default composite benchmark against an equal-weight combination of independent factor sleeves to isolate cross-factor agreement versus sleeve diversification.

### Command
```bash
PYTHONPATH=src python scripts/run_composite_vs_sleeves.py
```

### Setup
- Universe: `liquid_us`
- Universe mode: `dynamic`
- Dates: `{START}` to `{END}`
- Rebalance: `weekly`
- Top N: `50`
- Weighting: `equal`
- Costs: `10 bps`
- Factors: `momentum_12_1`, `reversal_1m`, `low_vol_20`, `gross_profitability`
- Composite: equal factor weights, existing normalized linear aggregation
- Sleeve combination: fixed 25% return-level combination of the four single-factor sleeves

### Key Results
- Composite CAGR / Sharpe / MaxDD / Turnover: `{comp['CAGR']:.4f}` / `{comp['Sharpe']:.4f}` / `{comp['MaxDD']:.4f}` / `{comp['Turnover']:.4f}`
- Sleeve-combo CAGR / Sharpe / MaxDD / Turnover: `{sleeves['CAGR']:.4f}` / `{sleeves['Sharpe']:.4f}` / `{sleeves['MaxDD']:.4f}` / `{sleeves['Turnover']:.4f}`
- Delta composite minus sleeve-combo:
  - CAGR: `{delta['delta_CAGR']:.4f}`
  - Sharpe: `{delta['delta_Sharpe']:.4f}`
  - MaxDD: `{delta['delta_MaxDD']:.4f}`
  - Turnover: `{delta['delta_Turnover']:.4f}`

### Initial Interpretation
- Composite {'outperformed' if comp['Sharpe'] > sleeves['Sharpe'] else 'underperformed'} the sleeve combination on Sharpe.
- Sleeve combination {'improved' if sleeves['MaxDD'] > comp['MaxDD'] else 'did not improve'} drawdown relative to the composite.
- Results bundle: `{outdir}`
"""
    with DOC_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n")


def main() -> None:
    global START, END, MAX_TICKERS, RESULTS_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max_tickers", type=int, default=MAX_TICKERS)
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--skip_log", action="store_true")
    args = parser.parse_args()

    START = str(args.start)
    END = str(args.end)
    MAX_TICKERS = int(args.max_tickers)
    RESULTS_ROOT = Path(str(args.results_root))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = RESULTS_ROOT / ts
    outdir.mkdir(parents=True, exist_ok=True)
    run_cache: dict[str, Any] = {}

    payloads: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    composite_spec = next(spec for spec in STRATEGY_SPECS if str(spec["strategy"]) == "composite_benchmark")
    print("Running composite_benchmark...")
    composite_summary, composite_outdir = run_backtest(**_run_config(composite_spec), run_cache=run_cache)
    payloads.append(
        _strategy_payload(
            name="composite_benchmark",
            summary=composite_summary,
            outdir=composite_outdir,
        )
    )
    run_manifest.append(
        {
            "strategy": "composite_benchmark",
            "factor_names": list(composite_spec["factor_names"]),
            "factor_weights": list(composite_spec["factor_weights"]),
            "notes": str(composite_spec["notes"]),
            "backtest_outdir": str(composite_outdir),
            "summary_path": str(Path(composite_outdir) / "summary.json"),
            "annual_turnover_from_summary": extract_annual_turnover(
                summary=composite_summary,
                outdir=composite_outdir,
            ),
        }
    )

    sleeve_payloads, sleeve_manifest = run_sleeve_backtests(run_cache=run_cache)
    payloads.extend(sleeve_payloads)
    run_manifest.extend(sleeve_manifest)

    combined_payload = _combined_sleeve_payload(sleeve_payloads)
    payloads.append(combined_payload)

    metrics_rows: list[dict[str, Any]] = []
    for payload in payloads:
        turnover_override = None
        if payload["outdir"] is not None:
            turnover_override = extract_annual_turnover(
                summary=payload["summary"],
                outdir=payload["outdir"],
            )
        metrics_rows.append(
            _portfolio_metrics(
                name=payload["name"],
                daily_return=payload["daily_return"],
                turnover=payload["daily_turnover"],
                annual_turnover_override=turnover_override,
            )
        )
    metrics_df = pd.DataFrame(metrics_rows)

    daily_returns_df = pd.concat([p["daily_return"] for p in payloads], axis=1).sort_index().fillna(0.0)
    daily_turnover_df = (
        pd.concat([p["daily_turnover"] for p in payloads], axis=1).sort_index().fillna(0.0)
    )
    sleeve_corr_df = daily_returns_df[SLEEVE_NAMES].corr()
    sleeve_overlap_df = _average_overlap_matrix(
        {name: next(p["holdings"] for p in sleeve_payloads if p["name"] == name) for name in SLEEVE_NAMES}
    )
    relative_df = _relative_comparison(metrics_df)
    holdings_summary_df = _holdings_summary(payloads)

    metrics_path = outdir / "metrics.csv"
    corr_path = outdir / "sleeve_correlations.csv"
    overlap_path = outdir / "sleeve_overlap_matrix.csv"
    daily_returns_path = outdir / "daily_returns.csv"
    daily_turnover_path = outdir / "daily_turnover.csv"
    holdings_summary_path = outdir / "holdings_summary.csv"
    relative_path = outdir / "relative_comparison.csv"
    manifest_path = outdir / "manifest.json"

    metrics_df.to_csv(metrics_path, index=False, float_format="%.10g")
    sleeve_corr_df.to_csv(corr_path, float_format="%.10g")
    sleeve_overlap_df.to_csv(overlap_path, float_format="%.10g")
    daily_returns_df.to_csv(daily_returns_path, index_label="date", float_format="%.10g")
    daily_turnover_df.to_csv(daily_turnover_path, index_label="date", float_format="%.10g")
    holdings_summary_df.to_csv(holdings_summary_path, index=False, float_format="%.10g")
    relative_df.to_csv(relative_path, index=False, float_format="%.10g")

    manifest = {
        "run_timestamp_utc": ts,
        "results_dir": str(outdir),
        "date_range": {"start": START, "end": END},
        "universe": UNIVERSE,
        "universe_mode": UNIVERSE_MODE,
        "top_n": TOP_N,
        "rebalance": REBALANCE,
        "weighting": WEIGHTING,
        "costs_bps": COSTS_BPS,
        "max_tickers": MAX_TICKERS,
        "factor_list": FACTORS,
        "benchmark_factor_weights": [0.25, 0.25, 0.25, 0.25],
        "notes": [
            "sleeve_combined_equal_weight is a return-level combination of the four single-factor sleeves.",
            "All sleeves reuse the default run_backtest normalized factor pipeline and causal execution behavior.",
        ],
        "runs": run_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _copy_latest(
        files={
            "metrics.csv": metrics_path,
            "sleeve_correlations.csv": corr_path,
            "sleeve_overlap_matrix.csv": overlap_path,
            "daily_returns.csv": daily_returns_path,
            "daily_turnover.csv": daily_turnover_path,
            "holdings_summary.csv": holdings_summary_path,
            "relative_comparison.csv": relative_path,
            "manifest.json": manifest_path,
        },
        latest_root=RESULTS_ROOT / "latest",
    )

    if not bool(args.skip_log):
        _append_experiment_log(outdir=outdir, metrics_df=metrics_df, relative_df=relative_df)
    _print_summary(metrics_df=metrics_df, corr_df=sleeve_corr_df, overlap_df=sleeve_overlap_df)
    print("")
    print(f"Saved results: {outdir}")


if __name__ == "__main__":
    main()
