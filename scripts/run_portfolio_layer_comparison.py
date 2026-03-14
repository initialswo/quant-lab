"""Run a fixed four-way portfolio-layer comparison experiment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.runner import run_backtest


FULL_START = "2005-01-01"
FULL_END = "2024-12-31"
SUBPERIODS: list[tuple[str, str]] = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]

BASE_CONFIG: dict[str, object] = {
    "start": FULL_START,
    "end": FULL_END,
    "factor_name": ["momentum_12_1", "reversal_1m", "low_vol_20"],
    "factor_names": ["momentum_12_1", "reversal_1m", "low_vol_20"],
    "factor_weights": [0.5, 0.3, 0.2],
    "top_n": 50,
    "rebalance": "monthly",
    "max_tickers": 2000,
    "data_source": "parquet",
    "data_cache_dir": "data/equities",
    "costs_bps": 10.0,
}

VARIANTS: list[dict[str, object]] = [
    {
        "Variant": "Baseline",
        "target_vol": 0.0,
        "regime_filter": False,
        "regime_benchmark": "SPY",
    },
    {
        "Variant": "VolTarget",
        "target_vol": 0.15,
        "port_vol_lookback": 20,
        "max_leverage": 1.5,
        "regime_filter": False,
        "regime_benchmark": "SPY",
    },
    {
        "Variant": "Regime",
        "target_vol": 0.0,
        "regime_filter": True,
        "regime_benchmark": "SPY",
    },
    {
        "Variant": "VolTarget+Regime",
        "target_vol": 0.15,
        "port_vol_lookback": 20,
        "max_leverage": 1.5,
        "regime_filter": True,
        "regime_benchmark": "SPY",
    },
]


def _fmt_float(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.4f}"


def _period_label(start: str, end: str) -> str:
    return f"{start[:4]}-{end[:4]}"


def _run_once(start: str, end: str, variant_cfg: dict[str, object]) -> dict:
    cfg = dict(BASE_CONFIG)
    cfg.update(variant_cfg)
    cfg["start"] = start
    cfg["end"] = end
    cfg.pop("Variant", None)

    summary, _ = run_backtest(**cfg)
    if str(summary.get("DataSource", "")).lower() != "parquet":
        raise ValueError(f"Unexpected DataSource={summary.get('DataSource')}; expected parquet.")
    if int(summary.get("DataFetchedOrRefreshed", 0)) != 0:
        raise ValueError(
            "Backtest unexpectedly refreshed/fetched external data: "
            f"DataFetchedOrRefreshed={summary.get('DataFetchedOrRefreshed')}"
        )
    return summary


def _full_row(variant: str, summary: dict) -> dict:
    return {
        "Variant": variant,
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "TickersUsed": int(summary.get("TickersUsed", 0)),
        "MissingTickersCount": int(summary.get("MissingTickersCount", 0)),
        "LeverageAvg": float(summary.get("LeverageAvg", float("nan"))),
        "LeverageMax": float(summary.get("LeverageMax", float("nan"))),
        "RegimeFilter": bool(summary.get("RegimeFilter", False)),
        "RegimeBenchmark": str(summary.get("RegimeBenchmark", "SPY")),
        "regime_pct_bull": float(summary.get("regime_pct_bull", float("nan"))),
        "regime_pct_bear_or_volatile": float(
            summary.get("regime_pct_bear_or_volatile", float("nan"))
        ),
        "TargetVol": float(summary.get("TargetVol", float("nan"))),
        "PortVolLookback": int(summary.get("PortVolLookback", 20)),
        "MaxLeverage": float(summary.get("MaxLeverage", 1.0)),
        "SanityWarningsCount": int(summary.get("SanityWarningsCount", 0)),
        "Outdir": str(summary.get("Outdir", "")),
    }


def _subperiod_row(variant: str, start: str, end: str, summary: dict) -> dict:
    return {
        "Variant": variant,
        "Period": _period_label(start, end),
        "Start": start,
        "End": end,
        "Sharpe": float(summary.get("Sharpe", float("nan"))),
        "CAGR": float(summary.get("CAGR", float("nan"))),
        "Vol": float(summary.get("Vol", float("nan"))),
        "MaxDD": float(summary.get("MaxDD", float("nan"))),
        "LeverageAvg": float(summary.get("LeverageAvg", float("nan"))),
        "LeverageMax": float(summary.get("LeverageMax", float("nan"))),
        "regime_pct_bull": float(summary.get("regime_pct_bull", float("nan"))),
        "regime_pct_bear_or_volatile": float(
            summary.get("regime_pct_bear_or_volatile", float("nan"))
        ),
        "SanityWarningsCount": int(summary.get("SanityWarningsCount", 0)),
        "Outdir": str(summary.get("Outdir", "")),
    }


def _stability_table(subperiod_df: pd.DataFrame) -> pd.DataFrame:
    grp = subperiod_df.groupby("Variant", dropna=False)["Sharpe"]
    stats = grp.agg(["mean", "std", "min", "max"]).reset_index()
    stats = stats.rename(
        columns={
            "mean": "MeanSubSharpe",
            "std": "StdSubSharpe",
            "min": "MinSubSharpe",
            "max": "MaxSubSharpe",
        }
    )
    return stats


def main() -> None:
    full_rows: list[dict] = []
    subperiod_rows: list[dict] = []

    for variant_cfg in VARIANTS:
        variant = str(variant_cfg["Variant"])
        full_summary = _run_once(start=FULL_START, end=FULL_END, variant_cfg=variant_cfg)
        full_rows.append(_full_row(variant=variant, summary=full_summary))

        for start, end in SUBPERIODS:
            sub_summary = _run_once(start=start, end=end, variant_cfg=variant_cfg)
            subperiod_rows.append(
                _subperiod_row(variant=variant, start=start, end=end, summary=sub_summary)
            )

    full_df = pd.DataFrame(full_rows)
    subperiod_df = pd.DataFrame(subperiod_rows)
    stability_df = _stability_table(subperiod_df=subperiod_df)

    table_full = full_df[
        [
            "Variant",
            "CAGR",
            "Vol",
            "Sharpe",
            "MaxDD",
            "LeverageAvg",
            "LeverageMax",
            "regime_pct_bull",
            "regime_pct_bear_or_volatile",
        ]
    ].copy()
    table_full.columns = [
        "Variant",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "LevAvg",
        "LevMax",
        "BullPct",
        "BearPct",
    ]
    for col in ["CAGR", "Vol", "Sharpe", "MaxDD", "LevAvg", "LevMax", "BullPct", "BearPct"]:
        table_full[col] = table_full[col].map(_fmt_float)

    table_stability = stability_df[
        ["Variant", "MeanSubSharpe", "StdSubSharpe", "MinSubSharpe", "MaxSubSharpe"]
    ].copy()
    for col in ["MeanSubSharpe", "StdSubSharpe", "MinSubSharpe", "MaxSubSharpe"]:
        table_stability[col] = table_stability[col].map(_fmt_float)

    print("FULL-PERIOD PORTFOLIO LAYER COMPARISON")
    print("--------------------------------------")
    print(table_full.to_string(index=False))
    print("")
    print("SUBPERIOD STABILITY COMPARISON")
    print("------------------------------")
    print(table_stability.to_string(index=False))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / f"portfolio_layer_comparison_{ts}.csv"
    full_json = outdir / f"portfolio_layer_comparison_{ts}.json"
    sub_csv = outdir / f"portfolio_layer_subperiods_{ts}.csv"

    full_df.to_csv(full_csv, index=False, float_format="%.10g")
    subperiod_df.to_csv(sub_csv, index=False, float_format="%.10g")
    full_json.write_text(
        json.dumps(
            {
                "config": {
                    "full_period": {"start": FULL_START, "end": FULL_END},
                    "subperiods": [{"start": s, "end": e} for s, e in SUBPERIODS],
                    "base": BASE_CONFIG,
                    "variants": VARIANTS,
                },
                "full_period_results": full_rows,
                "subperiod_results": subperiod_rows,
                "subperiod_stability": stability_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"Saved CSV: {full_csv}")
    print(f"Saved JSON: {full_json}")
    print(f"Saved Subperiod CSV: {sub_csv}")


if __name__ == "__main__":
    main()
