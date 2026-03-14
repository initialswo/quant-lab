"""Run reusable cross-sectional diagnostics for a factor panel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from quant_lab.data.fundamentals import align_fundamentals_to_daily_panel, load_fundamentals_file
from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.data.universe_dynamic import compute_eligibility_matrix
from quant_lab.factors.registry import compute_factor
from quant_lab.research.cs_factor_diagnostics import (
    run_cross_sectional_factor_diagnostics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUBPERIODS: list[tuple[str, str, str]] = [
    ("2005-2009", "2005-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2024", "2020-01-01", "2024-12-31"),
]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run cross-sectional factor diagnostics")
    p.add_argument("--factor", default="gross_profitability")
    p.add_argument("--fundamentals-path", default="")
    p.add_argument("--price-path", default="data/equities/daily_ohlcv.parquet")
    p.add_argument("--data-root", default="data/equities")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--universe", default="sp500")
    p.add_argument("--max-tickers", type=int, default=2000)
    p.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="monthly")
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--fallback-lag-days", type=int, default=60)
    p.add_argument("--min-price", type=float, default=0.0)
    p.add_argument("--min-avg-dollar-volume", type=float, default=0.0)
    p.add_argument("--liquidity-lookback", type=int, default=20)
    p.add_argument("--universe-min-history-days", type=int, default=300)
    p.add_argument("--universe-valid-lookback", type=int, default=252)
    p.add_argument("--universe-min-valid-frac", type=float, default=0.98)
    p.add_argument("--min-obs-ic", type=int, default=5)
    p.add_argument("--min-overlap-corr", type=int, default=20)
    return p


def _load_prices(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_path = Path(str(args.price_path))
    if str(args.universe).lower() in {"", "all", "none"} and price_path.exists():
        frame = pd.read_parquet(price_path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        need = {"date", "ticker", "close", "volume"}
        missing = need.difference(frame.columns)
        if missing:
            raise ValueError(f"price parquet missing columns: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[
            (frame["date"] >= pd.Timestamp(str(args.start)))
            & (frame["date"] <= pd.Timestamp(str(args.end)))
        ].copy()
        frame = frame.sort_values(["date", "ticker"])
        if int(args.max_tickers) > 0:
            keep = sorted(frame["ticker"].astype(str).unique())[: int(args.max_tickers)]
            frame = frame.loc[frame["ticker"].isin(keep)]
        close = frame.pivot(index="date", columns="ticker", values="close").sort_index().astype(float)
        volume = frame.pivot(index="date", columns="ticker", values="volume").sort_index().astype(float)
        close.columns.name = None
        volume.columns.name = None
        return close, volume

    loader = load_ohlcv_for_research(
        start=str(args.start),
        end=str(args.end),
        universe=None if str(args.universe).lower() in {"", "all", "none"} else str(args.universe),
        max_tickers=int(args.max_tickers),
        store_root=str(args.data_root),
    )
    close = loader.panels.get("close", pd.DataFrame()).astype(float)
    volume = loader.panels.get("volume", pd.DataFrame()).astype(float)
    return close, volume


def _build_factor(
    factor_name: str,
    close: pd.DataFrame,
    fundamentals_path: str,
    fallback_lag_days: int,
) -> pd.DataFrame:
    f = str(factor_name).strip()
    if f in {"gross_profitability", "earnings_yield", "roa"}:
        if not str(fundamentals_path).strip():
            raise ValueError("--fundamentals-path is required for fundamentals-based factors")
        fundamentals = load_fundamentals_file(
            path=str(fundamentals_path),
            fallback_lag_days=int(fallback_lag_days),
        )
        aligned = align_fundamentals_to_daily_panel(
            fundamentals=fundamentals,
            price_index=pd.DatetimeIndex(close.index),
            price_columns=close.columns,
        )
        return compute_factor(f, close, fundamentals_aligned=aligned)
    return compute_factor(f, close)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    return obj


def _plot_cumulative_ic(ic_by_date: pd.DataFrame, out_path: Path) -> bool:
    if ic_by_date.empty:
        return False
    ic_col = "rank_ic" if "rank_ic" in ic_by_date.columns else ("IC" if "IC" in ic_by_date.columns else None)
    if ic_col is None:
        return False

    frame = ic_by_date.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[ic_col] = pd.to_numeric(frame[ic_col], errors="coerce")
    frame = frame.dropna(subset=["Date", ic_col]).sort_values("Date")
    if frame.empty:
        return False

    frame["cumulative_ic"] = frame[ic_col].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frame["Date"], frame["cumulative_ic"], linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Cumulative Rank IC")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative sum of rank_ic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_quantile_mean_returns(q_summary: pd.DataFrame, out_path: Path) -> bool:
    if q_summary.empty or "Period" not in q_summary.columns:
        return False
    overall = q_summary.loc[q_summary["Period"] == "overall"]
    if overall.empty:
        return False
    row = overall.iloc[0]
    q_cols = [c for c in q_summary.columns if str(c).startswith("Q")]
    if not q_cols:
        return False

    values = pd.to_numeric(pd.Series([row[c] for c in q_cols], index=q_cols), errors="coerce").dropna()
    if values.empty:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(values.index.tolist(), values.values.tolist())
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Average Forward Return by Quantile")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Mean forward return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    args = _parser().parse_args()
    close, volume = _load_prices(args)
    if close.empty:
        raise ValueError("Empty close panel after loading/filtering.")

    factor = _build_factor(
        factor_name=str(args.factor),
        close=close,
        fundamentals_path=str(args.fundamentals_path),
        fallback_lag_days=int(args.fallback_lag_days),
    )

    elig = compute_eligibility_matrix(
        close=close,
        min_history_days=int(args.universe_min_history_days),
        valid_lookback=int(args.universe_valid_lookback),
        min_valid_frac=float(args.universe_min_valid_frac),
        min_price=float(args.min_price),
        volume=volume,
        min_avg_dollar_volume=float(args.min_avg_dollar_volume),
        liquidity_lookback=int(args.liquidity_lookback),
    )

    peer_factors: dict[str, pd.DataFrame] = {
        "momentum_12_1": compute_factor("momentum_12_1", close),
        "reversal_1m": compute_factor("reversal_1m", close),
        "low_vol_20": compute_factor("low_vol_20", close),
    }
    if str(args.factor).strip() != "gross_profitability" and str(args.fundamentals_path).strip():
        fundamentals_peer = load_fundamentals_file(
            path=str(args.fundamentals_path),
            fallback_lag_days=int(args.fallback_lag_days),
        )
        aligned_peer = align_fundamentals_to_daily_panel(
            fundamentals=fundamentals_peer,
            price_index=pd.DatetimeIndex(close.index),
            price_columns=close.columns,
        )
        peer_factors["gross_profitability"] = compute_factor(
            "gross_profitability", close, fundamentals_aligned=aligned_peer
        )

    report = run_cross_sectional_factor_diagnostics(
        factor_scores=factor,
        close=close,
        eligibility_mask=elig,
        rebalance=str(args.rebalance),
        quantiles=int(args.quantiles),
        horizon=int(args.horizon),
        subperiods=SUBPERIODS,
        peer_factors=peer_factors,
        min_obs_ic=int(args.min_obs_ic),
        min_overlap_corr=int(args.min_overlap_corr),
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"factor_diagnostics_{str(args.factor)}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    coverage_by_date = pd.DataFrame(report["coverage_by_date"]).reset_index().rename(columns={"index": "Date"})
    ic_by_date = pd.DataFrame(report["ic_by_date"]).reset_index().rename(columns={"index": "Date"})
    q_by_date = pd.DataFrame(report["quantile_returns_by_date"]).reset_index().rename(columns={"index": "Date"})
    q_summary = pd.DataFrame(report["quantile_summary"]).copy()
    corr_summary = pd.DataFrame(report["factor_correlation_summary"]).copy()

    coverage_path = outdir / "coverage_by_date.csv"
    ic_path = outdir / "ic_by_date.csv"
    q_by_date_path = outdir / "quantile_returns_by_date.csv"
    q_summary_path = outdir / "quantile_summary.csv"
    corr_summary_path = outdir / "factor_correlation_summary.csv"
    summary_json_path = outdir / "diagnostics_summary.json"
    cumulative_ic_plot_path = outdir / "cumulative_ic.png"
    quantile_mean_returns_plot_path = outdir / "quantile_mean_returns.png"

    coverage_by_date.to_csv(coverage_path, index=False)
    ic_by_date.to_csv(ic_path, index=False)
    q_by_date.to_csv(q_by_date_path, index=False)
    q_summary.to_csv(q_summary_path, index=False)
    corr_summary.to_csv(corr_summary_path, index=False)

    summary_payload = {
        "factor": str(args.factor),
        "start": str(args.start),
        "end": str(args.end),
        "rebalance": str(args.rebalance),
        "horizon": int(args.horizon),
        "quantiles": int(args.quantiles),
        "config": report.get("config", {}),
        "coverage_summary": report.get("coverage_summary", {}),
        "ic_summary": report.get("ic_summary", {}),
        "quantile_summary": q_summary.to_dict(orient="records"),
        "factor_correlation_summary": corr_summary.to_dict(orient="records"),
        "outdir": str(outdir),
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(summary_payload), f, indent=2)

    wrote_cum_ic = _plot_cumulative_ic(ic_by_date=ic_by_date, out_path=cumulative_ic_plot_path)
    wrote_quantile_plot = _plot_quantile_mean_returns(q_summary=q_summary, out_path=quantile_mean_returns_plot_path)

    ic_overall = report.get("ic_summary", {}).get("overall", {})
    coverage_overall = report.get("coverage_summary", {}).get("overall", {})
    q_overall = q_summary.loc[q_summary["Period"] == "overall"]
    spread_mean = float(q_overall.iloc[0]["SpreadMean"]) if not q_overall.empty else float("nan")
    spread_sl = float(q_overall.iloc[0]["SpreadSharpeLike"]) if not q_overall.empty else float("nan")

    print("FACTOR DIAGNOSTICS SUMMARY")
    print("--------------------------")
    print(f"Factor: {args.factor}")
    print(f"Range: {args.start} to {args.end}")
    print(f"Rebalance: {args.rebalance} | Horizon: {args.horizon} | Quantiles: {args.quantiles}")
    print(
        "Coverage(valid names): "
        f"mean={coverage_overall.get('valid_names', {}).get('mean')} "
        f"median={coverage_overall.get('valid_names', {}).get('median')}"
    )
    print(
        "Coverage fraction: "
        f"mean={coverage_overall.get('coverage_fraction', {}).get('mean')} "
        f"median={coverage_overall.get('coverage_fraction', {}).get('median')}"
    )
    print(
        "Rank IC: "
        f"mean={ic_overall.get('mean_ic')} std={ic_overall.get('std_ic')} "
        f"IR={ic_overall.get('ic_ir')} hit={ic_overall.get('ic_hit_rate')} n={ic_overall.get('count')}"
    )
    print(f"Quantile spread: mean={spread_mean} sharpe_like={spread_sl}")
    if not corr_summary.empty:
        print("Factor correlations (avg/median):")
        for _, row in corr_summary.iterrows():
            print(
                f"  {row['Peer']}: avg={row['AverageCorrelation']} "
                f"median={row['MedianCorrelation']} n={row['Count']}"
            )

    print("\nArtifacts:")
    print(f"- {summary_json_path}")
    print(f"- {coverage_path}")
    print(f"- {ic_path}")
    print(f"- {q_by_date_path}")
    print(f"- {q_summary_path}")
    print(f"- {corr_summary_path}")
    if wrote_cum_ic:
        print(f"- {cumulative_ic_plot_path}")
    if wrote_quantile_plot:
        print(f"- {quantile_mean_returns_plot_path}")


if __name__ == "__main__":
    main()
