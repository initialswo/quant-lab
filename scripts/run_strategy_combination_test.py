"""Analyze diversification between lead factor benchmark and sector rotation sleeve."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lab.engine.metrics import compute_metrics


def _find_latest_lead_backtest_summary() -> Path:
    root = Path("results")
    candidates: list[tuple[float, Path]] = []
    for p in root.glob("*/summary.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(d.get("FactorNames", "")) != "momentum_12_1;reversal_1m;low_vol_20;gross_profitability":
            continue
        if str(d.get("PortfolioMode", "composite")) != "composite":
            continue
        if str(d.get("Start", "")) != "2005-01-01" or str(d.get("End", "")) != "2024-12-31":
            continue
        if int(d.get("RankBuffer", -1)) != 20:
            continue
        if not bool(d.get("DynamicFactorWeights", False)):
            continue
        if float(d.get("TargetVol", 0.0)) != 0.14:
            continue
        if float(d.get("BearExposureScale", 0.0)) != 1.0:
            continue
        fn = str(d.get("FactorNeutralization", "none")).lower()
        if fn not in {"none", "(na)"}:
            continue
        candidates.append((p.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError("No matching lead backtest summary.json found in results/*")
    candidates.sort()
    return candidates[-1][1]


def _drawdown_series(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.astype(float).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def main() -> None:
    lead_summary = _find_latest_lead_backtest_summary()
    lead_dir = lead_summary.parent
    factor_path = lead_dir / "equity_curve.csv"

    sector_path = Path("results/sector_rotation_benchmark/sector_rotation_returns.csv")
    if not sector_path.exists():
        all_sector = sorted(Path("results/sector_rotation_benchmark").glob("*/sector_rotation_returns.csv"))
        if not all_sector:
            raise FileNotFoundError("No sector_rotation_returns.csv found")
        sector_path = all_sector[-1]

    factor = (
        pd.read_csv(factor_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "factor_return"})
        .set_index("date")
        .sort_index()
    )
    sector = (
        pd.read_csv(sector_path, parse_dates=["date"])[["date", "returns"]]
        .rename(columns={"returns": "sector_return"})
        .set_index("date")
        .sort_index()
    )
    aligned = factor.join(sector, how="inner").dropna().copy()

    corr = float(aligned["factor_return"].corr(aligned["sector_return"]))
    rolling_corr = aligned["factor_return"].rolling(252).corr(aligned["sector_return"])

    dd_factor = _drawdown_series(aligned["factor_return"])
    dd_sector = _drawdown_series(aligned["sector_return"])
    in_dd_factor = dd_factor < 0.0
    in_dd_sector = dd_sector < 0.0
    overlap_any = in_dd_factor | in_dd_sector
    overlap_both = in_dd_factor & in_dd_sector
    overlap_fraction_all_days = float(overlap_both.mean())
    overlap_fraction_when_any_dd = float(overlap_both.sum() / overlap_any.sum()) if overlap_any.any() else float("nan")

    portfolios = {
        "100% Factor": (1.0, 0.0),
        "100% Sector": (0.0, 1.0),
        "80/20": (0.8, 0.2),
        "70/30": (0.7, 0.3),
        "60/40": (0.6, 0.4),
    }

    combined = aligned.copy()
    rows: list[dict[str, float | str]] = []
    for name, (wf, ws) in portfolios.items():
        col = f"portfolio_{name.replace('%', 'pct').replace('/', '_').replace(' ', '')}"
        combined[col] = wf * combined["factor_return"] + ws * combined["sector_return"]
        m = compute_metrics(combined[col])
        rows.append(
            {
                "Portfolio": name,
                "CAGR": float(m["CAGR"]),
                "Vol": float(m["Vol"]),
                "Sharpe": float(m["Sharpe"]),
                "MaxDD": float(m["MaxDD"]),
            }
        )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results/strategy_combination_test") / ts
    outdir.mkdir(parents=True, exist_ok=True)

    combined_out = outdir / "combined_returns.csv"
    corr_out = outdir / "strategy_correlation.csv"
    summary_out = outdir / "portfolio_summary.csv"

    combined.assign(
        drawdown_factor=dd_factor,
        drawdown_sector=dd_sector,
        rolling_corr_252=rolling_corr,
        in_drawdown_factor=in_dd_factor.astype(int),
        in_drawdown_sector=in_dd_sector.astype(int),
        in_drawdown_both=overlap_both.astype(int),
    ).reset_index().to_csv(combined_out, index=False, float_format="%.10g")

    corr_df = pd.DataFrame(
        [
            {
                "factor_vs_sector_corr": corr,
                "rolling_corr_252_mean": float(rolling_corr.mean()),
                "rolling_corr_252_median": float(rolling_corr.median()),
                "rolling_corr_252_min": float(rolling_corr.min()),
                "rolling_corr_252_max": float(rolling_corr.max()),
                "drawdown_overlap_all_days": overlap_fraction_all_days,
                "drawdown_overlap_when_any_drawdown": overlap_fraction_when_any_dd,
                "lead_backtest_dir": str(lead_dir),
                "sector_returns_path": str(sector_path),
            }
        ]
    )
    corr_df.to_csv(corr_out, index=False, float_format="%.10g")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_out, index=False, float_format="%.10g")

    latest_dir = Path("results/strategy_combination_test")
    combined.assign(
        drawdown_factor=dd_factor,
        drawdown_sector=dd_sector,
        rolling_corr_252=rolling_corr,
        in_drawdown_factor=in_dd_factor.astype(int),
        in_drawdown_sector=in_dd_sector.astype(int),
        in_drawdown_both=overlap_both.astype(int),
    ).reset_index().to_csv(latest_dir / "combined_returns.csv", index=False, float_format="%.10g")
    corr_df.to_csv(latest_dir / "strategy_correlation.csv", index=False, float_format="%.10g")
    summary_df.to_csv(latest_dir / "portfolio_summary.csv", index=False, float_format="%.10g")

    print("STRATEGY COMBINATION TEST")
    print("-------------------------")
    print(corr_df.to_string(index=False))
    print("")
    print(summary_df.to_string(index=False))
    print("")
    print(f"Saved: {combined_out}")
    print(f"Saved: {corr_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
