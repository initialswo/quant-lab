"""Helpers for collecting sweep-level metrics from run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def extract_annual_turnover(summary: dict[str, Any], outdir: str | Path) -> float:
    """Return annual turnover from summary when present, else derive from equity artifact."""
    val = summary.get("AnnualTurnover")
    if val is not None:
        try:
            out = float(val)
            if pd.notna(out):
                return out
        except Exception:
            pass

    equity_path = Path(outdir) / "equity.csv"
    if not equity_path.exists():
        return float("nan")
    equity = pd.read_csv(equity_path, usecols=["Turnover"])
    if "Turnover" not in equity.columns or equity.empty:
        return float("nan")
    turnover = pd.to_numeric(equity["Turnover"], errors="coerce")
    if turnover.notna().sum() == 0:
        return float("nan")
    return float(turnover.mean() * 252.0)
