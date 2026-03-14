from __future__ import annotations

import math
from pathlib import Path

from quant_lab.research.sweep_metrics import extract_annual_turnover


def test_extract_annual_turnover_prefers_summary_value(tmp_path: Path) -> None:
    out = extract_annual_turnover({"AnnualTurnover": 12.5}, tmp_path)
    assert out == 12.5


def test_extract_annual_turnover_falls_back_to_equity_csv(tmp_path: Path) -> None:
    eq = tmp_path / "equity.csv"
    eq.write_text("Turnover\n0.0\n0.5\n1.0\n", encoding="utf-8")
    out = extract_annual_turnover({}, tmp_path)
    assert math.isclose(out, ((0.0 + 0.5 + 1.0) / 3.0) * 252.0)
