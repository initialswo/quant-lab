from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_portfolio_breadth_sweep.py"
    spec = importlib.util.spec_from_file_location("run_portfolio_breadth_sweep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_to_result_row_uses_expected_summary_fields(tmp_path: Path) -> None:
    mod = _load_module()
    summary = {
        "CAGR": 0.11,
        "Vol": 0.17,
        "Sharpe": 0.68,
        "MaxDD": -0.41,
        "EligibleOnRebalanceMedian": 172.5,
        "FinalTradableOnRebalanceMedian": 50.0,
        "RebalanceSkippedCount": 5,
    }
    outdir = tmp_path
    (outdir / "equity.csv").write_text("Turnover\n0.0\n0.5\n1.0\n", encoding="utf-8")

    row = mod.summary_to_result_row(summary=summary, outdir=str(outdir), universe="sp500", top_n=50)

    assert row == {
        "universe": "sp500",
        "top_n": 50,
        "CAGR": 0.11,
        "Vol": 0.17,
        "Sharpe": 0.68,
        "MaxDD": -0.41,
        "AnnualTurnover": 126.0,
        "EligMedian": 172.5,
        "TradableMedian": 50.0,
        "RebalanceSkipped": 5,
    }


def test_build_sharpe_summary_pivots_top_n_by_universe() -> None:
    mod = _load_module()
    df = pd.DataFrame(
        [
            {"universe": "sp500", "top_n": 20, "Sharpe": 0.6},
            {"universe": "liquid_us", "top_n": 20, "Sharpe": 0.8},
            {"universe": "sp500", "top_n": 50, "Sharpe": 0.7},
            {"universe": "liquid_us", "top_n": 50, "Sharpe": 0.9},
        ]
    )

    out = mod.build_sharpe_summary(df)

    assert out.columns.tolist() == ["top_n", "liquid_us", "sp500"]
    assert out.to_dict(orient="records") == [
        {"top_n": 20, "liquid_us": 0.8, "sp500": 0.6},
        {"top_n": 50, "liquid_us": 0.9, "sp500": 0.7},
    ]


def test_build_run_config_only_varies_top_n_and_universe_mode() -> None:
    mod = _load_module()

    sp500_cfg = mod.build_run_config("sp500", 20)
    liquid_cfg = mod.build_run_config("liquid_us", 200)

    assert sp500_cfg["factor_names"] == [
        "momentum_12_1",
        "reversal_1m",
        "low_vol_20",
        "gross_profitability",
    ]
    assert sp500_cfg["factor_weights"] == [0.25, 0.25, 0.25, 0.25]
    assert sp500_cfg["rebalance"] == "weekly"
    assert sp500_cfg["costs_bps"] == 10.0
    assert sp500_cfg["top_n"] == 20
    assert sp500_cfg["universe"] == "sp500"
    assert "universe_mode" not in sp500_cfg

    assert liquid_cfg["top_n"] == 200
    assert liquid_cfg["universe"] == "liquid_us"
    assert liquid_cfg["universe_mode"] == "dynamic"
