from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_subperiod_robustness.py"
    spec = importlib.util.spec_from_file_location("run_subperiod_robustness", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_formatters_and_summary_pivot() -> None:
    mod = _load_module()
    assert mod.format_weight_set([0.25, 0.25, 0.25, 0.25]) == "0.25,0.25,0.25,0.25"
    assert mod.format_period("2005-01-01", "2010-12-31") == "2005-01-01→2010-12-31"

    df = pd.DataFrame(
        [
            {
                "universe": "sp500",
                "period": "2005-01-01→2010-12-31",
                "weight_set": "0.25,0.25,0.25,0.25",
                "Sharpe": 0.6,
            },
            {
                "universe": "sp500",
                "period": "2005-01-01→2010-12-31",
                "weight_set": "0.30,0.20,0.10,0.40",
                "Sharpe": 0.7,
            },
            {
                "universe": "liquid_us",
                "period": "2005-01-01→2010-12-31",
                "weight_set": "0.25,0.25,0.25,0.25",
                "Sharpe": 0.8,
            },
            {
                "universe": "liquid_us",
                "period": "2005-01-01→2010-12-31",
                "weight_set": "0.30,0.20,0.10,0.40",
                "Sharpe": 0.9,
            },
        ]
    )
    out = mod.build_sharpe_summary(df)
    assert out.columns.tolist() == [
        "universe",
        "period",
        "0.25,0.25,0.25,0.25",
        "0.30,0.20,0.10,0.40",
    ]


def test_build_run_config_sets_window_and_universe_mode() -> None:
    mod = _load_module()
    sp500_cfg = mod.build_run_config("sp500", [0.25, 0.25, 0.25, 0.25], "2005-01-01", "2010-12-31")
    liquid_cfg = mod.build_run_config("liquid_us", [0.20, 0.30, 0.10, 0.40], "2020-01-01", "2024-12-31")

    assert sp500_cfg["top_n"] == 50
    assert sp500_cfg["start"] == "2005-01-01"
    assert sp500_cfg["end"] == "2010-12-31"
    assert "universe_mode" not in sp500_cfg

    assert liquid_cfg["start"] == "2020-01-01"
    assert liquid_cfg["end"] == "2024-12-31"
    assert liquid_cfg["universe_mode"] == "dynamic"
