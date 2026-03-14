from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_factor_weight_sweep.py"
    spec = importlib.util.spec_from_file_location("run_factor_weight_sweep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_format_weight_set_and_pivot() -> None:
    mod = _load_module()
    assert mod.format_weight_set([0.2, 0.3, 0.1, 0.4]) == "0.20,0.30,0.10,0.40"

    df = pd.DataFrame(
        [
            {"universe": "sp500", "weight_set": "0.25,0.25,0.25,0.25", "Sharpe": 0.68},
            {"universe": "liquid_us", "weight_set": "0.25,0.25,0.25,0.25", "Sharpe": 0.79},
            {"universe": "sp500", "weight_set": "0.20,0.30,0.10,0.40", "Sharpe": 0.70},
            {"universe": "liquid_us", "weight_set": "0.20,0.30,0.10,0.40", "Sharpe": 0.81},
        ]
    )
    out = mod.build_sharpe_summary(df)
    assert out.columns.tolist() == ["weight_set", "liquid_us", "sp500"]
    assert out.to_dict(orient="records") == [
        {"weight_set": "0.20,0.30,0.10,0.40", "liquid_us": 0.81, "sp500": 0.70},
        {"weight_set": "0.25,0.25,0.25,0.25", "liquid_us": 0.79, "sp500": 0.68},
    ]


def test_build_run_config_sets_fixed_top_n_and_universe_mode() -> None:
    mod = _load_module()
    sp500_cfg = mod.build_run_config("sp500", [0.25, 0.25, 0.25, 0.25])
    liquid_cfg = mod.build_run_config("liquid_us", [0.20, 0.30, 0.10, 0.40])

    assert sp500_cfg["top_n"] == 50
    assert sp500_cfg["factor_names"] == [
        "momentum_12_1",
        "reversal_1m",
        "low_vol_20",
        "gross_profitability",
    ]
    assert sp500_cfg["factor_weights"] == [0.25, 0.25, 0.25, 0.25]
    assert "universe_mode" not in sp500_cfg

    assert liquid_cfg["factor_weights"] == [0.2, 0.3, 0.1, 0.4]
    assert liquid_cfg["universe_mode"] == "dynamic"
