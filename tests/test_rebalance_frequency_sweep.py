from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from quant_lab.strategies.topn import rebalance_mask


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_rebalance_frequency_sweep.py"
    spec = importlib.util.spec_from_file_location("run_rebalance_frequency_sweep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_biweekly_mask_selects_alternating_weeks() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    weekly = rebalance_mask(idx, "weekly")
    biweekly = rebalance_mask(idx, "biweekly")

    assert int(biweekly.sum()) > 0
    assert int(biweekly.sum()) < int(weekly.sum())
    assert biweekly.loc[weekly[weekly].index].sum() == biweekly.sum()


def test_build_run_config_and_pivot() -> None:
    mod = _load_module()
    cfg = mod.build_run_config("liquid_us", "biweekly")
    assert cfg["rebalance"] == "biweekly"
    assert cfg["factor_weights"] == [0.25, 0.25, 0.25, 0.25]
    assert cfg["universe_mode"] == "dynamic"

    df = pd.DataFrame(
        [
            {"universe": "sp500", "rebalance_frequency": "weekly", "Sharpe": 0.68},
            {"universe": "liquid_us", "rebalance_frequency": "weekly", "Sharpe": 0.79},
            {"universe": "sp500", "rebalance_frequency": "biweekly", "Sharpe": 0.70},
            {"universe": "liquid_us", "rebalance_frequency": "biweekly", "Sharpe": 0.77},
            {"universe": "sp500", "rebalance_frequency": "monthly", "Sharpe": 0.66},
            {"universe": "liquid_us", "rebalance_frequency": "monthly", "Sharpe": 0.75},
        ]
    )
    out = mod.build_sharpe_summary(df)
    assert out.columns.tolist() == ["rebalance_frequency", "liquid_us", "sp500"]
    assert out["rebalance_frequency"].tolist() == ["weekly", "biweekly", "monthly"]
