from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_portfolio_construction_experiment.py"
    spec = importlib.util.spec_from_file_location("run_portfolio_construction_experiment", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_run_config_sets_expected_weighting_variants() -> None:
    mod = _load_module()
    args = mod._parse_args.__globals__["argparse"].Namespace(
        start="2010-01-01",
        end="2024-12-31",
        output_dir="results/portfolio_construction_experiment",
        fundamentals_path="data/fundamentals/fundamentals_fmp.parquet",
    )

    equal_cfg = mod.build_run_config(args=args, strategy="equal_weight")
    inv_vol_cfg = mod.build_run_config(args=args, strategy="inv_vol_weight")
    capped_cfg = mod.build_run_config(args=args, strategy="capped_inv_vol_weight")

    assert equal_cfg["factor_names"] == ["gross_profitability", "reversal_1m"]
    assert equal_cfg["factor_weights"] == [0.7, 0.3]
    assert equal_cfg["weighting"] == "equal"
    assert equal_cfg["max_weight"] == 0.15

    assert inv_vol_cfg["weighting"] == "inv_vol"
    assert inv_vol_cfg["max_weight"] == 0.15

    assert capped_cfg["weighting"] == "inv_vol"
    assert capped_cfg["max_weight"] == 0.05


def test_build_summary_frame_identifies_best_variant_vs_equal_weight() -> None:
    mod = _load_module()
    df = pd.DataFrame(
        [
            {"Strategy": "equal_weight", "CAGR": 0.12, "Vol": 0.18, "Sharpe": 0.80, "MaxDD": -0.40, "Turnover": 9.5},
            {"Strategy": "inv_vol_weight", "CAGR": 0.13, "Vol": 0.17, "Sharpe": 0.86, "MaxDD": -0.36, "Turnover": 9.2},
            {
                "Strategy": "capped_inv_vol_weight",
                "CAGR": 0.125,
                "Vol": 0.17,
                "Sharpe": 0.84,
                "MaxDD": -0.37,
                "Turnover": 9.0,
            },
        ]
    )

    out = mod.build_summary_frame(df)

    assert out.loc[0, "best_strategy"] == "inv_vol_weight"
    assert bool(out.loc[0, "improves_over_equal_weight"]) is True
    assert float(out.loc[0, "sharpe_delta_vs_equal_weight"]) == pytest.approx(0.06)
