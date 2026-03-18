from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_momentum_neutralized_baseline_test.py"
    spec = importlib.util.spec_from_file_location("run_momentum_neutralized_baseline_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_run_config_uses_canonical_baseline() -> None:
    mod = _load_module()
    args = mod._parse_args.__globals__["argparse"].Namespace(
        start="2010-01-01",
        end="2024-12-31",
        output_dir="results/momentum_neutralized_baseline_test",
        data_source="parquet",
        data_cache_dir="data/equities",
        fundamentals_path="data/fundamentals/fundamentals_fmp.parquet",
        max_tickers=2000,
    )

    cfg = mod.build_run_config(args)

    assert cfg["universe"] == "liquid_us"
    assert cfg["universe_mode"] == "dynamic"
    assert cfg["top_n"] == 75
    assert cfg["rebalance"] == "weekly"
    assert cfg["costs_bps"] == 10.0
    assert cfg["factor_names"] == ["gross_profitability", "reversal_1m"]
    assert cfg["factor_weights"] == [0.7, 0.3]
    assert cfg["factor_aggregation_method"] == "linear"
    assert cfg["save_artifacts"] is True


def test_neutralize_scores_against_momentum_removes_linear_component() -> None:
    mod = _load_module()
    idx = pd.DatetimeIndex(["2024-01-05"])
    momentum = pd.DataFrame([{"A": -1.0, "B": 0.0, "C": 1.0, "D": 2.0}], index=idx)
    scores = pd.DataFrame([{"A": -1.0, "B": 1.0, "C": 3.0, "D": 5.0}], index=idx)

    residual = mod.neutralize_scores_against_momentum(scores=scores, momentum=momentum)
    row = residual.loc[idx[0]]

    assert row.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-10)


def test_assess_dependency_flags_hidden_momentum_dependence() -> None:
    mod = _load_module()
    results_df = pd.DataFrame(
        [
            {"Strategy": "baseline", "CAGR": 0.12, "Vol": 0.18, "Sharpe": 0.80, "MaxDD": -0.40, "Turnover": 10.0},
            {
                "Strategy": "momentum_neutralized_baseline",
                "CAGR": 0.08,
                "Vol": 0.18,
                "Sharpe": 0.62,
                "MaxDD": -0.42,
                "Turnover": 9.8,
            },
        ]
    )
    exposure_df = pd.DataFrame(
        [
            {
                "Strategy": "baseline",
                "Mean Portfolio Momentum": 0.10,
                "Mean Universe Momentum": 0.04,
                "Mean Momentum Exposure": 0.06,
                "Percent Positive Exposure": 72.0,
            },
            {
                "Strategy": "momentum_neutralized_baseline",
                "Mean Portfolio Momentum": 0.05,
                "Mean Universe Momentum": 0.04,
                "Mean Momentum Exposure": 0.01,
                "Percent Positive Exposure": 48.0,
            },
        ]
    )

    out = mod.assess_dependency(results_df=results_df, exposure_summary_df=exposure_df)

    assert out["performance_materially_reduced"] is True
    assert out["momentum_exposure_materially_reduced"] is True
    assert "depends meaningfully" in out["dependency_assessment"]
