from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_momentum_exposure_diagnostic.py"
    spec = importlib.util.spec_from_file_location("run_momentum_exposure_diagnostic", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_run_config_uses_canonical_baseline() -> None:
    mod = _load_module()
    args = mod._parse_args.__globals__["argparse"].Namespace(
        start="2010-01-01",
        end="2024-12-31",
        output_dir="results/momentum_exposure_diagnostic",
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


def test_compute_momentum_exposure_timeseries_and_summary() -> None:
    mod = _load_module()
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-12"])

    holdings = pd.DataFrame(
        [
            {"AAA": 0.5, "BBB": 0.5, "CCC": 0.0},
            {"AAA": 0.0, "BBB": 0.5, "CCC": 0.5},
        ],
        index=idx,
    )
    eligibility = pd.DataFrame(
        [
            {"AAA": True, "BBB": True, "CCC": True},
            {"AAA": False, "BBB": True, "CCC": True},
        ],
        index=idx,
    )
    momentum = pd.DataFrame(
        [
            {"AAA": 0.10, "BBB": 0.20, "CCC": -0.10},
            {"AAA": 0.30, "BBB": 0.05, "CCC": 0.15},
        ],
        index=idx,
    )

    exposure_df = mod.compute_momentum_exposure_timeseries(
        holdings=holdings,
        eligibility=eligibility,
        momentum=momentum,
        rebalance_dates=idx,
    )
    assert exposure_df["portfolio_count"].tolist() == [2, 2]
    assert exposure_df["universe_count"].tolist() == [3, 2]
    assert exposure_df["portfolio_momentum"].tolist() == pytest.approx([0.15, 0.10])
    assert exposure_df["universe_momentum"].tolist() == pytest.approx([0.0666666667, 0.10])
    assert exposure_df["momentum_exposure"].tolist() == pytest.approx([0.0833333333, 0.0])

    summary_df = mod.build_summary_frame(exposure_df)
    metric_map = dict(zip(summary_df["Metric"], summary_df["Value"]))
    assert metric_map["mean portfolio momentum"] == pytest.approx(0.125)
    assert metric_map["mean universe momentum"] == pytest.approx(0.0833333333)
    assert metric_map["mean momentum exposure"] == pytest.approx(0.0416666667)
    assert metric_map["percent positive exposure"] == pytest.approx(50.0)
