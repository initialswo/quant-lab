from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_composite_multifactor.py"
    spec = importlib.util.spec_from_file_location("run_composite_multifactor", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_strategy_specs_includes_expected_default_comparisons() -> None:
    mod = _load_module()
    specs = mod.build_strategy_specs(
        factors=list(mod.DEFAULT_FACTORS),
        compare_single_factors=True,
        compare_alt_sets=True,
    )
    names = [str(spec["strategy_name"]) for spec in specs]
    assert names == [
        "composite_core",
        "composite_price_only",
        "composite_fundamental_only",
        "composite_plus_lowvol",
        "momentum_12_1",
        "reversal_1m",
        "gross_profitability",
        "book_to_market",
    ]


def test_build_run_config_sets_mean_rank_and_dynamic_universe() -> None:
    mod = _load_module()
    args = mod._parse_args.__globals__["argparse"].Namespace(
        start="2010-01-01",
        end="2024-12-31",
        universe="liquid_us",
        top_n=10,
        rebalance="weekly",
        costs_bps=10.0,
        slippage_bps=5.0,
    )
    cfg = mod.build_run_config(args=args, factors=["momentum_12_1", "reversal_1m"])
    assert cfg["factor_names"] == ["momentum_12_1", "reversal_1m"]
    assert cfg["factor_weights"] == [0.5, 0.5]
    assert cfg["factor_aggregation_method"] == "mean_rank"
    assert cfg["use_sector_neutralization"] is False
    assert cfg["use_size_neutralization"] is False
    assert cfg["universe_mode"] == "dynamic"
    assert cfg["slippage_bps"] == 5.0


def test_artifact_stats_reads_holdings_and_daily_returns(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "equity.csv").write_text(
        "Date,Equity,DailyReturn,Turnover\n"
        "2024-01-01,1.0,0.00,0.0\n"
        "2024-01-02,1.1,0.10,0.5\n"
        "2024-01-03,1.0,-0.090909,0.0\n"
        "2024-01-04,1.2,0.20,0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "holdings.csv").write_text(
        "Date,A,B,C\n"
        "2024-01-01,0.5,0.5,0.0\n"
        "2024-01-02,0.5,0.5,0.0\n"
        "2024-01-03,0.0,0.5,0.5\n"
        "2024-01-04,0.0,0.5,0.5\n",
        encoding="utf-8",
    )
    stats = mod.artifact_stats(outdir=tmp_path, rebalance="daily", skipped_count=1)
    assert stats["hit_rate"] == 0.5
    assert stats["median_selected_names"] == 2.0
    assert stats["n_rebalance_dates"] == 3
