from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab.engine import runner


def _mock_fetch_ohlcv_with_summary(
    tickers: list[str],
    start: str,
    end: str,
    **_: object,
) -> tuple[dict[str, pd.DataFrame], dict]:
    idx = pd.date_range(start, end, freq="B")
    out: dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers):
        base = 100.0 + i * 3.0
        trend = np.linspace(0.0, 40.0, len(idx))
        seasonal = 1.5 * np.sin(np.linspace(0, 15, len(idx)) + i)
        close = base + trend + seasonal
        out[t] = pd.DataFrame({"Close": close, "Adj Close": close}, index=idx)
    summary = {
        "source": "mock",
        "tickers_loaded": len(out),
        "earliest_date": str(idx.min().date()) if len(idx) else "",
        "latest_date": str(idx.max().date()) if len(idx) else "",
        "cache_dir": "data/ohlcv",
        "cached_files_used": 0,
        "fetched_or_refreshed": len(out),
    }
    return out, summary


def _patch_data_sources(monkeypatch) -> None:
    monkeypatch.setattr(runner, "load_sp500_tickers", lambda: ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(runner, "fetch_ohlcv_with_summary", _mock_fetch_ohlcv_with_summary)
    monkeypatch.setattr(runner, "append_registry_row", lambda row: None)
    monkeypatch.setattr(runner, "_git_commit", lambda: "deadbeef")


def _mock_fetch_partial_lowercase(
    tickers: list[str],
    start: str,
    end: str,
    **_: object,
) -> tuple[dict[str, pd.DataFrame], dict]:
    idx = pd.date_range(start, end, freq="B")
    out: dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers[:2]):
        base = 100.0 + i
        close = base + np.linspace(0.0, 5.0, len(idx))
        # Intentionally lowercase keys to exercise resilient lookup.
        out[t.lower()] = pd.DataFrame({"Close": close, "Adj Close": close}, index=idx)
    # Add one malformed ticker that should be classified as rejected, not missing.
    if len(tickers) >= 3:
        bad = tickers[2].lower()
        out[bad] = pd.DataFrame({"Open": np.ones(len(idx))}, index=idx)
    summary = {
        "source": "mock",
        "tickers_loaded": len(out),
        "earliest_date": str(idx.min().date()) if len(idx) else "",
        "latest_date": str(idx.max().date()) if len(idx) else "",
        "cache_dir": "data/ohlcv",
        "cached_files_used": 0,
        "fetched_or_refreshed": len(out),
    }
    return out, summary


def test_backtest_artifacts_toggle_and_regime_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_data_sources(monkeypatch)

    summary0, outdir0 = runner.run_backtest(
        start="2018-01-01",
        end="2024-01-01",
        max_tickers=4,
        top_n=2,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name=["momentum_6_1", "low_vol_20"],
        factor_names=["momentum_6_1", "low_vol_20"],
        factor_weights=[0.6, 0.4],
        save_artifacts=False,
    )
    out0 = Path(outdir0)
    assert summary0["SaveArtifacts"] is False
    assert not (out0 / "equity_curve.csv").exists()
    assert not (out0 / "holdings.csv").exists()
    assert not (out0 / "regime_label.csv").exists()
    assert not (out0 / "factor_weights_timeseries.csv").exists()

    summary1, outdir1 = runner.run_backtest(
        start="2018-01-01",
        end="2024-01-01",
        max_tickers=4,
        top_n=2,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name=["momentum_6_1", "low_vol_20"],
        factor_names=["momentum_6_1", "low_vol_20"],
        factor_weights=[0.6, 0.4],
        regime_filter=True,
        regime_bull_weights="momentum_6_1:0.7,low_vol_20:0.3",
        regime_bear_weights="momentum_6_1:0.3,low_vol_20:0.7",
        save_artifacts=True,
    )
    out1 = Path(outdir1)
    assert summary1["SaveArtifacts"] is True
    assert (out1 / "equity_curve.csv").exists()
    assert (out1 / "holdings.csv").exists()
    assert (out1 / "regime_label.csv").exists()
    assert (out1 / "factor_weights_timeseries.csv").exists()

    fw = pd.read_csv(out1 / "factor_weights_timeseries.csv")
    assert fw.columns.tolist()[1:] == ["momentum_6_1", "low_vol_20"]


def test_walkforward_summary_includes_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_data_sources(monkeypatch)

    summary, outdir = runner.run_walkforward(
        start="2016-01-01",
        end="2024-01-01",
        train_years=2,
        test_years=1,
        max_tickers=4,
        top_n=2,
        rebalance="daily",
        costs_bps=5.0,
        factor_name=["momentum_6_1", "low_vol_20"],
        factor_names=["momentum_6_1", "low_vol_20"],
        factor_weights=[0.5, 0.5],
        save_artifacts=False,
    )

    assert summary["Mode"] == "walkforward"
    assert "Windows" in summary
    assert isinstance(summary["Windows"], list)
    assert len(summary["Windows"]) == summary["NumWindows"]
    required = {
        "window_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "EffectiveCostBpsAvg",
        "TurnoverAvg",
        "LeverageAvg",
        "LeverageMax",
    }
    assert required.issubset(summary["Windows"][0].keys())

    out = Path(outdir)
    assert not (out / "windows.csv").exists()
    assert summary["EligibleTickersMin"] > 0
    assert summary["EligibleTickersMax"] > 0
    assert summary["EligibleOnRebalanceMin"] > 0
    assert summary["EligibleOnRebalanceMax"] > 0


def test_universe_dataset_build_saves_in_static_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_data_sources(monkeypatch)

    summary, outdir = runner.run_walkforward(
        start="2018-01-01",
        end="2022-01-01",
        train_years=2,
        test_years=1,
        max_tickers=4,
        top_n=2,
        rebalance="monthly",
        costs_bps=5.0,
        universe_mode="static",
        universe_dataset_mode="build",
        universe_dataset_freq="daily",
        universe_dataset_save=True,
        save_artifacts=False,
    )

    out = Path(outdir)
    membership_path = out / "universe_membership.csv"
    summary_path = out / "universe_summary.csv"
    assert membership_path.exists()
    assert summary_path.exists()
    assert summary.get("UniverseDatasetSaved") is True
    assert str(summary.get("UniverseDatasetMembershipPath", "")).endswith(
        "universe_membership.csv"
    )
    assert str(summary.get("UniverseDatasetSummaryPath", "")).endswith("universe_summary.csv")
    membership = pd.read_csv(membership_path, index_col=0, parse_dates=True)
    assert not membership.empty
    assert str(membership.index.min().date()) == str(summary.get("UniverseDatasetStartDate"))
    assert str(membership.index.max().date()) == str(summary.get("UniverseDatasetEndDate"))
    assert int(len(membership)) == int(summary.get("UniverseDatasetRows"))
    assert str(membership.index.min().date()) == "2018-01-01"


def test_universe_loading_diagnostics_observational_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "load_sp500_tickers", lambda: ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(runner, "fetch_ohlcv_with_summary", _mock_fetch_partial_lowercase)
    monkeypatch.setattr(runner, "append_registry_row", lambda row: None)
    monkeypatch.setattr(runner, "_git_commit", lambda: "deadbeef")

    summary, _ = runner.run_backtest(
        start="2020-01-01",
        end="2021-01-01",
        max_tickers=4,
        top_n=1,
        rebalance="monthly",
        costs_bps=5.0,
    )

    assert summary["RequestedTickersCount"] == 4
    assert summary["LoadedTickersCount"] == 2
    assert summary["MissingTickersCount"] == 1
    assert summary["RejectedTickersCount"] == 1
    assert set(summary["MissingTickersSample"]).issubset({"AAA", "BBB", "CCC", "DDD"})
    assert set(summary["RejectedTickersSample"]).issubset({"AAA", "BBB", "CCC", "DDD"})
    assert summary["TickersUsed"] == 2
