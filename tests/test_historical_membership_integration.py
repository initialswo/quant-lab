from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab.data.historical_membership import load_historical_membership
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
        base = 100.0 + i * 5.0
        close = base + np.linspace(0.0, 20.0, len(idx))
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
    monkeypatch.setattr(runner, "load_sp500_tickers", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(runner, "fetch_ohlcv_with_summary", _mock_fetch_ohlcv_with_summary)
    monkeypatch.setattr(runner, "append_registry_row", lambda row: None)
    monkeypatch.setattr(runner, "_git_commit", lambda: "deadbeef")


def _write_membership_csv(path: Path) -> None:
    rows = [
        {"date": "2018-01-01", "ticker": "AAA", "is_member": 1},
        {"date": "2018-01-01", "ticker": "BBB", "is_member": 0},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_historical_membership_masks_by_date(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    raw = pd.DataFrame(
        [
            {"date": "2020-01-01", "ticker": "AAA", "is_member": 1},
            {"date": "2020-01-01", "ticker": "BBB", "is_member": 0},
            {"date": "2020-01-06", "ticker": "BBB", "is_member": 1},
        ]
    )
    path = tmp_path / "hist_membership.csv"
    raw.to_csv(path, index=False)
    loaded = load_historical_membership(
        path=str(path),
        index=idx,
        columns=["AAA", "BBB"],
    )
    assert bool(loaded.loc[idx[1], "AAA"])
    assert not bool(loaded.loc[idx[1], "BBB"])
    assert bool(loaded.loc[idx[-1], "BBB"])


def test_historical_membership_ffill_from_non_trading_dates(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-02", periods=5, freq="B")
    raw = pd.DataFrame(
        [
            {"date": "2020-01-01", "ticker": "AAA", "is_member": 1},  # non-trading day
            {"date": "2020-01-01", "ticker": "BBB", "is_member": 0},
            {"date": "2020-01-06", "ticker": "BBB", "is_member": 1},
        ]
    )
    path = tmp_path / "hist_membership_non_trading.csv"
    raw.to_csv(path, index=False)

    loaded = load_historical_membership(
        path=str(path),
        index=idx,
        columns=["AAA", "BBB"],
    )

    assert bool(loaded.loc[idx[0], "AAA"])  # ffilled from 2020-01-01
    assert not bool(loaded.loc[idx[0], "BBB"])
    assert bool(loaded.loc[idx[-1], "BBB"])  # turned on at 2020-01-06


def test_backtest_historical_membership_masks_requested_tickers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_data_sources(monkeypatch)
    membership_path = tmp_path / "membership.csv"
    _write_membership_csv(membership_path)

    summary, _ = runner.run_backtest(
        start="2018-01-01",
        end="2021-12-31",
        max_tickers=2,
        top_n=1,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        historical_membership_path=str(membership_path),
        universe_skip_below_min_tickers=False,
    )

    assert summary["RequestedTickersCount"] == 2
    assert summary["LoadedTickersCount"] == 2
    assert summary["SourceAvailableOnRebalanceMax"] == 2
    assert summary["MembershipOnRebalanceMax"] == 1
    assert summary["FinalTradableOnRebalanceMax"] <= 1
    assert str(summary["HistoricalMembershipPath"]).endswith("membership.csv")


def test_walkforward_membership_reporting_fields_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_data_sources(monkeypatch)
    membership_path = tmp_path / "membership.csv"
    _write_membership_csv(membership_path)

    summary, _ = runner.run_walkforward(
        start="2018-01-01",
        end="2024-01-01",
        train_years=2,
        test_years=1,
        max_tickers=2,
        top_n=1,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        historical_membership_path=str(membership_path),
        universe_skip_below_min_tickers=False,
    )

    assert str(summary["HistoricalMembershipPath"]).endswith("membership.csv")
    assert summary["SourceAvailableOnRebalanceMax"] >= summary["MembershipOnRebalanceMax"]
    assert summary["MembershipOnRebalanceMax"] >= summary["FinalTradableOnRebalanceMax"]
    assert summary["Windows"]
    first = summary["Windows"][0]
    for key in [
        "SourceAvailableOnRebalanceMedian",
        "MembershipOnRebalanceMedian",
        "EligibilityFilteredOnRebalanceMedian",
        "FinalTradableOnRebalanceMedian",
    ]:
        assert key in first


def test_membership_is_post_load_only_for_walkforward(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_data_sources(monkeypatch)
    membership_path = tmp_path / "membership.csv"
    _write_membership_csv(membership_path)

    base_summary, _ = runner.run_walkforward(
        start="2018-01-01",
        end="2024-01-01",
        train_years=2,
        test_years=1,
        max_tickers=2,
        top_n=1,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        universe_skip_below_min_tickers=False,
    )
    member_summary, _ = runner.run_walkforward(
        start="2018-01-01",
        end="2024-01-01",
        train_years=2,
        test_years=1,
        max_tickers=2,
        top_n=1,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        historical_membership_path=str(membership_path),
        universe_skip_below_min_tickers=False,
    )

    assert base_summary["RequestedTickersCount"] == member_summary["RequestedTickersCount"]
    assert base_summary["LoadedTickersCount"] == member_summary["LoadedTickersCount"]
    assert base_summary["MissingTickersCount"] == member_summary["MissingTickersCount"]
    assert base_summary["RejectedTickersCount"] == member_summary["RejectedTickersCount"]
    assert (
        base_summary["SourceAvailableOnRebalanceMax"]
        == member_summary["SourceAvailableOnRebalanceMax"]
    )
    assert member_summary["MembershipOnRebalanceMax"] <= member_summary["SourceAvailableOnRebalanceMax"]


def test_membership_does_not_change_prefetch_arguments_or_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def _spy_fetch(
        tickers: list[str],
        start: str,
        end: str,
        **_: object,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        calls.append(list(tickers))
        return _mock_fetch_ohlcv_with_summary(tickers=tickers, start=start, end=end)

    monkeypatch.setattr(runner, "load_sp500_tickers", lambda: ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(runner, "fetch_ohlcv_with_summary", _spy_fetch)
    monkeypatch.setattr(runner, "append_registry_row", lambda row: None)
    monkeypatch.setattr(runner, "_git_commit", lambda: "deadbeef")

    membership_path = tmp_path / "membership.csv"
    _write_membership_csv(membership_path)

    summary0, _ = runner.run_walkforward(
        start="2018-01-01",
        end="2024-01-01",
        train_years=2,
        test_years=1,
        max_tickers=4,
        top_n=2,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        universe_skip_below_min_tickers=False,
    )
    summary1, _ = runner.run_walkforward(
        start="2018-01-01",
        end="2024-01-01",
        train_years=2,
        test_years=1,
        max_tickers=4,
        top_n=2,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        historical_membership_path=str(membership_path),
        universe_skip_below_min_tickers=False,
    )

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert summary0["FetchRequestTickersCount"] == summary1["FetchRequestTickersCount"]
    assert summary0["FetchedKeysCount"] == summary1["FetchedKeysCount"]
    assert summary0["PostFetchFilteredKeysCount"] == summary1["PostFetchFilteredKeysCount"]
    assert summary0["FetchedKeysCount"] > 0
    assert summary0["LoadedTickersCount"] == summary1["LoadedTickersCount"]


def test_membership_does_not_change_prefetch_for_backtest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def _spy_fetch(
        tickers: list[str],
        start: str,
        end: str,
        **_: object,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        calls.append(list(tickers))
        return _mock_fetch_ohlcv_with_summary(tickers=tickers, start=start, end=end)

    monkeypatch.setattr(runner, "load_sp500_tickers", lambda: ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(runner, "fetch_ohlcv_with_summary", _spy_fetch)
    monkeypatch.setattr(runner, "append_registry_row", lambda row: None)
    monkeypatch.setattr(runner, "_git_commit", lambda: "deadbeef")

    membership_path = tmp_path / "membership.csv"
    _write_membership_csv(membership_path)

    summary0, _ = runner.run_backtest(
        start="2018-01-01",
        end="2023-12-29",
        max_tickers=4,
        top_n=2,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        universe_skip_below_min_tickers=False,
    )
    summary1, _ = runner.run_backtest(
        start="2018-01-01",
        end="2023-12-29",
        max_tickers=4,
        top_n=2,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        historical_membership_path=str(membership_path),
        universe_skip_below_min_tickers=False,
    )

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert summary0["FetchRequestTickersCount"] == summary1["FetchRequestTickersCount"]
    assert summary0["FetchedKeysCount"] == summary1["FetchedKeysCount"]
    assert summary0["PostFetchFilteredKeysCount"] == summary1["PostFetchFilteredKeysCount"]
    assert summary0["LoadedTickersCount"] == summary1["LoadedTickersCount"]


def test_membership_overlap_produces_nonzero_tradable_counts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_data_sources(monkeypatch)
    membership_path = tmp_path / "membership.csv"
    # Use non-trading date to exercise robust alignment and ensure overlap.
    pd.DataFrame(
        [
            {"date": "2017-12-31", "ticker": "AAA", "is_member": 1},
            {"date": "2017-12-31", "ticker": "BBB", "is_member": 1},
            {"date": "2017-12-31", "ticker": "CCC", "is_member": 0},
            {"date": "2017-12-31", "ticker": "DDD", "is_member": 0},
        ]
    ).to_csv(membership_path, index=False)

    summary, _ = runner.run_walkforward(
        start="2018-01-01",
        end="2024-01-01",
        train_years=2,
        test_years=1,
        max_tickers=4,
        top_n=1,
        rebalance="monthly",
        costs_bps=5.0,
        factor_name="low_vol_20",
        factor_names=["low_vol_20"],
        historical_membership_path=str(membership_path),
        universe_skip_below_min_tickers=False,
    )

    assert summary["MembershipOnRebalanceMax"] >= 1
    assert summary["FinalTradableOnRebalanceMax"] >= 1
