from __future__ import annotations

import pandas as pd

from quant_lab.data import fetch
from quant_lab.data.sources.base import DataSourceBase, normalize_ohlcv
from quant_lab.data.sources.stooq_source import StooqSource


class _DummySource(DataSourceBase):
    source_name = "dummy"

    def __init__(self, cache_dir: str, refresh: bool = False, bulk_prepare: bool = False) -> None:
        super().__init__(cache_dir=cache_dir, refresh=refresh, bulk_prepare=bulk_prepare)
        self.calls = 0

    def fetch_symbol_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        self.calls += 1
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        return pd.DataFrame(
            {
                "Open": [1.0, 1.1, 1.2],
                "High": [1.1, 1.2, 1.3],
                "Low": [0.9, 1.0, 1.1],
                "Close": [1.0, 1.1, 1.2],
                "Volume": [100, 110, 120],
            },
            index=idx,
        )


class _RangeDummySource(DataSourceBase):
    source_name = "dummy_range"

    def __init__(self, cache_dir: str, refresh: bool = False, bulk_prepare: bool = False) -> None:
        super().__init__(cache_dir=cache_dir, refresh=refresh, bulk_prepare=bulk_prepare)
        self.calls = 0

    def fetch_symbol_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        self.calls += 1
        start_ts = pd.Timestamp(start or "2020-01-01")
        end_ts = pd.Timestamp(end or "2020-12-31")
        idx = pd.date_range(start_ts, end_ts, freq="B")
        px = pd.Series(range(len(idx)), index=idx, dtype=float) + 100.0
        return pd.DataFrame(
            {
                "Open": px,
                "High": px + 1.0,
                "Low": px - 1.0,
                "Close": px,
                "Volume": 1000,
            },
            index=idx,
        )


def test_cache_roundtrip(tmp_path) -> None:
    src = _DummySource(cache_dir=str(tmp_path), refresh=False)
    out1 = src.fetch_bulk(["AAA"], start="2020-01-01", end="2020-01-03")
    assert "AAA" in out1
    assert src.fetched == 1
    assert src.calls == 1

    out2 = src.fetch_bulk(["AAA"], start="2020-01-01", end="2020-01-03")
    assert "AAA" in out2
    assert src.cache_hits >= 1
    assert src.calls == 1


def test_cache_hit_when_range_fully_covered(tmp_path) -> None:
    src = _RangeDummySource(cache_dir=str(tmp_path), refresh=False)
    out1 = src.fetch_bulk(["AAA"], start="2020-01-01", end="2020-12-31")
    assert "AAA" in out1
    assert src.calls == 1
    out2 = src.fetch_bulk(["AAA"], start="2020-03-01", end="2020-10-01")
    assert "AAA" in out2
    assert src.calls == 1
    assert out2["AAA"].index.min() >= pd.Timestamp("2020-03-01")
    assert out2["AAA"].index.max() <= pd.Timestamp("2020-10-01")


def test_cache_refresh_when_range_exceeds_coverage(tmp_path) -> None:
    src = _RangeDummySource(cache_dir=str(tmp_path), refresh=False)
    # Seed cache with narrow 2020 range.
    src.fetch_bulk(["AAA"], start="2020-01-01", end="2020-12-31")
    assert src.calls == 1
    # Request much larger range: should refresh and overwrite cache.
    out = src.fetch_bulk(["AAA"], start="2005-01-01", end="2024-12-31")
    assert "AAA" in out
    assert src.calls == 2
    cached = src.load_from_cache("AAA")
    assert cached is not None
    expected_min = pd.date_range("2005-01-01", periods=1, freq="B")[0]
    assert cached.index.min() <= expected_min
    assert cached.index.max() >= pd.Timestamp("2024-12-31")


def test_normalize_ohlcv_schema() -> None:
    idx = pd.date_range("2021-01-01", periods=2, freq="B")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.1, 2.1],
            "Low": [0.9, 1.9],
            "Close": [1.0, 2.0],
            "Volume": [100, 200],
        },
        index=idx,
    )
    norm = normalize_ohlcv(raw)
    assert {"Open", "High", "Low", "Close", "Volume", "Adj Close"}.issubset(norm.columns)
    assert norm.index.is_monotonic_increasing


def test_source_selection_plumbing(monkeypatch, tmp_path) -> None:
    dummy = _DummySource(cache_dir=str(tmp_path))

    def _fake_build_source(data_source: str, cache_dir: str, refresh: bool, bulk_prepare: bool):
        assert data_source == "dummy"
        assert cache_dir == str(tmp_path)
        assert refresh is True
        assert bulk_prepare is True
        return dummy

    monkeypatch.setattr(fetch, "_build_source", _fake_build_source)
    out, summary = fetch.fetch_ohlcv_with_summary(
        tickers=["AAA"],
        start="2020-01-01",
        end="2020-01-10",
        cache_dir=str(tmp_path),
        data_source="dummy",
        refresh=True,
        bulk_prepare=True,
    )
    assert "AAA" in out
    assert summary["source"] == "dummy"


def test_fetch_ohlcv_backward_compatible(monkeypatch, tmp_path) -> None:
    dummy = _DummySource(cache_dir=str(tmp_path))
    monkeypatch.setattr(
        fetch,
        "_build_source",
        lambda data_source, cache_dir, refresh, bulk_prepare: dummy,
    )
    out = fetch.fetch_ohlcv(
        tickers=["AAA", "BBB"],
        start="2020-01-01",
        end="2020-01-10",
        cache_dir=str(tmp_path),
        data_source="default",
        refresh=False,
        bulk_prepare=False,
    )
    assert set(out.keys()) == {"AAA", "BBB"}


def test_summary_reflects_refreshed_longer_span(monkeypatch, tmp_path) -> None:
    src = _RangeDummySource(cache_dir=str(tmp_path), refresh=False)

    monkeypatch.setattr(
        fetch,
        "_build_source",
        lambda data_source, cache_dir, refresh, bulk_prepare: src,
    )

    fetch.fetch_ohlcv_with_summary(
        tickers=["AAA"],
        start="2020-01-01",
        end="2020-12-31",
        cache_dir=str(tmp_path),
        data_source="dummy",
        refresh=False,
        bulk_prepare=False,
    )
    _, summary = fetch.fetch_ohlcv_with_summary(
        tickers=["AAA"],
        start="2005-01-01",
        end="2024-12-31",
        cache_dir=str(tmp_path),
        data_source="dummy",
        refresh=False,
        bulk_prepare=False,
    )
    expected_min = str(pd.date_range("2005-01-01", periods=1, freq="B")[0].date())
    assert summary["earliest_date"] <= expected_min
    assert summary["latest_date"] >= "2024-12-31"


def test_cache_tolerates_non_trading_boundary_dates(tmp_path) -> None:
    src = _RangeDummySource(cache_dir=str(tmp_path), refresh=False)
    src.fetch_bulk(["AAA"], start="2005-01-01", end="2024-12-31")
    assert src.calls == 1
    # Weekend/holiday-like bounds should still count as covered and hit cache.
    src.fetch_bulk(["AAA"], start="2005-01-01", end="2024-01-01")
    assert src.calls == 1


def test_stooq_source_returns_non_empty_with_mocked_csv(monkeypatch, tmp_path) -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    raw = pd.DataFrame(
        {
            "Date": idx.strftime("%Y-%m-%d"),
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0, 1, 2, 3, 4],
            "Close": [1, 2, 3, 4, 5],
            "Volume": [100, 110, 120, 130, 140],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: raw.copy())
    src = StooqSource(cache_dir=str(tmp_path), refresh=True)
    out = src.fetch_bulk(["AAPL"], start="2020-01-01", end="2020-01-31")
    assert "AAPL" in out
    assert not out["AAPL"].empty
    assert "Close" in out["AAPL"].columns
