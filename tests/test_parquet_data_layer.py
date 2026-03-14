from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from quant_lab.data.loaders import load_ohlcv_for_research
from quant_lab.data.parquet_store import EquityParquetStore
from quant_lab.data.universe_registry import UniverseRegistry


HAS_PARQUET = bool(importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"))
pytestmark = pytest.mark.skipif(not HAS_PARQUET, reason="Requires pyarrow or fastparquet for parquet tests")


def _sample_daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-02"],
            "ticker": ["AAA", "AAA", "BBB", "AAA"],
            "open": [10.0, 11.0, 20.0, 10.5],
            "high": [11.0, 12.0, 21.0, 11.5],
            "low": [9.5, 10.5, 19.5, 10.0],
            "close": [10.8, 11.2, 20.5, 10.9],
            "volume": [1000, 1100, 2000, 1050],
            "source": ["test", "test", "test", "test"],
        }
    )


def test_daily_roundtrip_and_dedup(tmp_path) -> None:
    store = EquityParquetStore(root=tmp_path / "equities")
    written = store.upsert_daily_ohlcv(_sample_daily())
    assert written == 3

    loaded = store.load_daily_ohlcv(start="2024-01-01", end="2024-01-31")
    # AAA@2024-01-02 appears twice; keep last write only.
    assert len(loaded.loc[(loaded["date"] == pd.Timestamp("2024-01-02")) & (loaded["ticker"] == "AAA")]) == 1
    aaa_row = loaded.loc[
        (loaded["date"] == pd.Timestamp("2024-01-02")) & (loaded["ticker"] == "AAA")
    ].iloc[0]
    assert float(aaa_row["close"]) == pytest.approx(10.9)


def test_load_by_date_range_and_ticker_subset(tmp_path) -> None:
    store = EquityParquetStore(root=tmp_path / "equities")
    store.upsert_daily_ohlcv(_sample_daily())
    out = store.load_daily_ohlcv(start="2024-01-03", end="2024-01-03", tickers=["AAA"])
    assert out["ticker"].unique().tolist() == ["AAA"]
    assert out["date"].min() == pd.Timestamp("2024-01-03")
    assert out["date"].max() == pd.Timestamp("2024-01-03")


def test_load_by_universe_membership(tmp_path) -> None:
    store = EquityParquetStore(root=tmp_path / "equities")
    store.upsert_daily_ohlcv(_sample_daily())
    membership = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "universe": ["sp500", "sp500", "sp500"],
            "ticker": ["AAA", "AAA", "BBB"],
            "in_universe": [1, 1, 0],
        }
    )
    store.upsert_universe_membership(membership)
    registry = UniverseRegistry(store=store)
    out = store.load_daily_ohlcv_by_universe(
        universe="sp500",
        start="2024-01-01",
        end="2024-01-31",
        registry=registry,
    )
    assert set(out["ticker"].unique()) == {"AAA"}


def test_loader_returns_panels_and_diagnostics(tmp_path) -> None:
    store = EquityParquetStore(root=tmp_path / "equities")
    store.upsert_daily_ohlcv(_sample_daily())
    result = load_ohlcv_for_research(
        start="2024-01-01",
        end="2024-01-31",
        tickers=["AAA", "BBB"],
        store_root=tmp_path / "equities",
    )
    assert "close" in result.panels
    assert not result.panels["close"].empty
    assert result.diagnostics["tickers_loaded"] == 2


def test_ingest_sp500_historical_membership_csv(tmp_path) -> None:
    csv_path = tmp_path / "sp500_hist.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "ticker": ["AAA", "BBB", "AAA"],
            "is_member": [1, 0, 1],
        }
    ).to_csv(csv_path, index=False)

    store = EquityParquetStore(root=tmp_path / "equities")
    registry = UniverseRegistry(store=store)
    rows = registry.ingest_sp500_historical_csv(str(csv_path))
    assert rows == 3

    loaded = store.load_universe_membership(universe="sp500")
    assert set(loaded.columns) == {"date", "universe", "ticker", "in_universe"}
    assert loaded["universe"].nunique() == 1
    assert loaded["universe"].iloc[0] == "sp500"
