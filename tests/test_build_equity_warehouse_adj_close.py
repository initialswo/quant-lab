from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_equity_warehouse_preserves_vendor_adj_close_and_phase3_precedence(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    equities_root = tmp_path / "equities"
    warehouse_root = tmp_path / "warehouse"
    fundamentals_path = tmp_path / "fundamentals_fmp.parquet"
    equities_root.mkdir()
    warehouse_root.mkdir()

    daily = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "ticker": "ABC.US",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "adj_close": 8.0,
                "volume": 1000,
                "source": "tiingo",
            },
            {
                "date": "2024-01-02",
                "ticker": "ABC.US",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "adj_close": 7.5,
                "volume": 1000,
                "source": "tiingo_phase3_bulk",
            },
        ]
    )
    daily.to_parquet(equities_root / "daily_ohlcv.parquet", index=False)

    metadata = pd.DataFrame(
        [
            {
                "ticker": "ABC.US",
                "name": "ABC Corp",
                "exchange": "NYSE",
                "sector": "Industrials",
                "industry": "Machinery",
                "first_date": "2024-01-02",
                "last_date": "2024-01-02",
                "active_flag": True,
            }
        ]
    )
    metadata.to_parquet(equities_root / "metadata.parquet", index=False)

    membership = pd.DataFrame(
        [{"date": "2024-01-02", "universe": "sp500", "ticker": "ABC.US", "in_universe": True}]
    )
    membership.to_parquet(equities_root / "universe_membership.parquet", index=False)

    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "ABC",
                "period_end": "2023-12-31",
                "available_date": "2024-02-15",
                "revenue": 1.0,
                "cogs": 0.5,
                "gross_profit": 0.5,
                "total_assets": 2.0,
                "shareholders_equity": 1.0,
                "net_income": 0.2,
                "shares_outstanding": 100.0,
            }
        ]
    )
    fundamentals.to_parquet(fundamentals_path, index=False)

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_equity_warehouse.py",
            "--equities-root",
            str(equities_root),
            "--fundamentals-path",
            str(fundamentals_path),
            "--warehouse-root",
            str(warehouse_root),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    prices = pd.read_parquet(warehouse_root / "equity_prices_daily.parquet")
    assert len(prices) == 1
    row = prices.iloc[0]
    assert row["source"] == "tiingo_phase3_bulk"
    assert float(row["adj_close"]) == 7.5
    assert float(row["close"]) == 10.0
