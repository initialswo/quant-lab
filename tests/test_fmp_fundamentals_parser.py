from __future__ import annotations

import pandas as pd

from quant_lab.data.fmp_fundamentals import (
    INTERNAL_COLUMNS,
    parse_fmp_statements_payload,
)


def _income_payload() -> list[dict]:
    return [
        {
            "date": "2024-12-31",
            "revenue": 1000,
            "costOfRevenue": 600,
            "grossProfit": 400,
            "netIncome": 120,
            "weightedAverageShsOutDil": 55.0,
            "filingDate": "2025-02-10",
            "acceptedDate": "2025-02-10 09:20:00",
        },
        {
            "date": "2024-09-30",
            "revenue": 800,
            "costOfRevenue": 500,
            "grossProfit": None,
            "netIncome": 90,
            "weightedAverageShsOut": 50.0,
            "acceptedDate": None,
            "filingDate": None,
        },
    ]


def _balance_payload() -> list[dict]:
    return [
        {
            "date": "2024-12-31",
            "totalAssets": 5000,
            "totalStockholdersEquity": 2100,
        },
        {
            "date": "2024-09-30",
            "totalAssets": 4800,
            "totalStockholdersEquity": 2000,
        },
    ]


def test_fmp_mapping_and_merge_correctness() -> None:
    out = parse_fmp_statements_payload(
        ticker="brk.b",
        income_payload=_income_payload(),
        balance_payload=_balance_payload(),
        available_lag_days=60,
    )
    assert not out.empty
    assert list(out.columns) == INTERNAL_COLUMNS
    assert set(out["ticker"].unique()) == {"BRK-B"}

    row = out.loc[out["period_end"] == pd.Timestamp("2024-12-31")].iloc[0]
    assert float(row["revenue"]) == 1000.0
    assert float(row["cogs"]) == 600.0
    assert float(row["gross_profit"]) == 400.0
    assert float(row["net_income"]) == 120.0
    assert float(row["shares_outstanding"]) == 55.0
    assert float(row["total_assets"]) == 5000.0
    assert float(row["shareholders_equity"]) == 2100.0


def test_available_date_logic_with_fallback_and_filing_date() -> None:
    out = parse_fmp_statements_payload(
        ticker="AAPL",
        income_payload=_income_payload(),
        balance_payload=_balance_payload(),
        available_lag_days=60,
    )
    filed_row = out.loc[out["period_end"] == pd.Timestamp("2024-12-31")].iloc[0]
    fallback_row = out.loc[out["period_end"] == pd.Timestamp("2024-09-30")].iloc[0]
    assert filed_row["available_date"] == pd.Timestamp("2025-02-10")
    assert fallback_row["available_date"] == pd.Timestamp("2024-11-29")


def test_gross_profit_fallback_when_missing() -> None:
    out = parse_fmp_statements_payload(
        ticker="AAPL",
        income_payload=_income_payload(),
        balance_payload=_balance_payload(),
        available_lag_days=60,
    )
    row = out.loc[out["period_end"] == pd.Timestamp("2024-09-30")].iloc[0]
    assert abs(float(row["gross_profit"]) - 300.0) < 1e-12


def test_empty_payload_graceful() -> None:
    out = parse_fmp_statements_payload(
        ticker="AAPL",
        income_payload=[],
        balance_payload=[],
        available_lag_days=60,
    )
    assert out.empty
    assert list(out.columns) == INTERNAL_COLUMNS
