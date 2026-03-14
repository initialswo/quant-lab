from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.tiingo_fundamentals import (
    build_tiingo_payload_summary,
    parse_tiingo_statements_payload,
)


def _load_builder_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_fundamentals_database.py"
    spec = importlib.util.spec_from_file_location("build_fundamentals_database", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_payload() -> list[dict]:
    p = Path(__file__).resolve().parent / "data" / "tiingo_statements_sample.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_parser_extracts_expected_numeric_fields() -> None:
    payload = _sample_payload()
    out = parse_tiingo_statements_payload("AAPL", payload, available_lag_days=60)
    assert not out.empty
    assert set(
        [
            "ticker",
            "period_end",
            "available_date",
            "revenue",
            "cogs",
            "gross_profit",
            "total_assets",
            "shareholders_equity",
            "net_income",
        ]
    ).issubset(set(out.columns))
    first = out.iloc[0]
    assert first["ticker"] == "AAPL"
    assert float(first["revenue"]) > 0.0
    assert float(first["cogs"]) > 0.0
    assert float(first["total_assets"]) > 0.0
    assert float(first["shareholders_equity"]) > 0.0
    assert float(first["net_income"]) > 0.0


def test_parser_fallback_gross_profit_when_missing() -> None:
    payload = _sample_payload()
    out = parse_tiingo_statements_payload("AAPL", payload, available_lag_days=60)
    # second record has no grossProfit, should fallback to revenue - costRev
    row = out.loc[out["period_end"] == pd.Timestamp("2024-09-28")].iloc[0]
    assert abs(float(row["gross_profit"]) - (float(row["revenue"]) - float(row["cogs"]))) < 1e-9


def test_empty_payload_is_handled_gracefully() -> None:
    out = parse_tiingo_statements_payload("UNKNOWN", [], available_lag_days=60)
    assert out.empty


def test_parser_extracts_shares_outstanding_when_present() -> None:
    payload = [
        {
            "date": "2024-12-31",
            "statementData": {
                "incomeStatement": [
                    {"dataCode": "revenue", "value": 1000},
                    {"dataCode": "costRev", "value": 500},
                    {"dataCode": "netinc", "value": 200},
                    {"dataCode": "weightedAverageShsOut", "value": 40},
                ],
                "balanceSheet": [
                    {"dataCode": "totalAssets", "value": 3000},
                    {"dataCode": "equity", "value": 1200},
                ],
            },
        }
    ]
    out = parse_tiingo_statements_payload("AAPL", payload, available_lag_days=60)
    row = out.iloc[0]
    assert float(row["shares_outstanding"]) == 40.0


def test_debug_mode_writes_raw_payload_files(tmp_path: Path) -> None:
    mod = _load_builder_module()
    raw = {
        "endpoint": "/tiingo/fundamentals/AAPL/statements",
        "status_code": 200,
        "payload": _sample_payload(),
        "error": None,
    }
    mod._write_debug_payload("tiingo", "AAPL", raw, tmp_path)
    assert (tmp_path / "tiingo_aapl_raw.json").exists()
    assert (tmp_path / "tiingo_aapl_summary.json").exists()


def test_payload_summary_contains_structure_keys() -> None:
    payload = _sample_payload()
    summary = build_tiingo_payload_summary(
        endpoint="/tiingo/fundamentals/AAPL/statements",
        status_code=200,
        payload=payload,
    )
    assert summary["row_count"] == 2
    assert "statementData" in summary["top_level_keys_first_record"]
    assert "incomeStatement" in summary["statement_data_sections_first_record"]
