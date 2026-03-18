from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "validate_equity_warehouse.py"
    spec = importlib.util.spec_from_file_location("validate_equity_warehouse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reporting_delay_audit_flags_negative_delays() -> None:
    mod = _load_module()
    df = pd.DataFrame(
        [
            {
                "ticker_id": "t1",
                "canonical_symbol": "AAA",
                "raw_source_symbol": "AAA",
                "period_end": "2024-12-31",
                "available_date": "2024-12-15",
            },
            {
                "ticker_id": "t2",
                "canonical_symbol": "BBB",
                "raw_source_symbol": "BBB",
                "period_end": "2024-12-31",
                "available_date": "2025-02-01",
            },
        ]
    )
    audit, bad_rows = mod._reporting_delay_audit(df)
    assert bad_rows == 1
    row = audit.iloc[0]
    assert int(row["negative_reporting_delay_rows"]) == 1
    assert float(row["min_reporting_delay_days"]) == -16.0
    assert "AAA" in str(row["sample_negative_rows"])
