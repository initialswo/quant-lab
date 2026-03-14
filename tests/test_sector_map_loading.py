from __future__ import annotations

import pandas as pd

from quant_lab.engine.runner import _load_sector_map


def test_load_sector_map_accepts_symbol_and_gics_columns(tmp_path) -> None:
    p = tmp_path / "sp500_like.csv"
    pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB", "BRK.B"],
            "GICS Sector": ["Tech", "Health Care", "Financials"],
        }
    ).to_csv(p, index=False)

    out = _load_sector_map(str(p), ["AAA", "BBB", "BRK-B", "MISSING"])
    assert out is not None
    assert out["AAA"] == "Tech"
    assert out["BBB"] == "Health Care"
    assert out["BRK-B"] == "Financials"
    assert out["MISSING"] == "UNKNOWN"
