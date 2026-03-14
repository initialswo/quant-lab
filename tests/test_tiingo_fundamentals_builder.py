from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.tiingo_fundamentals import INTERNAL_COLUMNS, normalize_internal_fundamentals_frame


def _load_builder_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_fundamentals_database.py"
    spec = importlib.util.spec_from_file_location("build_fundamentals_database", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_api_key_env_fallback(monkeypatch) -> None:
    mod = _load_builder_module()
    monkeypatch.setenv("TIINGO_API_KEY", "env_key_123")
    assert mod._resolve_api_key("tiingo", "", "") == "env_key_123"


def test_api_key_cli_override_precedence(monkeypatch) -> None:
    mod = _load_builder_module()
    monkeypatch.setenv("TIINGO_API_KEY", "env_key_123")
    assert mod._resolve_api_key("tiingo", "", "cli_key_456") == "cli_key_456"


def test_api_key_missing_raises(monkeypatch) -> None:
    mod = _load_builder_module()
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Tiingo API key is missing"):
        mod._resolve_api_key("tiingo", "", "")


def test_fmp_api_key_env_fallback(monkeypatch) -> None:
    mod = _load_builder_module()
    monkeypatch.setenv("FMP_API_KEY", "fmp_env_key_123")
    assert mod._resolve_api_key("fmp", "", "") == "fmp_env_key_123"


def test_fmp_api_key_cli_override_precedence(monkeypatch) -> None:
    mod = _load_builder_module()
    monkeypatch.setenv("FMP_API_KEY", "fmp_env_key_123")
    assert mod._resolve_api_key("fmp", "fmp_cli_key_456", "") == "fmp_cli_key_456"


def test_fmp_api_key_missing_raises(monkeypatch) -> None:
    mod = _load_builder_module()
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FMP API key is missing"):
        mod._resolve_api_key("fmp", "", "")


def test_dotenv_loading_path_works(tmp_path: Path, monkeypatch) -> None:
    mod = _load_builder_module()
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TIINGO_API_KEY=from_dotenv_file\nFMP_API_KEY=fmp_from_dotenv_file\n", encoding="utf-8")
    loaded = mod._load_dotenv_from_repo_root(repo_root=tmp_path)
    assert loaded
    assert mod._resolve_api_key("tiingo", "", "") == "from_dotenv_file"
    assert mod._resolve_api_key("fmp", "", "") == "fmp_from_dotenv_file"


def test_schema_output_matches_internal_schema() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": "brk.b",
                "period_end": "2020-03-31",
                "available_date": "2020-06-01",
                "revenue": 1000,
                "cogs": 600,
                "gross_profit": None,
                "total_assets": 5000,
                "shareholders_equity": 2000,
                "net_income": 100,
            },
            {
                "ticker": "BRK.B",
                "period_end": "2020-03-31",
                "available_date": "2020-06-01",
                "revenue": 1000,
                "cogs": 600,
                "gross_profit": 400,
                "total_assets": 5000,
                "shareholders_equity": 2000,
                "net_income": 100,
            },
        ]
    )
    out = normalize_internal_fundamentals_frame(raw)
    assert list(out.columns) == INTERNAL_COLUMNS
    assert int(len(out)) == 1
    assert out.iloc[0]["ticker"] == "BRK-B"
    assert abs(float(out.iloc[0]["gross_profit"]) - 400.0) < 1e-12
