from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "staged_ingest_tiingo_daily_dry_run.py"
    spec = importlib.util.spec_from_file_location("staged_ingest_tiingo_daily_dry_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tiingo_ingest_retries_and_resumes(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_load_env_key", lambda: "token")
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    attempts: dict[str, int] = {"aaa": 0, "bbb": 0}
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}
            self.text = ""

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, params=None, headers=None, timeout=None):
            ticker = url.split("/")[-2]
            calls.append(ticker)
            attempts[ticker] += 1
            if ticker == "aaa" and attempts[ticker] == 1:
                return FakeResponse(429, {"detail": "rate limited"}, headers={"Retry-After": "0"})
            return FakeResponse(
                200,
                [
                    {
                        "date": "2024-01-02",
                        "open": 1.0,
                        "high": 1.5,
                        "low": 0.9,
                        "close": 1.2,
                        "volume": 10,
                        "adjClose": 1.1,
                    }
                ],
            )

    monkeypatch.setattr(mod.requests, "Session", lambda: FakeSession())

    out_dir = tmp_path / "tiingo"
    summary = mod.run_staged_tiingo_dry_run(
        cohort=["AAA", "BBB"],
        out_dir=out_dir,
        source_label="tiingo_phase3_bulk",
        batch_size=1,
        batch_pause_seconds=0.0,
        max_retries=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        resume=True,
    )

    assert summary["tickers_succeeded"] == 2
    assert attempts["aaa"] == 2
    assert attempts["bbb"] == 1
    report = pd.read_csv(out_dir / "tiingo_fetch_report.csv")
    assert int(report.loc[report["ticker"] == "AAA", "retries_used"].iloc[0]) == 1
    assert (out_dir / "ticker_chunks" / "AAA.parquet").exists()
    assert (out_dir / "ticker_chunks" / "BBB.parquet").exists()

    resumed_calls: list[str] = []

    class ResumeSession:
        def get(self, url, params=None, headers=None, timeout=None):
            resumed_calls.append(url)
            raise AssertionError("resume path should not refetch completed tickers")

    monkeypatch.setattr(mod.requests, "Session", lambda: ResumeSession())
    resumed = mod.run_staged_tiingo_dry_run(
        cohort=["AAA", "BBB"],
        out_dir=out_dir,
        source_label="tiingo_phase3_bulk",
        batch_size=1,
        batch_pause_seconds=0.0,
        max_retries=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        resume=True,
    )

    assert resumed["tickers_succeeded"] == 2
    assert resumed_calls == []
