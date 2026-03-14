from __future__ import annotations

import pandas as pd

from quant_lab.data import sp500_membership


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_build_sp500_membership_uses_fetched_html(monkeypatch) -> None:
    html = """
    <html><body>
      <table>
        <thead><tr><th>Symbol</th><th>Security</th></tr></thead>
        <tbody>
          <tr><td>AAPL</td><td>Apple</td></tr>
          <tr><td>MSFT</td><td>Microsoft</td></tr>
        </tbody>
      </table>
      <table>
        <thead><tr><th>Date</th><th>Added ticker</th><th>Removed ticker</th></tr></thead>
        <tbody>
          <tr><td>2020-01-02</td><td>GOOG</td><td>MSFT</td></tr>
        </tbody>
      </table>
    </body></html>
    """

    def _fake_get(url: str, headers: dict, timeout: int):  # noqa: ANN001
        assert "wikipedia.org" in url
        assert "User-Agent" in headers
        assert timeout == 30
        return _FakeResponse(html, status_code=200)

    monkeypatch.setattr(sp500_membership.requests, "get", _fake_get)
    out = sp500_membership.build_sp500_historical_membership(
        start_date="2020-01-01",
        end_date="2020-01-03",
    )
    assert {"date", "ticker", "is_member"} == set(out.columns)
    assert not out.empty
    # Current members should exist in output.
    assert set(out["ticker"].unique()) >= {"AAPL", "MSFT"}


def test_fetch_error_has_useful_message(monkeypatch) -> None:
    class _Err(sp500_membership.requests.RequestException):
        pass

    def _boom(url: str, headers: dict, timeout: int):  # noqa: ANN001
        raise _Err("boom")

    monkeypatch.setattr(sp500_membership.requests, "get", _boom)
    try:
        sp500_membership._load_wikipedia_tables()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        msg = str(exc).lower()
        assert "failed to fetch" in msg
        assert "wikipedia.org" in msg
