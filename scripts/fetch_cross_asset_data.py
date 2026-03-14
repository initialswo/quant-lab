"""Fetch minimal cross-asset ETF OHLCV history from FMP into local parquet files."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


SYMBOLS = [
    "SPY",
    "TLT",
    "GLD",
    "DBC",
    "VNQ",
    "SHY",
    "EFA",
    "EEM",
    "IEF",
    "LQD",
    "TIP",
    "UUP",
    "FXE",
]
START_DATE = "2000-01-01"
BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"


def _load_api_key() -> str:
    key = str(os.getenv("FMP_API_KEY", "")).strip()
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            txt = line.strip()
            if not txt or txt.startswith("#") or "=" not in txt:
                continue
            k, v = txt.split("=", 1)
            if k.strip() == "FMP_API_KEY":
                key = v.strip().strip("\"'").strip()
                if key:
                    return key
    raise ValueError("FMP_API_KEY not found in environment or .env")


def _fetch_symbol(symbol: str, api_key: str) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "from": START_DATE,
        "to": date.today().isoformat(),
        "apikey": api_key,
    }
    url = BASE_URL + "?" + urlencode(params)
    req = Request(
        url=url,
        headers={
            "User-Agent": "quant-lab-cross-asset-fetch/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"FMP HTTP error for {symbol}: {exc}") from exc
    except URLError as exc:
        raise RuntimeError(f"FMP network error for {symbol}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"FMP decode error for {symbol}: {exc}") from exc

    if isinstance(payload, list):
        hist = payload
    elif isinstance(payload, dict):
        hist = payload.get("historical", [])
    else:
        hist = []
    if not isinstance(hist, list) or len(hist) == 0:
        raise RuntimeError(f"No historical data returned for {symbol}")

    frame = pd.DataFrame(hist)
    need = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{symbol} response missing fields: {missing}")
    out = frame[need].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"{symbol} dataset empty after parsing")
    if out["date"].duplicated().any():
        dups = int(out["date"].duplicated().sum())
        raise RuntimeError(f"{symbol} has duplicate dates ({dups})")
    if not out["date"].is_monotonic_increasing:
        raise RuntimeError(f"{symbol} date ordering is not monotonic increasing")
    return out


def main() -> None:
    api_key = _load_api_key()
    out_root = Path("data/cross_asset")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for symbol in SYMBOLS:
        df = _fetch_symbol(symbol=symbol, api_key=api_key)
        out_path = out_root / f"{symbol}.parquet"
        df.to_parquet(out_path, index=False)
        row = {
            "symbol": symbol,
            "start_date": str(df["date"].iloc[0].date()),
            "end_date": str(df["date"].iloc[-1].date()),
            "rows": int(len(df)),
            "path": str(out_path),
        }
        rows.append(row)
        print(
            f"{symbol}: start={row['start_date']} end={row['end_date']} "
            f"rows={row['rows']} path={row['path']}"
        )

    report = pd.DataFrame(rows)
    report_path = out_root / "download_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
