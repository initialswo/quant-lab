"""
Build local fundamentals parquet database (FMP default source, Tiingo optional).

Setup:
1) Create repo-root `.env`
2) Add `FMP_API_KEY=...` (and optionally `TIINGO_API_KEY=...`)
3) Run this script without passing the key on CLI (optional CLI override supported).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:  # pragma: no cover - exercised when dotenv is available in runtime env
    from dotenv import load_dotenv as _dotenv_load
except Exception:  # pragma: no cover - fallback path
    def _dotenv_load(dotenv_path: Path, override: bool = False) -> bool:
        p = Path(dotenv_path)
        if not p.exists():
            return False
        loaded_any = False
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip()
            if not key:
                continue
            if (not override) and (key in os.environ):
                continue
            os.environ[key] = val
            loaded_any = True
        return loaded_any

from quant_lab.data.fmp_fundamentals import (
    INTERNAL_COLUMNS as FMP_INTERNAL_COLUMNS,
    build_fmp_payload_summary,
    fetch_fmp_fundamentals_frame,
    normalize_internal_fundamentals_frame as normalize_fmp_internal_fundamentals_frame,
)
from quant_lab.data.fundamentals import normalize_ticker_symbol
from quant_lab.data.tiingo_fundamentals import (
    INTERNAL_COLUMNS as TIINGO_INTERNAL_COLUMNS,
    build_tiingo_payload_summary,
    fetch_tiingo_fundamentals_frame,
    normalize_internal_fundamentals_frame as normalize_tiingo_internal_fundamentals_frame,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build fundamentals parquet database from FMP (default) or Tiingo.",
    )
    p.add_argument("--source", default="fmp", choices=["fmp", "tiingo"])
    p.add_argument("--tickers-file", default="", help="CSV/TXT file with ticker symbols.")
    p.add_argument("--tickers", default="", help="Optional comma-separated tickers.")
    p.add_argument("--fmp-api-key", default="", help="Optional FMP API key override.")
    p.add_argument("--tiingo-api-key", default="", help="Optional Tiingo API key override.")
    p.add_argument(
        "--output",
        default="data/fundamentals/fundamentals.parquet",
        help="Output parquet path.",
    )
    p.add_argument("--available-lag-days", type=int, default=60)
    p.add_argument("--limit", type=int, default=120)
    p.add_argument(
        "--statement-period",
        choices=["annual", "quarter"],
        default="quarter",
        help="Statement periodicity where supported by provider (used by FMP).",
    )
    p.add_argument("--start-date", default="", help="Optional Tiingo statements startDate (YYYY-MM-DD).")
    p.add_argument("--end-date", default="", help="Optional Tiingo statements endDate (YYYY-MM-DD).")
    p.add_argument("--as-reported", type=int, choices=[0, 1], default=0, help="Use Tiingo asReported=true if set.")
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--pause-seconds", type=float, default=0.25)
    p.add_argument("--batch-pause-seconds", type=float, default=1.0)
    p.add_argument("--timeout-seconds", type=float, default=20.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--debug-raw", action="store_true", help="Save raw provider payload JSON and structural summaries.")
    p.add_argument("--debug-tickers", default="AAPL,MSFT,GOOGL")
    p.add_argument("--debug-output-dir", default="results/fundamentals_debug")
    return p


def _load_dotenv_from_repo_root(repo_root: Path | None = None) -> bool:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    return bool(_dotenv_load(dotenv_path=env_path, override=False))


def _resolve_api_key(source: str, fmp_cli: str | None, tiingo_cli: str | None) -> str:
    src = str(source).strip().lower()
    if src == "fmp":
        key = str(fmp_cli or "").strip() or str(os.getenv("FMP_API_KEY", "")).strip()
        if not key:
            raise ValueError("FMP API key is missing. Provide --fmp-api-key or set FMP_API_KEY in repo-root .env.")
        return key
    if src == "tiingo":
        key = str(tiingo_cli or "").strip() or str(os.getenv("TIINGO_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "Tiingo API key is missing. Provide --tiingo-api-key or set TIINGO_API_KEY in repo-root .env."
            )
        return key
    raise ValueError(f"Unsupported source: {source}")


def _extract_tickers_from_df(df: pd.DataFrame) -> list[str]:
    preferred_cols = ["ticker", "symbol", "Symbol", "Ticker", "SYMBOL", "TICKER"]
    col = next((c for c in preferred_cols if c in df.columns), df.columns[0] if len(df.columns) else None)
    if col is None:
        return []
    vals = [normalize_ticker_symbol(x) for x in df[col].astype(str).tolist() if str(x).strip()]
    return sorted(set([x for x in vals if x]))


def _load_tickers(tickers_file: str, tickers_csv: str) -> list[str]:
    from_file: list[str] = []
    if str(tickers_file).strip():
        p = Path(tickers_file)
        if not p.exists():
            raise FileNotFoundError(f"Tickers file not found: {p}")
        if p.suffix.lower() == ".csv":
            from_file = _extract_tickers_from_df(pd.read_csv(p))
        else:
            lines = p.read_text(encoding="utf-8").splitlines()
            from_file = sorted(
                set(
                    normalize_ticker_symbol(x)
                    for x in lines
                    if str(x).strip() and not str(x).strip().startswith("#")
                )
            )

    from_arg = sorted(
        set(
            normalize_ticker_symbol(x)
            for x in str(tickers_csv).split(",")
            if str(x).strip()
        )
    )
    out = sorted(set(from_file) | set(from_arg))
    if not out:
        raise ValueError("No tickers supplied. Provide --tickers-file and/or --tickers.")
    return out


def _parse_debug_tickers(raw: str) -> list[str]:
    out = sorted(set(normalize_ticker_symbol(x) for x in str(raw).split(",") if str(x).strip()))
    return out or ["AAPL", "MSFT", "GOOGL"]


def _print_summary(out: pd.DataFrame, out_path: Path) -> None:
    print("[INFO] Fundamentals database build complete.")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Total rows: {len(out)}")
    print(f"[INFO] Unique tickers: {out['ticker'].nunique() if not out.empty else 0}")
    if out.empty:
        return

    pmin = out["period_end"].min()
    pmax = out["period_end"].max()
    print(f"[INFO] period_end min/max: {pmin.date()} -> {pmax.date()}")

    null_counts = out[
        [
            "period_end",
            "available_date",
            "revenue",
            "cogs",
            "gross_profit",
            "total_assets",
            "net_income",
            "shares_outstanding",
        ]
    ].isna().sum()
    nonnull_counts = out[
        [
            "revenue",
            "cogs",
            "gross_profit",
            "total_assets",
            "shareholders_equity",
            "net_income",
            "shares_outstanding",
        ]
    ].notna().sum()
    print("[INFO] Null counts:")
    for k, v in null_counts.to_dict().items():
        print(f"  - {k}: {int(v)}")
    print("[INFO] Non-null counts (core numeric fields):")
    for k, v in nonnull_counts.to_dict().items():
        print(f"  - {k}: {int(v)}")

    print("[INFO] Sample rows:")
    print(out.head(5).to_string(index=False))


def _write_debug_payload(source: str, ticker: str, raw: dict, debug_output_dir: Path) -> None:
    src = str(source).lower()
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = debug_output_dir / f"{src}_{ticker.lower()}_raw.json"
    summary_path = debug_output_dir / f"{src}_{ticker.lower()}_summary.json"

    payload = {"source": src, "ticker": ticker, "raw": raw}
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if src == "fmp":
        summary = build_fmp_payload_summary(raw)
        status_str = f"income={summary.get('income_status')} balance={summary.get('balance_status')}"
    else:
        summary = build_tiingo_payload_summary(
            endpoint=str(raw.get("endpoint")),
            status_code=raw.get("status_code"),
            payload=raw.get("payload"),
        )
        summary["error"] = raw.get("error")
        status_str = f"status={summary.get('status_code')}"
    summary["source"] = src
    summary["ticker"] = ticker
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DEBUG] {src.upper()} {ticker} {status_str}")
    print(f"[DEBUG] wrote {payload_path}")
    print(f"[DEBUG] wrote {summary_path}")


def _fetch_one(source: str, session: requests.Session, ticker: str, api_key: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    src = str(source).lower()
    if src == "fmp":
        return fetch_fmp_fundamentals_frame(
            session=session,
            ticker=ticker,
            api_key=api_key,
            available_lag_days=int(args.available_lag_days),
            limit=int(args.limit),
            statement_period=str(args.statement_period),
            timeout_seconds=float(args.timeout_seconds),
            max_retries=int(args.max_retries),
        )
    return fetch_tiingo_fundamentals_frame(
        session=session,
        ticker=ticker,
        api_key=api_key,
        available_lag_days=int(args.available_lag_days),
        as_reported=bool(int(args.as_reported)),
        start_date=str(args.start_date).strip() or None,
        end_date=str(args.end_date).strip() or None,
        timeout_seconds=float(args.timeout_seconds),
        max_retries=int(args.max_retries),
    )


def main() -> None:
    args = _build_parser().parse_args()
    _ = _load_dotenv_from_repo_root()
    source = str(args.source).lower()
    api_key = _resolve_api_key(source=source, fmp_cli=args.fmp_api_key, tiingo_cli=args.tiingo_api_key)

    if int(args.available_lag_days) < 0:
        raise ValueError("--available-lag-days must be >= 0")
    if int(args.limit) <= 0:
        raise ValueError("--limit must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")

    tickers_supplied = bool(str(args.tickers_file).strip() or str(args.tickers).strip())
    debug_tickers = _parse_debug_tickers(str(args.debug_tickers))
    tickers = _load_tickers(str(args.tickers_file), str(args.tickers)) if tickers_supplied else debug_tickers

    total = len(tickers)
    batches = int(math.ceil(total / int(args.batch_size)))
    print(f"[INFO] Source: {source}")
    if source == "fmp":
        print(f"[INFO] FMP params: period={args.statement_period} limit={int(args.limit)}")
    else:
        print(
            f"[INFO] Tiingo params: asReported={bool(int(args.as_reported))} "
            f"startDate={args.start_date or '-'} endDate={args.end_date or '-'}"
        )
    print(f"[INFO] Loaded {total} tickers.")
    print(f"[INFO] Processing in {batches} batch(es) of up to {int(args.batch_size)}.")

    session = requests.Session()
    debug_dir = Path(str(args.debug_output_dir))
    debug_set = set(debug_tickers)

    if bool(args.debug_raw):
        print(f"[INFO] Debug raw mode enabled. Debug tickers: {','.join(debug_tickers)}")
        for tkr in debug_tickers:
            _, raw_dbg = _fetch_one(source=source, session=session, ticker=tkr, api_key=api_key, args=args)
            _write_debug_payload(source=source, ticker=tkr, raw=raw_dbg, debug_output_dir=debug_dir)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for b in range(batches):
        start_i = b * int(args.batch_size)
        end_i = min((b + 1) * int(args.batch_size), total)
        batch = tickers[start_i:end_i]
        print(f"[INFO] Batch {b + 1}/{batches} ({start_i + 1}-{end_i})")
        for i, ticker in enumerate(batch, start=start_i + 1):
            print(f"[INFO] [{i}/{total}] Fetching {ticker} ...")
            try:
                one, raw = _fetch_one(source=source, session=session, ticker=ticker, api_key=api_key, args=args)
                if bool(args.debug_raw) and ticker in debug_set:
                    _write_debug_payload(source=source, ticker=ticker, raw=raw, debug_output_dir=debug_dir)
                if one.empty:
                    failed.append(ticker)
                else:
                    frames.append(one)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed ticker {ticker}: {exc}")
                failed.append(ticker)
            time.sleep(max(0.0, float(args.pause_seconds)))
        if b < batches - 1:
            time.sleep(max(0.0, float(args.batch_pause_seconds)))

    cols = FMP_INTERNAL_COLUMNS if source == "fmp" else TIINGO_INTERNAL_COLUMNS
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    if source == "fmp":
        out = normalize_fmp_internal_fundamentals_frame(combined)
    else:
        out = normalize_tiingo_internal_fundamentals_frame(combined)

    out_path = Path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    _print_summary(out, out_path)

    if failed:
        sample = ", ".join(failed[:10])
        print(f"[WARN] Failed/empty tickers: {len(failed)} (sample: {sample})")


if __name__ == "__main__":
    main()
