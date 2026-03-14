"""Fetch SPDR sector ETF data and store local parquet artifacts."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from quant_lab.data.fetch import fetch_ohlcv_with_summary
from quant_lab.data.parquet_store import EquityParquetStore
from quant_lab.research.sector_rotation import DEFAULT_SECTOR_UNIVERSE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch default SPDR sector ETF data into local parquet files.")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--data-source", default="default", choices=["default", "stooq"])
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--store-root", default="data/equities")
    p.add_argument("--out-root", default="data/sector_etfs")
    return p.parse_args()


def _resolve_cache_dir(data_source: str, cache_dir: str) -> str:
    if str(cache_dir).strip():
        return str(cache_dir).strip()
    src = str(data_source).lower().strip()
    if src == "stooq":
        return "data/cache/stooq"
    return "data/cache/current"


def main() -> None:
    args = parse_args()
    cache_dir = _resolve_cache_dir(data_source=args.data_source, cache_dir=args.cache_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ohlcv_map, summary = fetch_ohlcv_with_summary(
        tickers=DEFAULT_SECTOR_UNIVERSE,
        start=str(args.start),
        end=str(args.end),
        cache_dir=cache_dir,
        data_source=str(args.data_source),
        refresh=bool(args.refresh),
        bulk_prepare=False,
    )
    if not ohlcv_map:
        raise RuntimeError("No sector ETF data was fetched/loaded.")

    store = EquityParquetStore(root=Path(args.store_root))
    rows: list[dict[str, object]] = []
    for symbol in DEFAULT_SECTOR_UNIVERSE:
        frame = ohlcv_map.get(symbol)
        if frame is None or frame.empty:
            continue
        normalized = pd.DataFrame(
            {
                "date": pd.to_datetime(frame.index, errors="coerce"),
                "open": pd.to_numeric(frame["Open"], errors="coerce"),
                "high": pd.to_numeric(frame["High"], errors="coerce"),
                "low": pd.to_numeric(frame["Low"], errors="coerce"),
                "close": pd.to_numeric(frame["Close"], errors="coerce"),
                "volume": pd.to_numeric(frame["Volume"], errors="coerce"),
            }
        ).dropna(subset=["date"])
        normalized = normalized.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        if normalized.empty:
            continue

        normalized.to_parquet(out_root / f"{symbol}.parquet", index=False)
        upsert_df = normalized.copy()
        upsert_df["ticker"] = symbol
        upsert_df["source"] = f"sector_{str(args.data_source).lower().strip()}"
        store.upsert_daily_ohlcv(upsert_df)

        rows.append(
            {
                "symbol": symbol,
                "start_date": str(normalized["date"].iloc[0].date()),
                "end_date": str(normalized["date"].iloc[-1].date()),
                "rows": int(len(normalized)),
                "path": str(out_root / f"{symbol}.parquet"),
            }
        )
        print(
            f"{symbol}: start={rows[-1]['start_date']} end={rows[-1]['end_date']} "
            f"rows={rows[-1]['rows']} path={rows[-1]['path']}"
        )

    if not rows:
        raise RuntimeError("No sector ETF files were written.")

    store.refresh_metadata_from_daily(source=f"sector_{str(args.data_source).lower().strip()}")

    report = pd.DataFrame(rows)
    report_path = out_root / "download_report.csv"
    report.to_csv(report_path, index=False)

    missing = [s for s in DEFAULT_SECTOR_UNIVERSE if s not in report["symbol"].tolist()]
    print("\nSECTOR ETF FETCH SUMMARY")
    print("------------------------")
    print(f"Loaded symbols: {', '.join(report['symbol'].tolist())}")
    print(f"Missing symbols: {', '.join(missing) if missing else '(none)'}")
    print(f"Data source summary: {summary}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
