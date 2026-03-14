"""Build a persisted Tiingo-based manifest for U.S. common equities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.data.tiingo_universe import (
    DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST,
    build_manifest_frame,
    fetch_tiingo_meta,
    filter_common_equities,
    load_tiingo_api_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_TIINGO_US_COMMON_EQUITY_MANIFEST))
    parser.add_argument("--summary-output", default="")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if str(args.summary_output).strip() else output_path.with_suffix('.summary.json')

    meta = fetch_tiingo_meta(load_tiingo_api_key())
    filtered, filter_stats = filter_common_equities(meta)
    manifest = build_manifest_frame(filtered)
    manifest.to_csv(output_path, index=False)

    summary = {
        'output_path': str(output_path),
        'summary_path': str(summary_path),
        'ticker_count': int(manifest['ticker'].nunique()) if 'ticker' in manifest.columns else int(len(manifest)),
        'columns': list(manifest.columns),
        'sample_head': manifest['ticker'].head(20).tolist() if 'ticker' in manifest.columns else [],
        'sample_tail': manifest['ticker'].tail(20).tolist() if 'ticker' in manifest.columns else [],
        'active_ticker_count': int(manifest['isActive'].fillna(False).astype(bool).sum()) if 'isActive' in manifest.columns else None,
        'inactive_ticker_count': int((~manifest['isActive'].fillna(False).astype(bool)).sum()) if 'isActive' in manifest.columns else None,
        'filter_stats': filter_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
