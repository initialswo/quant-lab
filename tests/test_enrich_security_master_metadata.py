from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from quant_lab.data.security_master_metadata import enrich_security_master_metadata


def test_enrich_security_master_metadata_fills_missing_and_preserves_existing() -> None:
    security_master = pd.DataFrame(
        [
            {
                'ticker_id': 'T0000001',
                'canonical_symbol': 'AAA',
                'selected_raw_symbol': 'AAA-US',
                'name': pd.NA,
                'exchange': pd.NA,
                'sector': pd.NA,
                'industry': pd.NA,
                'updated_at': '2026-03-15T00:00:00+00:00',
            },
            {
                'ticker_id': 'T0000002',
                'canonical_symbol': 'BBB',
                'selected_raw_symbol': 'BBB-US',
                'name': 'Existing BBB',
                'exchange': 'NYSE',
                'sector': pd.NA,
                'industry': pd.NA,
                'updated_at': '2026-03-15T00:00:00+00:00',
            },
        ]
    )
    bulk_meta = pd.DataFrame(
        [
            {'norm': 'AAA', 'name': 'AAA Corp', 'sector': 'Field not available for free/evaluation', 'industry': 'Semiconductors'},
            {'norm': 'BBB', 'name': 'BBB Corp', 'sector': 'Industrials', 'industry': 'Machinery'},
        ]
    )
    daily_meta = pd.DataFrame(
        [
            {'norm': 'AAA', 'name': 'AAA Daily Name', 'exchange': 'NASDAQ'},
            {'norm': 'BBB', 'name': 'BBB Daily Name', 'exchange': 'NASDAQ'},
        ]
    )

    enriched, summary = enrich_security_master_metadata(
        security_master=security_master,
        bulk_meta=bulk_meta,
        daily_meta=daily_meta,
    )

    aaa = enriched.loc[enriched['canonical_symbol'] == 'AAA'].iloc[0]
    assert aaa['name'] == 'AAA Daily Name'
    assert aaa['exchange'] == 'NASDAQ'
    assert pd.isna(aaa['sector'])
    assert aaa['industry'] == 'Semiconductors'

    bbb = enriched.loc[enriched['canonical_symbol'] == 'BBB'].iloc[0]
    assert bbb['name'] == 'Existing BBB'
    assert bbb['exchange'] == 'NYSE'
    assert bbb['sector'] == 'Industrials'
    assert bbb['industry'] == 'Machinery'

    assert summary['before_nulls'] == {'name': 1, 'exchange': 1, 'sector': 2, 'industry': 2}
    assert summary['after_nulls'] == {'name': 0, 'exchange': 0, 'sector': 1, 'industry': 0}
    assert summary['changed_rows'] == 2
