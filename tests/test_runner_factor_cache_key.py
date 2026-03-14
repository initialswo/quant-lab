from __future__ import annotations

from quant_lab.engine.runner import _factor_raw_cache_key


def test_factor_raw_cache_key_excludes_top_n_breadth_parameter() -> None:
    base = dict(
        factor_names=["momentum_12_1", "reversal_1m", "low_vol_20", "gross_profitability"],
        raw_factor_params={},
        start="2005-01-01",
        end="2024-12-31",
        data_source="parquet",
        data_cache_dir="data/equities",
        price_fill_mode="ffill",
        drop_bad_tickers=False,
        universe_mode="dynamic",
        max_tickers=2000,
        close_columns=["A", "B", "C"],
    )

    key1 = _factor_raw_cache_key(**base)
    key2 = _factor_raw_cache_key(**base)

    assert key1 == key2


def test_factor_raw_cache_key_changes_when_factor_inputs_change() -> None:
    base = dict(
        factor_names=["momentum_12_1", "reversal_1m"],
        raw_factor_params={},
        start="2005-01-01",
        end="2024-12-31",
        data_source="parquet",
        data_cache_dir="data/equities",
        price_fill_mode="ffill",
        drop_bad_tickers=False,
        universe_mode="static",
        max_tickers=2000,
        close_columns=["A", "B", "C"],
    )

    key1 = _factor_raw_cache_key(**base)
    key2 = _factor_raw_cache_key(**{**base, "factor_names": ["momentum_12_1"]})
    key3 = _factor_raw_cache_key(**{**base, "close_columns": ["A", "B"]})

    assert key1 != key2
    assert key1 != key3
