from __future__ import annotations

from quant_lab.engine.runner import _cache_get, _cache_put


def test_cache_get_put_roundtrip() -> None:
    cache: dict = {}
    key = ("k", 1)
    assert _cache_get(cache, "bucket", key) is None
    _cache_put(cache, "bucket", key, {"x": 1})
    out = _cache_get(cache, "bucket", key)
    assert out == {"x": 1}


def test_cache_isolated_by_bucket() -> None:
    cache: dict = {}
    _cache_put(cache, "a", "k", 1)
    _cache_put(cache, "b", "k", 2)
    assert _cache_get(cache, "a", "k") == 1
    assert _cache_get(cache, "b", "k") == 2
