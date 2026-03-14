"""Default research-policy windows for Quant Lab experiments.

These are convenience defaults for research scripts, not engine constraints.

Price-only research can start earlier because the required inputs are broadly
available in the price history. Fundamental-only and mixed research start later
to reflect the thinner and less stable fundamentals coverage in earlier years.
"""

from __future__ import annotations

from typing import Mapping


PRICE_RESEARCH_WINDOW: dict[str, str] = {
    "start": "2005-01-01",
    "end": "2024-12-31",
}

FUNDAMENTAL_RESEARCH_WINDOW: dict[str, str] = {
    "start": "2010-01-01",
    "end": "2024-12-31",
}

MIXED_RESEARCH_WINDOW: dict[str, str] = {
    "start": "2010-01-01",
    "end": "2024-12-31",
}


def _copy_window(window: Mapping[str, str]) -> dict[str, str]:
    return {
        "start": str(window["start"]),
        "end": str(window["end"]),
    }


def get_price_window() -> dict[str, str]:
    return _copy_window(PRICE_RESEARCH_WINDOW)


def get_fundamental_window() -> dict[str, str]:
    return _copy_window(FUNDAMENTAL_RESEARCH_WINDOW)


def get_mixed_window() -> dict[str, str]:
    return _copy_window(MIXED_RESEARCH_WINDOW)


def resolve_window(
    default_window: Mapping[str, str],
    start: str | None = None,
    end: str | None = None,
) -> tuple[str, str]:
    window = _copy_window(default_window)
    return (
        str(start) if start is not None else window["start"],
        str(end) if end is not None else window["end"],
    )
