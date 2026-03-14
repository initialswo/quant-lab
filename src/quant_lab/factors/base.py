"""Factor plugin protocol."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class FactorProtocol(Protocol):
    """Lightweight contract each factor module must satisfy."""

    FACTOR_NAME: str

    @staticmethod
    def compute(close: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Return a float score DataFrame aligned to `close`."""
        ...
