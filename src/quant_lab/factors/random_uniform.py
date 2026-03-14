"""Random uniform placebo factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAME = "random_uniform"


def compute(close: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate deterministic placebo scores from Uniform(-1, 1).

    Returns a Date x Ticker panel aligned to `close`.
    """
    rng = np.random.default_rng(int(seed))
    values = rng.uniform(-1.0, 1.0, size=close.shape)
    return pd.DataFrame(values, index=close.index, columns=close.columns, dtype=float)

