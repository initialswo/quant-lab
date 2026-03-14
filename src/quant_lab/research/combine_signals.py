"""Reusable multi-factor signal combination helper for research workflows."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from quant_lab.factors.combine import combine_factor_scores
from quant_lab.factors.normalize import normalize_scores


def _resolve_weights(
    names: list[str],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if not names:
        raise ValueError("No factor panels were provided.")
    if weights is None:
        eq = 1.0 / float(len(names))
        return {n: eq for n in names}
    missing = [n for n in names if n not in weights]
    extra = [k for k in weights.keys() if k not in names]
    if missing or extra:
        raise ValueError(
            f"weights keys must match factor names exactly; missing={missing}, extra={extra}"
        )
    w = np.asarray([float(weights[n]) for n in names], dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite numbers.")
    total = float(w.sum())
    if abs(total) < 1e-12:
        raise ValueError("weights sum must be non-zero.")
    w = w / total
    return {n: float(v) for n, v in zip(names, w)}


def combine_factor_panels(
    factor_panels: dict[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
    normalize: str = "zscore",
) -> pd.DataFrame:
    """
    Align, normalize each factor panel cross-sectionally, then combine by weights.

    Returned panel shape follows the first factor panel's index/columns (runner-compatible).
    """
    if not factor_panels:
        raise ValueError("factor_panels cannot be empty.")
    names = list(factor_panels.keys())
    norm = {
        n: normalize_scores(factor_panels[n].astype(float), method=normalize)
        for n in names
    }
    w = _resolve_weights(names=names, weights=weights)
    return combine_factor_scores(norm, w, require_all_factors=False)

