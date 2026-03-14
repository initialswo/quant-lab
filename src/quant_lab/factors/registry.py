"""Factor discovery and execution helpers."""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from types import ModuleType
from typing import Any

import pandas as pd

import quant_lab.factors as factors_pkg

_EXCLUDED_MODULES = {"base", "registry", "normalize", "combine", "neutralize", "orthogonalize", "__init__"}


def _iter_factor_modules() -> list[tuple[str, ModuleType]]:
    modules: list[tuple[str, ModuleType]] = []
    for module_info in pkgutil.iter_modules(factors_pkg.__path__):
        if module_info.name in _EXCLUDED_MODULES:
            continue
        module_name = f"{factors_pkg.__name__}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive import guard
            warnings.warn(
                f"Skipping factor module '{module_name}': import failed ({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        modules.append((module_name, module))
    return modules


def _validate_factor_module(module_name: str, module: ModuleType) -> tuple[str, object] | None:
    factor_name = getattr(module, "FACTOR_NAME", None)
    compute = getattr(module, "compute", None)
    if not isinstance(factor_name, str) or not factor_name.strip():
        warnings.warn(
            f"Skipping factor module '{module_name}': missing valid FACTOR_NAME",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not callable(compute):
        warnings.warn(
            f"Skipping factor module '{module_name}': missing callable compute(close, **params)",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return factor_name.strip(), compute


def list_factors() -> list[str]:
    """Return all discoverable factor names."""
    names: set[str] = set()
    for module_name, module in _iter_factor_modules():
        valid = _validate_factor_module(module_name, module)
        if valid is None:
            continue
        names.add(valid[0])
    return sorted(names)


def load_factor(name: str) -> ModuleType:
    """Load one factor module by FACTOR_NAME."""
    target = name.strip()
    for module_name, module in _iter_factor_modules():
        valid = _validate_factor_module(module_name, module)
        if valid is None:
            continue
        factor_name, _ = valid
        if factor_name == target:
            return module
    available = ", ".join(list_factors()) or "(none)"
    raise ValueError(f"Unknown factor '{name}'. Available factors: {available}")


def compute_factor(name: str, close: pd.DataFrame, **params: Any) -> pd.DataFrame:
    """Compute a factor by name and return a float DataFrame aligned to `close`."""
    module = load_factor(name)
    compute = getattr(module, "compute")
    scores = compute(close, **params)
    if not isinstance(scores, pd.DataFrame):
        raise TypeError(f"Factor '{name}' returned {type(scores)!r}; expected pandas.DataFrame.")
    aligned = scores.reindex(index=close.index, columns=close.columns)
    return aligned.astype(float)



def compute_factors(
    factor_names: list[str],
    close: pd.DataFrame,
    factor_params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute multiple factors by name with optional per-factor params."""
    params_map = factor_params or {}
    out: dict[str, pd.DataFrame] = {}
    for name in factor_names:
        params = params_map.get(name, {}) or {}
        if not isinstance(params, dict):
            raise TypeError(f"factor_params[{name!r}] must be a dict, got {type(params)!r}")
        out[name] = compute_factor(name, close, **params)
    return out
