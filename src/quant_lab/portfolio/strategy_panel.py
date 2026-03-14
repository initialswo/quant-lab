"""Strategy return panel alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class StrategyPanel:
    """Aligned panel of strategy daily return series."""

    returns: pd.DataFrame

    def __init__(self, series_map: dict[str, pd.Series]):
        if not series_map:
            raise ValueError("series_map cannot be empty.")
        frames: list[pd.Series] = []
        for name, series in series_map.items():
            if not str(name).strip():
                raise ValueError("strategy names must be non-empty.")
            s = pd.Series(series, dtype=float).copy()
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[~s.index.isna()].sort_index()
            s.name = str(name).strip()
            frames.append(s)
        panel = pd.concat(frames, axis=1, join="inner").dropna(how="any").copy()
        panel.index = pd.DatetimeIndex(panel.index)
        panel.index.name = "date"
        object.__setattr__(self, "returns", panel)

    @property
    def strategies(self) -> list[str]:
        return [str(c) for c in self.returns.columns]

    @property
    def start_date(self) -> pd.Timestamp | None:
        if self.returns.empty:
            return None
        return pd.Timestamp(self.returns.index.min())

    @property
    def end_date(self) -> pd.Timestamp | None:
        if self.returns.empty:
            return None
        return pd.Timestamp(self.returns.index.max())

    @classmethod
    def from_csv_files(cls, mapping: dict[str, Path]) -> StrategyPanel:
        """Build StrategyPanel from CSV artifacts with date/returns columns."""
        if not mapping:
            raise ValueError("mapping cannot be empty.")
        series_map: dict[str, pd.Series] = {}
        for name, path in mapping.items():
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"strategy returns file not found: {p}")
            df = pd.read_csv(p, parse_dates=["date"])
            need = {"date", "returns"}
            if not need.issubset(set(df.columns)):
                raise ValueError(f"{p} must contain columns: {sorted(need)}")
            s = (
                pd.to_numeric(df["returns"], errors="coerce")
                .rename(str(name))
            )
            s.index = pd.to_datetime(df["date"], errors="coerce")
            s = s[~s.index.isna()]
            series_map[str(name)] = s
        return cls(series_map=series_map)

