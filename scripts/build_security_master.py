"""Build canonical security master for the local warehouse.

Phase 1 uses existing local parquet datasets only.
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def normalize_symbol(raw: object) -> str:
    s = str(raw or "").strip().upper()
    if s.endswith(".US"):
        s = s[:-3]
    s = s.replace(".", "-")
    return s


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required dataset: {path}")
    if columns is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=columns)


def _collect_symbol_rows(
    daily: pd.DataFrame,
    metadata: pd.DataFrame,
    membership: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    d = daily[["ticker"]].copy()
    d["source_table"] = "daily_ohlcv"
    frames.append(d)

    m = metadata[["ticker"]].copy()
    m["source_table"] = "metadata"
    frames.append(m)

    u = membership[["ticker"]].copy()
    u["source_table"] = "universe_membership"
    frames.append(u)

    f = fundamentals[["ticker"]].copy()
    f["source_table"] = "fundamentals_fmp"
    frames.append(f)

    rows = pd.concat(frames, ignore_index=True)
    rows["raw_source_symbol"] = rows["ticker"].astype(str).str.strip().str.upper()
    rows["canonical_symbol"] = rows["raw_source_symbol"].map(normalize_symbol)
    rows = rows.loc[rows["canonical_symbol"].ne("")].copy()
    return rows[["source_table", "raw_source_symbol", "canonical_symbol"]]


def _collapse_symbol(raw_symbol: str) -> str:
    txt = str(raw_symbol).strip().upper()
    if txt.endswith(".US"):
        txt = txt[:-3]
    txt = txt.replace(".", "-")
    txt = re.sub(r"[^A-Z0-9-]+", "", txt)
    return txt


def _build_id_map(canonical_symbols: list[str], existing: pd.DataFrame | None = None) -> pd.DataFrame:
    if existing is None or existing.empty:
        existing_map = pd.DataFrame(columns=["ticker_id", "canonical_symbol"])
    else:
        existing_map = existing[["ticker_id", "canonical_symbol"]].copy()
        existing_map["canonical_symbol"] = existing_map["canonical_symbol"].astype(str).map(normalize_symbol)
        existing_map = existing_map.drop_duplicates(subset=["canonical_symbol"], keep="first")

    known = set(existing_map["canonical_symbol"].astype(str))
    new_symbols = sorted(s for s in canonical_symbols if s not in known)

    if existing_map.empty:
        next_id = 1
    else:
        nums = (
            existing_map["ticker_id"]
            .astype(str)
            .str.extract(r"(\d+)$", expand=False)
            .dropna()
            .astype(int)
        )
        next_id = int(nums.max()) + 1 if not nums.empty else 1

    new_rows = []
    for s in new_symbols:
        new_rows.append({"ticker_id": f"T{next_id:07d}", "canonical_symbol": s})
        next_id += 1

    all_map = pd.concat([existing_map, pd.DataFrame(new_rows)], ignore_index=True)
    all_map = all_map.drop_duplicates(subset=["canonical_symbol"], keep="first")
    return all_map.sort_values("canonical_symbol").reset_index(drop=True)


def _build_stability_report(existing: pd.DataFrame | None, current: pd.DataFrame) -> pd.DataFrame:
    curr = current[["ticker_id", "canonical_symbol"]].copy()
    if existing is None or existing.empty:
        out = curr.copy()
        out["previous_ticker_id"] = pd.NA
        out["status"] = "new"
        out["stability_issue"] = False
        return out[["canonical_symbol", "previous_ticker_id", "ticker_id", "status", "stability_issue"]]

    prev = existing[["ticker_id", "canonical_symbol"]].copy()
    prev["canonical_symbol"] = prev["canonical_symbol"].astype(str).map(normalize_symbol)
    prev = prev.drop_duplicates(subset=["canonical_symbol"], keep="first")
    merged = curr.merge(
        prev.rename(columns={"ticker_id": "previous_ticker_id"}),
        on="canonical_symbol",
        how="left",
    )
    merged["status"] = "new"
    reused = merged["previous_ticker_id"].notna() & merged["previous_ticker_id"].eq(merged["ticker_id"])
    changed = merged["previous_ticker_id"].notna() & (~merged["previous_ticker_id"].eq(merged["ticker_id"]))
    merged.loc[reused, "status"] = "reused"
    merged.loc[changed, "status"] = "changed"
    merged["stability_issue"] = merged["status"].eq("changed")
    return merged[["canonical_symbol", "previous_ticker_id", "ticker_id", "status", "stability_issue"]]


def _build_collision_report(symbol_rows: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    grp = (
        symbol_rows.groupby("canonical_symbol")
        .agg(
            raw_symbol_count=("raw_source_symbol", "nunique"),
            source_table_count=("source_table", "nunique"),
            raw_symbol_examples=("raw_source_symbol", lambda x: ";".join(sorted(set(x.astype(str)))[:20])),
            source_tables=("source_table", lambda x: ";".join(sorted(set(x.astype(str))))),
        )
        .reset_index()
    )
    grp = grp.merge(id_map[["canonical_symbol", "ticker_id"]], on="canonical_symbol", how="left")
    grp["collision_resolved"] = grp["raw_symbol_count"] > 1

    def _is_ambiguous_row(row: pd.Series) -> bool:
        if int(row.get("raw_symbol_count", 0)) <= 1:
            return False
        examples = [x for x in str(row.get("raw_symbol_examples", "")).split(";") if x]
        collapsed = sorted({_collapse_symbol(x) for x in examples})
        # If normalized-collapsed aliases diverge, require manual review.
        return len(collapsed) > 1

    grp["ambiguous_mapping_flag"] = grp.apply(_is_ambiguous_row, axis=1)
    grp["source"] = "phase1_local"
    return grp[
        [
            "ticker_id",
            "canonical_symbol",
            "raw_symbol_count",
            "source_table_count",
            "raw_symbol_examples",
            "source_tables",
            "collision_resolved",
            "ambiguous_mapping_flag",
            "source",
        ]
    ].sort_values(["ambiguous_mapping_flag", "raw_symbol_count", "canonical_symbol"], ascending=[False, False, True]).reset_index(drop=True)


def _to_bool_flag(val: object) -> bool:
    if isinstance(val, bool):
        return bool(val)
    if pd.isna(val):
        return False
    txt = str(val).strip().lower()
    return txt in {"1", "true", "t", "yes", "y"}


def _consolidate_metadata(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolidate metadata rows by canonical symbol with deterministic precedence.

    Policy:
    1) Normalize ticker to canonical_symbol.
    2) Prefer row with latest `last_date` (most up-to-date lifecycle).
    3) Tie-breaker: prefer non-.US raw symbol over .US suffix.
    4) Tie-breaker: earliest `first_date`, then lexical ticker.
    5) Consolidated active_flag is taken from the selected representative row.
    """
    md = metadata.copy()
    md["ticker"] = md["ticker"].astype(str).str.strip().str.upper()
    md["canonical_symbol"] = md["ticker"].map(normalize_symbol)
    md["first_date"] = pd.to_datetime(md["first_date"], errors="coerce").dt.normalize()
    md["last_date"] = pd.to_datetime(md["last_date"], errors="coerce").dt.normalize()
    md["active_flag"] = md["active_flag"].map(_to_bool_flag)
    md["_is_us_suffix"] = md["ticker"].str.endswith(".US")

    def _pick(g: pd.DataFrame) -> pd.Series:
        ordered = g.sort_values(
            ["last_date", "_is_us_suffix", "first_date", "ticker"],
            ascending=[False, True, True, True],
        )
        return ordered.iloc[0]

    picked = (
        md.groupby("canonical_symbol", as_index=False, group_keys=False)
        .apply(_pick)
        .reset_index(drop=True)
    )
    picked = picked[
        [
            "canonical_symbol",
            "ticker",
            "name",
            "exchange",
            "sector",
            "industry",
            "first_date",
            "last_date",
            "active_flag",
        ]
    ].rename(columns={"ticker": "selected_raw_symbol", "active_flag": "consolidated_active_flag"})

    counts = (
        md.groupby("canonical_symbol")
        .agg(
            metadata_row_count=("ticker", "size"),
            raw_symbol_count=("ticker", "nunique"),
            active_true_count=("active_flag", lambda x: int(pd.Series(x).astype(bool).sum())),
            active_false_count=("active_flag", lambda x: int((~pd.Series(x).astype(bool)).sum())),
            raw_symbol_examples=("ticker", lambda x: ";".join(sorted(set(x.astype(str)))[:20])),
        )
        .reset_index()
    )
    counts["active_flag_conflict"] = (
        (counts["active_true_count"] > 0) & (counts["active_false_count"] > 0)
    )
    counts["consolidation_rule"] = "latest_last_date_then_symbol_preference"

    consolidated = picked.merge(counts, on="canonical_symbol", how="left")
    return consolidated, counts


def build_security_master(
    equities_root: Path,
    fundamentals_path: Path,
    warehouse_root: Path,
    existing_security_master_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    daily = _read_parquet(equities_root / "daily_ohlcv.parquet", columns=["date", "ticker"])
    metadata = _read_parquet(
        equities_root / "metadata.parquet",
        columns=["ticker", "name", "exchange", "sector", "industry", "first_date", "last_date", "active_flag"],
    )
    membership = _read_parquet(equities_root / "universe_membership.parquet", columns=["ticker"])
    fundamentals = _read_parquet(fundamentals_path, columns=["ticker"])

    symbol_rows = _collect_symbol_rows(
        daily=daily,
        metadata=metadata,
        membership=membership,
        fundamentals=fundamentals,
    )

    existing_path = (
        Path(existing_security_master_path)
        if existing_security_master_path is not None
        else (warehouse_root / "security_master.parquet")
    )
    existing = pd.read_parquet(existing_path) if existing_path.exists() else None

    id_map = _build_id_map(
        canonical_symbols=sorted(symbol_rows["canonical_symbol"].unique().tolist()),
        existing=existing,
    )
    stability = _build_stability_report(existing=existing, current=id_map)
    collisions = _build_collision_report(symbol_rows=symbol_rows, id_map=id_map)

    raw_counts = (
        symbol_rows.groupby("canonical_symbol")["raw_source_symbol"]
        .agg(lambda x: sorted(set(x.astype(str)))[:10])
        .rename("raw_symbols")
        .reset_index()
    )
    raw_example = raw_counts.copy()
    raw_example["raw_symbol_example"] = raw_example["raw_symbols"].map(lambda xs: xs[0] if len(xs) else "")
    raw_example["raw_symbol_count"] = raw_example["raw_symbols"].map(len)

    price_span = daily.copy()
    price_span["canonical_symbol"] = price_span["ticker"].map(normalize_symbol)
    price_span["date"] = pd.to_datetime(price_span["date"], errors="coerce").dt.normalize()
    price_span = (
        price_span.groupby("canonical_symbol")["date"]
        .agg(price_first_date="min", price_last_date="max")
        .reset_index()
    )

    md_consolidated, md_counts = _consolidate_metadata(metadata)

    sm = id_map.merge(raw_example[["canonical_symbol", "raw_symbol_example", "raw_symbol_count"]], on="canonical_symbol", how="left")
    sm = sm.merge(price_span, on="canonical_symbol", how="left")
    sm = sm.merge(
        md_consolidated[
            [
                "canonical_symbol",
                "selected_raw_symbol",
                "name",
                "exchange",
                "sector",
                "industry",
                "first_date",
                "last_date",
                "consolidated_active_flag",
                "metadata_row_count",
                "active_flag_conflict",
                "consolidation_rule",
            ]
        ],
        on="canonical_symbol",
        how="left",
    )

    sm["source"] = "phase1_local"
    sm["created_at"] = datetime.now(UTC).isoformat()
    sm["updated_at"] = sm["created_at"]

    cols = [
        "ticker_id",
        "canonical_symbol",
        "raw_symbol_example",
        "raw_symbol_count",
        "selected_raw_symbol",
        "name",
        "exchange",
        "sector",
        "industry",
        "first_date",
        "last_date",
        "consolidated_active_flag",
        "metadata_row_count",
        "active_flag_conflict",
        "consolidation_rule",
        "price_first_date",
        "price_last_date",
        "source",
        "created_at",
        "updated_at",
    ]
    sm = sm[cols].sort_values("ticker_id").reset_index(drop=True)

    diag = {
        "unique_canonical_symbols": int(sm["canonical_symbol"].nunique()),
        "existing_mapping_rows": int(0 if existing is None else len(existing)),
        "new_mapping_rows": int(len(sm) - (0 if existing is None else len(existing))),
        "ids_reused": int(stability["status"].eq("reused").sum()),
        "ids_new": int(stability["status"].eq("new").sum()),
        "ids_changed": int(stability["status"].eq("changed").sum()),
        "collisions_resolved": int(collisions["collision_resolved"].sum()),
        "ambiguous_mappings": int(collisions["ambiguous_mapping_flag"].sum()),
        "metadata_active_flag_conflicts": int(md_counts["active_flag_conflict"].sum()),
    }
    return sm, stability, collisions, diag


def main() -> None:
    p = argparse.ArgumentParser(description="Build warehouse security_master from local parquet datasets.")
    p.add_argument("--equities-root", default="data/equities")
    p.add_argument("--fundamentals-path", default="data/fundamentals/fundamentals_fmp.parquet")
    p.add_argument("--warehouse-root", default="data/warehouse")
    p.add_argument("--existing-security-master-path", default="")
    args = p.parse_args()

    equities_root = Path(args.equities_root)
    fundamentals_path = Path(args.fundamentals_path)
    warehouse_root = Path(args.warehouse_root)
    warehouse_root.mkdir(parents=True, exist_ok=True)

    sm, stability, collisions, diag = build_security_master(
        equities_root=equities_root,
        fundamentals_path=fundamentals_path,
        warehouse_root=warehouse_root,
        existing_security_master_path=(
            Path(args.existing_security_master_path)
            if str(args.existing_security_master_path).strip()
            else None
        ),
    )

    out_path = warehouse_root / "security_master.parquet"
    sm.to_parquet(out_path, index=False)
    stability_path = warehouse_root / "ticker_id_stability_report.parquet"
    collisions_path = warehouse_root / "symbol_collision_report.parquet"
    stability.to_parquet(stability_path, index=False)
    collisions.to_parquet(collisions_path, index=False)

    print(f"saved: {out_path}")
    print(f"saved: {stability_path}")
    print(f"saved: {collisions_path}")
    print(
        "security_master summary: "
        f"rows={len(sm)} unique_canonical={diag['unique_canonical_symbols']} "
        f"existing_map_rows={diag['existing_mapping_rows']} new_map_rows={diag['new_mapping_rows']} "
        f"ids_reused={diag['ids_reused']} ids_new={diag['ids_new']} ids_changed={diag['ids_changed']} "
        f"collisions_resolved={diag['collisions_resolved']} ambiguous={diag['ambiguous_mappings']}"
    )


if __name__ == "__main__":
    main()
