from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.config import SILVER_DIR
from core.audit import AuditLogger

BRONZE_SUFFIX_RE = re.compile(r"_bronze_\d{8}_\d{6}$")


def _match_table_name(bronze_stem: str, source_table: str) -> bool:
    if bronze_stem == source_table:
        return True
    base = BRONZE_SUFFIX_RE.sub("", bronze_stem)
    return base == source_table


def _apply_rule(df: pd.DataFrame, col: str, logic: str) -> pd.Series:
    orig_logic = logic or ""
    logic = orig_logic.lower()
    series = df[col] if col in df.columns else pd.Series([pd.NA] * len(df))

    # mapping rule support: logic starts with 'map:' or contains mapping pairs like Key=Value;Key2=Value2
    if orig_logic.startswith("map:") or logic.startswith("map:") or ("=" in orig_logic and ";" in orig_logic):
        map_text = orig_logic.split(":", 1)[1] if orig_logic.startswith("map:") or logic.startswith("map:") else orig_logic
        mapping = {}
        for part in map_text.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                mapping[k.strip()] = v.strip()

        # apply mapping with case-insensitive match, preserve mapped value
        def map_value(x):
            if pd.isna(x):
                return x
            xs = str(x)
            if xs in mapping:
                return mapping[xs]
            low = xs.lower()
            for lk, rv in mapping.items():
                if lk.lower() == low:
                    return rv
            return xs

        return series.astype(object).apply(map_value)

    if "drop null" in logic or "dropna" in logic or "drop null" == logic:
        # handled at DataFrame level
        return series
    if "fill" in logic:
        if "mean" in logic:
            val = series.mean()
            return series.fillna(val)
        if "median" in logic:
            val = series.median()
            return series.fillna(val)
        if "mode" in logic:
            modes = series.mode()
            val = modes.iloc[0] if not modes.empty else pd.NA
            return series.fillna(val)
        if "0" in logic or "zero" in logic:
            return series.fillna(0)
    if "date" in logic or "datetime" in logic or "standardize_date" in logic:
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.dt.strftime("%Y-%m-%d")
    if "int" in logic and "float" not in logic:
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if "float" in logic or "numeric" in logic or "clean_numeric" in logic:
        return pd.to_numeric(series, errors="coerce")
    if "lower" in logic or "lowercase" in logic:
        return series.astype(str).str.lower()
    if "upper" in logic or "uppercase" in logic:
        return series.astype(str).str.upper()
    if "title" in logic or "title case" in logic:
        return series.astype(str).str.title()
    if "strip" in logic or "trim" in logic:
        return series.astype(str).str.strip()
    # fallback: return original column
    return series


def run(bronze_paths: List[str], sttm_path: str, run_id: str) -> List[str]:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    sttm_p = Path(sttm_path)
    with sttm_p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        sttm_rows = [r for r in reader]

    written: List[str] = []

    for bp in bronze_paths:
        bpath = Path(bp)
        stem = bpath.stem
        # Determine base name (strip _bronze_... if present)
        base = BRONZE_SUFFIX_RE.sub("", stem)

        # Determine relevant STTM rows: match source_table to stem or base
        rows = [r for r in sttm_rows if _match_table_name(stem, (r.get("source_table") or "").strip())]
        if not rows:
            # no rules for this bronze file; skip
            continue

        df = pd.read_parquet(bpath)
        original_len = len(df)

        # Apply per-column rules while preserving all original bronze columns.
        dropna_subsets: List[str] = []
        for r in rows:
            src_col = (r.get("source_column") or "").strip()
            tgt_col = (r.get("target_column") or src_col).strip()
            orig_logic = (r.get("transformation_logic") or "").strip()
            logic = orig_logic.lower()
            t_type = (r.get("transformation_type") or "").strip().lower()

            if t_type == "metadata_injection":
                # metadata handled at Bronze; Silver keeps it unless STTM says otherwise
                continue

            # If logic indicates drop null, collect for later
            if "drop null" in logic or "dropna" in logic:
                dropna_subsets.append(src_col)

            # Apply transformation
            series = _apply_rule(df, src_col, orig_logic)
            df[tgt_col] = series

        # handle drop null across collected subsets
        if dropna_subsets:
            df = df.dropna(subset=dropna_subsets)

        # deduplicate if any rule contains 'dedup' or 'deduplicate'
        if any("dedup" in (r.get("transformation_logic") or "").lower() or "deduplicate" in (r.get("transformation_logic") or "").lower() for r in rows):
            df = df.drop_duplicates()

        # Insert surrogate key as first column
        pk_name = f"pk_{base}_silver_id"
        df.insert(0, pk_name, range(1, len(df) + 1))

        # Preserve all columns from the transformed silver dataset.
        df_final = df

        out_path = SILVER_DIR / f"{base}_silver_{run_id}.parquet"
        df_final.to_parquet(out_path, index=False)
        written.append(str(out_path))

        AuditLogger().log(
            agent="silver_agent",
            action="run",
            bronze_path=str(bpath),
            output_path=str(out_path),
            input_shape=[int(original_len), int(len(df.columns))],
            output_shape=[int(len(df_final)), int(len(df_final.columns))],
            run_id=run_id,
        )

    return written
