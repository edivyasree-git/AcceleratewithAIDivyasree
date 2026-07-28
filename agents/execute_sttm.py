from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.audit import AuditLogger
from core.config import LANDING_DIR, BRONZE_DIR


def _find_landing_file(table_name: str) -> Path:
    # Match exact stem or stem that startswith table_name
    for p in LANDING_DIR.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if stem == table_name or stem.startswith(table_name):
            return p
    raise FileNotFoundError(f"No landing file found for table '{table_name}' in {LANDING_DIR}")


def _apply_transformation(series: pd.Series, t_type: str, t_logic: str) -> pd.Series:
    if t_type in ("copy", "passthrough"):
        return series
    if t_type in ("type_cast", "datetime", "standardize_date"):
        # Try parsing to datetime; keep as datetime64[ns]
        try:
            out = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
            return out
        except Exception:
            return series
    if t_type in ("clean_numeric", "numeric", "type_cast_numeric"):
        num = pd.to_numeric(series, errors="coerce")
        # Simple rule: set negative values to NaN
        num[num < 0] = pd.NA
        return num
    if t_type in ("standardize_category", "standardize_text"):
        return series.astype(str).str.strip().str.title()
    # Fallback: return original
    return series


def execute_bronze(sttm_csv_path: str, run_id: str) -> List[str]:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    sttm_path = Path(sttm_csv_path)

    with sttm_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader]

    # Group rows by source_table
    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    metadata_rows: List[Dict[str, Any]] = []
    for r in rows:
        src_table = r.get("source_table", "").strip()
        if src_table == "_" or src_table == "":
            metadata_rows.append(r)
        else:
            rows_by_source.setdefault(src_table, []).append(r)

    written: List[str] = []
    for src_table, mappings in rows_by_source.items():
        try:
            landing_file = _find_landing_file(src_table)
        except FileNotFoundError as e:
            # Skip tables we cannot find
            continue
        df = pd.read_csv(landing_file, low_memory=False)
        out_dfs: Dict[str, pd.DataFrame] = {}

        # For each mapping row, apply transformation and place into target table frame
        for m in mappings:
            tgt_table = m.get("target_table") or src_table
            tgt_col = m.get("target_column")
            src_col = m.get("source_column")
            t_type = (m.get("transformation_type") or "").strip().lower()
            t_logic = m.get("transformation_logic") or ""

            if tgt_table not in out_dfs:
                out_dfs[tgt_table] = pd.DataFrame(index=df.index)

            if src_col and src_col in df.columns:
                series = df[src_col]
                transformed = _apply_transformation(series, t_type, t_logic)
                out_dfs[tgt_table][tgt_col] = transformed
            else:
                # If source column missing, create NaN column
                out_dfs[tgt_table][tgt_col] = pd.NA

        # Inject metadata columns for this source table if metadata rows exist
        for tgt_table, out_df in out_dfs.items():
            for meta in metadata_rows:
                col = meta.get("source_column")
                if not col:
                    continue
                if col == "_load_timestamp":
                    out_df["_load_timestamp"] = datetime.now(timezone.utc).isoformat()
                elif col == "_source_file":
                    out_df["_source_file"] = landing_file.name

            # Write parquet file per target_table (append run_id to filename)
            out_path = BRONZE_DIR / f"{tgt_table}_{run_id}.parquet"
            out_df.to_parquet(out_path, index=False)
            written.append(str(out_path))

    AuditLogger().log(agent="execute_sttm", action="execute_bronze", sttm=str(sttm_path), outputs=written, run_id=run_id)
    return written
