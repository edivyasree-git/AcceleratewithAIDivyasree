from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from core.config import BRONZE_DIR
from core.audit import AuditLogger


def _cast_column(series: pd.Series, target_type: str) -> pd.Series:
    t = target_type.strip().lower()
    if t in ("datetime", "date", "standardize_date"):
        return pd.to_datetime(series, errors="coerce")
    if t == "float":
        return pd.to_numeric(series, errors="coerce")
    if t == "int":
        # Use pandas nullable integer type
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if t == "str":
        return series.astype(str)
    # unknown -> try numeric else keep original
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return series


def run(input_files: List[str], sttm_path: str, run_id: str) -> List[str]:
    """Apply Bronze STTM to input CSV files and write parquet outputs.

    Returns list of written parquet file paths.
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    sttm_p = Path(sttm_path)

    # Read STTM
    import csv

    with sttm_p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        sttm_rows = [r for r in reader]

    written: List[str] = []

    for fp in input_files:
        path = Path(fp)
        table = path.stem  # extract table name from filename stem
        # Filter STTM rows for this table
        rules = [r for r in sttm_rows if (r.get("source_table") or "").strip() == table]
        if not rules:
            # no rules for this table; skip
            continue

        df = pd.read_csv(fp, low_memory=False)
        out_df = pd.DataFrame(index=df.index)

        for r in rules:
            src_col = (r.get("source_column") or "").strip()
            tgt_col = (r.get("target_column") or src_col).strip()
            t_type = (r.get("transformation_type") or "").strip().lower()
            t_logic = (r.get("transformation_logic") or "").strip().lower()

            # Skip metadata injection here
            if t_type == "metadata_injection":
                continue

            # determine action
            if t_type in ("passthrough", "copy", "pass-through"):
                # copy as-is if column exists
                if src_col in df.columns:
                    out_df[tgt_col] = df[src_col]
                else:
                    out_df[tgt_col] = pd.NA
                continue

            # type_cast or other cast-like operations
            target_t = ""  # determine type to cast to
            # Look for explicit keywords in t_logic
            if "date" in t_type or "date" in t_logic:
                target_t = "datetime"
            elif "int" in t_logic or t_type == "int":
                target_t = "int"
            elif "float" in t_logic or t_type == "float":
                target_t = "float"
            elif "str" in t_logic or t_type == "str":
                target_t = "str"
            else:
                # fallback: if source dtype is numeric, cast to numeric, else copy
                if src_col in df.columns and pd.api.types.is_numeric_dtype(df[src_col].dtype):
                    target_t = "float"
                else:
                    target_t = "str"

            if src_col in df.columns:
                out_df[tgt_col] = _cast_column(df[src_col], target_t)
            else:
                out_df[tgt_col] = pd.NA

        # Inject metadata columns
        out_df["_load_timestamp"] = datetime.now(timezone.utc).isoformat()
        out_df["_source_file"] = str(fp)

        # Write parquet
        out_path = BRONZE_DIR / f"{table}_bronze_{run_id}.parquet"
        out_df.to_parquet(out_path, index=False)
        written.append(str(out_path))

        # Audit log input/output shapes
        AuditLogger().log(
            agent="bronze_agent",
            action="run",
            table=table,
            input_shape=[int(df.shape[0]), int(df.shape[1])],
            output_shape=[int(out_df.shape[0]), int(out_df.shape[1])],
            output_path=str(out_path),
            run_id=run_id,
        )

    return written
