from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from core.audit import AuditLogger
from core.config import PROFILES_DIR

DATE_PATTERN_MDY = re.compile(r"^\d{2}/\d{2}/\d{4}$")
DATE_PATTERN_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _sample_values(series: pd.Series, max_samples: int = 5) -> list[Any]:
    values = series.dropna().astype(str)
    uniques = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            uniques.append(value)
            if len(uniques) >= max_samples:
                break
    return uniques


def _quality_flags(series: pd.Series) -> list[str]:
    flags: list[str] = []
    if series.dtype == object:
        values = series.dropna().astype(str)
        if not values.empty:
            total = len(values)
            md = sum(bool(DATE_PATTERN_MDY.match(v)) for v in values)
            yd = sum(bool(DATE_PATTERN_YMD.match(v)) for v in values)
            if total and md / total > 0.1 and yd / total > 0.1:
                flags.append("mixed_date_formats")
            avg_len = values.map(len).mean()
            if avg_len < 6:
                flags.append("possible_abbreviations")
    return flags


def _column_profile(series: pd.Series) -> dict[str, Any]:
    null_count = int(series.isna().sum())
    row_count = len(series)
    null_pct = float(null_count / row_count) if row_count else 0.0
    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_pct": null_pct,
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": _sample_values(series),
        "quality_flags": _quality_flags(series),
    }
    if is_numeric_dtype(series.dtype) and row_count:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            profile["min"] = None
            profile["max"] = None
            profile["mean"] = None
        else:
            min_val = numeric.min()
            max_val = numeric.max()
            mean_val = numeric.mean()
            profile["min"] = min_val.item() if hasattr(min_val, "item") else min_val
            profile["max"] = max_val.item() if hasattr(max_val, "item") else max_val
            profile["mean"] = mean_val.item() if hasattr(mean_val, "item") else float(mean_val)
    return profile


def profile(file_paths: list[str], run_id: str) -> str:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"profile_combined_{run_id}.json"

    tables: dict[str, dict[str, Any]] = {}
    join_key_candidates: dict[str, list[str]] = {}
    id_columns: dict[str, set[str]] = {}

    for fp in file_paths:
        path = Path(fp)
        table_name = path.stem
        df = pd.read_csv(fp, low_memory=False)
        columns: dict[str, Any] = {}
        for col in df.columns:
            columns[col] = _column_profile(df[col])
            if col.endswith("_id"):
                id_columns.setdefault(col, set()).add(table_name)

        tables[table_name] = {
            "row_count": int(len(df)),
            "columns": columns,
        }

    candidate_join_keys: dict[str, list[str]] = {
        col: sorted(list(tables))
        for col, tables in id_columns.items()
        if len(tables) >= 2
    }

    output = {
        "run_id": run_id,
        "tables": tables,
        "candidate_join_keys": candidate_join_keys,
    }

    with profile_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    AuditLogger().log(agent="profiler", action="completed", profile_path=str(profile_path))
    return str(profile_path)
