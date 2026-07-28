from __future__ import annotations

import csv
import json
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_openai import ChatOpenAI

from core.audit import AuditLogger
from core.config import (
    GITHUB_BASE_URL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    LLM_PROVIDER,
    STTM_DIR,
)

REQUIRED_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]

CSV_FENCE_RE = re.compile(r"```(?:csv)?\s*(.*?)```", re.S | re.I)


def _llm_client() -> ChatOpenAI:
    if LLM_PROVIDER != "github":
        raise RuntimeError("Only GitHub Models provider is supported by this STTM generator")
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is required for LLM requests")
    return ChatOpenAI(api_key=GITHUB_TOKEN, base_url=GITHUB_BASE_URL, model=GITHUB_MODEL)


def _llm_query(prompt: str) -> str:
    client = _llm_client()
    response = client.invoke(prompt)
    return getattr(response, "content", response)


def _extract_csv(text: str) -> str:
    match = CSV_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _validate_and_save(csv_text: str, out_path: Path) -> None:
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("STTM output contains no CSV header")

    headers = [name.strip() for name in reader.fieldnames]
    missing = [col for col in REQUIRED_COLUMNS if col not in headers]
    if missing:
        raise ValueError(f"STTM output is missing required columns: {missing}")

    allowed_types = {
        'bronze': {
            'copy', 'passthrough', 'pass-through', 'date', 'int', 'float', 'str',
            'standardize_date', 'data_cleaning', 'metadata_injection', 'surrogate_key',
            'map', 'lowercase', 'uppercase', 'title case', 'strip', 'clean_numeric',
        },
        'silver': {
            'drop null', 'fill null', 'date', 'lowercase', 'uppercase', 'title case',
            'strip', 'deduplicate', 'surrogate_key', 'map', 'clean_numeric',
        },
        'gold': {
            'join_left', 'group_by', 'aggregate', 'map', 'copy', 'rename',
        },
    }

    def _get_row_logic(row: dict[str, Any]) -> str:
        t_type = (row.get('transformation_type') or '').strip()
        logic = (row.get('transformation_logic') or '').strip()
        if logic:
            return logic
        if t_type.lower().startswith('aggregate('):
            inner = t_type[t_type.find('(')+1:t_type.rfind(')')]
            return inner.strip()
        if t_type.lower() == 'aggregate' and row.get('source_column'):
            return f"SUM({row['source_column']})"
        return t_type

    def _is_allowed_type(value: str, layer: str) -> bool:
        if not value:
            return True
        v = value.strip().lower()
        if layer == 'gold':
            if v.startswith('join_left:') or v.startswith('aggregate(') or v.startswith('map:'):
                return True
        if layer == 'silver':
            if v.startswith('fill') or v.startswith('drop null') or v.startswith('map:'):
                return True
        return v in allowed_types.get(layer, set())

    if out_path.name.startswith('sttm_gold_'):
        layer = 'gold'
    elif out_path.name.startswith('sttm_silver_'):
        layer = 'silver'
    elif out_path.name.startswith('sttm_bronze_'):
        layer = 'bronze'
    else:
        layer = 'bronze'

    invalid_types = []
    for row in reader:
        t_type = (row.get('transformation_type') or '').strip()
        logic = (row.get('transformation_logic') or '').strip()
        row_logic = _get_row_logic(row)
        if not _is_allowed_type(t_type, layer) and not _is_allowed_type(logic, layer) and not _is_allowed_type(row_logic, layer):
            invalid_types.append(t_type or logic or row_logic)
    if invalid_types:
        invalid_list = sorted(set(invalid_types))
        raise ValueError(f"STTM contains unsupported transformation types for {out_path.name}: {invalid_list}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(csv_text.strip() + "\n", encoding="utf-8")


def _read_profile(profile_path: str) -> dict[str, Any]:
    with open(profile_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_sttm(sttm_path: str) -> str:
    with open(sttm_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _table_schema_text(df: pd.DataFrame, table_name: str) -> str:
    lines = [f"Table: {table_name}"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().astype(str).head(3).tolist()
        lines.append(f"  - {col}: {dtype}, samples={sample_values}")
    return "\n".join(lines)


def _profile_schema_text(profile: dict[str, Any]) -> str:
    lines = []
    for table_name, table in profile.get("tables", {}).items():
        lines.append(f"Table: {table_name}")
        for col_name, col_meta in table.get("columns", {}).items():
            dtype = col_meta.get("dtype")
            flags = col_meta.get("quality_flags", [])
            sample = col_meta.get("sample_values", [])[:3]
            lines.append(f"  - {col_name}: {dtype}, flags={flags}, samples={sample}")
    return "\n".join(lines)


def _build_bronze_prompt(profile: dict[str, Any], business_intent: str) -> str:
    profile_text = _profile_schema_text(profile)
    prompt = f"""
You are generating a source-to-target mapping CSV for the Bronze layer.
The output must be valid CSV with these columns: {', '.join(REQUIRED_COLUMNS)}.
Generate a transformation rule for every source column in every table.
Include metadata injection rows for _load_timestamp and _source_file.
Do not produce any explanation text outside the CSV.

Business intent: {business_intent}

Profile:
{profile_text}
"""
    return prompt.strip()


def _build_silver_prompt(bronze_schema_text: str, bronze_sttm_text: str, business_intent: str) -> str:
    prompt = f"""
You are generating a source-to-target mapping CSV for the Silver layer.
The output must be valid CSV with these columns: {', '.join(REQUIRED_COLUMNS)}.
Use the Bronze schema and the Bronze STTM as context.
Provide cleansing transformations only.
Possible transformation_type values: drop null, fill null, date, lowercase, uppercase, title case, strip, deduplicate, surrogate_key.
Do not change the schema names or table names unless required for cleansed outputs.
Do not produce any explanation text outside the CSV.

Business intent: {business_intent}

Bronze schema:
{bronze_schema_text}

Bronze STTM:
{bronze_sttm_text}
"""
    return prompt.strip()


def _build_gold_prompt(silver_schema_text: str, silver_sttm_text: str, business_intent: str) -> str:
    prompt = f"""
You are generating a source-to-target mapping CSV for the Gold layer.
The output must be valid CSV with these columns: {', '.join(REQUIRED_COLUMNS)}.
Use the Silver schema and the Silver STTM as context.
Create transformations that answer the business intent.
Supported gold transformations include join_left:table_a:table_b:key, group_by, aggregate(SUM/AVG/COUNT/MAX/MIN).
Do not produce any explanation text outside the CSV.

Business intent: {business_intent}

Silver schema:
{silver_schema_text}

Silver STTM:
{silver_sttm_text}
"""
    return prompt.strip()


def generate_bronze_sttm(profile_path: str, business_intent: str, run_id: str) -> str:
    profile = _read_profile(profile_path)
    prompt = _build_bronze_prompt(profile, business_intent)
    raw = _llm_query(prompt)
    csv_text = _extract_csv(raw)
    out_path = STTM_DIR / f"sttm_bronze_{run_id}.csv"
    _validate_and_save(csv_text, out_path)
    AuditLogger().log(agent="sttm_generator", action="generate_bronze_sttm", bronze_sttm_path=str(out_path), run_id=run_id)
    return str(out_path)


def generate_silver_sttm(bronze_paths: list[str], bronze_sttm_path: str, business_intent: str, run_id: str) -> str:
    schema_lines: list[str] = []
    for fp in bronze_paths:
        df = pd.read_parquet(fp)
        schema_lines.append(_table_schema_text(df, Path(fp).stem))
    bronze_schema_text = "\n\n".join(schema_lines)
    bronze_sttm_text = _read_sttm(bronze_sttm_path)
    prompt = _build_silver_prompt(bronze_schema_text, bronze_sttm_text, business_intent)
    raw = _llm_query(prompt)
    csv_text = _extract_csv(raw)
    out_path = STTM_DIR / f"sttm_silver_{run_id}.csv"
    _validate_and_save(csv_text, out_path)
    AuditLogger().log(agent="sttm_generator", action="generate_silver_sttm", silver_sttm_path=str(out_path), run_id=run_id)
    return str(out_path)


def generate_gold_sttm(silver_paths: list[str], silver_sttm_path: str, business_intent: str, run_id: str) -> str:
    schema_lines: list[str] = []
    for fp in silver_paths:
        df = pd.read_parquet(fp)
        schema_lines.append(_table_schema_text(df, Path(fp).stem))
    silver_schema_text = "\n\n".join(schema_lines)
    silver_sttm_text = _read_sttm(silver_sttm_path)
    prompt = _build_gold_prompt(silver_schema_text, silver_sttm_text, business_intent)
    raw = _llm_query(prompt)
    csv_text = _extract_csv(raw)
    out_path = STTM_DIR / f"sttm_gold_{run_id}.csv"
    _validate_and_save(csv_text, out_path)
    AuditLogger().log(agent="sttm_generator", action="generate_gold_sttm", gold_sttm_path=str(out_path), run_id=run_id)
    return str(out_path)
