from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd
import plotly.express as px
from langchain_openai import ChatOpenAI

from core.audit import AuditLogger
from core.config import GITHUB_BASE_URL, GITHUB_MODEL, GITHUB_TOKEN, REPORTS_DIR

CSV_FENCE_RE = re.compile(r"```(?:csv)?\s*(.*?)```", re.S | re.I)
SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.S | re.I)
JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.S | re.I)


def _llm_client() -> ChatOpenAI:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is required for LLM requests")
    return ChatOpenAI(api_key=GITHUB_TOKEN, base_url=GITHUB_BASE_URL, model=GITHUB_MODEL)


def _llm_query(prompt: str) -> str:
    client = _llm_client()
    resp = client.invoke(prompt)
    return getattr(resp, "content", resp)


def _extract_sql(text: str) -> str:
    m = SQL_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: look for a SELECT..
    sel = re.search(r"(SELECT[\s\S]+)", text, re.I)
    return sel.group(1).strip() if sel else ""


def _extract_json_block(text: str) -> str:
    m = JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_narrative(text: str) -> str:
    # Look for 'NARRATIVE:' marker
    parts = re.split(r"NARRATIVE:\s*", text, flags=re.I)
    if len(parts) >= 2:
        # narrative is after marker up to CHARTS: or end
        narrative = re.split(r"CHARTS:\s*", parts[1], flags=re.I)[0].strip()
        return narrative
    # fallback: return text after SQL block
    after_sql = re.split(SQL_FENCE_RE, text)
    return after_sql[-1].strip() if after_sql else text.strip()


def _build_prompt(schema_text: str, business_intent: str) -> str:
    return f"""
You are an analyst assistant. Given the schema and sample data below, produce three sections:

SQL:
Provide a single DuckDB-compatible SELECT query (fenced in ```sql) that answers the business question exactly.

NARRATIVE:
Write a concise 2-3 paragraph executive summary of the results.

CHARTS:
Return a JSON array (fenced in ```json) of chart specs: {{"type": "bar|pie|line", "x": "column", "y": "column", "title": "string"}}.

Business question: {business_intent}

Schema and samples:
{schema_text}
""".strip()


def generate_report(gold_paths: List[str], business_intent: str, run_id: str) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # load gold tables into duckdb via pandas registration
    conn = duckdb.connect(database=':memory:')
    table_names: List[str] = []
    schema_lines: List[str] = []
    for p in gold_paths:
        path = Path(p)
        name = path.stem
        df = pd.read_parquet(path)
        conn.register(name, df)
        table_names.append(name)
        schema_lines.append(f"Table: {name}")
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(3).tolist()
            schema_lines.append(f"  - {col}: {str(df[col].dtype)}, samples={sample}")

    schema_text = "\n".join(schema_lines)
    prompt = _build_prompt(schema_text, business_intent)
    raw = _llm_query(prompt)

    sql = _extract_sql(raw)
    narrative = _extract_narrative(raw)
    charts_text = _extract_json_block(raw)
    charts = []
    try:
        charts = json.loads(charts_text) if charts_text else []
    except Exception:
        charts = []

    # execute SQL with retry
    try:
        df_result = conn.execute(sql).df()
    except Exception:
        # fallback: select first gold table
        if table_names:
            df_result = conn.execute(f"SELECT * FROM {table_names[0]} LIMIT 20").df()
        else:
            df_result = pd.DataFrame()

    # build charts html fragments
    chart_html_parts: List[str] = []
    for spec in charts:
        try:
            typ = spec.get('type')
            x = spec.get('x')
            y = spec.get('y')
            title = spec.get('title', '')
            if typ == 'bar':
                fig = px.bar(df_result, x=x, y=y, title=title)
            elif typ == 'pie':
                fig = px.pie(df_result, names=x, values=y, title=title)
            elif typ == 'line':
                fig = px.line(df_result, x=x, y=y, title=title)
            else:
                continue
            chart_html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception:
            continue

    table_html = df_result.to_html(index=False)

    timestamp = datetime.now(timezone.utc).isoformat()
    html_parts: List[str] = []
    html_parts.append(f"<h1>Report: {business_intent}</h1>")
    html_parts.append(f"<h3>Run: {run_id} &nbsp; Generated: {timestamp}</h3>")
    html_parts.append(f"<h2>Executive Summary</h2><div>{narrative}</div>")
    html_parts.append("<h2>Charts</h2>")
    html_parts.extend(chart_html_parts)
    html_parts.append("<h2>Data</h2>")
    html_parts.append(table_html)
    html_parts.append("<h2>SQL</h2>")
    html_parts.append(f"<pre>{sql}</pre>")

    full_html = "\n".join([
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>Report {run_id}</title>",
        "</head>",
        "<body>",
        *html_parts,
        "</body>",
        "</html>",
    ])

    html_path = REPORTS_DIR / f"report_{run_id}.html"
    json_path = REPORTS_DIR / f"report_{run_id}.json"
    html_path.write_text(full_html, encoding="utf-8")
    json_path.write_text(json.dumps({"sql": sql, "narrative": narrative, "charts": charts}, indent=2), encoding="utf-8")

    AuditLogger().log(agent="reporter", action="generate_report", report=str(html_path), run_id=run_id)
    return str(html_path)
