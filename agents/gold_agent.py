from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from core.config import GOLD_DIR
from core.audit import AuditLogger

BRONZE_SILVER_SUFFIX_RE = re.compile(r"_(bronze|silver)_\d{8}_\d{6}$")
AGG_RE = re.compile(r"(?P<func>SUM|AVG|COUNT|MAX|MIN)\((?P<col>[^)]+)\)\s*(?:AS\s+(?P<alias>\w+))?", re.I)
JOIN_RE = re.compile(r"join_left:(?P<a>[^:]+):(?P<b>[^:]+):(?P<key>\w+)", re.I)


def _normalize_name(name: str) -> str:
    return BRONZE_SILVER_SUFFIX_RE.sub("", name)


def _get_row_logic(row: Dict[str, Any]) -> str:
    t_type = (row.get('transformation_type') or '').strip()
    logic = (row.get('transformation_logic') or '').strip()
    if logic:
        return logic
    if t_type.lower().startswith('aggregate('):
        inner = t_type[t_type.find('(')+1:t_type.rfind(')')]
        return inner.strip()
    if t_type.lower() == 'aggregate' and row.get('source_column'):
        return f"SUM({row['source_column']})"
    if t_type.lower().startswith('join_left'):
        return t_type
    return t_type


def _row_type(row: Dict[str, Any]) -> str:
    t_type = (row.get('transformation_type') or '').strip().lower()
    logic = _get_row_logic(row).lower()
    if t_type.startswith('join_left') or logic.startswith('join_left'):
        return 'join'
    if t_type.startswith('aggregate(') or 'aggregate(' in logic or t_type == 'aggregate':
        return 'aggregate'
    if t_type.startswith('group_by') or 'group by' in logic or t_type == 'group_by':
        return 'group_by'
    if t_type.startswith('map') or logic.startswith('map:') or t_type == 'map':
        return 'map'
    return t_type


def _find_table(tables: Dict[str, pd.DataFrame], name: str) -> Tuple[str, pd.DataFrame]:
    # Try exact match, normalized match, and with _silver suffix variants
    if name in tables:
        return name, tables[name]
    norm = _normalize_name(name)
    for k in list(tables.keys()):
        if _normalize_name(k) == norm:
            return k, tables[k]
    # try adding _silver if needed
    cand = name if name.endswith("_silver") else f"{name}_silver"
    if cand in tables:
        return cand, tables[cand]
    raise KeyError(f"Table '{name}' not found in provided silver tables")


def _parse_aggregate(logic: str) -> Tuple[str, str, str]:
    m = AGG_RE.search(logic)
    if not m:
        raise ValueError(f"Unsupported aggregate logic: {logic}")
    func = m.group('func').upper()
    col = m.group('col')
    alias = m.group('alias') or f"{func.lower()}_{col}"
    return func, col, alias


def run(silver_paths: List[str], sttm_path: str, business_intent: str, run_id: str) -> List[str]:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    sttm_p = Path(sttm_path)
    with sttm_p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        sttm_rows = [r for r in reader]

    # load silver tables into dict keyed by stem
    tables: Dict[str, pd.DataFrame] = {}
    for p in silver_paths:
        path = Path(p)
        tables[path.stem] = pd.read_parquet(path)

    # Group STTM rows by target_table, preserving order of first appearance.
    targets: Dict[str, List[Dict[str, Any]]] = {}
    target_order: List[str] = []
    for r in sttm_rows:
        tgt = (r.get('target_table') or '').strip()
        if not tgt:
            continue
        if tgt not in targets:
            target_order.append(tgt)
            targets[tgt] = []
        targets[tgt].append(r)

    written: List[str] = []

    for target_table in target_order:
        rows = targets[target_table]
        # For each target, find join rules, group_by, aggregate, map, and renames
        def row_type(r: Dict[str, Any]) -> str:
            t_type = (r.get('transformation_type') or '').strip().lower()
            logic = (r.get('transformation_logic') or '').strip().lower()
            if t_type.startswith('join_left') or logic.startswith('join_left'):
                return 'join'
            if t_type.startswith('aggregate(') or 'aggregate(' in logic:
                return 'aggregate'
            if t_type.startswith('group_by') or 'group by' in logic:
                return 'group_by'
            if t_type.startswith('map') or logic.startswith('map:'):
                return 'map'
            if t_type == 'aggregate':
                return 'aggregate'
            if t_type == 'group_by':
                return 'group_by'
            if t_type == 'map':
                return 'map'
            return t_type

        join_rules = [r for r in rows if row_type(r) == 'join']
        group_by_cols = [r.get('target_column') for r in rows if row_type(r) == 'group_by']
        aggregate_rules = [r for r in rows if row_type(r) == 'aggregate']
        map_rules = [r for r in rows if row_type(r) == 'map']
        rename_rules = [r for r in rows if (r.get('source_column') and r.get('target_column') and r.get('source_column') != r.get('target_column'))]

        # Start dataframe assembly
        base_df: pd.DataFrame | None = None
        if join_rules:
            # Support join rules of form join_left:A:B:key
            for jr in join_rules:
                logic = _get_row_logic(jr).strip()
                m = JOIN_RE.search(logic)
                if not m:
                    continue
                a, b, key = m.group('a'), m.group('b'), m.group('key')
                try:
                    a_name, a_df = _find_table(tables, a)
                except KeyError:
                    continue
                try:
                    b_name, b_df = _find_table(tables, b)
                except KeyError:
                    continue
                left_suf = f"_{_normalize_name(a_name)}"
                right_suf = f"_{_normalize_name(b_name)}"
                if base_df is None:
                    base_df = a_df.merge(b_df, on=key, how='left', suffixes=(left_suf, right_suf))
                else:
                    base_df = base_df.merge(b_df, on=key, how='left', suffixes=("", f"_{_normalize_name(b_name)}"))

        if base_df is None:
            # Choose a base table from the target name or from map rules if needed
            try:
                _, df = _find_table(tables, target_table)
                base_df = df.copy()
            except KeyError:
                if map_rules:
                    src_table = (map_rules[0].get('source_table') or '').strip()
                    if src_table:
                        try:
                            _, df = _find_table(tables, src_table)
                            base_df = df.copy()
                        except KeyError:
                            pass
                if base_df is None:
                    first_key = next(iter(tables))
                    base_df = tables[first_key].copy()

        if base_df is None:
            continue

        # Apply renames / select columns
        for r in rename_rules:
            src = (r.get('source_column') or '').strip()
            tgt = (r.get('target_column') or '').strip()
            if src in base_df.columns:
                base_df[tgt] = base_df[src]

        # Prepare aggregates
        agg_map: Dict[str, List[Tuple[str, str]]] = {}
        for ar in aggregate_rules:
            logic = _get_row_logic(ar).strip()
            func, col, alias = _parse_aggregate(logic)
            func = func.upper()
            alias = alias.strip()
            if func == 'SUM':
                base_func = 'sum'
            elif func == 'AVG':
                base_func = 'mean'
            elif func == 'COUNT':
                base_func = 'count'
            elif func == 'MAX':
                base_func = 'max'
            elif func == 'MIN':
                base_func = 'min'
            else:
                raise ValueError(f"Unsupported aggregate function: {func}")
            agg_map[alias] = (base_func, col)

        group_cols = [c for c in (group_by_cols or []) if c]

        if agg_map:
            if group_cols:
                agg_result = base_df.groupby(group_cols).apply(
                    lambda g: pd.Series({alias: getattr(g[source_col], func_name)() for alias, (func_name, source_col) in agg_map.items()})
                ).reset_index()
                df_final = agg_result
            else:
                agg_values = {alias: getattr(base_df[col], func)() for alias, (func, col) in agg_map.items()}
                df_final = pd.DataFrame([agg_values])
        elif map_rules:
            df_final = None
            for mr in map_rules:
                logic = _get_row_logic(mr).strip().lower()
                src = (mr.get('source_column') or '').strip()
                tgt = (mr.get('target_column') or '').strip()
                if 'highest total_sales' in logic or 'highest total sales' in logic or 'max total_sales' in logic:
                    if 'total_sales' in base_df.columns and src in base_df.columns:
                        idx = base_df['total_sales'].idxmax()
                        df_final = pd.DataFrame([{tgt: base_df.at[idx, src]}])
                        break
                elif logic.startswith('map:'):
                    if src in base_df.columns:
                        df_final = pd.DataFrame({tgt: base_df[src]})
                        break
                elif src in base_df.columns:
                    df_final = pd.DataFrame({tgt: base_df[src]})
                    break
            if df_final is None:
                df_final = base_df.copy()
        else:
            target_cols = [(r.get('target_column') or '').strip() for r in rows if (r.get('target_column') or '').strip()]
            df_final = base_df.loc[:, [c for c in target_cols if c in base_df.columns]]

        # Ensure df_final is a DataFrame
        if not isinstance(df_final, pd.DataFrame):
            df_final = pd.DataFrame(df_final)

        # Insert surrogate key if not already present
        pk_name = 'pk_gold_id'
        if pk_name not in df_final.columns:
            df_final.insert(0, pk_name, range(1, len(df_final) + 1))

        # Keep only STTM-approved target columns (plus pk)
        approved = [ (r.get('target_column') or '').strip() for r in rows if (r.get('target_column') or '').strip() ]
        final_cols = [pk_name] + [c for c in approved if c in df_final.columns]
        # Ensure unique column names in df_final to satisfy parquet writers
        cols = list(df_final.columns)
        seen: Dict[str, int] = {}
        unique_cols: List[str] = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_c = f"{c}_{seen[c]}"
            else:
                seen[c] = 0
                new_c = c
            unique_cols.append(new_c)
        df_final.columns = unique_cols
        # now select final columns (they may have been renamed to unique names)
        selected = [pk_name]
        for c in approved:
            # find the first column that matches original name among unique_cols
            match = next((uc for uc in unique_cols if uc == c or uc.startswith(c + "_")), None)
            if match:
                selected.append(match)
        # deduplicate selected while preserving order
        seen_sel = set()
        uniq_selected = []
        for col in selected:
            if col not in seen_sel:
                seen_sel.add(col)
                uniq_selected.append(col)
        df_final = df_final.loc[:, uniq_selected]

        out_path = GOLD_DIR / f"{target_table}_{run_id}.parquet"
        df_final.to_parquet(out_path, index=False)
        written.append(str(out_path))
        # make this derived gold table available for subsequent STTM targets in the same run
        tables[target_table] = df_final

        AuditLogger().log(
            agent='gold_agent',
            action='run',
            target_table=target_table,
            output_path=str(out_path),
            input_tables=list(tables.keys()),
            output_shape=[int(len(df_final)), int(len(df_final.columns))],
            run_id=run_id,
        )

    return written
