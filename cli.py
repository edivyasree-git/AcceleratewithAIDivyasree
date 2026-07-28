from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

from core.config import ensure_dirs, LANDING_DIR, STTM_DIR
from core.audit import new_run_id, AuditLogger
from core.state import PipelineState

from agents import profiler, sttm_generator, bronze_agent, silver_agent, gold_agent, reporter


def banner(text: str) -> None:
    print("=" * 80)
    print(text)
    print("=" * 80)


def display_sttm(sttm_path: str, layer: str) -> None:
    try:
        from tabulate import tabulate
    except Exception:
        tabulate = None

    df = pd.read_csv(sttm_path)
    print(f"STTM ({layer}): {sttm_path}")
    if tabulate:
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    else:
        # fallback
        print(df.to_string(index=False))


def _open_in_editor(path: Path) -> None:
    editor = (Path(sys.argv[0]).parent / "")  # noop to keep style
    EDITOR = (  # respect env var then sensible defaults
        subprocess.os.environ.get("EDITOR")
        or subprocess.os.environ.get("VISUAL")
        or ("notepad" if subprocess.os.name == "nt" else "nano")
    )
    try:
        subprocess.run([EDITOR, str(path)], check=False)
    except Exception:
        # last resort: open with default application
        try:
            if subprocess.os.name == "nt":
                subprocess.run(["notepad", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception:
            print(f"Could not open editor for {path}")


def hitl_gate(layer: str, sttm_path: str) -> bool:
    while True:
        display_sttm(sttm_path, layer)
        ans = input("[y]es / [e]dit then re-review / [n]o abort > ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            print(f"Aborting at {layer} HITL gate")
            return False
        if ans in ("e", "edit"):
            _open_in_editor(Path(sttm_path))
            print("Re-displaying updated STTM...")
            continue
        print("Please answer y, e, or n")


def _copy_to_landing(src: Path) -> Path:
    ensure_dirs()
    dest = LANDING_DIR / src.name
    try:
        if src.resolve() == dest.resolve():
            return dest
    except Exception:
        pass
    shutil.copy2(src, dest)
    return dest


def run_pipeline(files: List[str], intent: str) -> PipelineState:
    ensure_dirs()
    run_id = new_run_id()
    state = PipelineState(run_id=run_id, input_files=files, business_intent=intent)
    audit = AuditLogger(run_id=run_id)

    # Phase 1 - Profile + Bronze STTM
    banner("Phase 1: Profile -> Bronze STTM")
    # copy files into landing if needed
    landing_paths: List[str] = []
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"Skipping missing file: {f}")
            continue
        dest = _copy_to_landing(p)
        landing_paths.append(str(dest))

    state.profile_path = profiler.profile(landing_paths, run_id)
    state.bronze_sttm_path = sttm_generator.generate_bronze_sttm(state.profile_path, intent, run_id)
    audit.log(agent="cli", action="generated_bronze_sttm", bronze_sttm=state.bronze_sttm_path)
    if not hitl_gate("Bronze", state.bronze_sttm_path):
        return state

    # Phase 2 - Bronze execution + Silver STTM
    banner("Phase 2: Bronze Execution -> Silver STTM")
    state.bronze_paths = bronze_agent.run(landing_paths, state.bronze_sttm_path, run_id)
    state.silver_sttm_path = sttm_generator.generate_silver_sttm(state.bronze_paths, state.bronze_sttm_path, intent, run_id)
    audit.log(agent="cli", action="generated_silver_sttm", silver_sttm=state.silver_sttm_path)
    if not hitl_gate("Silver", state.silver_sttm_path):
        return state

    # Phase 3 - Silver execution + Gold STTM
    banner("Phase 3: Silver Execution -> Gold STTM")
    state.silver_paths = silver_agent.run(state.bronze_paths, state.silver_sttm_path, run_id)
    state.gold_sttm_path = sttm_generator.generate_gold_sttm(state.silver_paths, state.silver_sttm_path, intent, run_id)
    audit.log(agent="cli", action="generated_gold_sttm", gold_sttm=state.gold_sttm_path)
    if not hitl_gate("Gold", state.gold_sttm_path):
        return state

    # Phase 4 - Gold execution + Report
    banner("Phase 4: Gold Execution -> Report")
    state.gold_paths = gold_agent.run(state.silver_paths, state.gold_sttm_path, intent, run_id)
    state.report_path = reporter.generate_report(state.gold_paths, intent, run_id)
    audit.log(agent="cli", action="completed_pipeline", report=state.report_path)

    print("Run completed")
    print(f"run_id: {run_id}")
    print(f"report: {state.report_path}")
    print(f"audit log: {audit.log_path}")
    return state


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive ETL pipeline with HITL gates")
    p.add_argument("--files", nargs="+", help="Input files (CSV) to ingest")
    p.add_argument("--intent", help="Business intent / analytic goal")
    args = p.parse_args()

    files = args.files
    intent = args.intent
    if not files:
        files = input("Enter input file paths (space-separated): ").strip().split()
    if not intent:
        intent = input("Enter business intent / question: ").strip()

    run_pipeline(files, intent)


if __name__ == "__main__":
    main()
