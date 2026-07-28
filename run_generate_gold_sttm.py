from __future__ import annotations

import argparse
from core.audit import new_run_id
from agents.sttm_generator import generate_gold_sttm


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Gold STTM CSV from Silver tables")
    p.add_argument("--silver-paths", nargs="+", required=True, help="Paths to silver parquet files")
    p.add_argument("--silver-sttm", required=True, help="Path to the silver STTM CSV used as context")
    p.add_argument("--intent", required=True, help="Business intent for Gold STTM generation")
    p.add_argument("--run-id", required=False, help="Optional run id; generated if omitted")
    args = p.parse_args()

    run_id = args.run_id or new_run_id()
    out = generate_gold_sttm(args.silver_paths, args.silver_sttm, args.intent, run_id)
    print(out)


if __name__ == "__main__":
    main()
