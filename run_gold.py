from __future__ import annotations

import argparse
from core.audit import new_run_id
from agents import sttm_generator, gold_agent, reporter


def main() -> None:
    p = argparse.ArgumentParser(description="Run Gold STTM generation and/or Gold agent")
    p.add_argument("--silver-paths", nargs="+", required=True, help="Silver parquet files")
    p.add_argument("--silver-sttm", required=True, help="Existing silver STTM CSV path")
    p.add_argument("--gold-sttm", required=False, help="Existing gold STTM CSV path to use instead of generating one")
    p.add_argument("--intent", required=False, help="Business intent")
    p.add_argument("--run-id", required=False, help="Optional run id")
    p.add_argument("--generate-only", action="store_true", help="Only generate gold STTM and exit")
    args = p.parse_args()

    run_id = args.run_id or new_run_id()

    if args.gold_sttm:
        gold_sttm = args.gold_sttm
        print(f"using existing gold STTM: {gold_sttm}")
    else:
        if not args.intent:
            raise ValueError("--intent is required when generating a gold STTM")
        gold_sttm = sttm_generator.generate_gold_sttm(args.silver_paths, args.silver_sttm, args.intent, run_id)
        print(f"gold sttm written: {gold_sttm}")
        if args.generate_only:
            return

    gold_paths = gold_agent.run(args.silver_paths, gold_sttm, args.intent or "", run_id)
    print("gold parquet outputs:")
    for pth in gold_paths:
        print(f"  {pth}")

    report = reporter.generate_report(gold_paths, args.intent or "", run_id)
    print(f"report: {report}")


if __name__ == "__main__":
    main()
