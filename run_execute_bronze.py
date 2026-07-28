from agents.execute_sttm import execute_bronze
from core.audit import new_run_id

STTM_PATH = "data/sttm/sttm_bronze_20260728_124545.csv"

if __name__ == "__main__":
    run_id = new_run_id()
    outputs = execute_bronze(STTM_PATH, run_id)
    print("Wrote:")
    for p in outputs:
        print(p)
