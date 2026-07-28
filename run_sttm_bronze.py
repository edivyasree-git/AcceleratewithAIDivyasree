from agents.sttm_generator import generate_bronze_sttm
from core.audit import new_run_id
import os

# Adjust these as needed
PROFILE_PATH = os.getenv("PROFILE_PATH", "data/profiles/profile_combined_20260728_123341.json")
BUSINESS_INTENT = os.getenv("BUSINESS_INTENT", "")


def main():
    run_id = new_run_id()
    out = generate_bronze_sttm(PROFILE_PATH, BUSINESS_INTENT, run_id)
    print("Bronze STTM written to:", out)


if __name__ == "__main__":
    main()
