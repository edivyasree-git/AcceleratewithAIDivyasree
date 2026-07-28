from agents.profiler import profile
from core.audit import new_run_id

path = profile(
    [
        "data/landing/sales_data_1.csv",
        "data/landing/products 2.csv",
    ],
    new_run_id(),
)
print("Profile written to:", path)
