from agents.silver_agent import run
paths = ['data/bronze_layer/sales_data_1_bronze_20260728_125152.parquet']
out = run(paths, 'data/sttm/sttm_silver_category_normalize_20260728_131200.csv', '20260728_131200')
print(out)
