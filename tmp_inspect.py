from pathlib import Path
import pandas as pd
for path in ['data/silver_layer/sales_data_1_silver_20260728_131200.parquet', 'data/gold_layer/gold_sales_summary_20260728_142146.parquet', 'data/gold_layer/gold_top_category_20260728_142146.parquet']:
    p = Path(path)
    print('===', p)
    if p.exists():
        df = pd.read_parquet(p)
        print(df.head(10).to_string(index=False))
        print('columns:', list(df.columns))
        print('shape:', df.shape)
    else:
        print('missing', p)
