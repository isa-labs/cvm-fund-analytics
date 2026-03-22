import sys, pandas as pd
sys.path.insert(0, '.')
from src.database import get_or_load

# Via DuckDB
daily_db = get_or_load(['2025-01'])
print("DuckDB columns:", daily_db.columns.tolist())
print("DuckDB CNPJ_BASE sample:", daily_db['CNPJ_BASE'].head(3).tolist())

# Via CSV
df_csv = pd.read_csv('data/raw/inf_diario_fi_202501.csv', dtype={'CNPJ_FUNDO_CLASSE': str}, low_memory=False)
df_csv['CNPJ_BASE'] = df_csv['CNPJ_FUNDO_CLASSE'].str[:18]
print("\nCSV columns:", df_csv.columns.tolist())
print("CSV CNPJ_BASE sample:", df_csv['CNPJ_BASE'].head(3).tolist())