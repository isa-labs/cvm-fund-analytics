"""
DuckDB persistence layer for CVM Fund Analytics
Handles:
- Creating and managing the local DuckDB database
- Inserting CVM daily NAV data
- Querying data by month range
- Checking which months are already persisted
Database location: data/cvm.duckdb
"""

import os
import glob
import duckdb
import pandas as pd
import dask.dataframe as dd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cvm.duckdb")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def get_connection() -> duckdb.DuckDBPyConnection:
    # Returns a connection to the local DuckDB database
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)

def setup_schema(con: duckdb.DuckDBPyConnection) -> None:
    # Creates the main table if it doesn't exist
    con.execute("""
        CREATE TABLE IF NOT EXISTS inf_diario (
            TP_FUNDO_CLASSE   VARCHAR,
            CNPJ_FUNDO_CLASSE VARCHAR,
            CNPJ_BASE         VARCHAR,
            ID_SUBCLASSE      VARCHAR,
            DT_COMPTC         DATE,
            VL_TOTAL          DOUBLE,
            VL_QUOTA          DOUBLE,
            VL_PATRIM_LIQ     DOUBLE,
            CAPTC_DIA         DOUBLE,
            RESG_DIA          DOUBLE,
            NR_COTST          DOUBLE,
            year_month        VARCHAR  -- e.g. "2025-01"
        )
    """)
    # Index for fast month filtering
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_year_month
        ON inf_diario (year_month)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_cnpj_base
        ON inf_diario (CNPJ_BASE)
    """)

def get_loaded_months(con: duckdb.DuckDBPyConnection) -> list[str]:
    # Returns list of year_month values already in the database
    try:
        result = con.execute(
            "SELECT DISTINCT year_month FROM inf_diario ORDER BY year_month"
        ).fetchall()
        return [r[0] for r in result]
    except Exception:
        return []

def insert_month(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    year_month: str,
) -> None:
    """
    Inserts one month of CVM data into DuckDB
    Skips if the month is already loaded
    con: DuckDB connection
    df: DataFrame from ingest.download_month()
    year_month: "2025-01"
    """
    loaded = get_loaded_months(con)
    if year_month in loaded:
        print(f"  {year_month} already in database, skipping")
        return

    df = df.copy()
    df["year_month"] = year_month
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"])

    # Ensure CNPJ_BASE exists
    if "CNPJ_BASE" not in df.columns:
        df["CNPJ_BASE"] = df["CNPJ_FUNDO_CLASSE"].str[:18]

    # Select only columns that match the schema
    cols = [
        "TP_FUNDO_CLASSE", "CNPJ_FUNDO_CLASSE", "CNPJ_BASE",
        "ID_SUBCLASSE", "DT_COMPTC", "VL_TOTAL", "VL_QUOTA",
        "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "NR_COTST", "year_month"
    ]
    available = [c for c in cols if c in df.columns]
    df = df[available]

    con.execute("INSERT INTO inf_diario SELECT * FROM df")
    print(f"  Inserted {len(df):,} rows for {year_month}")

def load_months(
    con: duckdb.DuckDBPyConnection,
    months: list[str],
) -> pd.DataFrame:
    """
    Queries the database for the given months
    months: list of "YYYY-MM" strings
    Returns: pd.DataFrame
    """
    placeholders = ", ".join(f"'{m}'" for m in months)
    df = con.execute(f"""
        SELECT
            TP_FUNDO_CLASSE,
            CNPJ_FUNDO_CLASSE,
            CNPJ_BASE,
            ID_SUBCLASSE,
            DT_COMPTC,
            VL_TOTAL,
            VL_QUOTA,
            VL_PATRIM_LIQ,
            CAPTC_DIA,
            RESG_DIA,
            NR_COTST
        FROM inf_diario
        WHERE year_month IN ({placeholders})
        ORDER BY CNPJ_BASE, DT_COMPTC
    """).df()
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"])
    return df

def sync_from_csv(con: duckdb.DuckDBPyConnection) -> list[str]:
    """
    Scans data/raw/ for CSV files and inserts any months not yet in the database
    Returns list of newly inserted months
    """
    setup_schema(con)
    loaded = get_loaded_months(con)
    inserted = []

    for path in sorted(glob.glob(os.path.join(RAW_DIR, "inf_diario_fi_*.csv"))):
        base = os.path.basename(path)
        ym_raw = base.replace("inf_diario_fi_", "").replace(".csv", "")
        if len(ym_raw) != 6:
            continue
        year_month = f"{ym_raw[:4]}-{ym_raw[4:]}"

        if year_month in loaded:
            continue

        print(f"Syncing {year_month} from CSV to DuckDB")
        import dask.dataframe as dd
        ddf = dd.read_csv(path, dtype={"CNPJ_FUNDO_CLASSE": str}, assume_missing=True)
        df = ddf.compute()
        insert_month(con, df, year_month)
        inserted.append(year_month)

    return inserted

def get_or_load(months: list[str]) -> pd.DataFrame:
    """
    Main entry point. For each month:
    - If in DuckDB → query from DB
    - If CSV exists on disk → insert into DB, then query
    - If neither → download from CVM, save CSV, insert into DB
    months: list of "YYYY-MM" strings
    Returns: pd.DataFrame with columns [CNPJ_FUNDO_CLASSE, CNPJ_BASE, DT_COMPTC, VL_QUOTA]
    """
    from src.ingest import download_month

    con = get_connection()
    setup_schema(con)
    loaded = get_loaded_months(con)

    for ym in months:
        if ym in loaded:
            continue  # already in DB

        ym_raw = ym.replace("-", "")
        csv_path = os.path.join(RAW_DIR, f"inf_diario_fi_{ym_raw}.csv")
        year, month = int(ym[:4]), int(ym[5:])

        if os.path.exists(csv_path):
            print(f"Loading {ym} from CSV")
            ddf = dd.read_csv(csv_path, dtype={"CNPJ_FUNDO_CLASSE": str}, assume_missing=True)
            df = ddf.compute()
        else:
            print(f"Downloading {ym} from CVM")
            df = download_month(year, month, save=True)

        insert_month(con, df, ym)

    return load_months(con, months)