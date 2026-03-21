"""
Downloads and consolidates CVM daily fund portfolio data
CVM publishes monthly files with daily NAV (cota) and net assets (PL)
for all registered funds in Brazil
Source: https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/
"""

import os
import zipfile
import io
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

CVM_BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def build_url(year: int, month: int) -> str:
    return f"{CVM_BASE_URL}inf_diario_fi_{year}{month:02d}.zip"

def download_month(year: int, month: int, save: bool = True) -> pd.DataFrame:
    """
    Downloads one month of CVM daily fund data
    year: int
    month: int
    save: bool
        If True, saves raw CSV to data/raw/
    Returns: pd.DataFrame
    """
    url = build_url(year, month)
    print(f"Fetching {url} ...")

    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    # CVM files are zipped; extract the CSV inside
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f,
                sep=";",
                encoding="latin-1",
                dtype={"CNPJ_FUNDO_CLASSE": str},
            )
        df["CNPJ_BASE"] = df["CNPJ_FUNDO_CLASSE"].str[:18]

    if save:
        os.makedirs(RAW_DIR, exist_ok=True)
        path = os.path.join(RAW_DIR, f"inf_diario_fi_{year}{month:02d}.csv")
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
    return df

def download_range(start: str, end: str) -> pd.DataFrame:
    """
    Downloads multiple months and concatenates into one DataFrame
    start: str  ("2025-01")
    end: str  ("2025-12")
    Returns: pd.DataFrame
    """
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")

    frames = []
    current = start_dt
    while current <= end_dt:
        try:
            df = download_month(current.year, current.month)
            frames.append(df)
        except Exception as e:
            print(f"  Warning: could not fetch {current.year}-{current.month:02d}: {e}")
        # advance one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    if not frames:
        raise ValueError("No data downloaded, check date range or connectivity")
    combined = pd.concat(frames, ignore_index=True)
    combined["DT_COMPTC"] = pd.to_datetime(combined["DT_COMPTC"])
    return combined

def load_register() -> pd.DataFrame:
    """
    Downloads the CVM fund register (cadastro) which contains fund name,
    type (classe), CNPJ and status
    Source: https://dados.cvm.gov.br/dados/FI/CAD/DADOS/
    """
    url = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
    print(f"Fetching fund register from: {url}")
    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.content.decode("latin-1")),
        sep=";",
        dtype={"CNPJ_FUNDO": str},
        low_memory=False,
    )
    return df