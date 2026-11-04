"""
data_loader.py — Load and validate the Supermarket Sales dataset.
"""
import sys
import requests
import pandas as pd

# Make sure src/ is on the path when run standalone
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from config import DATASET_URL, RAW_CSV


def download_dataset() -> pd.DataFrame:
    """Download the CSV from a public URL and cache it locally."""
    if RAW_CSV.exists():
        print(f"[data_loader] Using cached dataset: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV)
        return df

    print(f"[data_loader] Downloading dataset from:\n  {DATASET_URL}")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    RAW_CSV.write_bytes(response.content)
    print(f"[data_loader] Saved to {RAW_CSV}  ({RAW_CSV.stat().st_size:,} bytes)")
    df = pd.read_csv(RAW_CSV)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Print a structured validation report for the raw dataset."""
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"  Rows        : {df.shape[0]:,}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print("\n-- Column names & dtypes " + "-" * 34)
    for col in df.columns:
        print(f"  {col:<35} {str(df[col].dtype):<12}")
    print("\n-- Missing values " + "-" * 41)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values detected.")
    else:
        print(missing[missing > 0].to_string())
    print(f"\n-- Duplicate rows : {df.duplicated().sum()}")
    print("\n-- First 5 rows " + "-" * 43)
    print(df.head().to_string())
    print("\n-- Descriptive statistics (numerical) " + "-" * 21)
    print(df.describe().to_string())
    print("=" * 60 + "\n")


def load_data(validate: bool = True) -> pd.DataFrame:
    """Top-level entry point: download + optional validation."""
    df = download_dataset()
    if validate:
        validate_dataset(df)
    return df


if __name__ == "__main__":
    load_data()
