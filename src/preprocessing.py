"""
preprocessing.py — Data cleaning, feature engineering, encoding, scaling, and splitting.
"""
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL, MODEL_FEATURE_COLS,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED,
    PROCESSED_CSV, SCALER_PATH
)


# -- Helpers -------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return a DataFrame summarising IQR-based outlier counts."""
    records = []
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        records.append({
            "Feature": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower fence": round(lower, 4),
            "Upper fence": round(upper, 4),
            "Outlier count": n_out,
            "Outlier %": round(100 * n_out / len(df), 2),
        })
    return pd.DataFrame(records)


# -- Main pipeline -------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw DataFrame."""
    df = df.copy()

    # 1. Remove exact duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup:
        print(f"  [clean] Removing {n_dup} duplicate rows.")
        df = df.drop_duplicates()
    else:
        print("  [clean] No duplicate rows found.")

    # 2. Drop rows with missing values (none expected, but handle gracefully)
    n_missing_before = df.isnull().sum().sum()
    if n_missing_before:
        print(f"  [clean] Imputing / dropping {n_missing_before} missing cells.")
        for col in NUMERICAL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Validate non-negative monetary and quantity values
    for col in ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                print(f"  [clean] WARNING: {neg} negative values in '{col}'. Removing.")
                df = df[df[col] >= 0]

    # 4. Standardise column names (strip leading/trailing whitespace)
    df.columns = [c.strip() for c in df.columns]

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date/time derived features and ordinal encodings."""
    df = df.copy()

    # -- Date/time features ----------------------------------------------------
    if "Date" in df.columns:
        df["Date_parsed"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
        df["Month"]     = df["Date_parsed"].dt.month
        df["DayOfWeek"] = df["Date_parsed"].dt.dayofweek   # 0 = Monday
        df["Day"]       = df["Date_parsed"].dt.day

    if "Time" in df.columns:
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour

    # -- Label-encode categoricals ---------------------------------------------
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    print(f"  [engineer] Features added: Month, DayOfWeek, Day, Hour + {len(CATEGORICAL_COLS)} label-encoded cols.")
    return df, encoders


def split_data(df: pd.DataFrame):
    """Split into train / val / test; fit scaler on train only."""
    available_features = [f for f in MODEL_FEATURE_COLS if f in df.columns]
    missing_feats = [f for f in MODEL_FEATURE_COLS if f not in df.columns]
    if missing_feats:
        print(f"  [split] WARNING: Missing engineered features: {missing_feats}")

    X = df[available_features].values
    y = df[TARGET_COL].values

    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_SEED
    )
    # Val / test split from temp
    val_frac = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_frac),
        random_state=RANDOM_SEED
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"  [split] Scaler saved to {SCALER_PATH}")
    print(f"  [split] Train : {X_train_sc.shape[0]}  |  Val : {X_val_sc.shape[0]}  |  Test : {X_test_sc.shape[0]}")
    print(f"  [split] Features used: {available_features}")

    return (X_train_sc, y_train,
            X_val_sc,   y_val,
            X_test_sc,  y_test,
            scaler, available_features)


def run_preprocessing(df: pd.DataFrame):
    """Full preprocessing pipeline; returns cleaned df + split arrays."""
    print("\n[preprocessing] Starting preprocessing pipeline …")

    df_clean = clean_data(df)

    # Outlier report (detect only, do NOT remove sales extremes)
    print("\n-- Outlier Detection (IQR method) -------------------------")
    num_cols_present = [c for c in NUMERICAL_COLS if c in df_clean.columns]
    outlier_report = detect_outliers_iqr(df_clean, num_cols_present)
    print(outlier_report.to_string(index=False))
    print("  NOTE: Legitimate extreme sales values are preserved.")

    df_feat, encoders = engineer_features(df_clean)

    # Save processed dataset
    df_feat.to_csv(PROCESSED_CSV, index=False)
    print(f"\n  [preprocessing] Processed CSV saved to {PROCESSED_CSV}")

    splits = split_data(df_feat)

    print("\n[preprocessing] Pipeline complete.\n")
    return df_clean, df_feat, encoders, splits


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_data
    df = load_data(validate=False)
    run_preprocessing(df)
