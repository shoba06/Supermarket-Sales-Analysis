"""
statistical_analysis.py — Univariate, bivariate, and multivariate statistical summaries.
"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NUMERICAL_COLS, CATEGORICAL_COLS, TABLES_DIR


def univariate_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended descriptive statistics for numerical columns."""
    rows = []
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    for col in num_cols:
        s = df[col].dropna()
        rows.append({
            "Feature"  : col,
            "Count"    : len(s),
            "Mean"     : round(s.mean(), 4),
            "Median"   : round(s.median(), 4),
            "Mode"     : round(s.mode().iloc[0], 4) if len(s.mode()) else np.nan,
            "Std"      : round(s.std(), 4),
            "Variance" : round(s.var(), 4),
            "Min"      : round(s.min(), 4),
            "Max"      : round(s.max(), 4),
            "Q1"       : round(s.quantile(0.25), 4),
            "Q3"       : round(s.quantile(0.75), 4),
            "IQR"      : round(s.quantile(0.75) - s.quantile(0.25), 4),
            "Skewness" : round(float(stats.skew(s)), 4),
            "Kurtosis" : round(float(stats.kurtosis(s)), 4),
        })
    result = pd.DataFrame(rows)
    out_path = TABLES_DIR / "univariate_numerical.csv"
    result.to_csv(out_path, index=False)
    print(f"  [stats] Univariate numerical summary saved to {out_path}")
    return result


def univariate_categorical(df: pd.DataFrame) -> dict:
    """Compute frequency distributions for categorical columns."""
    freq_tables = {}
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols:
        vc = df[col].value_counts(normalize=False)
        pct = df[col].value_counts(normalize=True) * 100
        tbl = pd.DataFrame({"Count": vc, "Percentage": pct.round(2)})
        freq_tables[col] = tbl
        out_path = TABLES_DIR / f"freq_{col.replace(' ', '_')}.csv"
        tbl.to_csv(out_path)
    print(f"  [stats] Categorical frequency tables saved to {TABLES_DIR}")
    return freq_tables


def bivariate_analysis(df: pd.DataFrame) -> None:
    """Group-level means for key business questions."""
    print("\n-- Bivariate Analysis -----------------------------------------")
    analyses = [
        ("Product line", "gross income", "Gross income by product line"),
        ("Branch",       "gross income", "Gross income by branch"),
        ("Product line", "Rating",       "Rating by product line"),
        ("Gender",       "Total",        "Total sales by gender"),
        ("Payment",      "Total",        "Total sales by payment method"),
        ("Customer type","Total",        "Total sales by customer type"),
    ]
    results = {}
    for group_col, value_col, label in analyses:
        if group_col in df.columns and value_col in df.columns:
            grp = df.groupby(group_col)[value_col].agg(["mean", "median", "sum", "count"])
            grp.columns = ["Mean", "Median", "Sum", "Count"]
            grp = grp.round(4)
            print(f"\n  {label}:")
            print(grp.to_string())
            results[label] = grp
            grp.to_csv(TABLES_DIR / f"bivariate_{label.replace(' ', '_')}.csv")
    return results


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for numerical features."""
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    corr = df[num_cols].corr(method="pearson").round(4)
    out_path = TABLES_DIR / "correlation_matrix.csv"
    corr.to_csv(out_path)
    print(f"\n  [stats] Correlation matrix saved to {out_path}")
    return corr


def run_statistical_analysis(df: pd.DataFrame):
    """Full statistical analysis pipeline."""
    print("\n[statistical_analysis] Running statistical analysis …")
    num_summary  = univariate_numerical(df)
    print("\n-- Univariate Numerical Summary -------------------------------")
    print(num_summary.to_string(index=False))

    cat_freq     = univariate_categorical(df)
    print("\n-- Categorical Frequency Distributions ------------------------")
    for col, tbl in cat_freq.items():
        print(f"\n  {col}:")
        print(tbl.to_string())

    biv_results  = bivariate_analysis(df)
    corr_matrix  = correlation_analysis(df)

    print("\n-- Correlation Matrix ------------------------------------------")
    print(corr_matrix.to_string())

    print("\n[statistical_analysis] Complete.\n")
    return num_summary, cat_freq, biv_results, corr_matrix
