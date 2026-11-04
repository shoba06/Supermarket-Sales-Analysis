"""
eda.py — Thin EDA orchestration layer (calls statistical_analysis + visualizations).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statistical_analysis import run_statistical_analysis
from visualizations import run_eda_visualizations


def run_eda(df, df_feat):
    """Run full EDA: stats + 9 EDA visualizations."""
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Statistical analysis on cleaned (pre-encoding) data
    num_summary, cat_freq, biv_results, corr_matrix = run_statistical_analysis(df)

    # Visualizations on feature-engineered data (has Hour, DayOfWeek, etc.)
    run_eda_visualizations(df_feat)

    return num_summary, cat_freq, biv_results, corr_matrix
