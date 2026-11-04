"""
config.py — Central configuration for the Supermarket Sales Analysis project.
"""
import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR   = BASE_DIR / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
TABLES_DIR    = OUTPUTS_DIR / "tables"
METRICS_DIR   = OUTPUTS_DIR / "metrics"
SRC_DIR       = BASE_DIR / "src"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_URL  = "https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv"
RAW_CSV      = RAW_DIR / "supermarket_sales.csv"
PROCESSED_CSV = PROCESSED_DIR / "supermarket_sales_processed.csv"

# ── Feature definitions ───────────────────────────────────────────────────────
CATEGORICAL_COLS = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]
NUMERICAL_COLS   = ["Unit price", "Quantity", "Tax 5%", "Total", "cogs",
                    "gross margin percentage", "gross income", "Rating"]
DATETIME_COL     = "Date"
TIME_COL         = "Time"
ID_COL           = "Invoice ID"

# Target: predict 'Rating' (customer satisfaction score, 1–10).
# 'Rating' is NOT mathematically derived from other columns — it is a genuine
# survey response, so it's a clean regression target with no target leakage.
TARGET_COL = "Rating"

# Features used for modelling (exclude ID, target, and Date/Time strings after
# engineering; also exclude 'gross margin percentage' which is constant).
MODEL_FEATURE_COLS = [
    "Unit price", "Quantity", "Tax 5%", "cogs", "gross income",
    "Branch_enc", "City_enc", "Customer type_enc", "Gender_enc",
    "Product line_enc", "Payment_enc",
    "Hour", "DayOfWeek", "Month"
]

# ── Splits ────────────────────────────────────────────────────────────────────
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.15
TEST_SIZE  = 0.15
RANDOM_SEED = 42

# ── Model hyper-parameters ────────────────────────────────────────────────────
HIDDEN_SIZES    = [128, 64]
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 32
MAX_EPOCHS      = 100
EARLY_STOP_PAT  = 10          # patience in epochs
DROPOUT_RATE    = 0.2

# ── Paths for model artefacts ─────────────────────────────────────────────────
MODEL_CHECKPOINT = OUTPUTS_DIR / "best_model.pt"
SCALER_PATH      = PROCESSED_DIR / "scaler.joblib"

# ── Plotting ──────────────────────────────────────────────────────────────────
FIGURE_DPI = 120
PALETTE    = "Set2"

# Ensure all output directories exist at import time
for _d in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
