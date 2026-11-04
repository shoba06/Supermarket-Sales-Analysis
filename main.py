"""
main.py — Single entry point for the Supermarket Sales Analysis project.

Usage:
    python main.py

Executes the full pipeline:
  1. Load & validate dataset
  2. Preprocessing (clean, engineer, encode, scale, split)
  3. Statistical analysis
  4. EDA visualizations
  5. Hyperparameter experiments
  6. Final model training
  7. Test-set evaluation
  8. Evaluation plots
  9. Feature importance
  10. Save all outputs
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# -- Step 0: Imports -----------------------------------------------------------
print("=" * 65)
print("  SUPERMARKET SALES ANALYSIS — DEEP LEARNING PROJECT")
print("=" * 65)
t_start = time.time()

from src.config import (
    METRICS_DIR, FIGURES_DIR, MODEL_CHECKPOINT,
    HIDDEN_SIZES, LEARNING_RATE, BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PAT,
    DROPOUT_RATE, RANDOM_SEED
)
from src.data_loader       import load_data
from src.preprocessing     import run_preprocessing
from src.eda               import run_eda
from src.train             import train_model, run_hyperparameter_experiments
from src.evaluate          import evaluate_model, compute_permutation_importance
from src.visualizations    import (
    plot_training_curves, plot_actual_vs_predicted,
    plot_residuals, plot_hyperparameter_comparison, plot_feature_importance
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & VALIDATE DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading and validating dataset …")
df_raw = load_data(validate=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Preprocessing …")
df_clean, df_feat, encoders, splits = run_preprocessing(df_raw)
(X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names) = splits

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 & 4 — STATISTICAL ANALYSIS + EDA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3-4] EDA + Statistical Analysis …")
num_summary, cat_freq, biv_results, corr_matrix = run_eda(df_clean, df_feat)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — HYPERPARAMETER EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Hyperparameter experiments …")
exp_results = run_hyperparameter_experiments(X_train, y_train, X_val, y_val)
exp_df = pd.DataFrame(exp_results)
print("\n-- Experiment Results ------------------------------------------")
print(exp_df.to_string(index=False))
exp_out = METRICS_DIR / "hyperparameter_experiments.csv"
exp_df.to_csv(exp_out, index=False)
print(f"  Saved to {exp_out}")

# Select best config by lowest Val_MSE
best_cfg_row = exp_df.loc[exp_df["Val_MSE"].idxmin()]
best_cfg_name = best_cfg_row["Config"]
print(f"\n  Best configuration: {best_cfg_name}  (Val MSE = {best_cfg_row['Val_MSE']:.6f})")
plot_hyperparameter_comparison(exp_df)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TRAIN BEST MODEL (full epochs)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[STEP 6] Training final model with best configuration …")
import ast
best_hidden = ast.literal_eval(best_cfg_row["HiddenSizes"])
best_lr     = float(best_cfg_row["LR"])
best_bs     = int(best_cfg_row["BatchSize"])

train_result = train_model(
    X_train, y_train, X_val, y_val,
    hidden_sizes = best_hidden,
    lr           = best_lr,
    batch_size   = best_bs,
    max_epochs   = MAX_EPOCHS,
    patience     = EARLY_STOP_PAT,
    dropout      = DROPOUT_RATE,
    save_path    = MODEL_CHECKPOINT,
    verbose      = True,
)
model        = train_result["model"]
device       = train_result["device"]
train_losses = train_result["train_losses"]
val_losses   = train_result["val_losses"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Evaluating on held-out test set …")
test_metrics, y_pred_test = evaluate_model(
    model, X_test, y_test, device,
    feature_names=feature_names, split_name="Test"
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 8] Generating evaluation plots …")
plot_training_curves(train_losses, val_losses)
plot_actual_vs_predicted(y_test, y_pred_test, split_name="Test")
plot_residuals(y_test, y_pred_test, split_name="Test")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 9] Computing feature importance …")
importances, feat_names = compute_permutation_importance(
    model, X_test, y_test, device, feature_names=feature_names
)
plot_feature_importance(feat_names, importances.tolist())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — VERIFY OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 10] Verifying outputs …")
fig_files = list(FIGURES_DIR.glob("*.png"))
met_files = list(METRICS_DIR.glob("*"))
print(f"  Figures : {len(fig_files)} PNG files in {FIGURES_DIR}")
for f in sorted(fig_files):
    print(f"    {f.name}")
print(f"  Metrics : {len(met_files)} files in {METRICS_DIR}")
for f in sorted(met_files):
    print(f"    {f.name}")
print(f"  Model   : {MODEL_CHECKPOINT}  (exists={MODEL_CHECKPOINT.exists()})")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print("\n" + "=" * 65)
print("  FINAL COMPLETION SUMMARY")
print("=" * 65)
print(f"  Dataset       : Supermarket Sales (selva86 / GitHub mirror of Kaggle aungpyaeap)")
print(f"  Rows × Cols   : {df_raw.shape[0]} × {df_raw.shape[1]}")
print(f"  Original cols : {list(df_raw.columns)}")
print(f"  EDA Plots     : {len([f for f in fig_files if f.name.startswith('0')])} plots (01–09)")
print(f"  Model Plots   : {len([f for f in fig_files if not f.name.startswith('0')])} plots (10–14)")
print(f"  DL Model      : MLP  {X_train.shape[1]} → {best_hidden} → 1  (regression)")
print(f"  Target        : Rating (1–10, customer satisfaction survey, no leakage)")
print(f"  Best config   : {best_cfg_name}")
print(f"  Training      : {len(train_losses)} epochs, early-stop patience={EARLY_STOP_PAT}")
print(f"  Test MAE      : {test_metrics['MAE']:.4f}")
print(f"  Test MSE      : {test_metrics['MSE']:.4f}")
print(f"  Test RMSE     : {test_metrics['RMSE']:.4f}")
print(f"  Test R²       : {test_metrics['R2']:.4f}")
print(f"  Model saved   : {MODEL_CHECKPOINT}")
print(f"  Figures dir   : {FIGURES_DIR}")
print(f"  Metrics dir   : {METRICS_DIR}")
print(f"  Total runtime : {elapsed:.1f} seconds")
print("=" * 65)
print("  PROJECT COMPLETE")
print("=" * 65)
