"""
evaluate.py — Model evaluation on held-out test set + feature importance.
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import METRICS_DIR, MODEL_CHECKPOINT


class TorchMLPWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper around the PyTorch MLP for permutation importance."""

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def fit(self, X, y):
        return self   # Already trained

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t).cpu().numpy()


def evaluate_model(
    model,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    device,
    feature_names: list = None,
    split_name:    str  = "Test",
) -> dict:
    """Evaluate a trained MLP on the test set; return metrics dict."""
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_t).cpu().numpy()

    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {
        "split":  split_name,
        "MAE":    round(mae,  6),
        "MSE":    round(mse,  6),
        "RMSE":   round(rmse, 6),
        "R2":     round(r2,   6),
        "n_samples": len(y_test),
    }

    print(f"\n── {split_name} Set Evaluation Results ────────────────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  n    : {len(y_test)}")

    # Save metrics
    out_path = METRICS_DIR / f"metrics_{split_name.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {out_path}")

    return metrics, y_pred


def compute_permutation_importance(
    model, X_test: np.ndarray, y_test: np.ndarray, device,
    feature_names: list, n_repeats: int = 10, seed: int = 42
) -> tuple:
    """Compute permutation feature importance using the sklearn wrapper."""
    print("\n[evaluate] Computing permutation feature importance …")
    wrapper = TorchMLPWrapper(model, device)
    result  = permutation_importance(
        wrapper, X_test, y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    importances = result.importances_mean   # negative MSE deltas; higher = more important
    # Convert to positive "importance" — more negative delta = less important
    importances = -importances              # flip sign so positive = important

    print("  Permutation Feature Importance (sorted):")
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        name = feature_names[i] if feature_names else f"f{i}"
        print(f"    {name:<30} {importances[i]:+.6f}")
    print("  NOTE: Correlation ≠ causation. Importance reflects predictive contribution only.")

    # Save
    import pandas as pd
    imp_df = pd.DataFrame({
        "Feature":    [feature_names[i] if feature_names else f"f{i}" for i in range(len(importances))],
        "Importance": importances.round(6),
    }).sort_values("Importance", ascending=False)
    out_path = METRICS_DIR / "feature_importance.csv"
    imp_df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return importances, feature_names
