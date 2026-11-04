"""
train.py — Training loop with early stopping, checkpointing, and experiment tracking.
"""
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MAX_EPOCHS, EARLY_STOP_PAT, BATCH_SIZE, LEARNING_RATE,
    HIDDEN_SIZES, DROPOUT_RATE, MODEL_CHECKPOINT, RANDOM_SEED
)
from model import build_model


# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    hidden_sizes: list  = None,
    lr:           float = LEARNING_RATE,
    batch_size:   int   = BATCH_SIZE,
    max_epochs:   int   = MAX_EPOCHS,
    patience:     int   = EARLY_STOP_PAT,
    dropout:      float = DROPOUT_RATE,
    save_path:    Path  = MODEL_CHECKPOINT,
    verbose:      bool  = True,
) -> dict:
    """
    Train the MLP; return a dict with the best model, epoch histories, and metadata.
    """
    if hidden_sizes is None:
        hidden_sizes = HIDDEN_SIZES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [train] Device: {device}")

    input_dim = X_train.shape[1]
    model = build_model(input_dim, hidden_sizes=hidden_sizes, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    best_epoch    = 0

    for epoch in range(1, max_epochs + 1):
        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        batch_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                val_batch_losses.append(criterion(preds, y_b).item())
        val_loss = float(np.mean(val_batch_losses))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}/{max_epochs} | "
                  f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        if no_improve >= patience:
            print(f"  [train] Early stopping at epoch {epoch} "
                  f"(best val MSE={best_val_loss:.4f} at epoch {best_epoch}).")
            break

    # Restore best weights and save checkpoint
    model.load_state_dict(best_state)
    torch.save({"model_state": best_state,
                "input_dim":   input_dim,
                "hidden_sizes": hidden_sizes,
                "dropout":     dropout,
                "best_epoch":  best_epoch,
                "best_val_mse": best_val_loss}, save_path)
    print(f"  [train] Best model saved to {save_path}")

    return {
        "model":        model,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_epoch":   best_epoch,
        "best_val_mse": best_val_loss,
        "device":       device,
    }


def run_hyperparameter_experiments(
    X_train, y_train, X_val, y_val
) -> list:
    """
    Compare a small set of hyperparameter configurations on the validation set.
    Returns a list of result dicts.
    """
    configs = [
        {"name": "LR=1e-3 / H=[128,64]",   "lr": 1e-3, "hidden": [128, 64],  "bs": 32},
        {"name": "LR=1e-4 / H=[128,64]",   "lr": 1e-4, "hidden": [128, 64],  "bs": 32},
        {"name": "LR=1e-3 / H=[64,32]",    "lr": 1e-3, "hidden": [64, 32],   "bs": 32},
        {"name": "LR=1e-3 / H=[256,128]",  "lr": 1e-3, "hidden": [256, 128], "bs": 64},
        {"name": "LR=1e-3 / H=[128,64,32]","lr": 1e-3, "hidden": [128,64,32],"bs": 32},
    ]
    results = []
    print("\n[train] Running hyperparameter experiments (5 configs × 40 epochs max) …")
    for cfg in configs:
        print(f"\n  Config: {cfg['name']}")
        res = train_model(
            X_train, y_train, X_val, y_val,
            hidden_sizes = cfg["hidden"],
            lr           = cfg["lr"],
            batch_size   = cfg["bs"],
            max_epochs   = 40,     # reduced for speed
            patience     = 8,
            save_path    = MODEL_CHECKPOINT.parent / f"exp_{cfg['name'].replace('/', '_').replace(' ','_')}.pt",
            verbose      = False,
        )
        results.append({
            "Config":       cfg["name"],
            "LR":           cfg["lr"],
            "HiddenSizes":  str(cfg["hidden"]),
            "BatchSize":    cfg["bs"],
            "Best_Epoch":   res["best_epoch"],
            "Val_MSE":      round(res["best_val_mse"], 6),
        })
        print(f"    Val MSE: {res['best_val_mse']:.6f}  (epoch {res['best_epoch']})")
    return results
