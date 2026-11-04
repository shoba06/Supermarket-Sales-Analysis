# Supermarket Sales Analysis — Deep Learning Project

## Overview
This project performs Exploratory Data Analysis (EDA) and deep learning on the
**Supermarket Sales** dataset (1,000 transactions, 17 original features).



## Dataset — Feature Limitation Note
The original dataset has **17 columns** (not 15+ independent features).  
Several columns are mathematically derived from others:
- `Tax 5%` = 5% of `cogs`
- `Total` = `cogs` + `Tax 5%`
- `gross income` = `Tax 5%` (by construction at 4.762% margin)
- `gross margin percentage` is constant (4.761904762%)

The project documents this clearly and uses `Rating` as the prediction target
because it is a genuine survey score unrelated to the financial calculations.

## Project Structure
```
supermarket-sales-analysis/
├── data/raw/           — Raw downloaded CSV
├── data/processed/     — Cleaned CSV + scaler.joblib
├── outputs/
│   ├── figures/        — 14 PNG plots
│   ├── tables/         — Statistical summary CSVs
│   └── metrics/        — JSON metrics + experiment CSV
├── src/
│   ├── config.py       — Paths, hyperparameters, feature lists
│   ├── data_loader.py  — Download + validate
│   ├── preprocessing.py — Clean, engineer, encode, scale, split
│   ├── statistical_analysis.py — Univariate / bivariate / correlation stats
│   ├── eda.py          — EDA orchestrator
│   ├── visualizations.py — 9 EDA + 5 evaluation plots
│   ├── model.py        — PyTorch MLP definition
│   ├── train.py        — Training loop + hyperparameter experiments
│   └── evaluate.py     — Metrics + permutation importance
├── main.py             — Single entry point
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. (Optional) Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py
```

All outputs are written to `outputs/figures/`, `outputs/tables/`, and
`outputs/metrics/`. The best model checkpoint is saved to `outputs/best_model.pt`.

## Deep Learning Model
- **Architecture:** MLP (input → [128, 64] → 1 neuron)
- **Target:** `Rating` (customer satisfaction, 1–10 scale)
- **Rationale:** Rating is the only column not derivable from other features
- **Loss:** MSE  |  **Optimizer:** Adam  |  **Scheduler:** ReduceLROnPlateau
- **Regularisation:** BatchNorm + Dropout (0.2) + Early Stopping (patience=10)
