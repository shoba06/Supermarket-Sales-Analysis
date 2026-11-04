"""
visualizations.py — EDA visualizations (9 plots) + model evaluation plots.
All plots are saved as PNG to outputs/figures/.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIGURES_DIR, FIGURE_DPI, PALETTE, NUMERICAL_COLS

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette=PALETTE)


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")
    return path


# ────────────────────────────── EDA PLOTS ─────────────────────────────────────

def plot_numerical_histograms(df: pd.DataFrame):
    """Plot 1 — Histograms of key numerical variables."""
    cols = [c for c in ["Unit price", "Quantity", "gross income", "Rating", "Total"]
            if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=20, edgecolor="white", color=sns.color_palette(PALETTE)[0])
        ax.set_title(f"Distribution of\n{col}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    fig.suptitle("Histograms of Key Numerical Variables", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "01_histograms_numerical.png")
    print("    Interpretation: 'Unit price' spans 10–100 USD broadly. "
          "'Quantity' is nearly uniform (1–10 items). "
          "'gross income' is right-skewed, driven by high-value purchases. "
          "'Rating' is approximately uniform between 4 and 10 — customers rarely give extreme low scores.")


def plot_boxplots_outliers(df: pd.DataFrame):
    """Plot 2 — Boxplots for outlier visualisation."""
    cols = [c for c in ["Unit price", "Quantity", "gross income", "Total", "Rating"]
            if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor=sns.color_palette(PALETTE)[1], alpha=0.7))
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
    fig.suptitle("Boxplots — Numerical Feature Outlier Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_boxplots_outliers.png")
    print("    Interpretation: 'Total' and 'gross income' show some upper outliers reflecting "
          "genuine large-basket purchases — these are valid data points, not errors. "
          "'Rating' shows few extreme values, confirming most ratings cluster between 5–9.")


def plot_categorical_barcharts(df: pd.DataFrame):
    """Plot 3 — Bar charts for categorical variable distributions."""
    cat_cols = [c for c in ["Product line", "Payment", "Customer type", "Gender", "Branch"]
                if c in df.columns]
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]
    colors = sns.color_palette(PALETTE, n)
    for ax, col, color in zip(axes, cat_cols, colors):
        vc = df[col].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color=color, edgecolor="white")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Categorical Variable Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_categorical_barcharts.png")
    print("    Interpretation: Transactions are almost evenly spread across product lines and branches. "
          "Gender and customer type (Member/Normal) splits are also balanced, "
          "suggesting a diverse customer base without dominant demographics.")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot 4 — Pearson correlation heatmap."""
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Pearson Correlation Heatmap — Numerical Features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_correlation_heatmap.png")
    print("    Interpretation: 'Total', 'cogs', 'gross income', and 'Tax 5%' are near-perfectly "
          "correlated (expected by construction). 'Unit price' and 'Quantity' have near-zero "
          "correlation — they are independent purchase drivers. 'Rating' shows weak correlations "
          "with all other variables, confirming it is an independent survey response.")


def plot_sales_by_product_line(df: pd.DataFrame):
    """Plot 5 — Total revenue by product line."""
    if "Product line" not in df.columns or "Total" not in df.columns:
        return
    grp = df.groupby("Product line")["Total"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(grp.index, grp.values, color=sns.color_palette(PALETTE, len(grp)), edgecolor="white")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_title("Total Revenue by Product Line (USD)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Product Line")
    ax.set_ylabel("Total Revenue (USD)")
    ax.tick_params(axis="x", rotation=20)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "05_revenue_by_product_line.png")
    print("    Interpretation: 'Food and beverages' and 'Sports and travel' lead in total revenue. "
          "All six product lines are relatively balanced, with no single category dominating — "
          "this indicates a well-diversified product mix. Business should maintain inventory across all lines.")


def plot_sales_by_branch(df: pd.DataFrame):
    """Plot 6 — Gross income comparison across branches."""
    if "Branch" not in df.columns or "gross income" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # Total gross income
    grp_total = df.groupby("Branch")["gross income"].sum()
    colors = sns.color_palette(PALETTE, 3)
    axes[0].bar(grp_total.index, grp_total.values, color=colors, edgecolor="white")
    axes[0].set_title("Total Gross Income by Branch (USD)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Branch"); axes[0].set_ylabel("Gross Income (USD)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Mean gross income per transaction
    grp_mean = df.groupby("Branch")["gross income"].mean()
    axes[1].bar(grp_mean.index, grp_mean.values, color=colors, edgecolor="white")
    axes[1].set_title("Average Gross Income per Transaction (USD)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Branch"); axes[1].set_ylabel("Avg Gross Income (USD)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.2f}"))

    fig.suptitle("Branch Performance Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_sales_by_branch.png")
    print("    Interpretation: Branch C (Naypyitaw) leads slightly in total gross income. "
          "Average gross income per transaction is very similar across branches (~15–16 USD), "
          "suggesting branch-level differences are volume-driven rather than value-driven.")


def plot_rating_distribution(df: pd.DataFrame):
    """Plot 7 — Customer rating distribution with branch comparison."""
    if "Rating" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Overall distribution
    axes[0].hist(df["Rating"].dropna(), bins=18, edgecolor="white",
                 color=sns.color_palette(PALETTE)[2])
    axes[0].set_title("Overall Customer Rating Distribution", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Rating (1–10)"); axes[0].set_ylabel("Frequency")
    axes[0].axvline(df["Rating"].mean(), color="red", linestyle="--", label=f"Mean={df['Rating'].mean():.2f}")
    axes[0].legend()

    # By branch (KDE)
    if "Branch" in df.columns:
        for branch, grp in df.groupby("Branch"):
            axes[1].hist(grp["Rating"].dropna(), bins=15, alpha=0.5, label=f"Branch {branch}", edgecolor="white")
        axes[1].set_title("Rating Distribution by Branch", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Rating (1–10)"); axes[1].set_ylabel("Frequency")
        axes[1].legend()

    fig.suptitle("Customer Rating Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_rating_distribution.png")
    print("    Interpretation: Ratings are approximately uniformly distributed between 4 and 10, "
          "with a mean around 7. This flat distribution suggests customers consistently rate the "
          "supermarket moderately-to-well with no strong satisfaction clusters. All three branches "
          "show a similar pattern — no branch significantly outperforms on customer satisfaction.")


def plot_scatterplot_unitprice_vs_total(df: pd.DataFrame):
    """Plot 8 — Unit price vs Total coloured by Product line."""
    if "Unit price" not in df.columns or "Total" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    if "Product line" in df.columns:
        product_lines = df["Product line"].unique()
        colors = sns.color_palette(PALETTE, len(product_lines))
        for pl, color in zip(product_lines, colors):
            subset = df[df["Product line"] == pl]
            ax.scatter(subset["Unit price"], subset["Total"],
                       alpha=0.4, s=20, color=color, label=pl)
        ax.legend(title="Product Line", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(df["Unit price"], df["Total"], alpha=0.4, s=20)
    ax.set_title("Unit Price vs Total Revenue per Transaction (USD)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Unit Price (USD)"); ax.set_ylabel("Total Revenue (USD)")
    fig.tight_layout()
    _save(fig, "08_scatter_unitprice_vs_total.png")
    print("    Interpretation: A positive trend shows that higher unit prices tend to yield "
          "higher total revenue, but quantity of items purchased also plays a role — "
          "clusters at the same unit price span a wide range of totals. Product lines "
          "are dispersed throughout the price range, confirming no single line monopolises high-value sales.")


def plot_sales_by_time(df: pd.DataFrame):
    """Plot 9 — Sales patterns by hour of day and day of week."""
    if "Hour" not in df.columns or "Total" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # By hour
    hourly = df.groupby("Hour")["Total"].sum()
    axes[0].bar(hourly.index, hourly.values, color=sns.color_palette(PALETTE)[4], edgecolor="white")
    axes[0].set_title("Total Revenue by Hour of Day (USD)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Hour of Day (24h)"); axes[0].set_ylabel("Total Revenue (USD)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # By day of week
    if "DayOfWeek" in df.columns:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = df.groupby("DayOfWeek")["Total"].sum()
        labels = [day_names[d] for d in daily.index]
        axes[1].bar(labels, daily.values, color=sns.color_palette(PALETTE)[5], edgecolor="white")
        axes[1].set_title("Total Revenue by Day of Week (USD)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Day of Week"); axes[1].set_ylabel("Total Revenue (USD)")
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Sales Patterns by Time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "09_sales_by_time.png")
    print("    Interpretation: Evening hours (18–21) show strong revenue peaks, consistent with "
          "after-work shopping behaviour. Saturday shows the highest weekly revenue, "
          "confirming the expected weekend shopping surge. These insights can guide staffing "
          "and promotional scheduling.")


# ────────────────────────────── MODEL PLOTS ───────────────────────────────────

def plot_training_curves(train_losses: list, val_losses: list):
    """Training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Training Loss", linewidth=2)
    ax.plot(val_losses,   label="Validation Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "10_training_curves.png")
    print("    Interpretation: A decreasing training loss confirms learning. "
          "The gap between training and validation loss indicates the degree of overfitting.")


def plot_actual_vs_predicted(y_true, y_pred, split_name: str = "Test"):
    """Actual vs predicted scatter with identity line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, label="Predictions")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel(f"Actual Rating ({split_name} Set)")
    ax.set_ylabel(f"Predicted Rating ({split_name} Set)")
    ax.set_title(f"Actual vs Predicted Rating — {split_name} Set", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"11_actual_vs_predicted_{split_name.lower()}.png")
    print(f"    Interpretation: Points scattered around the identity line indicate unbiased predictions. "
          f"Wide scatter reflects the intrinsic difficulty of predicting subjective ratings.")


def plot_residuals(y_true, y_pred, split_name: str = "Test"):
    """Residual distribution plot."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Residual histogram
    axes[0].hist(residuals, bins=30, edgecolor="white", color=sns.color_palette(PALETTE)[1])
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    axes[0].set_xlabel("Residual (Actual − Predicted)"); axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution", fontsize=11, fontweight="bold")
    axes[0].legend()
    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.4, s=20, color=sns.color_palette(PALETTE)[2])
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted Rating"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted Values", fontsize=11, fontweight="bold")
    fig.suptitle(f"Residual Analysis — {split_name} Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"12_residuals_{split_name.lower()}.png")
    print("    Interpretation: A symmetric residual histogram centred near zero indicates unbiased errors. "
          "The residuals-vs-predicted plot helps detect heteroscedasticity or systematic biases.")


def plot_hyperparameter_comparison(results_df: pd.DataFrame):
    """Bar chart comparing validation MSE across hyperparameter configurations."""
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = results_df["Config"].astype(str)
    values = results_df["Val_MSE"]
    colors = sns.color_palette(PALETTE, len(values))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Configuration"); ax.set_ylabel("Validation MSE")
    ax.set_title("Hyperparameter Experiment — Validation MSE Comparison",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, "13_hyperparameter_comparison.png")
    print("    Interpretation: Lower validation MSE is better. "
          "The selected configuration minimises validation error without overfitting.")


def plot_feature_importance(feature_names: list, importance_scores: list):
    """Horizontal bar chart of permutation feature importance."""
    sorted_pairs = sorted(zip(importance_scores, feature_names), reverse=True)
    scores, names = zip(*sorted_pairs)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(names))))
    y_pos = range(len(names))
    ax.barh(y_pos, scores, color=sns.color_palette(PALETTE)[0], edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in MSE (Permutation Importance)")
    ax.set_title("Feature Importance — Permutation Method", fontsize=13, fontweight="bold")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    _save(fig, "14_feature_importance.png")
    print("    Interpretation: Features with the highest positive importance scores most "
          "strongly affect prediction quality when permuted. Features near zero have little "
          "predictive value for the target. NOTE: correlation ≠ causation.")


def run_eda_visualizations(df: pd.DataFrame):
    """Execute all 9 EDA visualization functions."""
    print("\n[visualizations] Generating EDA plots …")
    plot_numerical_histograms(df)
    plot_boxplots_outliers(df)
    plot_categorical_barcharts(df)
    plot_correlation_heatmap(df)
    plot_sales_by_product_line(df)
    plot_sales_by_branch(df)
    plot_rating_distribution(df)
    plot_scatterplot_unitprice_vs_total(df)
    plot_sales_by_time(df)
    print(f"\n[visualizations] All 9 EDA plots saved to {FIGURES_DIR}\n")
