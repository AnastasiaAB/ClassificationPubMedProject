# %%
"""
Step 3b: Visualize baseline model errors on the held-out test set.

Input:  baseline_test_predictions.csv (produced by baseline_model.py)
Output: baseline_error_analysis.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line if running interactively / in a notebook
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PREDICTIONS_CSV = "baseline_test_predictions.csv"
OUTPUT_PNG = "baseline_error_analysis.png"

def main():
    df = pd.read_csv(PREDICTIONS_CSV)
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    y_proba = df["y_proba"]
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    labels = ["category_a\n(cancer)", "category_b\n(cardio)"]
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color=color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Held-out Test Set (n={len(df)})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Right panel: predicted probability distribution, correct vs incorrect ---
    ax2 = axes[1]
    ax2.hist(y_proba[correct], bins=30, alpha=0.6, label="Correct", color="#2c7fb8")
    ax2.hist(y_proba[~correct], bins=30, alpha=0.8, label="Misclassified", color="#d95f02")
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Predicted P(category_b)")
    ax2.set_ylabel("Count")
    ax2.set_title("Predicted Probability Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved visualization -> {OUTPUT_PNG}")

    # Quick numeric summary alongside the plot
    n_errors = (~correct).sum()
    print(f"Misclassified: {n_errors} of {len(df)} ({n_errors/len(df)*100:.1f}%)")
    print(f"  true=category_b, pred=category_a: {((y_true==1)&(y_pred==0)).sum()}")
    print(f"  true=category_a, pred=category_b: {((y_true==0)&(y_pred==1)).sum()}")

if __name__ == "__main__":
    main()
# %%
