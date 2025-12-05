#!/usr/bin/env python3
"""
Evaluation plots for Meta-AC classifier:
- ROC curve & confusion matrix
- Simplified impact views (scatter + box/hist)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import auc, confusion_matrix, roc_curve

from meta_ac.config import OUTPUT_DIR, PREDICTIONS_PATH

DATA_PATH = PREDICTIONS_PATH
EVAL_PLOT_PATH = OUTPUT_DIR / "meta_ac_eval.png"
SCATTER_PLOT_PATH = OUTPUT_DIR / "meta_ac_scatter.png"
BOX_PLOT_PATH = OUTPUT_DIR / "meta_ac_box.png"
HIST_PLOT_PATH = OUTPUT_DIR / "meta_ac_hist.png"


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names to expected schema
    if "prob_accept" in df.columns and "probability" not in df.columns:
        df = df.rename(columns={"prob_accept": "probability"})
    if "true_label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"true_label": "label"})
    required = {"probability", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Prediction CSV missing required columns: {required - set(df.columns)}")
    return df


def main() -> None:
    sns.set_theme(style="white", context="talk", palette="Set2")
    df = load_predictions(DATA_PATH)
    y_true = df["label"].to_numpy()
    y_score = df["probability"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="C0")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (threshold=0.5)")

    plt.tight_layout()
    EVAL_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(EVAL_PLOT_PATH, dpi=300)
    print(f"Saved evaluation plot to {EVAL_PLOT_PATH}")

    # Simplified scatter: Raw Avg Rating vs Probability (color by variance)
    raw = df.get("raw_avg")
    variance = df.get("variance", pd.Series(0.0, index=df.index))
    if raw is not None:
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            raw,
            y_score,
            c=variance.fillna(0.0),
            cmap="magma_r",
            s=60,
            alpha=0.75,
            linewidths=0.2,
            edgecolors="k",
        )
        cbar = plt.colorbar(sc, shrink=0.85)
        cbar.set_label("Rating Variance", rotation=270, labelpad=14)
        plt.xlabel("Raw Avg Rating")
        plt.ylabel("Predicted Probability")
        plt.title("Raw Rating vs Predicted Probability")
        plt.tight_layout()
        plt.savefig(SCATTER_PLOT_PATH, dpi=300)
        print(f"Saved scatter plot to {SCATTER_PLOT_PATH}")
    else:
        print("Raw_avg not found; skipping scatter plot.")

    # Box/strip plot of probability by label
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df["label"], y=y_score, color="#a7cbe3", fliersize=0)
    sns.stripplot(
        x=df["label"],
        y=y_score,
        color="k",
        alpha=0.35,
        jitter=0.18,
        size=4,
    )
    plt.xlabel("True Label")
    plt.ylabel("Predicted Probability")
    plt.title("Probability Distribution by Label")
    plt.tight_layout()
    plt.savefig(BOX_PLOT_PATH, dpi=300)
    print(f"Saved box plot to {BOX_PLOT_PATH}")

    # Histogram / density of predicted probabilities
    plt.figure(figsize=(8, 6))
    sns.histplot(y_score, bins=25, kde=True, color="#4C72B0", edgecolor="white", alpha=0.85)
    plt.xlabel("Predicted Probability")
    plt.title("Predicted Probability Distribution")
    plt.tight_layout()
    plt.savefig(HIST_PLOT_PATH, dpi=300)
    print(f"Saved histogram to {HIST_PLOT_PATH}")


if __name__ == "__main__":
    main()
