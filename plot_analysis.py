#!/usr/bin/env python3
"""
Evaluation plots for Meta-AC classifier: ROC curve and confusion matrix.
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
OUTPUT_PATH = OUTPUT_DIR / "meta_ac_eval.png"


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"probability", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Prediction CSV missing required columns: {required - set(df.columns)}")
    return df


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved evaluation plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
