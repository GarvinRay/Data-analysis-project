#!/usr/bin/env python3
"""
Visualization script for Meta-AC probability adjustments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path("meta_ac_predictions.csv")
OUTPUT_PATH = Path("meta_ac_impact.png")


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    df = pd.read_csv(path)
    expected_columns = {"raw_avg", "variance", "final_prob"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "sentiment_change" not in df.columns:
        df["sentiment_change"] = 0.0
    return df


def normalize_raw_scores(df: pd.DataFrame) -> pd.Series:
    """
    Normalize raw average ratings into [0, 1], assuming 0-10 scale.
    """
    return df["raw_avg"] / 10.0


def annotate_top_deltas(ax, df: pd.DataFrame, delta_col: str, top_n: int = 3):
    """
    Annotate the top-N papers with the largest probability adjustments.
    """
    top_rows = df.nlargest(top_n, delta_col)
    for _, row in top_rows.iterrows():
        ax.annotate(
            row.get("title", "Paper"),
            (row["raw_avg"], row["final_prob"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df = load_predictions(DATA_PATH)
    df = df.dropna(subset=["raw_avg", "final_prob"]).copy()

    df["raw_norm"] = normalize_raw_scores(df)
    df["sentiment_change"] = df.get("sentiment_change", 0.0)
    df["delta"] = df["final_prob"] - df["raw_norm"]

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["raw_avg"],
        df["final_prob"],
        c=df["variance"].fillna(0.0),
        cmap="Reds",
        s=50 + 200 * df["sentiment_change"].fillna(0.0).clip(lower=0.0),
        alpha=0.8,
        edgecolor="k",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Rating Variance", rotation=270, labelpad=15)

    # Reference line representing raw score ≈ probability
    x_vals = np.linspace(df["raw_avg"].min(), df["raw_avg"].max(), 100)
    plt.plot(x_vals, x_vals / 10.0, linestyle="--", color="gray", label="Raw Score = Probability")

    annotate_top_deltas(plt.gca(), df, "delta", top_n=3)

    plt.xlabel("Raw Avg Rating")
    plt.ylabel("Meta-AC Probability")
    plt.title("Meta-AC Adjustments vs. Raw Ratings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
