#!/usr/bin/env python3
"""
Train a logistic regression model to combine Meta-AC features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from agents import ArgumentAgent, BayesianAgent
from meta_ac.models import PaperRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Meta-AC logistic regression.")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("meta_ac_stats_sampled.csv"),
        help="CSV with quantitative features (default: %(default)s).",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("final_predictions.csv"),
        help="CSV with Meta-AC prediction details (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("meta_ac_model.pkl"),
        help="Output path for trained model (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("meta_ac_dataset_sampled.json"),
        help="JSON corpus used to recompute sentiment if needed.",
    )
    return parser.parse_args()


def load_and_merge(
    stats_path: Path, predictions_path: Path, dataset_path: Path
) -> pd.DataFrame:
    stats_df = pd.read_csv(stats_path)
    if "paper_id" not in stats_df.columns:
        raise ValueError("meta_ac_stats.csv must contain a 'paper_id' column.")
    preds_df = load_predictions(stats_df, predictions_path, dataset_path)
    merged = pd.merge(stats_df, preds_df, on="paper_id", suffixes=("_stats", "_pred"))
    return merged


def load_predictions(
    stats_df: pd.DataFrame, predictions_path: Path, dataset_path: Path
) -> pd.DataFrame:
    if predictions_path.exists():
        raw_preds = pd.read_csv(predictions_path)
    else:
        raw_preds = pd.DataFrame()

    preds_df = pd.DataFrame({"paper_id": stats_df["paper_id"]})

    if not raw_preds.empty:
        if "paper_id" not in raw_preds.columns:
            if len(raw_preds) > len(stats_df):
                raise ValueError(
                    "Predictions file is longer than stats file and lacks 'paper_id'; "
                    "cannot align rows."
                )
            raw_preds = raw_preds.copy()
            raw_preds["paper_id"] = stats_df.loc[: len(raw_preds) - 1, "paper_id"].values
        preds_df = preds_df.merge(raw_preds, on="paper_id", how="left")

    if (
        "sentiment_score" not in preds_df.columns
        or preds_df["sentiment_score"].isna().all()
    ):
        sentiment_df = recompute_sentiments(stats_df, dataset_path)
        preds_df = preds_df.drop(columns=[col for col in ["sentiment_score"] if col in preds_df])
        preds_df = preds_df.merge(sentiment_df, on="paper_id", how="left")

    if "decision" not in preds_df.columns and "decision" in raw_preds.columns:
        preds_df["decision"] = raw_preds["decision"]

    return preds_df



def load_dataset_records(dataset_path: Path) -> Dict[str, PaperRecord]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset JSON not found at {dataset_path}; cannot recompute sentiment scores."
        )
    with dataset_path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    records = [PaperRecord.from_dict(item) for item in payload]
    return {
        record.paper_id: record
        for record in records
        if record.paper_id
    }


def recompute_sentiments(stats_df: pd.DataFrame, dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset JSON not found at {dataset_path}; "
            "cannot recompute sentiment scores."
        )
    record_lookup = load_dataset_records(dataset_path)
    argument_agent = ArgumentAgent()
    sentiment_rows = []
    for paper_id in stats_df["paper_id"]:
        paper = record_lookup.get(paper_id)
        pairs = (
            [pair.to_dict() for pair in paper.review_rebuttal_pairs]
            if paper
            else None
        )
        first_pair = pairs[0] if pairs else None
        if first_pair:
            result = argument_agent.analyze_pair(first_pair)
        else:
            result = {"sentiment_change": 0.0}
        sentiment = float(result.get("sentiment_change", 0.0))
        sentiment_score = float(np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0))
        sentiment_rows.append({"paper_id": paper_id, "sentiment_score": sentiment_score})
    return pd.DataFrame(sentiment_rows)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "avg_rating": ["avg_rating_stats", "avg_rating"],
        "rating_variance": ["rating_variance_stats", "rating_variance"],
        "num_reviews": ["num_reviews_stats", "num_reviews"],
        "confidence_weighted_avg": [
            "confidence_weighted_avg_stats",
            "confidence_weighted_avg",
        ],
    }
    normalized = df.copy()
    for target, candidates in mapping.items():
        for col in candidates:
            if col in normalized.columns:
                normalized[target] = normalized[col]
                break
    return normalized


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = normalize_columns(df)
    required_columns = [
        "avg_rating",
        "rating_variance",
        "num_reviews",
        "confidence_weighted_avg",
        "sentiment_score",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Recompute reliability/calibrated score the same way BayesianAgent does.
    bayes = BayesianAgent()
    calibrations = []
    reliabilities = []
    for _, row in df.iterrows():
        calibrations.append(bayes.calibrate_score(row))
        reliabilities.append(bayes.get_reliability_score(row))

    features = pd.DataFrame(
        {
            "calibrated_score": calibrations,
            "variance": df["rating_variance"],
            "sentiment_score": df["sentiment_score"],
            "reliability": reliabilities,
        }
    ).fillna(0.0)

    decision_column = None
    for candidate in ("decision_stats", "decision_pred", "decision"):
        if candidate in df.columns:
            decision_column = candidate
            break
    if decision_column is None:
        raise ValueError("Could not locate decision labels in merged data.")

    decision_series = df[decision_column]
    if "decision_pred" in df.columns:
        decision_series = decision_series.fillna(df["decision_pred"])

    targets = decision_series.astype(int).to_numpy()
    return features.to_numpy(), targets


def main() -> None:
    args = parse_args()
    merged_df = load_and_merge(args.stats_path, args.predictions_path, args.dataset_path)
    X, y = prepare_features(merged_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("Coefficients:")
    for name, coef in zip(
        ["calibrated_score", "variance", "sentiment_score", "reliability"],
        model.coef_[0],
    ):
        print(f"  {name}: {coef:.4f}")

    joblib.dump(model, args.model_path)
    print(f"Saved logistic regression model to {args.model_path}")


if __name__ == "__main__":
    main()
