#!/usr/bin/env python3
"""
Train and evaluate an MLP classifier for Meta-AC acceptance prediction.
Features are extracted via agents.BayesianAgent and agents.ArgumentAgent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from agents import ArgumentAgent, BayesianAgent
from meta_ac.models import PaperRecord

CSV_PATH = Path("meta_ac_stats_sampled.csv")
JSON_PATH = Path("meta_ac_dataset_sampled.json")


def load_data_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def load_json_records(path: Path) -> List[PaperRecord]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    return [PaperRecord.from_dict(item) for item in payload]


def merge_sources(
    df: pd.DataFrame, records: Sequence[PaperRecord]
) -> List[Tuple[pd.Series, PaperRecord]]:
    lookup = {rec.paper_id: rec for rec in records if rec.paper_id}
    merged: List[Tuple[pd.Series, PaperRecord]] = []
    for idx, row in df.iterrows():
        paper_id = row.get("paper_id")
        record = lookup.get(paper_id)
        if record is None and idx < len(records):
            record = records[idx]
        if record is None:
            continue
        merged.append((row, record))
    return merged


def extract_rebuttal_score(arg_agent: ArgumentAgent, record: PaperRecord) -> float:
    """
    Use ArgumentAgent to score rebuttal quality; return a score in [0, 1].
    """
    pairs = record.review_rebuttal_pairs
    if not pairs:
        return 0.5  # neutral default
    first_pair = pairs[0].to_dict()
    try:
        result = arg_agent.analyze_pair(first_pair)
        sentiment = float(result.get("sentiment_change", 0.0))
        # include debate adjustment if present
        sentiment += float(result.get("debate_adjustment", 0.0))
        score = float(np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0))
        return score
    except Exception:
        return 0.5


def extract_features_and_labels(
    merged: Sequence[Tuple[pd.Series, PaperRecord]],
    bayes: BayesianAgent,
    arg: ArgumentAgent,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X: List[List[float]] = []
    y: List[int] = []
    ids: List[str] = []

    for row, record in tqdm(merged, desc="Extracting features", unit="paper"):
        try:
            avg_rating = _safe_float(row.get("avg_rating")) or 0.0
            variance = _safe_float(row.get("rating_variance")) or 0.0
            num_reviews = _safe_float(row.get("num_reviews")) or 0.0
            conf_avg = _safe_float(row.get("confidence_weighted_avg")) or 0.0
            calibrated = bayes.calibrate_score(row)
            rebuttal_score = extract_rebuttal_score(arg, record)

            decision_value = row.get("decision")
            if pd.isna(decision_value):
                decision_value = record.decision

            X.append([avg_rating, variance, calibrated, rebuttal_score])
            y.append(int(decision_value))
            ids.append(record.paper_id or f"idx_{len(ids)}")
        except Exception:
            # Skip problematic records but continue overall pipeline
            continue

    return np.array(X, dtype=float), np.array(y, dtype=int), ids


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, ids: List[str]
) -> Tuple[MLPClassifier, StandardScaler, np.ndarray]:
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        random_state=42,
        max_iter=500,
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    mis_idx = [ids[i] for i, y_p, y_t in zip(idx_test, y_pred, y_test) if y_p != y_t]
    if mis_idx:
        print("Misclassified paper_ids (test set):", mis_idx)

    probs_all = clf.predict_proba(scaler.transform(X))[:, 1]
    return clf, scaler, probs_all


def main() -> None:
    df = load_data_frame(CSV_PATH)
    records = load_json_records(JSON_PATH)
    merged = merge_sources(df, records)
    if not merged:
        raise RuntimeError("No merged records were available.")

    bayes = BayesianAgent()
    arg = ArgumentAgent()

    X, y, ids = extract_features_and_labels(merged, bayes, arg)
    if X.size == 0:
        raise RuntimeError("No features extracted; aborting.")

    model, scaler, probs = train_and_evaluate(X, y, ids)

    # Attach probabilities to paper ids for inspection
    results = pd.DataFrame({
        "paper_id": ids,
        "probability": probs,
        "label": y,
    })
    results.to_csv("final_predictions.csv", index=False)
    print("Saved predictions to final_predictions.csv")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
