#!/usr/bin/env python3
"""
Train and evaluate an MLP classifier for Meta-AC acceptance prediction.
Features are extracted via agents.BayesianAgent and agents.ArgumentAgent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from agents import ArgumentAgent, BayesianAgent, DomainAgent
from meta_ac.config import JSON_PATH, MODEL_PATH, OUTPUT_DIR, PREDICTIONS_PATH, STATS_PATH
from meta_ac.models import PaperRecord

# prevent HF tokenizers parallel warnings when forking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
        if "rebuttal_score" in result and "sentiment_change" not in result:
            try:
                sc = float(result["rebuttal_score"])
                result["sentiment_change"] = float(np.clip((sc * 2) - 1, -1.0, 1.0))
            except Exception:
                result["sentiment_change"] = 0.0
        sentiment = float(result.get("sentiment_change", 0.0))
        # include debate adjustment if present
        sentiment += float(result.get("debate_adjustment", 0.0))
        score = float(np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0))
        return score
    except Exception:
        return 0.5


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    ids: List[str],
    titles: List[str],
    raw_avgs: List[float],
    variances: List[float],
    sentiments: List[float],
    densities: List[float],
    novelties: List[float],
    model_type: str = "MLP",
    grid_search: bool = False,
) -> Tuple[Any, StandardScaler, np.ndarray, List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf: Any
    best_params: Dict[str, Any] | None = None
    if model_type.upper() == "TABNET":
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            print("TabNet not installed; falling back to MLP.")
            clf = MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="adam",
                alpha=0.01,
                random_state=42,
                max_iter=500,
            )
        else:
            clf = TabNetClassifier(seed=42, verbose=0)
    else:
        if grid_search:
            param_grid = {
                "hidden_layer_sizes": [(16, 8), (32, 16), (64, 32)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01],
            }
            base = MLPClassifier(
                activation="relu",
                solver="adam",
                random_state=42,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=15,
            )
            search = GridSearchCV(
                base,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train_scaled, y_train)
            clf = search.best_estimator_
            best_params = search.best_params_
            print(f"Grid search best params: {best_params}")
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="adam",
                alpha=0.01,
                random_state=42,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=15,
            )

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_prob_test = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    print(f"Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(report)

    mis_details: List[Dict[str, Any]] = []
    for i, y_p, y_t, prob in zip(idx_test, y_pred, y_test, y_prob_test):
        if y_p != y_t:
            mis_details.append({
                "paper_id": ids[i],
                "title": titles[i],
                "true_decision": int(y_t),
                "pred_prob": float(prob),
            })
    if mis_details:
        print("Misclassified samples (test set):")
        for item in mis_details:
            print(
                f" - {item['paper_id']} | {item['title']} | true={item['true_decision']} prob={item['pred_prob']:.3f}"
            )

    boundary_mask = (y_prob_test >= 0.4) & (y_prob_test <= 0.6)
    if boundary_mask.any():
        boundary_acc = accuracy_score(y_test[boundary_mask], y_pred[boundary_mask])
        print(f"Boundary Accuracy (0.4-0.6 prob band): {boundary_acc:.3f}")
    else:
        print("Boundary Accuracy: no samples in [0.4, 0.6] band.")
        boundary_acc = None

    probs_all = clf.predict_proba(scaler.transform(X))[:, 1]

    test_info = []
    for local_idx, (i, prob) in enumerate(zip(idx_test, y_prob_test)):
        test_info.append({
            "paper_id": ids[i],
            "title": titles[i],
            "raw_avg": raw_avgs[i],
            "variance": variances[i],
            "sentiment_score": sentiments[i],
            "density_score": densities[i] if i < len(densities) else None,
            "novelty_score": novelties[i] if i < len(novelties) else None,
            "prob_accept": prob,
            "true_label": int(y_test[local_idx]),
        })

    metrics = {
        "accuracy": float(acc),
        "classification_report": report,
        "boundary_accuracy": None if boundary_acc is None else float(boundary_acc),
        "best_params": best_params,
        "model_type": model_type,
        "grid_search": grid_search,
    }

    return clf, scaler, probs_all, test_info, metrics, mis_details


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-AC training pipeline")
    parser.add_argument(
        "--model-type",
        type=str,
        default="MLP",
        choices=["MLP", "TABNET"],
        help="Model architecture to use for classification.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable grid search over MLP hyperparameters.",
    )
    args = parser.parse_args()
    df_path = STATS_PATH if STATS_PATH.exists() else STATS_PATH.name
    json_path = JSON_PATH if JSON_PATH.exists() else JSON_PATH.name

    df = load_data_frame(Path(df_path))
    records = load_json_records(Path(json_path))
    merged = merge_sources(df, records)
    if not merged:
        raise RuntimeError("No merged records were available.")

    bayes = BayesianAgent()
    arg = ArgumentAgent()
    abstract_corpus = [
        (rec.abstract_clean or rec.abstract_raw or "") for rec in records
    ]
    domain = DomainAgent(abstract_corpus)

    cache_path = Path("llm_scores_cache.csv")
    cache_df = (
        pd.read_csv(cache_path)
        if cache_path.exists()
        else pd.DataFrame(columns=["paper_id", "llm_score"])
    )
    cache = dict(zip(cache_df.get("paper_id", []), cache_df.get("llm_score", [])))
    new_scores: List[Tuple[str, float]] = []

    X_rows: List[List[float]] = []
    y: List[int] = []
    ids: List[str] = []
    titles: List[str] = []
    raw_avgs: List[float] = []
    variances: List[float] = []
    sentiments: List[float] = []
    densities: List[float] = []
    novelties: List[float] = []

    for row, record in tqdm(merged, desc="Extracting features", unit="paper"):
        try:
            paper_id = record.paper_id or f"idx_{len(ids)}"
            llm_score = cache.get(paper_id)
            if llm_score is None:
                llm_score = extract_rebuttal_score(arg, record)
                new_scores.append((paper_id, llm_score))
            sentiments.append(llm_score)

            avg_rating = _safe_float(row.get("avg_rating")) or 0.0
            variance = _safe_float(row.get("rating_variance")) or 0.0
            calibrated = bayes.calibrate_score(row)
            try:
                domain_scores = domain.analyze_novelty(
                    record.abstract_clean or record.abstract_raw or ""
                )
                density_score = float(domain_scores.get("density_score", 0.5))
                novelty_score = float(domain_scores.get("novelty_score", 0.5))
            except Exception:
                density_score = 0.5
                novelty_score = 0.5
            densities.append(density_score)
            novelties.append(novelty_score)

            decision_value = row.get("decision")
            if pd.isna(decision_value):
                decision_value = record.decision

            X_rows.append(
                [avg_rating, variance, calibrated, llm_score, density_score, novelty_score]
            )
            y.append(int(decision_value))
            ids.append(paper_id)
            titles.append(record.title or paper_id)
            raw_avgs.append(avg_rating)
            variances.append(variance)
        except Exception:
            continue

    if new_scores:
        cache_updates = pd.DataFrame(new_scores, columns=["paper_id", "llm_score"])
        updated_cache = pd.concat(
            [cache_df, cache_updates], ignore_index=True
        ).drop_duplicates(subset=["paper_id"], keep="last")
        updated_cache.to_csv(cache_path, index=False)

    X = np.array(X_rows, dtype=float)
    y_arr = np.array(y, dtype=int)
    if X.size == 0:
        raise RuntimeError("No features extracted; aborting.")

    model, scaler, probs, test_info, metrics, mis_details = train_and_evaluate(
        X,
        y_arr,
        ids,
        titles,
        raw_avgs,
        variances,
        sentiments,
        densities,
        novelties,
        model_type=args.model_type,
        grid_search=args.grid_search,
    )

    # Attach probabilities to paper ids for inspection (test set only)
    results = pd.DataFrame(test_info)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved predictions to {PREDICTIONS_PATH}")

    # Persist metrics and misclassified samples for reproducibility
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "run_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")

    if mis_details:
        mis_path = OUTPUT_DIR / "misclassified.csv"
        pd.DataFrame(mis_details).to_csv(mis_path, index=False)
        print(f"Saved misclassified samples to {mis_path}")

    # Persist model + scaler and feature names
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [
        "avg_rating",
        "variance",
        "calibrated_score",
        "llm_score",
        "density_score",
        "novelty_score",
    ]
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model and scaler to {MODEL_PATH}")

    # Save readable weights if available
    weights_path = OUTPUT_DIR / "feature_weights.csv"
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        pd.DataFrame(
            {"feature": feature_names, "weight": coef}
        ).to_csv(weights_path, index=False)
        print(f"Saved linear weights to {weights_path}")
    elif hasattr(model, "coefs_") and len(model.coefs_) > 0:
        first_layer = model.coefs_[0]
        hidden_dim = first_layer.shape[1]
        hidden_cols = [f"h_{j}" for j in range(hidden_dim)]
        pd.DataFrame(first_layer, index=feature_names, columns=hidden_cols).to_csv(
            weights_path
        )
        print(f"Saved first-layer weights to {weights_path}")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
