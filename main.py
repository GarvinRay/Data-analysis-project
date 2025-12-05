#!/usr/bin/env python3
"""
Entry point for running the Meta-AC pipeline on real data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import requests
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from meta_ac.models import PaperRecord

CSV_PATH = Path("meta_ac_stats_sampled.csv")
JSON_PATH = Path("meta_ac_dataset_sampled.json")
DEEPSEEK_BASEURL = "https://api.deepseek.com/v1/chat/completions"
LLM_PROMPT = """
You are a strict and experienced Area Chair (AC) for ICLR. 
Your task is to evaluate the "effectiveness" of an Author's Rebuttal to a Reviewer's criticism.

Input data includes:
1. Reviewer's specific concern.
2. Author's rebuttal.

Please analyze:
1. **Responsiveness**: Did the author directly answer the question? (Avoid dodging)
2. **Evidence**: Did the author provide new experiments, citations, or theoretical proofs?
3. **Attitude**: Is the tone professional and constructive?

Output JSON format ONLY:
{
    "reasoning": "A short summary of why the rebuttal is strong or weak...",
    "rebuttal_score": <float between 0.0 and 1.0, where 0.0 is completely ignored/failed, 0.5 is partial, 1.0 is perfectly resolved>
}
"""


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
    """
    Align CSV rows with JSON records using paper_id; fall back to index order.
    """
    record_lookup = {record.paper_id: record for record in records if record.paper_id}
    merged: List[Tuple[pd.Series, PaperRecord]] = []
    for idx, row in df.iterrows():
        paper_id = row.get("paper_id")
        record = record_lookup.get(paper_id)
        if record is None and idx < len(records):
            record = records[idx]
        if record is None:
            continue
        merged.append((row, record))
    return merged


def display_report(
    row: pd.Series, record: PaperRecord, llm_score: float, prob: float
) -> None:
    title = record.title or f"Paper {record.paper_id}"
    decision_value = row.get("decision")
    if pd.isna(decision_value):
        decision_value = record.decision
    decision_label = _format_decision(decision_value)
    avg_rating = row.get("avg_rating", "N/A")
    print(f"=== Paper: {title} ===")
    print(f"Decision: {decision_label}")
    print(f"Raw Avg Rating: {avg_rating}")
    print(f"LLM Rebuttal Score: {llm_score:.2f}")
    print(f"MLP Predicted Probability: {prob:.2f}")
    print("-" * 50)


def _format_decision(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    try:
        num = float(value)
        if num >= 0.5:
            return "Accept"
        return "Reject"
    except (TypeError, ValueError):
        lowered = str(value).lower()
        if "oral" in lowered or "spotlight" in lowered or "accept" in lowered:
            return "Accept"
        if "reject" in lowered:
            return "Reject"
        return str(value)


def summarize_by_decision(results: Sequence[Dict[str, Any]]) -> None:
    if not results:
        print("No results available for summary.")
        return
    df = pd.DataFrame(results)
    if "decision" not in df.columns:
        print("Decision column missing; skipping summary.")
        return
    summary = (
        df.dropna(subset=["decision"])
        .groupby("decision")["final_prob"]
        .mean()
        .to_dict()
    )
    if not summary:
        print("No decision labels available for summary.")
        return
    for decision, avg in summary.items():
        print(f"Average Meta-AC probability for decision '{decision}': {avg:.2f}")


def main() -> None:
    df = load_data_frame(CSV_PATH)
    records = load_json_records(JSON_PATH)
    merged = merge_sources(df, records)
    if not merged:
        raise RuntimeError("No merged records were available.")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    results_buffer: List[Dict[str, Any]] = []
    feature_rows: List[List[float]] = []
    labels: List[int] = []
    llm_scores: List[float] = []

    for row, record in tqdm(merged, desc="Processing papers", unit="paper"):
        try:
            pairs = [pair.to_dict() for pair in record.review_rebuttal_pairs]
            first_pair = pairs[0] if pairs else {}
            llm_score = 0.0
            if first_pair:
                review_text = first_pair.get("review_text") or ""
                rebuttal_text = first_pair.get("rebuttal_text") or ""
                llm_score = call_deepseek(review_text, rebuttal_text, api_key)
            llm_scores.append(llm_score)

            decision_value = row.get("decision")
            if pd.isna(decision_value):
                decision_value = record.decision
            labels.append(int(decision_value))

            avg_rating = _safe_float(row.get("avg_rating")) or 0.0
            variance = _safe_float(row.get("rating_variance")) or 0.0
            num_reviews = _safe_float(row.get("num_reviews")) or 0.0
            conf_avg = _safe_float(row.get("confidence_weighted_avg")) or 0.0
            feature_rows.append([
                avg_rating,
                variance,
                num_reviews,
                conf_avg,
                llm_score,
            ])

            results_buffer.append({
                "title": record.title,
                "raw_avg": row.get("avg_rating"),
                "variance": row.get("rating_variance"),
                "llm_score": llm_score,
                "decision": decision_value,
            })
        except Exception as exc:
            title = record.title or record.paper_id
            print(f"Error processing '{title}': {exc}")
            continue

    if not feature_rows:
        raise RuntimeError("No features collected; aborting.")

    clf = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        random_state=42,
        max_iter=500,
    )
    clf.fit(feature_rows, labels)
    probs = clf.predict_proba(feature_rows)[:, 1]

    # Display reports with probabilities
    for (row, record), prob, llm_score in zip(merged, probs, llm_scores):
        display_report(row, record, llm_score, prob)

    # Attach probabilities and save
    for entry, prob in zip(results_buffer, probs):
        entry["final_prob"] = prob

    summarize_by_decision(results_buffer)

    if results_buffer:
        pd.DataFrame(results_buffer).to_csv("final_predictions.csv", index=False)
        print("Saved predictions to final_predictions.csv")


def call_deepseek(review_text: str, rebuttal_text: str, api_key: str | None) -> float:
    """
    Call DeepSeek API to score rebuttal quality (0-1). Returns 0.0 on failure.
    """
    if not api_key:
        return 0.0
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": LLM_PROMPT},
            {
                "role": "user",
                "content": f"评审意见:\n{review_text}\n\n作者回复:\n{rebuttal_text}",
            },
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            DEEPSEEK_BASEURL, headers=headers, json=payload, timeout=60
        )
        resp.raise_for_status()
        content = (
            resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        score = float(content.strip())
        if 0.0 <= score <= 1.0:
            return score
    except Exception:
        return 0.0
    return 0.0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
