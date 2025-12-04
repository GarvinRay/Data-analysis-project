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

from agents import ArgumentAgent, BayesianAgent, MetaAC


CSV_PATH = Path("meta_ac_stats.csv")
JSON_PATH = Path("meta_ac_dataset.json")


def load_data_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def load_json_records(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def merge_sources(
    df: pd.DataFrame, records: Sequence[Dict]
) -> List[Tuple[pd.Series, Dict]]:
    """
    Align CSV rows with JSON records using paper_id; fall back to index order.
    """
    record_lookup = {
        record.get("paper_id"): record for record in records if record.get("paper_id")
    }
    merged: List[Tuple[pd.Series, Dict]] = []
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
    row: pd.Series, record: Dict, text_result: Dict, meta_result: Dict
) -> None:
    title = record.get("title") or f"Paper {record.get('paper_id')}"
    decision_value = row.get("decision")
    if pd.isna(decision_value):
        decision_value = record.get("decision")
    decision_label = _format_decision(decision_value)
    avg_rating = row.get("avg_rating", "N/A")
    print(f"=== Paper: {title} ===")
    print(f"Decision: {decision_label}")
    print(f"Raw Avg Rating: {avg_rating}")
    print(f"Meta-AC Probability: {meta_result['final_probability']:.2f}")
    print(f"Reasoning: {meta_result['reasoning']}")
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
        print(
            f"Average Meta-AC probability for decision '{decision}': {avg:.2f}"
        )


def main() -> None:
    df = load_data_frame(CSV_PATH)
    records = load_json_records(JSON_PATH)
    merged = merge_sources(df, records)
    if not merged:
        raise RuntimeError("No merged records were available.")

    bayes = BayesianAgent()
    arg = ArgumentAgent()
    meta = MetaAC(bayes, arg)

    use_live_llm = bool(os.environ.get("DEEPSEEK_API_KEY"))
    results_buffer: List[Dict[str, Any]] = []

    for row, record in merged[:5]:
        pairs = record.get("review_rebuttal_pairs") or []
        first_pair = pairs[0] if pairs else {}
        if first_pair:
            if use_live_llm and hasattr(arg, "analyze_pair"):
                text_result = arg.analyze_pair(first_pair)
            else:
                text_result = arg.mock_analysis(first_pair)
        else:
            text_result = {"resolved": False, "sentiment_change": 0.0}

        meta_result = meta.predict(row, text_result)
        display_report(row, record, text_result, meta_result)

        decision_value = row.get("decision")
        if pd.isna(decision_value):
            decision_value = record.get("decision")
        results_buffer.append(
            {
                "title": record.get("title"),
                "raw_avg": row.get("avg_rating"),
                "variance": row.get("rating_variance"),
                "final_prob": meta_result["final_probability"],
                "decision": decision_value,
            }
        )

    summarize_by_decision(results_buffer)

    if results_buffer:
        pd.DataFrame(results_buffer).to_csv("meta_ac_predictions.csv", index=False)
        print("Saved predictions to meta_ac_predictions.csv")


if __name__ == "__main__":
    main()
