#!/usr/bin/env python3
"""
Process OpenReview JSONL data into numerical features and text pairs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_INPUT = "iclr_data.jsonl"
DEFAULT_CSV_OUTPUT = "processed_dataset.csv"
DEFAULT_TEXT_OUTPUT = "text_pairs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ML-ready features from OpenReview dumps.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        type=Path,
        help="Path to the raw OpenReview JSONL file.",
    )
    parser.add_argument(
        "--csv-output",
        default=DEFAULT_CSV_OUTPUT,
        type=Path,
        help="Destination CSV file for numerical features.",
    )
    parser.add_argument(
        "--text-output",
        default=DEFAULT_TEXT_OUTPUT,
        type=Path,
        help="Destination JSON file for review/rebuttal text pairs.",
    )
    return parser.parse_args()


NUMBER_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = NUMBER_PATTERN.match(str(value))
    if match:
        return float(match.group(1))
    return None


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def text_word_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(text.split())


def decision_to_label(decision_text: Optional[str]) -> int:
    if not decision_text:
        return 0
    lowered = decision_text.lower()
    if any(keyword in lowered for keyword in ("oral", "spotlight", "poster", "accept")):
        return 1
    return 0


def compute_features(record: Dict[str, Any]) -> Dict[str, Any]:
    reviews = record.get("reviews") or []
    ratings: List[float] = []
    weighted_entries: List[tuple[float, float]] = []
    rebuttal_lengths: List[int] = []
    has_rebuttal = False

    for review in reviews:
        rating = parse_numeric(review.get("rating"))
        confidence = parse_numeric(review.get("confidence"))
        response = review.get("author_response_text")
        if rating is not None:
            ratings.append(rating)
            if confidence is not None:
                weighted_entries.append((rating, confidence))
        if response:
            has_rebuttal = True
            rebuttal_lengths.append(text_word_count(response))

    avg_rating = mean(ratings) if ratings else None
    if len(ratings) >= 2:
        rating_variance = float(np.var(ratings))
    elif len(ratings) == 1:
        rating_variance = 0.0
    else:
        rating_variance = None
    confidence_score = None
    if weighted_entries:
        total_conf = sum(conf for _, conf in weighted_entries)
        if total_conf > 0:
            confidence_score = sum(r * c for r, c in weighted_entries) / total_conf

    rebuttal_length = float(np.mean(rebuttal_lengths)) if rebuttal_lengths else 0.0

    return {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "avg_rating": avg_rating,
        "rating_variance": rating_variance,
        "confidence_score": confidence_score,
        "has_rebuttal": has_rebuttal,
        "rebuttal_length": rebuttal_length,
        "label": decision_to_label(record.get("decision_text") or record.get("decision")),
    }


def build_text_pairs(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for record in records:
        for review in record.get("reviews") or []:
            pairs.append(
                {
                    "paper_id": record.get("paper_id"),
                    "review": (review.get("review_text") or "").strip(),
                    "rebuttal": (review.get("author_response_text") or "").strip(),
                    "reviewer_score": parse_numeric(review.get("rating")),
                }
            )
    return pairs


def main():
    args = parse_args()
    raw_records = list(iter_jsonl(args.input))
    feature_rows = [compute_features(record) for record in raw_records]
    text_pairs = build_text_pairs(raw_records)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(args.csv_output, index=False)

    with args.text_output.open("w", encoding="utf-8") as outfile:
        json.dump(text_pairs, outfile, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(feature_df)} rows to {args.csv_output} and "
        f"{len(text_pairs)} text pairs to {args.text_output}."
    )


if __name__ == "__main__":
    main()
