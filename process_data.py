#!/usr/bin/env python3
"""
Process OpenReview notes (Oral + Spotlight) into Meta-AC ready datasets.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_SOURCES: List[Tuple[str, int]] = [
    ("openreview_oral_results_50.json", 1),
    ("openreview_spotlight_results.json", 1),
]

CLEAN_RE = re.compile(r"[^A-Za-z0-9 .,;:?!'\"()-]+")
NUMBER_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform scraped OpenReview JSON into Meta-AC datasets."
    )
    parser.add_argument(
        "--input",
        action="append",
        metavar="PATH:LABEL",
        help=(
            "Input JSON and decision label (1=accept,0=reject). "
            "Provide multiple, e.g. rejects.json:0"
        ),
    )
    parser.add_argument(
        "--json-output",
        default="meta_ac_dataset.json",
        type=Path,
        help="Hierarchical JSON output path (default: %(default)s).",
    )
    parser.add_argument(
        "--csv-output",
        default="meta_ac_stats.csv",
        type=Path,
        help="Flat CSV output path (default: %(default)s).",
    )
    return parser.parse_args()


def parse_source(value: str) -> Tuple[Path, int]:
    try:
        path_str, label_str = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --input '{value}'. Expected format path:label"
        ) from exc
    label = int(label_str)
    if label not in (0, 1):
        raise argparse.ArgumentTypeError("Decision label must be 0 or 1.")
    return Path(path_str), label


def load_sources(args: argparse.Namespace) -> List[Tuple[Path, int]]:
    if args.input:
        return [parse_source(item) for item in args.input]
    return [(Path(path), label) for path, label in DEFAULT_SOURCES]


def extract_value(payload: Any) -> Any:
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def parse_numeric(value: Any) -> Optional[float]:
    raw = extract_value(value)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = NUMBER_RE.match(str(raw))
    if match:
        return float(match.group(1))
    return None


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = CLEAN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_keywords(value: Any) -> List[str]:
    normalized = extract_value(value)
    if isinstance(normalized, list):
        return [str(item).strip() for item in normalized if str(item).strip()]
    if isinstance(normalized, str):
        parts = re.split(r"[;,]", normalized)
        return [part.strip() for part in parts if part.strip()]
    return []


def build_review_text(content: Dict[str, Any]) -> str:
    ordered_fields = [
        "summary",
        "review",
        "strengths",
        "weaknesses",
        "questions",
        "comment",
        "main_review",
    ]
    pieces: List[str] = []
    for field in ordered_fields:
        value = extract_value(content.get(field))
        if value:
            label = field.replace("_", " ").title()
            pieces.append(f"{label}: {value}")
    # include any other textual fields that were not covered above
    for key, value in content.items():
        if key in ordered_fields or key in {"rating", "confidence"}:
            continue
        plain = extract_value(value)
        if isinstance(plain, str) and plain.strip():
            pieces.append(f"{key.replace('_', ' ').title()}: {plain}")
    return "\n\n".join(pieces).strip()


@dataclass
class PaperRecord:
    paper_id: Optional[str]
    decision: int
    title: Optional[str]
    abstract_raw: str
    abstract_clean: str
    keywords: List[str]
    url: Optional[str]
    ratings: List[float]
    confidences: List[float]
    avg_rating: Optional[float]
    rating_variance: Optional[float]
    confidence_weighted_avg: Optional[float]
    num_reviews: int
    num_rebuttals: int
    review_rebuttal_pairs: List[Dict[str, Any]]


def parse_paper(entry: Dict[str, Any], decision: int) -> Optional[PaperRecord]:
    notes = entry.get("data", {}).get("notes", [])
    if not notes:
        return None
    metadata_note = next(
        (note for note in notes if "abstract" in note.get("content", {})), None
    )
    if metadata_note is None:
        metadata_note = notes[0]
    meta_content = metadata_note.get("content", {})
    paper_id = entry.get("paper_id") or metadata_note.get("forum")
    title = extract_value(meta_content.get("title"))
    abstract_raw = extract_value(meta_content.get("abstract")) or ""
    abstract_clean = clean_text(abstract_raw)
    keywords = normalize_keywords(meta_content.get("keywords"))
    url = entry.get("url") or (f"https://openreview.net/forum?id={paper_id}" if paper_id else None)

    review_data = parse_reviews(notes)

    return PaperRecord(
        paper_id=paper_id,
        decision=decision,
        title=title,
        abstract_raw=abstract_raw,
        abstract_clean=abstract_clean,
        keywords=keywords,
        url=url,
        ratings=review_data["ratings"],
        confidences=review_data["confidences"],
        avg_rating=review_data["avg_rating"],
        rating_variance=review_data["rating_variance"],
        confidence_weighted_avg=review_data["confidence_weighted_avg"],
        num_reviews=review_data["num_reviews"],
        num_rebuttals=review_data["num_rebuttals"],
        review_rebuttal_pairs=review_data["pairs"],
    )


def parse_reviews(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    review_notes = [
        note
        for note in notes
        if any("Official_Review" in invitation for invitation in note.get("invitations", []))
    ]
    comment_notes = [
        note
        for note in notes
        if any("Official_Comment" in invitation for invitation in note.get("invitations", []))
    ]

    # Group official comments by the review (replyto) they answer so we can pair them.
    comment_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for comment in comment_notes:
        parent_id = comment.get("replyto")
        if not parent_id:
            continue
        comment_lookup.setdefault(parent_id, []).append(comment)

    ratings: List[float] = []
    confidences: List[float] = []
    weighted_entries: List[Tuple[float, float]] = []
    pairs: List[Dict[str, Any]] = []
    rebuttal_count = 0

    for review in review_notes:
        content = review.get("content", {})
        rating = parse_numeric(content.get("rating"))
        confidence = parse_numeric(content.get("confidence"))
        if rating is not None:
            ratings.append(rating)
        if confidence is not None:
            confidences.append(confidence)
        if rating is not None and confidence is not None:
            weighted_entries.append((rating, confidence))

        reviewer_id = None
        signatures = review.get("signatures") or []
        if signatures:
            reviewer_id = signatures[0]

        review_text = build_review_text(content)
        replies = comment_lookup.get(review.get("id"), [])
        if replies:
            for reply in replies:
                rebuttal_text = extract_value(reply.get("content", {}).get("comment"))
                rebuttal_count += 1
                pairs.append(
                    {
                        "review_id": review.get("id"),
                        "reviewer_id": reviewer_id,
                        "review_text": review_text,
                        "rebuttal_text": rebuttal_text.strip() if isinstance(rebuttal_text, str) else None,
                        "rating": rating,
                    }
                )
        else:
            pairs.append(
                {
                    "review_id": review.get("id"),
                    "reviewer_id": reviewer_id,
                    "review_text": review_text,
                    "rebuttal_text": None,
                    "rating": rating,
                }
            )

    avg_rating = float(mean(ratings)) if ratings else None
    if len(ratings) >= 2:
        rating_variance = float(pvariance(ratings))
    elif len(ratings) == 1:
        rating_variance = 0.0
    else:
        rating_variance = None

    confidence_weighted_avg = None
    if weighted_entries:
        total_conf = sum(conf for _, conf in weighted_entries)
        if total_conf > 0:
            confidence_weighted_avg = (
                sum(r * c for r, c in weighted_entries) / total_conf
            )

    return {
        "ratings": ratings,
        "confidences": confidences,
        "avg_rating": avg_rating,
        "rating_variance": rating_variance,
        "confidence_weighted_avg": confidence_weighted_avg,
        "num_reviews": len(review_notes),
        "num_rebuttals": rebuttal_count,
        "pairs": pairs,
    }


def record_to_dict(record: PaperRecord) -> Dict[str, Any]:
    return {
        "paper_id": record.paper_id,
        "decision": record.decision,
        "title": record.title,
        "abstract_raw": record.abstract_raw,
        "abstract_clean": record.abstract_clean,
        "keywords": record.keywords,
        "url": record.url,
        "ratings": record.ratings,
        "confidences": record.confidences,
        "avg_rating": record.avg_rating,
        "rating_variance": record.rating_variance,
        "confidence_weighted_avg": record.confidence_weighted_avg,
        "num_reviews": record.num_reviews,
        "num_rebuttals": record.num_rebuttals,
        "review_rebuttal_pairs": record.review_rebuttal_pairs,
    }


def build_stats_dataframe(records: Iterable[PaperRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "paper_id": record.paper_id,
                "decision": record.decision,
                "avg_rating": record.avg_rating,
                "rating_variance": record.rating_variance,
                "confidence_weighted_avg": record.confidence_weighted_avg,
                "num_reviews": record.num_reviews,
                "num_rebuttals": record.num_rebuttals,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    sources = load_sources(args)

    processed_records: List[PaperRecord] = []
    for path, label in sources:
        if not path.exists():
            raise FileNotFoundError(f"Input file {path} not found.")
        with path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        for entry in payload:
            record = parse_paper(entry, label)
            if record:
                processed_records.append(record)

    json_records = [record_to_dict(record) for record in processed_records]
    with args.json_output.open("w", encoding="utf-8") as outfile:
        json.dump(json_records, outfile, ensure_ascii=False, indent=2)

    stats_df = build_stats_dataframe(processed_records)
    stats_df.to_csv(args.csv_output, index=False)

    print(
        f"Wrote {len(json_records)} papers to {args.json_output} "
        f"and {len(stats_df)} rows to {args.csv_output}."
    )


if __name__ == "__main__":
    main()
