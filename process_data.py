#!/usr/bin/env python3
"""
Process OpenReview notes (Oral + Spotlight) into Meta-AC ready datasets.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from meta_ac.models import PaperRecord, ReviewRebuttalPair


@dataclass
class CategorizedRecord:
    record: PaperRecord
    category: str


DEFAULT_SOURCES: List[Tuple[str, int]] = [
    ("openreview_oral_results.json", 1),
    ("openreview_spotlight_results.json", 1),
    ("openreview_poster_results.json", 1),
    ("openreview_reject_results.json", 0),
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
        default="meta_ac_dataset_sampled.json",
        type=Path,
        help="Hierarchical JSON output path (default: %(default)s).",
    )
    parser.add_argument(
        "--csv-output",
        default="meta_ac_stats_sampled.csv",
        type=Path,
        help="Flat CSV output path (default: %(default)s).",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=300,
        help="Total number of papers to include after stratified sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
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


def parse_paper(entry: Dict[str, Any], decision: int) -> Optional[PaperRecord]:
    if not isinstance(entry, dict):
        return None
    data_payload = entry.get("data") or {}
    notes = data_payload.get("notes", [])
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
    url = entry.get("url") or (
        f"https://openreview.net/forum?id={paper_id}" if paper_id else None
    )

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
        if any(
            "Official_Review" in invitation
            for invitation in note.get("invitations", [])
        )
    ]
    comment_notes = [
        note
        for note in notes
        if any(
            "Official_Comment" in invitation
            for invitation in note.get("invitations", [])
        )
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
    pairs: List[ReviewRebuttalPair] = []
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
                    ReviewRebuttalPair(
                        review_id=review.get("id"),
                        reviewer_id=reviewer_id,
                        review_text=review_text,
                        rebuttal_text=rebuttal_text.strip()
                        if isinstance(rebuttal_text, str)
                        else None,
                        rating=rating,
                    )
                )
        else:
            pairs.append(
                ReviewRebuttalPair(
                    review_id=review.get("id"),
                    reviewer_id=reviewer_id,
                    review_text=review_text,
                    rebuttal_text=None,
                    rating=rating,
                )
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


def build_stats_dataframe(records: Iterable[PaperRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append({
            "paper_id": record.paper_id,
            "decision": record.decision,
            "avg_rating": record.avg_rating,
            "rating_variance": record.rating_variance,
            "confidence_weighted_avg": record.confidence_weighted_avg,
            "num_reviews": record.num_reviews,
            "num_rebuttals": record.num_rebuttals,
        })
    return pd.DataFrame(rows)


def infer_category(path: Path, label: int) -> str:
    if label == 0:
        return "reject"
    name = path.name.lower()
    if "oral" in name:
        return "oral"
    if "spotlight" in name:
        return "spotlight"
    if "poster" in name:
        return "poster"
    return "accept"


def allocate_counts(group_counts: Dict[str, int], target_total: int) -> Dict[str, int]:
    if target_total <= 0 or not group_counts:
        return {key: 0 for key in group_counts}

    total = sum(group_counts.values())
    allocations: Dict[str, int] = {}
    fractions: List[Tuple[float, str]] = []
    assigned = 0
    for category, count in group_counts.items():
        if count == 0:
            allocations[category] = 0
            fractions.append((0.0, category))
            continue
        share = (count / total) * target_total
        base = min(int(share), count)
        allocations[category] = base
        fractions.append((share - base, category))
        assigned += base

    remaining = min(target_total - assigned, target_total)
    fractions.sort(reverse=True)
    while remaining > 0:
        updated = False
        for _, category in fractions:
            available = group_counts[category]
            if allocations[category] < available:
                allocations[category] += 1
                remaining -= 1
                updated = True
                if remaining == 0:
                    break
        if not updated:
            break
    return allocations


def stratified_sample(
    records: List[CategorizedRecord], total_samples: int, seed: int
) -> List[CategorizedRecord]:
    rng = random.Random(seed)
    accepts = [item for item in records if item.record.decision == 1]
    rejects = [item for item in records if item.record.decision == 0]

    target_accepts = min(len(accepts), total_samples // 2)
    target_rejects = min(len(rejects), total_samples - target_accepts)

    accept_groups: Dict[str, List[CategorizedRecord]] = defaultdict(list)
    for item in accepts:
        accept_groups[item.category].append(item)

    group_counts = {k: len(v) for k, v in accept_groups.items()}
    allocations = allocate_counts(group_counts, target_accepts)

    sampled_accepts: List[CategorizedRecord] = []
    for category, count in allocations.items():
        if count > 0 and len(accept_groups[category]) >= count:
            sampled_accepts.extend(accept_groups[category][:count])

    sampled_rejects = rejects[:target_rejects] if target_rejects > 0 else []

    selected = sampled_accepts + sampled_rejects
    return selected


def main() -> None:
    args = parse_args()
    sources = load_sources(args)

    records_with_meta: List[CategorizedRecord] = []
    for path, label in sources:
        if not path.exists():
            raise FileNotFoundError(f"Input file {path} not found.")
        with path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        for entry in payload:
            record = parse_paper(entry, label)
            if record:
                records_with_meta.append(
                    CategorizedRecord(
                        record=record, category=infer_category(path, label)
                    )
                )

    sampled = stratified_sample(records_with_meta, args.total_samples, seed=args.seed)
    sampled_records = [item.record for item in sampled]

    count_summary = Counter(item.category for item in sampled)
    display_names = {
        "oral": "Oral",
        "spotlight": "Spotlight",
        "poster": "Poster",
        "reject": "Reject",
    }
    summary_parts = []
    for key in ("oral", "spotlight", "poster", "reject"):
        summary_parts.append(f"{count_summary.get(key, 0)} {display_names[key]}")
    print("Selected: " + ", ".join(summary_parts))

    json_records = [record.to_dict() for record in sampled_records]
    with args.json_output.open("w", encoding="utf-8") as outfile:
        json.dump(json_records, outfile, ensure_ascii=False, indent=2)

    stats_df = build_stats_dataframe(sampled_records)
    stats_df.to_csv(args.csv_output, index=False)

    print(
        f"Wrote {len(json_records)} papers to {args.json_output} "
        f"and {len(stats_df)} rows to {args.csv_output}."
    )


if __name__ == "__main__":
    main()
