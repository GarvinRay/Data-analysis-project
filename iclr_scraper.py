#!/usr/bin/env python3
"""
Scrape ICLR 2025 paper metadata, reviews, and rebuttals with openreview-py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openreview import api
from tqdm import tqdm


DEFAULT_VENUE_ID = "ICLR.cc/2025/Conference"
DEFAULT_BASEURL = "https://api2.openreview.net"
ACCEPT_KEYWORDS = ("oral", "spotlight", "poster")


def safe_get_all_notes(client: api.OpenReviewClient, invitation: str) -> List[Any]:
    try:
        return client.get_all_notes(invitation=invitation)
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ICLR peer review data into JSONL format."
    )
    parser.add_argument(
        "--venue-id",
        default=DEFAULT_VENUE_ID,
        help="OpenReview venue/group id to target (default: %(default)s).",
    )
    parser.add_argument(
        "--baseurl",
        default=DEFAULT_BASEURL,
        help="OpenReview API endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="iclr_data.jsonl",
        type=Path,
        help="Output JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--reject-ratio",
        default=1.0,
        type=float,
        help="How many rejects to sample relative to accepts (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reject sampling (default: %(default)s).",
    )
    parser.add_argument("--username", help="OpenReview username/email.")
    parser.add_argument("--password", help="OpenReview password or token.")
    return parser.parse_args()


def fetch_submissions(client: api.OpenReviewClient, venue_id: str) -> Dict[str, Any]:
    invitations = (
        f"{venue_id}/-/Blind_Submission",
        f"{venue_id}/-/Submission",
    )
    submissions: Dict[str, Any] = {}
    for invitation in invitations:
        notes = safe_get_all_notes(client, invitation)
        if notes:
            submissions = {note.forum: note for note in notes}
            break
    if not submissions:
        raise RuntimeError(f"Could not fetch submissions for {venue_id}.")
    return submissions


def fetch_decisions(client: api.OpenReviewClient, venue_id: str) -> List[Any]:
    invitation = f"{venue_id}/-/-/Decision"
    return client.get_all_notes(invitation=invitation)


def decision_label(decision_text: str) -> str:
    text = (decision_text or "").lower()
    if any(term in text for term in ACCEPT_KEYWORDS):
        return "Accept"
    return "Reject"


def filter_decisions(decisions: Iterable[Any]) -> Dict[str, List[Any]]:
    accepted, rejected = [], []
    for note in decisions:
        text = (note.content or {}).get("decision", "")
        label = decision_label(text)
        if label == "Accept":
            accepted.append(note)
        else:
            rejected.append(note)
    return {"accept": accepted, "reject": rejected}


def sample_rejects(
    rejected: List[Any], accept_count: int, ratio: float, seed: int
) -> List[Any]:
    if not rejected or accept_count <= 0 or ratio <= 0:
        return []
    sample_size = min(len(rejected), int(round(accept_count * ratio)))
    rng = random.Random(seed)
    return rng.sample(rejected, sample_size)


def normalize_keywords(value) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = [kw.strip() for kw in re.split(r"[;,]", value) if kw.strip()]
        return parts
    return []


def extract_text(note: Any) -> str | None:
    content = note.content or {}
    for field in ("review", "comment", "text", "content"):
        if field in content and content[field]:
            return content[field]
    return None


def gather_reviews_and_rebuttals(
    client: api.OpenReviewClient, venue_id: str, submission: Any
) -> List[dict]:
    number = getattr(submission, "number", None)
    if number is None:
        return []
    review_inv = f"{venue_id}/Paper{number}/-/Official_Review"
    comment_inv = f"{venue_id}/Paper{number}/-/Official_Comment"

    reviews = safe_get_all_notes(client, review_inv)
    comments = safe_get_all_notes(client, comment_inv)
    replies = defaultdict(list)
    for comment in comments:
        if getattr(comment, "replyto", None):
            replies[comment.replyto].append(comment)

    pairs: List[dict] = []
    for review in reviews:
        review_text = extract_text(review)
        rating = (review.content or {}).get("rating")
        confidence = (review.content or {}).get("confidence")
        linked_comments = replies.get(review.id)
        if linked_comments:
            for comment in linked_comments:
                pairs.append(
                    {
                        "review_text": review_text,
                        "rating": rating,
                        "confidence": confidence,
                        "author_response_text": extract_text(comment),
                    }
                )
        else:
            pairs.append(
                {
                    "review_text": review_text,
                    "rating": rating,
                    "confidence": confidence,
                    "author_response_text": None,
                }
            )
    return pairs


def build_record(
    client: api.OpenReviewClient,
    venue_id: str,
    decision_note: Any,
    submission_note: Any,
) -> dict:
    submission_content = submission_note.content or {}
    keywords = normalize_keywords(submission_content.get("keywords"))
    return {
        "paper_id": submission_note.forum,
        "title": submission_content.get("title"),
        "abstract": submission_content.get("abstract"),
        "keywords": keywords,
        "decision": decision_label((decision_note.content or {}).get("decision")),
        "decision_text": (decision_note.content or {}).get("decision"),
        "reviews": gather_reviews_and_rebuttals(client, venue_id, submission_note),
    }


def main():
    args = parse_args()
    client = api.OpenReviewClient(
        baseurl=args.baseurl, username=args.username, password=args.password
    )
    submissions = fetch_submissions(client, args.venue_id)
    decisions = fetch_decisions(client, args.venue_id)
    classified = filter_decisions(decisions)
    accept_notes = classified["accept"]
    reject_notes = sample_rejects(
        classified["reject"], len(accept_notes), args.reject_ratio, args.seed
    )
    selected = accept_notes + reject_notes
    tqdm_bar = tqdm(selected, desc="Downloading papers", unit="paper")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as outfile:
        for decision_note in tqdm_bar:
            submission_note = submissions.get(decision_note.forum)
            if submission_note is None:
                try:
                    submission_note = client.get_note(decision_note.forum)
                except Exception as exc:
                    tqdm_bar.write(
                        f"Skipping forum {decision_note.forum}: could not fetch submission ({exc})."
                    )
                    continue
            record = build_record(client, args.venue_id, decision_note, submission_note)
            outfile.write(json.dumps(record) + "\n")

    print(
        f"Wrote {len(selected)} papers "
        f"({len(accept_notes)} accepts, {len(reject_notes)} rejects) to {args.output}."
    )


if __name__ == "__main__":
    main()
