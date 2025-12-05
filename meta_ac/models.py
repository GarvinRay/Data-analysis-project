#!/usr/bin/env python3
"""
Shared data models used across Meta-AC components.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ReviewRebuttalPair:
    review_id: Optional[str]
    reviewer_id: Optional[str]
    review_text: str
    rebuttal_text: Optional[str]
    rating: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRebuttalPair":
        return cls(
            review_id=data.get("review_id"),
            reviewer_id=data.get("reviewer_id"),
            review_text=data.get("review_text", ""),
            rebuttal_text=data.get("rebuttal_text"),
            rating=data.get("rating"),
        )


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
    review_rebuttal_pairs: List[ReviewRebuttalPair]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["review_rebuttal_pairs"] = [
            pair.to_dict() for pair in self.review_rebuttal_pairs
        ]
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperRecord":
        return cls(
            paper_id=data.get("paper_id"),
            decision=int(data.get("decision", 0)),
            title=data.get("title"),
            abstract_raw=data.get("abstract_raw", ""),
            abstract_clean=data.get("abstract_clean", ""),
            keywords=list(data.get("keywords") or []),
            url=data.get("url"),
            ratings=list(data.get("ratings") or []),
            confidences=list(data.get("confidences") or []),
            avg_rating=data.get("avg_rating"),
            rating_variance=data.get("rating_variance"),
            confidence_weighted_avg=data.get("confidence_weighted_avg"),
            num_reviews=int(data.get("num_reviews") or 0),
            num_rebuttals=int(data.get("num_rebuttals") or 0),
            review_rebuttal_pairs=[
                ReviewRebuttalPair.from_dict(item)
                for item in data.get("review_rebuttal_pairs") or []
            ],
        )
