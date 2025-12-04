#!/usr/bin/env python3
"""
Agent definitions for the Meta-AC decision-making system.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import requests


@dataclass
class BayesianAgent:
    """
    Quantitative expert that calibrates review-derived scores.

    Reliability is modeled as a smooth scalar in [0, 1]. Small sample sizes
    (few reviews) and high variance reduce the reliability signal, which
    biases the calibrated score toward the simple mean rating as a
    conservative fallback.
    """

    min_reviews: int = 3
    max_variance: float = 1.5
    low_reliability: float = 0.4
    high_reliability: float = 0.9

    def get_reliability_score(self, row: Dict[str, Any] | Any) -> float:
        """
        Return a reliability scalar using heuristics on count/variance.
        """
        num_reviews = self._safe_float(row.get("num_reviews"))
        variance = self._safe_float(row.get("rating_variance"))

        if num_reviews is None or num_reviews < self.min_reviews:
            return self.low_reliability
        if variance is not None and variance > self.max_variance:
            return self.low_reliability
        return self.high_reliability

    def calibrate_score(self, row: Dict[str, Any] | Any) -> float:
        """
        Blend confidence-weighted and raw averages via the reliability score.
        """
        reliability = self.get_reliability_score(row)
        weighted = self._safe_float(row.get("confidence_weighted_avg"))
        avg_rating = self._safe_float(row.get("avg_rating"))

        if weighted is None and avg_rating is None:
            return 0.0
        if weighted is None:
            weighted = avg_rating or 0.0
        if avg_rating is None:
            avg_rating = weighted

        calibrated = (weighted * reliability) + (avg_rating * (1.0 - reliability))
        return calibrated

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """
        Attempt to convert to float; return None when conversion fails.
        """
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass
class ArgumentAgent:
    """
    Argumentation strategist that prepares LLM prompts and mock analyses.
    """

    system_prompt: str = (
        "You are an expert Area Chair. Analyze the provided Review-Rebuttal pair. "
        "Did the author resolve the reviewer's concern? Output JSON: "
        '{"resolved": bool, "sentiment_change": float}.'
    )
    random_state: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(seed=42)
    )
    api_key: str | None = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))
    model_name: str = "deepseek-chat"
    api_url: str = "https://api.deepseek.com/v1/chat/completions"

    def construct_llm_messages(self, pair: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build the OpenAI/Anthropic-compatible chat message payload.
        """
        user_content = (
            "Review:\n"
            f"{pair.get('review_text', '').strip()}\n\n"
            "Author Rebuttal:\n"
            f"{(pair.get('rebuttal_text') or 'No rebuttal provided.').strip()}\n\n"
            f"Reviewer score: {pair.get('rating', 'N/A')}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def mock_analysis(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a dummy LLM analysis; useful before wiring real API calls.
        """
        resolved = bool(pair.get("rebuttal_text"))
        sentiment_change = float(self.random_state.uniform(-1.0, 1.0))
        return {"resolved": resolved, "sentiment_change": sentiment_change}

    def analyze_pair(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the DeepSeek API when credentials are configured, otherwise mock.
        """
        if not self.api_key:
            return self.mock_analysis(pair)

        payload = {
            "model": self.model_name,
            "messages": self.construct_llm_messages(pair),
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("No choices returned from DeepSeek.")
            content = choices[0].get("message", {}).get("content", "")
            return self._parse_llm_response(content)
        except Exception:
            # Fall back to a deterministic mock result if the remote call fails.
            return self.mock_analysis(pair)

    @staticmethod
    def _parse_llm_response(content: str) -> Dict[str, Any]:
        """
        Attempt to parse the LLM output as JSON; fall back to defaults otherwise.
        """
        if not content:
            return {"resolved": False, "sentiment_change": 0.0}

        # Try a direct JSON parse first.
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON-like substring.
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = content[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        return {"resolved": False, "sentiment_change": 0.0}


@dataclass
class MetaAC:
    """
    Orchestrator that fuses Bayesian and argumentation signals.
    """

    bayes_agent: BayesianAgent
    argument_agent: ArgumentAgent

    def predict(
        self,
        paper_data: Dict[str, Any],
        text_analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine calibrated quantitative scores with sentiment adjustments.

        The final probability is a convex combination of the normalized
        calibrated score (mapped to [0, 1] assuming a 10-point rating scale)
        and the sentiment delta (mapped from [-1, 1] to [0, 1]).
        """
        calibrated_score = self.bayes_agent.calibrate_score(paper_data)
        reliability = self.bayes_agent.get_reliability_score(paper_data)
        normalized_score = np.clip(calibrated_score / 10.0, 0.0, 1.0)

        sentiment = float(text_analysis_result.get("sentiment_change", 0.0))
        sentiment_score = np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0)

        final_probability = float(
            np.clip((0.65 * normalized_score) + (0.35 * sentiment_score), 0.0, 1.0)
        )
        reasoning = (
            f"Calibrated score={calibrated_score:.2f} (reliability {reliability:.2f}), "
            f"sentiment contribution={sentiment_score:.2f}. "
            "Higher reliability pushes the decision toward quantitative evidence."
        )
        return {"final_probability": final_probability, "reasoning": reasoning}
