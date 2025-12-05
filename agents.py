#!/usr/bin/env python3
"""
Agent definitions for the Meta-AC decision-making system.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


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

    def get_reliability_score(
        self, row: Dict[str, Any] | Any, return_details: bool = False
    ) -> float | Tuple[float, str]:
        """
        Return a reliability scalar using heuristics on count/variance.
        """
        num_reviews = self._safe_float(row.get("num_reviews"))
        variance = self._safe_float(row.get("rating_variance"))

        if num_reviews is None or num_reviews < self.min_reviews:
            base = self.low_reliability
        elif variance is not None and variance > self.max_variance:
            base = self.low_reliability
        else:
            base = self.high_reliability

        review_text = self._extract_review_text(row)
        profile_weight, profile_reason = self.profile_reviewer(review_text)
        reliability = base * profile_weight

        if return_details:
            return reliability, profile_reason
        return reliability

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

    def profile_reviewer(self, review_text: str | None) -> Tuple[float, str]:
        """
        Infer reviewer persona based on textual clues.
        """
        if not review_text:
            return 1.0, "standard reviewer"

        words = review_text.strip().split()
        lower_text = review_text.lower()
        if len(words) < 50:
            return 0.5, "lazy reviewer penalty"

        nitpick_terms = ("typo", "formatting", "font", "missing citation")
        core_terms = ("method", "experiment", "novelty", "result", "theory")
        if any(term in lower_text for term in nitpick_terms) and not any(
            term in lower_text for term in core_terms
        ):
            return 0.8, "nitpicker focus penalty"
        return 1.0, "balanced reviewer"

    @staticmethod
    def _extract_review_text(row: Dict[str, Any] | Any) -> str | None:
        if hasattr(row, "get"):
            text = row.get("review_text")
            if text:
                return text
            pairs = row.get("review_rebuttal_pairs")
            if pairs:
                sample = pairs[0]
                if isinstance(sample, dict):
                    return sample.get("review_text")
        if isinstance(row, dict):
            pairs = row.get("review_rebuttal_pairs")
            if pairs:
                sample = pairs[0]
                if isinstance(sample, dict):
                    return sample.get("review_text")
        return None


@dataclass
class ArgumentAgent:
    """
    Argumentation strategist that prepares LLM prompts and mock analyses.
    """

    system_prompt: str = """
You are a seasoned and critical Area Chair (AC) for ICLR. 
You are reviewing a specific interaction between a Reviewer and an Author during the Rebuttal phase.

Your Goal: Quantify how effectively the author negated the reviewer's negative points.

Analyze the input based on:
1. **Responsiveness**: Did the author address the core issue, or did they pivot to a strawman argument?
2. **Substance**: Did they provide new experiments, baselines, or math derivations? (Solid Evidence >> Promises to fix)
3. **Conversion Potential**: Is this rebuttal strong enough to potentially change a reviewer's score?

Output JSON format ONLY:
{
    "analysis": "Brief step-by-step reasoning (max 50 words).",
    "rebuttal_score": <float between 0.0 and 1.0>
}

Scoring Guide:
- 0.0-0.2: Non-responsive, defensive, or completely failed to address the flaw.
- 0.3-0.5: Partial acknowledgement, promised to fix in camera-ready but showed no proof.
- 0.6-0.8: Good response with logical arguments or preliminary evidence.
- 0.9-1.0: Perfect rebuttal. New experiments/proofs provided, completely invalidating the reviewer's concern.
"""
    random_state: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(seed=42)
    )
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY")
    )
    model_name: str = "deepseek-chat"
    api_url: str = "https://api.deepseek.com/v3.2_speciale_expires_on_20251215"

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
            base_result = self.mock_analysis(pair)
        else:
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
            except Exception:
                base_result = self.mock_analysis(pair)
            else:
                base_result = self._parse_llm_response(content)

        debate = self.simulate_debate(pair)
        total_sentiment = float(base_result.get("sentiment_change", 0.0)) + float(
            debate.get("adjustment", 0.0)
        )
        base_result.update({
            "sentiment_change": total_sentiment,
            "debate_adjustment": debate.get("adjustment", 0.0),
            "debate_verdict": debate.get("verdict"),
            "debate_comment": debate.get("comment"),
        })
        return base_result

    @staticmethod
    def _parse_llm_response(content: str) -> Dict[str, Any]:
        """
        Attempt to parse the LLM output as JSON; fall back to defaults otherwise.
        """
        if not content:
            return {"resolved": False, "sentiment_change": 0.0}

        # Try a direct JSON parse first.
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass
        else:
            # Support new schema: {"analysis": "...", "rebuttal_score": 0-1}
            if "rebuttal_score" in data and "sentiment_change" not in data:
                try:
                    score = float(data["rebuttal_score"])
                    data["sentiment_change"] = float(
                        np.clip((score * 2) - 1, -1.0, 1.0)
                    )
                except Exception:
                    data["sentiment_change"] = 0.0
            return data

        # Fallback: extract JSON-like substring.
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = content[start : end + 1]
            try:
                data = json.loads(snippet)
                if "rebuttal_score" in data and "sentiment_change" not in data:
                    try:
                        score = float(data["rebuttal_score"])
                        data["sentiment_change"] = float(
                            np.clip((score * 2) - 1, -1.0, 1.0)
                        )
                    except Exception:
                        data["sentiment_change"] = 0.0
                return data
            except json.JSONDecodeError:
                pass
        return {"resolved": False, "sentiment_change": 0.0}

    def simulate_debate(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a counterfactual 'devil's advocate' pass on the rebuttal.
        """
        if not pair:
            return {"verdict": "UNKNOWN", "adjustment": 0.0, "comment": None}

        review = pair.get("review_text", "")
        rebuttal = pair.get("rebuttal_text", "")
        system_prompt = (
            "You are a seasoned Area Chair acting as a devil's advocate. "
            "Read the reviewer concern and the author's rebuttal. "
            "If you can find a logical flaw, missing evidence, or unresolved issue, "
            "state it in one sentence. If the rebuttal is solid, respond with 'SOLID'."
        )
        user_content = (
            "Reviewer Concern:\n"
            f"{review}\n\n"
            "Author Rebuttal:\n"
            f"{rebuttal or 'No rebuttal provided.'}"
        )

        if not self.api_key:
            return self._mock_debate(review, rebuttal)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
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
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            verdict = content.strip() if content else "UNKNOWN"
        except Exception:
            return self._mock_debate(review, rebuttal)

        verdict_upper = verdict.strip().upper()
        if verdict_upper.startswith("SOLID"):
            return {"verdict": "SOLID", "adjustment": 0.2, "comment": verdict}
        return {"verdict": "FLAW", "adjustment": -0.3, "comment": verdict}

    def _mock_debate(self, review: str, rebuttal: str) -> Dict[str, Any]:
        if rebuttal and self.random_state.random() > 0.4:
            return {"verdict": "SOLID", "adjustment": 0.2, "comment": "SOLID"}
        return {
            "verdict": "FLAW",
            "adjustment": -0.3,
            "comment": "Potential gap identified in rebuttal.",
        }


@dataclass
class DomainAgent:
    """
    Domain expert that judges novelty via embedding density.
    """

    abstracts: Sequence[str]
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence_transformers is required for DomainAgent. "
                "Install it via `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(self.model_name)
        self.corpus_embeddings = self.model.encode(
            list(self.abstracts), convert_to_numpy=True
        )

    def analyze_novelty(self, target_abstract: str | None) -> Dict[str, float]:
        """
        Measure local density and novelty against the abstract corpus.
        """
        if not target_abstract:
            return {"density_score": 0.0, "novelty_score": 0.0}

        target_embedding = self.model.encode(target_abstract, convert_to_numpy=True)
        similarities = self._cosine_similarity(target_embedding, self.corpus_embeddings)

        # Exclude self similarity by ignoring the maximum value.
        density = float(np.sum(similarities > 0.8))

        sorted_sim = np.sort(similarities)[::-1]
        top_neighbors = sorted_sim[1:6]  # skip the paper itself
        if top_neighbors.size == 0:
            novelty = 1.0
        else:
            distances = 1.0 - top_neighbors
            novelty = float(np.mean(distances))

        return {
            "density_score": density,
            "novelty_score": novelty,
        }

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vec_norm = np.linalg.norm(vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero
        valid = (vec_norm > 0) & (matrix_norms > 0)
        sims = np.zeros(len(matrix))
        if not valid.any():
            return sims
        sims[valid] = np.dot(matrix[valid], vec) / (matrix_norms[valid] * vec_norm)
        return sims


@dataclass
class MetaAC:
    """
    Orchestrator that fuses Bayesian and argumentation signals.
    """

    bayes_agent: BayesianAgent
    argument_agent: ArgumentAgent
    domain_agent: DomainAgent | None = None

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
        reliability, profile_reason = self.bayes_agent.get_reliability_score(
            paper_data, return_details=True
        )
        normalized_score = np.clip(calibrated_score / 10.0, 0.0, 1.0)

        sentiment = float(text_analysis_result.get("sentiment_change", 0.0))
        sentiment += float(text_analysis_result.get("debate_adjustment", 0.0))
        sentiment_score = np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0)
        debate_verdict = text_analysis_result.get("debate_verdict")

        novelty_bonus = 0.0
        if self.domain_agent:
            abstract = paper_data.get("abstract_clean") or paper_data.get("abstract")
            domain_stats = self.domain_agent.analyze_novelty(abstract)
            novelty_bonus = np.clip(domain_stats["novelty_score"], 0.0, 1.0) * 0.05
        else:
            domain_stats = {"density_score": None, "novelty_score": None}

        final_probability = float(
            np.clip(
                (0.6 * normalized_score) + (0.3 * sentiment_score) + novelty_bonus,
                0.0,
                1.0,
            )
        )
        debate_phrase = ""
        if debate_verdict:
            if debate_verdict == "SOLID":
                debate_phrase = "Simulation confirmed rebuttal solidity."
            else:
                debate_phrase = "Simulation flagged rebuttal weakness."

        reasoning = (
            f"Calibrated score={calibrated_score:.2f} (reliability {reliability:.2f}; "
            f"{profile_reason}), sentiment contribution={sentiment_score:.2f}, "
            f"novelty bonus={novelty_bonus:.2f}. {debate_phrase}"
        )
        return {
            "final_probability": final_probability,
            "reasoning": reasoning,
            "domain": domain_stats,
        }
