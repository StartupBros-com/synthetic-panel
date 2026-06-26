"""
SSR Scoring Module

Implementation of the SSR (Semantic Similarity Rating) methodology from PyMC Labs
(arXiv:2510.08338).

The algorithm compares LLM response embeddings to reference anchor statements
to produce probability distributions over Likert scale ratings.
"""

import os

from .config import is_multi_set_format
from .embeddings import get_embedding, get_embeddings_batch

__all__ = [
    "calculate_ssr_score",
    "SSRScoringError",
]

DEFAULT_DIST_TEMPERATURE = 1.0


def _calculate_single_set_distribution(
    response_embedding, ref_statements: dict, temperature: float
) -> list[float]:
    """
    Calculate probability distribution for a single reference statement set.

    Internal helper for calculate_ssr_score(). Implements Equations 7-9 from
    arXiv:2510.08338 for one set of anchor statements.

    Args:
        response_embedding: Embedding vector for the response text
        ref_statements: Dict mapping Likert scores (1-5) to anchor statement text
        temperature: Temperature parameter for distribution sharpness

    Returns:
        List of 5 probabilities for Likert scores 1-5
    """
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Get embeddings for reference statements in batch (single API call)
    scores = list(ref_statements.keys())
    statements = list(ref_statements.values())
    embeddings = get_embeddings_batch(statements)
    ref_embeddings = {int(scores[i]): embeddings[i] for i in range(len(scores))}

    # Calculate cosine similarities (Equation 7)
    similarities = {
        score: cosine_similarity(response_embedding, ref_emb)
        for score, ref_emb in ref_embeddings.items()
    }

    # Subtract minimum similarity (Equation 8)
    min_sim = min(similarities.values())
    adjusted = {k: v - min_sim for k, v in similarities.items()}

    # Normalize to raw probabilities
    total_adj = sum(adjusted.values())
    if total_adj > 0:
        raw_probs = {k: v / total_adj for k, v in adjusted.items()}
    else:
        raw_probs = {k: 0.2 for k in range(1, 6)}

    # Apply temperature scaling: p^(1/T) (Equation 9)
    temp_scaled = {
        k: (p ** (1.0 / temperature)) if p > 0 else 0 for k, p in raw_probs.items()
    }

    # Re-normalize after temperature scaling
    total_scaled = sum(temp_scaled.values())
    if total_scaled > 0:
        probs = [temp_scaled.get(i, 0) / total_scaled for i in range(1, 6)]
    else:
        probs = [0.2] * 5

    return probs


class SSRScoringError(Exception):
    """Raised when SSR scoring fails after retries (e.g. embedding API down)."""


def calculate_ssr_score(
    response_text: str,
    reference_statements: dict,
    temperature: float = None,
) -> tuple[float, list[float], str]:
    """
    Calculate SSR score by comparing response to reference anchor statements.

    Uses the SSR (Semantic Similarity Rating) methodology from PyMC Labs
    (arXiv:2510.08338). Per the paper, averaging across 6 distinct reference
    statement sets yields final probability distributions with reduced bias.

    Key steps per the paper's equations:
    1. Compute embeddings for response and reference anchor statements
    2. Calculate cosine similarities (Equation 7)
    3. Subtract minimum similarity for non-negative values (Equation 8)
    4. Normalize to probability distribution
    5. Apply power-law temperature scaling: p^(1/T) (Equation 9)
    6. Re-normalize and compute weighted average score
    7. Average across all 6 reference sets (paper methodology)

    Args:
        response_text: The LLM's textual response to analyze
        reference_statements: Either:
            - Dict mapping Likert scores (1-5) to anchor statements (legacy single-set)
            - Dict of sets {"set_1": {...}, "set_2": {...}} (paper-compliant 6-set)
        temperature: Controls distribution sharpness (default: SSR_DIST_TEMPERATURE env var or 1.0)
                    T=1.0 is neutral (paper's default)
                    T<1.0 = sharper/more peaked distributions
                    T>1.0 = flatter distributions with more uncertainty

    Returns:
        (weighted_score, probability_distribution, scoring_method) tuple.
        scoring_method is "ssr" for successful embedding-based scoring.

    Raises:
        SSRScoringError: If embedding API calls fail after retries.
    """
    # Temperature controls distribution sharpness per SSR paper methodology
    if temperature is None:
        temperature = float(
            os.environ.get("SSR_DIST_TEMPERATURE", str(DEFAULT_DIST_TEMPERATURE))
        )

    try:
        # Get response embedding once (reused for all reference sets)
        response_embedding = get_embedding(response_text)

        # Check format: multi-set (paper-compliant) or single-set (legacy)
        if is_multi_set_format(reference_statements):
            # Paper-compliant: average across all 6 reference sets
            all_probs = []
            for set_name, ref_set in reference_statements.items():
                set_probs = _calculate_single_set_distribution(
                    response_embedding, ref_set, temperature
                )
                all_probs.append(set_probs)

            # Average probability distributions across all sets
            n_sets = len(all_probs)
            probs = [sum(p[i] for p in all_probs) / n_sets for i in range(5)]
        else:
            # Legacy single-set format (backwards compatibility)
            probs = _calculate_single_set_distribution(
                response_embedding, reference_statements, temperature
            )

        # Weighted average score
        weighted_score = sum((i + 1) * p for i, p in enumerate(probs))

        return weighted_score, probs, "ssr"

    except Exception as e:
        from rich.console import Console

        console = Console()
        console.print(f"[red]SSR scoring failed: {e}[/red]")
        raise SSRScoringError(f"Embedding API call failed after retries: {e}") from e
