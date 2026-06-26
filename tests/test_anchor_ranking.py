from pathlib import Path

import pytest

from synthetic_panel import embeddings
from synthetic_panel.config import load_default_anchors, load_personas
from synthetic_panel.scoring import calculate_ssr_score


@pytest.mark.slow
@pytest.mark.model_dependent
def test_default_anchors_rank_purchase_intent_ladder_with_local_embeddings(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SSR_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("SSR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setenv("SSR_DIST_TEMPERATURE", "1.0")
    monkeypatch.setattr(embeddings, "_CACHE_DIR", tmp_path / "embedding-cache")
    monkeypatch.setattr(
        embeddings,
        "_CACHE_PATH_GZ",
        tmp_path / "embedding-cache" / "embeddings.json.gz",
    )
    monkeypatch.setattr(
        embeddings, "_CACHE_PATH_OLD", tmp_path / "embedding-cache" / "embeddings.json"
    )
    embeddings.EmbeddingCache._disk_cache = None

    anchors = load_default_anchors(Path("synthetic_panel/data/anchors.default.yaml"))
    personas = load_personas(Path("synthetic_panel/data/personas.example.yaml"))
    sample_persona = personas["alex"]

    assert sample_persona["name"] == "Alex Chen"

    reaction_ladder = {
        "negative": [
            "I would not buy this. It feels irrelevant, risky, and not worth my time, so I would leave without taking another step.",
            "This does not solve a problem I care about. I am not interested and would definitely pass.",
            "I feel skeptical and turned off. I would avoid this and would not sign up or continue.",
        ],
        "neutral": [
            "I am not sure yet. Some parts sound potentially useful, but I would need clearer details and proof before deciding.",
            "This might be relevant, but my interest is mixed. I would compare alternatives before taking any action.",
            "I feel neutral. I could consider it later, but nothing here makes me ready to proceed now.",
        ],
        "positive": [
            "This is exactly what I am looking for. I feel confident, interested, and ready to take the next step now.",
            "The value is clear and compelling. I would seriously consider buying this and would want to sign up soon.",
            "This strongly matches my needs. I am excited to continue and would be highly likely to act.",
        ],
    }

    rung_scores = {}
    rung_distributions = {}
    for rung, reactions in reaction_ladder.items():
        scores = []
        distributions = []
        for reaction in reactions:
            score, distribution, method = calculate_ssr_score(
                reaction,
                anchors,
            )
            assert method == "ssr"
            scores.append(score)
            distributions.append(distribution)

        rung_scores[rung] = sum(scores) / len(scores)
        rung_distributions[rung] = distributions

    assert rung_scores["negative"] < rung_scores["neutral"] < rung_scores["positive"]
    assert rung_scores["positive"] - rung_scores["negative"] > 0.3

    for distributions in rung_distributions.values():
        for distribution in distributions:
            assert max(distribution) - min(distribution) > 0.05
