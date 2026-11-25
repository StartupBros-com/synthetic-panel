import pytest


@pytest.fixture
def isolated_embedding_disk_cache(monkeypatch, tmp_path):
    """Keep this test's embedding cache and legacy fallback under one temporary directory."""
    from synthetic_panel import embeddings

    cache_dir = tmp_path / "embedding-cache"
    monkeypatch.setattr(embeddings, "_CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        embeddings,
        "_CACHE_PATH_GZ",
        cache_dir / "embeddings.json.gz",
    )
    monkeypatch.setattr(
        embeddings, "_CACHE_PATH_OLD", cache_dir / "embeddings.json"
    )
    monkeypatch.setattr(embeddings.EmbeddingCache, "_disk_cache", None)
