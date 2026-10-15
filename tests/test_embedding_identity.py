"""Offline characterization of cache and fingerprint embedding identities."""

import hashlib
import socket

import pytest

from synthetic_panel import config, embeddings


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected network, model initialization, or persistence")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ["_init_local_model", "_init_gemini_client", "_init_openai_client"]:
        monkeypatch.setattr(embeddings, name, forbidden)
    monkeypatch.setattr(embeddings.EmbeddingCache, "save_disk_cache", forbidden)
    monkeypatch.setattr(embeddings.EmbeddingCache, "get_disk_cache", forbidden)
    for name in [
        "SSR_GENERATION_PROVIDER",
        "SSR_MODEL",
        "SSR_DIST_TEMPERATURE",
        "SSR_N_SAMPLES",
    ]:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "raw_provider,provider,default_model",
    [
        (None, "local", "all-MiniLM-L6-v2"),
        ("local", "local", "all-MiniLM-L6-v2"),
        ("GeMiNi", "gemini", "gemini-embedding-001"),
        ("OPENAI", "openai", "provider-default"),
        ("Other", "other", "all-MiniLM-L6-v2"),
        ("", "", "all-MiniLM-L6-v2"),
        (" gemini ", " gemini ", "all-MiniLM-L6-v2"),
    ],
)
@pytest.mark.parametrize("raw_model", [None, "", " custom:Model "])
def test_cache_and_fingerprint_keep_exact_identity(
    monkeypatch, raw_provider, provider, default_model, raw_model
):
    for name, value in [
        ("SSR_EMBEDDING_PROVIDER", raw_provider),
        ("SSR_EMBEDDING_MODEL", raw_model),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    model = default_model if raw_model is None else raw_model
    text = "synthetic reaction é"
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    assert embeddings._get_cache_key(text) == f"{provider}:{model}:{digest}"
    result = config.get_config_fingerprint()
    assert result["embedding"] == {"provider": provider, "model": model}
    assert result["llm"] == {"provider": "agent", "model": "agent-provided"}
    assert result["ssr"] == {"dist_temperature": 1.0, "n_samples": 2}
    assert result["version"] == "1.0"
    assert result["fingerprint"] == (
        f"agent-provided:{provider}:{model}:T1.0:N2:"
        f"{result['data']['personas_hash'][:6]}:{result['data']['anchors_hash'][:6]}"
    )


def test_environment_is_resolved_on_every_call(monkeypatch):
    for provider, model in [
        ("local", "first"),
        ("GEMINI", "second"),
        ("openai", "third"),
    ]:
        monkeypatch.setenv("SSR_EMBEDDING_PROVIDER", provider)
        monkeypatch.setenv("SSR_EMBEDDING_MODEL", model)
        assert embeddings._get_cache_key("text").startswith(
            f"{provider.lower()}:{model}:"
        )
        assert config.get_config_fingerprint()["embedding"] == {
            "provider": provider.lower(),
            "model": model,
        }


@pytest.mark.parametrize("provider", ["local", "gemini", "openai", "unknown"])
def test_public_cache_callers_preserve_identity_order_and_saves(monkeypatch, provider):
    monkeypatch.setenv("SSR_EMBEDDING_PROVIDER", provider)
    monkeypatch.setenv("SSR_EMBEDDING_MODEL", "fixture-model")
    cache = {}
    calls = []
    saves = []
    monkeypatch.setattr(embeddings.EmbeddingCache, "get_disk_cache", lambda: cache)
    monkeypatch.setattr(
        embeddings.EmbeddingCache, "save_disk_cache", lambda: saves.append(dict(cache))
    )
    selected = provider if provider in ("gemini", "openai") else "local"

    def single(text):
        calls.append(("single", text))
        return [1.0]

    def batch(texts):
        calls.append(("batch", texts))
        return [[float(i + 2)] for i in range(len(texts))]

    monkeypatch.setattr(embeddings, f"_get_{selected}_embedding", single)
    monkeypatch.setattr(embeddings, f"_get_{selected}_embeddings_batch", batch)
    first = embeddings.get_embedding("first")
    assert embeddings.get_embedding("first") is first
    assert embeddings.get_embeddings_batch(["first", "second", "first", "third"]) == [
        [1.0],
        [2.0],
        [1.0],
        [3.0],
    ]
    assert embeddings.get_embeddings_batch(["third", "first"]) == [[3.0], [1.0]]
    assert embeddings.get_embeddings_batch([]) == []
    assert calls == [("single", "first"), ("batch", ["second", "third"])]
    assert len(saves) == 2
    assert cache == {
        f"{provider}:fixture-model:{hashlib.sha256(text.encode()).hexdigest()[:16]}": value
        for text, value in [("first", [1.0]), ("second", [2.0]), ("third", [3.0])]
    }
