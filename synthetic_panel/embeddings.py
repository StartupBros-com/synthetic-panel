"""
SSR Embeddings Module

Embedding generation with local-first multi-provider support and disk caching.

Supports:
- Gemini (gemini-embedding-001) - Best STS performance, MTEB 81.54
- OpenAI (configure SSR_EMBEDDING_MODEL for the provider model)
- Local (sentence-transformers all-MiniLM-L6-v2) - No API needed

Cache is keyed by (provider, model, text_hash) for automatic invalidation
when configuration changes.
"""

import gzip
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

__all__ = [
    "get_embedding",
    "get_embeddings_batch",
    "EmbeddingCache",
]

logger = logging.getLogger(__name__)

# Max retries and base delay for transient API errors (503, 429, timeouts)
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds, doubles each retry


def _retry_api_call(fn, *args, **kwargs):
    """Retry an API call with exponential backoff on transient errors."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            # Retry on transient errors: 503, 429, timeout, connection
            is_transient = any(
                s in err_str
                for s in [
                    "503",
                    "429",
                    "unavailable",
                    "timeout",
                    "connection",
                    "rate limit",
                    "overloaded",
                ]
            )
            if not is_transient or attempt == _MAX_RETRIES - 1:
                raise
            delay = _BASE_DELAY * (2**attempt)
            logger.warning(
                "Embedding API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1,
                _MAX_RETRIES,
                e,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # unreachable, but satisfies type checkers


# Cache paths
_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_PATH_GZ = _CACHE_DIR / "embeddings.json.gz"
_CACHE_PATH_OLD = _CACHE_DIR / "embeddings.json"  # Legacy uncompressed


class EmbeddingCache:
    """
    Singleton embedding cache with disk persistence and provider management.

    Replaces module-level global caches with a properly encapsulated class.
    """

    _instance = None
    _disk_cache: dict | None = None
    _provider_cache: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_disk_cache(cls) -> dict:
        """Load embedding cache from disk (supports gzip compression)."""
        if cls._disk_cache is not None:
            return cls._disk_cache

        # Try compressed cache first
        if _CACHE_PATH_GZ.exists():
            try:
                with gzip.open(_CACHE_PATH_GZ, "rt", encoding="utf-8") as f:
                    cls._disk_cache = json.load(f)
                return cls._disk_cache
            except (json.JSONDecodeError, IOError, gzip.BadGzipFile):
                cls._disk_cache = {}

        # Fall back to legacy uncompressed cache and migrate
        if _CACHE_PATH_OLD.exists():
            try:
                with open(_CACHE_PATH_OLD) as f:
                    cls._disk_cache = json.load(f)
                # Migrate: save as compressed, remove old file
                cls.save_disk_cache()
                _CACHE_PATH_OLD.unlink()
                return cls._disk_cache
            except (json.JSONDecodeError, IOError):
                cls._disk_cache = {}

        cls._disk_cache = {}
        return cls._disk_cache

    @classmethod
    def save_disk_cache(cls) -> None:
        """Save embedding cache to disk with gzip compression."""
        if cls._disk_cache is None:
            return

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        fd, temp_path = tempfile.mkstemp(dir=str(_CACHE_DIR), suffix=".gz.tmp")
        try:
            with os.fdopen(fd, "wb") as raw_file:
                with gzip.GzipFile(fileobj=raw_file, mode="wb", compresslevel=6) as gz:
                    gz.write(json.dumps(cls._disk_cache).encode("utf-8"))
            os.replace(temp_path, str(_CACHE_PATH_GZ))
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    @classmethod
    def get_provider(cls, provider_name: str):
        """Get cached provider client."""
        return cls._provider_cache.get(provider_name)

    @classmethod
    def set_provider(cls, provider_name: str, client):
        """Cache a provider client."""
        cls._provider_cache[provider_name] = client


def _get_cache_key(text: str) -> str:
    """Generate cache key including provider and model for auto-invalidation."""
    provider = os.environ.get("SSR_EMBEDDING_PROVIDER", "local").lower()
    if provider == "gemini":
        model = os.environ.get("SSR_EMBEDDING_MODEL", "gemini-embedding-001")
    elif provider == "openai":
        model = os.environ.get("SSR_EMBEDDING_MODEL", "provider-default")
    else:
        model = os.environ.get("SSR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"{provider}:{model}:{text_hash}"


# =============================================================================
# Gemini Provider
# =============================================================================


def _init_gemini_client():
    """Initialize Gemini client if needed."""
    if EmbeddingCache.get_provider("gemini_client") is None:
        # Support both GOOGLE_API_KEY (legacy) and GEMINI_API_KEY (new SDK default)
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY required for Gemini embeddings"
            )
        from google import genai

        EmbeddingCache.set_provider("gemini_client", genai.Client(api_key=api_key))
    return EmbeddingCache.get_provider("gemini_client")


def _get_gemini_embedding(text: str) -> list[float]:
    """Get embedding using Google's Gemini API (with retry on transient errors)."""
    client = _init_gemini_client()
    model = os.environ.get("SSR_EMBEDDING_MODEL", "gemini-embedding-001")

    def _call():
        return client.models.embed_content(
            model=model, contents=text, config={"task_type": "SEMANTIC_SIMILARITY"}
        )

    result = _retry_api_call(_call)
    return list(result.embeddings[0].values)


def _get_gemini_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in one Gemini API call (with retry on transient errors)."""
    client = _init_gemini_client()
    model = os.environ.get("SSR_EMBEDDING_MODEL", "gemini-embedding-001")

    def _call():
        return client.models.embed_content(
            model=model, contents=texts, config={"task_type": "SEMANTIC_SIMILARITY"}
        )

    result = _retry_api_call(_call)
    return [list(emb.values) for emb in result.embeddings]


# =============================================================================
# OpenAI Provider
# =============================================================================


def _init_openai_client():
    """Initialize OpenAI client if needed."""
    if EmbeddingCache.get_provider("openai_client") is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        import openai

        EmbeddingCache.set_provider("openai_client", openai.OpenAI())
    return EmbeddingCache.get_provider("openai_client")


def _get_openai_embedding(text: str) -> list[float]:
    """Get embedding using OpenAI API (with retry on transient errors)."""
    client = _init_openai_client()
    model = os.environ.get("SSR_EMBEDDING_MODEL")
    if not model:
        raise ValueError("SSR_EMBEDDING_MODEL required for OpenAI embeddings")

    def _call():
        return client.embeddings.create(model=model, input=text)

    response = _retry_api_call(_call)
    return response.data[0].embedding


def _get_openai_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in one OpenAI API call (with retry on transient errors)."""
    client = _init_openai_client()
    model = os.environ.get("SSR_EMBEDDING_MODEL")
    if not model:
        raise ValueError("SSR_EMBEDDING_MODEL required for OpenAI embeddings")

    def _call():
        return client.embeddings.create(model=model, input=texts)

    response = _retry_api_call(_call)
    return [item.embedding for item in response.data]


# =============================================================================
# Local Provider (sentence-transformers)
# =============================================================================


def _init_local_model():
    """Initialize local sentence-transformers model if needed."""
    if EmbeddingCache.get_provider("local_model") is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get("SSR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        EmbeddingCache.set_provider("local_model", SentenceTransformer(model_name))
    return EmbeddingCache.get_provider("local_model")


def _get_local_embedding(text: str) -> list[float]:
    """Get embedding using local sentence-transformers model."""
    model = _init_local_model()
    return model.encode(text).tolist()


def _get_local_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts using local model."""
    model = _init_local_model()
    # sentence-transformers natively supports batch encoding
    embeddings = model.encode(texts)
    return [emb.tolist() for emb in embeddings]


# =============================================================================
# Public API
# =============================================================================


def get_embedding(text: str) -> list[float]:
    """
    Get embedding vector for text using configured provider.

    Caches embeddings to disk to avoid redundant API calls for static
    reference statements. Cache is keyed by (provider, model, text_hash)
    so it auto-invalidates when configuration changes.

    Provider selection (SSR_EMBEDDING_PROVIDER env var):
    - gemini: Google's gemini-embedding-001 (best STS performance, MTEB 81.54)
    - openai: OpenAI embeddings (set SSR_EMBEDDING_MODEL explicitly)
    - local: sentence-transformers all-MiniLM-L6-v2 (no API needed)

    Returns embedding as list of floats.
    """
    # Check cache first
    cache = EmbeddingCache.get_disk_cache()
    cache_key = _get_cache_key(text)

    if cache_key in cache:
        return cache[cache_key]

    # Fetch from provider
    provider = os.environ.get("SSR_EMBEDDING_PROVIDER", "local").lower()

    if provider == "gemini":
        embedding = _get_gemini_embedding(text)
    elif provider == "openai":
        embedding = _get_openai_embedding(text)
    else:  # local
        embedding = _get_local_embedding(text)

    # Cache and persist
    cache[cache_key] = embedding
    EmbeddingCache.save_disk_cache()

    return embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Get embedding vectors for multiple texts in a single API call.

    Checks cache first, batches only uncached texts, then caches results.
    All providers support batch embedding, reducing API round trips.

    Returns embeddings in same order as input texts.
    """
    cache = EmbeddingCache.get_disk_cache()
    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    # Check cache for each text
    for i, text in enumerate(texts):
        cache_key = _get_cache_key(text)
        if cache_key in cache:
            results[i] = cache[cache_key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Batch fetch uncached texts
    if uncached_texts:
        provider = os.environ.get("SSR_EMBEDDING_PROVIDER", "local").lower()

        if provider == "gemini":
            embeddings = _get_gemini_embeddings_batch(uncached_texts)
        elif provider == "openai":
            embeddings = _get_openai_embeddings_batch(uncached_texts)
        else:  # local
            embeddings = _get_local_embeddings_batch(uncached_texts)

        # Cache and assign results
        for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
            cache_key = _get_cache_key(text)
            cache[cache_key] = embedding
            results[idx] = embedding

        EmbeddingCache.save_disk_cache()

    return results
