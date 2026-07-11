"""
SSR Configuration Module

Configuration, fingerprinting, personas, and caching utilities for the
Synthetic Customer Research Tester.

This module provides:
- Persona loading and prompt building
- Configuration fingerprinting for cache invalidation
- Stimulus prompt templates
- Response caching and retrieval
"""

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

__all__ = [
    "PersonaValidationError",
    "load_personas",
    "load_default_anchors",
    "get_persona_names",
    "resolve_persona_names",
    "get_config_fingerprint",
    "build_persona_prompt",
    "get_stimulus_prompt",
    "is_multi_set_format",
    "get_cached_result",
    "load_responses",
    "save_responses_to_disk",
    "RESPONSES_DIR",
    "RESPONSES_PATH_GZ",
]


class PersonaValidationError(ValueError):
    """Raised when a persona is incomplete or malformed."""


# Response storage directory (same as calibration data)
RESPONSES_DIR = Path(__file__).parent / "calibration_data"
RESPONSES_PATH_GZ = RESPONSES_DIR / "responses.json.gz"
_RESPONSES_PATH_OLD = RESPONSES_DIR / "responses.json"  # Legacy uncompressed
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PERSONAS_PATH = DATA_DIR / "personas.example.yaml"
DEFAULT_ANCHORS_PATH = DATA_DIR / "anchors.default.yaml"


_MINIMAL_REQUIRED_FIELDS = [
    "name",
    "demographics.age",
    "demographics.location",
    "demographics.family",
    "demographics.occupation",
    "demographics.income",
]

_RICH_REQUIRED_FIELDS = [
    "situation.time_available",
    "situation.capital_available",
    "situation.risk_tolerance",
    "situation.current_status",
    "motivations.primary",
    "motivations.secondary",
    "motivations.excitement_driver",
    "motivations.income_goal",
    "psychology.patience",
    "psychology.experience_level",
    "psychology.biggest_blocker",
    "psychology.secondary_blocker",
    "psychology.fear",
    "preferences.learning_format",
    "preferences.content_style",
    "preferences.needs",
    "messaging_triggers.positive",
    "messaging_triggers.negative",
]


def _persona_label(persona_data: dict) -> str:
    return str(persona_data.get("id") or persona_data.get("name") or "unknown")


def _get_required_field(persona_data: dict, dotted_path: str):
    current = persona_data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            label = _persona_label(persona_data)
            raise PersonaValidationError(
                f"persona '{label}' is missing required field '{dotted_path}'"
            )
        current = current[part]
    if current in (None, "") or current == []:
        label = _persona_label(persona_data)
        raise PersonaValidationError(
            f"persona '{label}' is missing required field '{dotted_path}'"
        )
    return current


def _validate_persona_for_prompt(persona_data: dict) -> None:
    if not isinstance(persona_data, dict):
        raise PersonaValidationError("persona must be a mapping")
    for field in _MINIMAL_REQUIRED_FIELDS:
        _get_required_field(persona_data, field)
    if "situation" in persona_data:
        for field in _RICH_REQUIRED_FIELDS:
            _get_required_field(persona_data, field)


def load_personas(path: str | Path | None = None) -> dict:
    """Load persona definitions from YAML."""
    import yaml

    personas_path = Path(path) if path else DEFAULT_PERSONAS_PATH
    with open(personas_path) as f:
        return yaml.safe_load(f)


def load_default_anchors(path: str | Path | None = None) -> dict:
    """Load the default SSR anchor statements from YAML."""
    import yaml

    anchors_path = Path(path) if path else DEFAULT_ANCHORS_PATH
    with open(anchors_path) as f:
        anchors = yaml.safe_load(f)
    if not isinstance(anchors, dict) or not anchors:
        raise ValueError(
            f"Anchor file must contain a non-empty mapping: {anchors_path}"
        )
    return anchors


def get_persona_names(personas_data: dict) -> list[str]:
    """Return configured persona keys, excluding non-persona metadata."""
    return [
        key
        for key, value in personas_data.items()
        if key != "reference_statements" and isinstance(value, dict) and "name" in value
    ]


def resolve_persona_names(
    personas_data: dict, requested_persona: str | None = None
) -> list[str]:
    """Resolve a requested persona or all configured personas."""
    persona_names = get_persona_names(personas_data)
    if requested_persona in (None, "all"):
        return persona_names
    return [requested_persona]


def get_config_fingerprint(personas_tested: list[str] | None = None) -> dict:
    """
    Generate a configuration fingerprint capturing all scoring-relevant settings.

    This metadata is stored with each calibration record to enable:
    - Detecting when old scores may be invalid due to config changes
    - Filtering records by configuration version
    - Understanding why scores differ across test runs

    Args:
        personas_tested: List of persona names used (for persona-specific hashing)

    Returns:
        Dictionary with all config that affects scores, suitable for storage in metadata
    """
    # Generation is handled out-of-process by the user's AI agent in v1.
    # Do not report a generation temperature here; this CLI cannot control it.
    provider = os.environ.get("SSR_GENERATION_PROVIDER", "agent")
    llm_model = os.environ.get("SSR_MODEL", "agent-provided")

    # Get embedding config
    embedding_provider = os.environ.get("SSR_EMBEDDING_PROVIDER", "local").lower()
    if embedding_provider == "gemini":
        embedding_model = os.environ.get("SSR_EMBEDDING_MODEL", "gemini-embedding-001")
    elif embedding_provider == "openai":
        embedding_model = os.environ.get("SSR_EMBEDDING_MODEL", "provider-default")
    else:
        embedding_model = os.environ.get("SSR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Get SSR algorithm config
    dist_temperature = float(os.environ.get("SSR_DIST_TEMPERATURE", "1.0"))
    n_samples = int(os.environ.get("SSR_N_SAMPLES", "2"))

    # Load persona and anchor data for version hashing
    import yaml

    personas_path = DEFAULT_PERSONAS_PATH
    personas_content = personas_path.read_text()
    personas_data = yaml.safe_load(personas_content)
    anchors_content = DEFAULT_ANCHORS_PATH.read_text()
    anchors_data = yaml.safe_load(anchors_content)

    # Always use full personas hash for fingerprint comparison (ensures consistency)
    # This way "current vs outdated" isn't affected by which personas were tested
    full_personas_hash = hashlib.sha256(personas_content.encode()).hexdigest()[:12]

    # If specific personas tested, also store a hash of just those (for detailed tracking)
    if personas_tested:
        relevant_personas = {
            k: v for k, v in personas_data.items() if k in personas_tested
        }
        tested_personas_hash = hashlib.sha256(
            yaml.dump(relevant_personas, sort_keys=True).encode()
        ).hexdigest()[:12]
    else:
        tested_personas_hash = full_personas_hash

    # Extract reference statements hash separately (this is critical for scoring)
    ref_hash = hashlib.sha256(
        yaml.dump(anchors_data, sort_keys=True).encode()
    ).hexdigest()[:12]

    # Build full fingerprint
    fingerprint = {
        "version": "1.0",  # Schema version for future compatibility
        "llm": {
            "provider": provider,
            "model": llm_model,
        },
        "embedding": {
            "provider": embedding_provider,
            "model": embedding_model,
        },
        "ssr": {
            "dist_temperature": dist_temperature,
            "n_samples": n_samples,
        },
        "data": {
            "personas_hash": full_personas_hash,  # Full hash for consistency
            "tested_personas_hash": tested_personas_hash,  # Which personas were actually tested
            "anchors_hash": ref_hash,
        },
        # Compact string for quick comparison - uses full_personas_hash for consistency
        "fingerprint": f"{llm_model}:{embedding_provider}:{embedding_model}:T{dist_temperature}:N{n_samples}:{full_personas_hash[:6]}:{ref_hash[:6]}",
    }

    return fingerprint


def build_persona_prompt(persona_data: dict) -> str:
    """
    Build persona prompt for LLM conditioning using third-person interview framing.

    Uses research-backed third-person framing (arXiv:2512.22725) which showed 13.9%
    improvement in human-LLM alignment over first-person prompting. The interview
    style reduces social desirability bias and produces more authentic responses.

    Supports both rich personas (full psychological profile) and minimal personas
    (demographics only). Minimal personas omit backstory per research showing
    simpler profiles often produce better human alignment (arXiv:2503.16527).
    """
    _validate_persona_for_prompt(persona_data)
    p = persona_data
    name = p["name"]
    is_minimal = "situation" not in p

    # Third-person interview framing (research-backed)
    prompt = f"""You are a market research analyst simulating how {name} would respond in a consumer research interview.

RESPONDENT PROFILE: {name}

DEMOGRAPHICS:
- Age: {p["demographics"]["age"]}
- Location: {p["demographics"]["location"]}
- Family: {p["demographics"]["family"]}
- Occupation: {p["demographics"]["occupation"]}
- Income: {p["demographics"]["income"]}"""

    # Rich personas include full psychological profile
    if not is_minimal:
        prompt += f"""

SITUATION:
- Time available: {p["situation"]["time_available"]}
- Capital available: {p["situation"]["capital_available"]}
- Risk tolerance: {p["situation"]["risk_tolerance"]}
- Current status: {p["situation"]["current_status"]}

MOTIVATIONS:
- Primary motivation: {p["motivations"]["primary"]}
- Secondary motivation: {p["motivations"]["secondary"]}
- What excites them: {p["motivations"]["excitement_driver"]}
- Income goal: {p["motivations"]["income_goal"]}

PSYCHOLOGY:
- Patience level: {p["psychology"]["patience"]}
- Experience level: {p["psychology"]["experience_level"]}
- Biggest blocker: {p["psychology"]["biggest_blocker"]}
- Secondary blocker: {p["psychology"]["secondary_blocker"]}
- Fear: {p["psychology"]["fear"]}

PREFERENCES:
- Learning format: {p["preferences"]["learning_format"]}
- Content style: {p["preferences"]["content_style"]}

What they need:
{chr(10).join("- " + need for need in p["preferences"]["needs"])}

Messages that resonate with them:
{chr(10).join("- " + msg for msg in p["messaging_triggers"]["positive"])}

Messages that turn them off:
{chr(10).join("- " + msg for msg in p["messaging_triggers"]["negative"])}

Provide {name}'s authentic response as they would give it to a neutral researcher. Capture their genuine reaction based on their situation, constraints, and psychology."""
    else:
        # Minimal personas: simple third-person instruction
        prompt += f"""

Provide {name}'s authentic response as someone with this demographic profile would naturally give it to a neutral researcher."""

    return prompt


def get_stimulus_prompt(stimulus_type: str, content: str, context: str = "") -> str:
    """
    Build stimulus presentation prompt using third-person interview framing.

    Uses interview-style questions per research (arXiv:2512.22725) showing
    third-person framing reduces social desirability bias in LLM responses.

    Args:
        stimulus_type: Type of stimulus (headline, email, offer, etc.)
        content: The content variation to present
        context: Optional background context prepended to the interview scenario.
                 Provides test-level framing (e.g., page description, pricing info)
                 that applies to all variations in a test.
    """
    prompts = {
        "headline": f"""INTERVIEW SCENARIO:
The respondent encounters this on a page:

"{content}"

INTERVIEWER: What is this person's honest, gut reaction to this headline?
- Does it speak to their situation and goals?
- Does it address their concerns or fears?
- Would it make them want to learn more?
- Would they consider taking the next step?

Provide their authentic response, being specific about WHY they feel this way.""",
        "email_subject": f"""INTERVIEW SCENARIO:
The respondent receives an email with this subject line:

"{content}"

INTERVIEWER: What is this person's honest reaction? Would they open this email?
Consider how relevant this feels to their current situation and goals.

Provide their authentic response.""",
        "offer": f"""INTERVIEW SCENARIO:
The respondent comes across this offer description:

---
{content}
---

INTERVIEWER: What is this person's honest reaction to this offer?
- Does this solve a problem they have?
- Is it clear what they would get?
- Does it feel worth the investment (time/money)?
- What concerns or objections do they have?
- Would they consider taking the next step?

Provide their authentic response, being specific about WHY based on their situation.""",
        "value_prop": f"""INTERVIEW SCENARIO:
The respondent reads this value proposition:

"{content}"

INTERVIEWER: What is this person's reaction? Does this resonate with what they're looking for?
What specifically appeals or doesn't appeal to them?

Provide their authentic response.""",
        "cta": f"""INTERVIEW SCENARIO:
The respondent sees this call-to-action button:

[{content}]

INTERVIEWER: How does this make them feel? Would they click it? Why or why not?

Provide their authentic response.""",
        "price": f"""INTERVIEW SCENARIO:
The respondent sees this pricing:

{content}

INTERVIEWER: What is this person's reaction given their budget and situation?
Does this feel like good value to them? What concerns do they have?

Provide their authentic response.""",
    }

    prompt = prompts.get(
        stimulus_type,
        f"""INTERVIEW SCENARIO:
The respondent receives this message:

---
{content}
---

INTERVIEWER: What is this person's honest reaction?
- Does it feel relevant to their situation and goals?
- What specifically appeals or does not appeal to them?
- Would they consider taking the next step?

Provide their authentic response, being specific about WHY they feel this way.""",
    )

    if context:
        prompt = f"BACKGROUND CONTEXT:\n{context.strip()}\n\n{prompt}"

    return prompt


def is_multi_set_format(reference_statements: dict) -> bool:
    """Check if reference_statements uses multi-set format (set_1, set_2, etc.)."""
    if not reference_statements:
        return False
    first_key = next(iter(reference_statements.keys()))
    return str(first_key).startswith("set_")


def load_responses() -> list[dict]:
    """Load existing responses from disk (supports gzip compression)."""
    # Try compressed file first
    if RESPONSES_PATH_GZ.exists():
        try:
            with gzip.open(RESPONSES_PATH_GZ, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, gzip.BadGzipFile):
            return []

    # Fall back to legacy uncompressed file and migrate
    if _RESPONSES_PATH_OLD.exists():
        try:
            with open(_RESPONSES_PATH_OLD, "r") as f:
                responses = json.load(f)
            # Migrate: save as compressed, then remove old file
            save_responses_to_disk(responses)
            _RESPONSES_PATH_OLD.unlink()
            return responses
        except (json.JSONDecodeError, IOError):
            return []

    return []


def save_responses_to_disk(responses: list[dict]) -> None:
    """Save responses to disk with gzip compression (atomic write)."""
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename
    fd, temp_path = tempfile.mkstemp(dir=str(RESPONSES_DIR), suffix=".gz.tmp")
    try:
        with os.fdopen(fd, "wb") as raw_file:
            with gzip.GzipFile(fileobj=raw_file, mode="wb", compresslevel=6) as gz:
                gz.write(json.dumps(responses, indent=2, default=str).encode("utf-8"))
        os.replace(temp_path, str(RESPONSES_PATH_GZ))
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _get_cached_responses(content: str, test_id: str, persona: str | None) -> list[str]:
    """
    Retrieve actual response text from responses.json for a cached result.

    Matches by test_id and variation_content to find the original LLM responses.
    Falls back to content matching if test_id doesn't match (for older cached data).
    """
    all_responses = load_responses()
    if not all_responses:
        return []

    # First try: exact match on test_id and content
    for record in all_responses:
        if (
            record.get("test_id") == test_id
            and record.get("variation_content") == content
            and (persona is None or record.get("persona") == persona)
        ):
            responses = record.get("responses", [])
            if responses and responses[0] != "[Cached]":
                return responses

    # Second try: match by content and persona (for cross-test cache hits)
    for record in reversed(all_responses):  # Most recent first
        if record.get("variation_content") == content and (
            persona is None or record.get("persona") == persona
        ):
            responses = record.get("responses", [])
            if responses and responses[0] != "[Cached]":
                return responses

    return []


def get_cached_result(
    content: str, test_type: str, personas: list[str], current_fingerprint: str
) -> dict | None:
    """
    Check if we have a cached result for this content with matching config.

    Returns cached result dict with score/distribution if found and fingerprint matches,
    otherwise returns None (indicating LLM call is needed).

    Cache hit requires BOTH:
    - Same content_hash (content + test_type + personas)
    - Same config fingerprint (LLM model, embedding model, personas version, etc.)
    """
    try:
        from calibration import CalibrationStore
    except ImportError:
        return None

    store = CalibrationStore()
    records = store.find_duplicate_content(content, test_type, personas)

    if not records:
        return None

    # Find records with matching fingerprint (most recent first)
    for record in sorted(records, key=lambda r: r.created_at, reverse=True):
        record_fp = record.metadata.get("config", {}).get("fingerprint", "")
        if record_fp == current_fingerprint:
            # Look up actual response text from responses.json
            cached_responses = _get_cached_responses(
                content, record.test_id, personas[0] if personas else None
            )
            return {
                "content": record.variation_content,
                "score": record.predicted_score,
                "distribution": record.predicted_distribution,
                "response": cached_responses[0] if cached_responses else "[Cached]",
                "all_responses": cached_responses if cached_responses else ["[Cached]"],
                "cached": True,
                "cached_from_test": record.test_id,
            }

    # Records exist but fingerprint differs - config changed, need fresh run
    return None
