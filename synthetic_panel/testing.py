"""
SSR Testing Module

Test execution orchestration including:
- Single variation testing (sync and async)
- Batch API support (50% cost reduction via Anthropic Message Batches)
- Full test suite execution
- Response storage and calibration tracking
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import (
    build_persona_prompt,
    get_cached_result,
    get_config_fingerprint,
    get_stimulus_prompt,
    load_default_anchors,
    load_personas,
    load_responses,
    resolve_persona_names,
    save_responses_to_disk,
    RESPONSES_PATH_GZ,
)
from .scoring import SSRScoringError, calculate_ssr_score

PROMPT_SCHEMA_VERSION = 1
DEFAULT_STIMULUS_TYPE = "headline"
DEFAULT_N_SAMPLES = 2
MIN_REACTION_LENGTH = 40
COLLAPSE_SCORE_SPREAD_FLOOR = 0.01
COLLAPSE_EMBEDDING_DISTANCE_FLOOR = 0.002


class SeamInputError(ValueError):
    """Raised for actionable prompt/reaction handoff input errors."""


def get_llm_client():
    raise NotImplementedError("generation is provided by the skill in v1")


def get_async_llm_client():
    raise NotImplementedError("generation is provided by the skill in v1")


def generate_response(*_args, **_kwargs):
    raise NotImplementedError("generation is provided by the skill in v1")


async def generate_response_async(*_args, **_kwargs):
    raise NotImplementedError("generation is provided by the skill in v1")


__all__ = [
    # Core testing
    "test_variation",
    "test_variation_async",
    "run_test",
    "run_test_async",
    "run_test_batch",
    # Batch API
    "prepare_batch_context",
    "_submit_batch",
    "_poll_batch",
    "_retrieve_batch_results",
    "_process_batch_results",
    # Storage and tracking
    "save_responses",
    "track_predictions",
    # Display
    "display_results",
    # Zero-key seam
    "PROMPT_SCHEMA_VERSION",
    "DEFAULT_STIMULUS_TYPE",
    "DEFAULT_N_SAMPLES",
    "SeamInputError",
    "emit_prompt_handoff",
    "write_prompt_handoff",
    "score_reaction_handoff",
]

console = Console()


def _average_distributions(distributions: list[list[float]]) -> list[float]:
    return [sum(d[i] for d in distributions) / len(distributions) for i in range(5)]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import numpy as np

    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    denom = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)
    if denom == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / denom)


def _reasoning_excerpt(reactions: list[str], max_chars: int = 280) -> str:
    for reaction in reactions:
        compact = " ".join(reaction.split())
        if compact:
            return compact[:max_chars]
    return ""


def _load_json_file(path: str | Path, label: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise SeamInputError(f"missing {label}: {json_path}")
    if json_path.stat().st_size == 0:
        raise SeamInputError(f"empty {label}: {json_path}")
    try:
        return json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        raise SeamInputError(f"invalid {label} JSON: {json_path}") from exc


def emit_prompt_handoff(
    personas_data: dict,
    variants: dict[str, str],
    stimulus_type: str = DEFAULT_STIMULUS_TYPE,
    n_samples: int = DEFAULT_N_SAMPLES,
    persona_ids: list[str] | None = None,
    context: str = "",
) -> dict:
    """Build the versioned prompt handoff object for external generation."""
    if not variants:
        raise SeamInputError("missing variants: provide at least one copy variant")
    if n_samples < 1:
        raise SeamInputError("invalid n_samples: must be at least 1")

    available_personas = resolve_persona_names(personas_data, None)
    selected_personas = persona_ids or available_personas
    missing_personas = [p for p in selected_personas if p not in personas_data]
    if missing_personas:
        raise SeamInputError(f"missing persona: {', '.join(missing_personas)}")
    if not selected_personas:
        raise SeamInputError("missing persona file: no personas found")

    records = []
    for variant_id, copy_text in variants.items():
        if not str(variant_id).strip():
            raise SeamInputError("invalid variant: variant_id cannot be empty")
        if not str(copy_text).strip():
            raise SeamInputError(f"empty variant copy: {variant_id}")
        for persona_id in selected_personas:
            persona_prompt = build_persona_prompt(personas_data[persona_id])
            stimulus_prompt = get_stimulus_prompt(stimulus_type, copy_text, context)
            full_prompt = f"{persona_prompt}\n\n---\n\n{stimulus_prompt}"
            for sample_index in range(n_samples):
                records.append(
                    {
                        "variant_id": str(variant_id),
                        "persona_id": persona_id,
                        "sample_index": sample_index,
                        "prompt": full_prompt,
                    }
                )

    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "stimulus_type": stimulus_type,
        "n_samples": n_samples,
        "variants": {str(k): str(v) for k, v in variants.items()},
        "personas": selected_personas,
        "records": records,
    }


def write_prompt_handoff(prompt_handoff: dict, output_path: str | Path) -> None:
    """Write a prompt handoff object as stable JSON."""
    Path(output_path).write_text(json.dumps(prompt_handoff, indent=2, sort_keys=True))


def score_reaction_handoff(
    reactions_path: str | Path,
    reference_statements: dict | None = None,
    dist_temperature: float = 1.0,
) -> dict:
    """Score a versioned reactions handoff with local embeddings and SSR anchors."""
    from .embeddings import get_embeddings_batch

    payload = _load_json_file(reactions_path, "reactions file")
    if payload.get("schema_version") != PROMPT_SCHEMA_VERSION:
        raise SeamInputError(
            f"unsupported schema_version: {payload.get('schema_version')!r}"
        )

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise SeamInputError(f"empty reactions file: {reactions_path}")

    variants = payload.get("variants") or {}
    personas = payload.get("personas") or []
    n_samples = int(payload.get("n_samples") or DEFAULT_N_SAMPLES)
    if not variants:
        raise SeamInputError("missing variants: reactions file has no variants")
    if not personas:
        raise SeamInputError("missing persona file: reactions file has no personas")

    reference_statements = reference_statements or load_default_anchors()
    expected_keys = {
        (variant_id, persona_id, sample_index)
        for variant_id in variants
        for persona_id in personas
        for sample_index in range(n_samples)
    }

    by_key: dict[tuple[str, str, int], dict] = {}
    issues: list[dict] = []
    for record in records:
        key = (
            str(record.get("variant_id")),
            str(record.get("persona_id")),
            int(record.get("sample_index", -1)),
        )
        if key in expected_keys:
            by_key[key] = record

    missing_keys = sorted(expected_keys - set(by_key))
    for variant_id, persona_id, sample_index in missing_keys:
        issues.append(
            {
                "code": "missing_sample",
                "variant_id": variant_id,
                "persona_id": persona_id,
                "sample_index": sample_index,
                "message": "missing expected reaction sample",
            }
        )

    sample_scores: dict[tuple[str, str], list[dict]] = defaultdict(list)
    variant_reactions: dict[str, list[str]] = defaultdict(list)
    reaction_texts_for_embeddings: list[str] = []
    reaction_embedding_keys: list[tuple[str, str, int]] = []

    for key in sorted(expected_keys):
        record = by_key.get(key)
        if not record:
            continue
        reaction = str(record.get("reaction", "")).strip()
        variant_id, persona_id, sample_index = key
        if len(reaction) < MIN_REACTION_LENGTH:
            issues.append(
                {
                    "code": "degenerate_reaction",
                    "variant_id": variant_id,
                    "persona_id": persona_id,
                    "sample_index": sample_index,
                    "message": "reaction is empty or too short",
                }
            )
            continue

        score, distribution, method = calculate_ssr_score(
            reaction,
            reference_statements,
            temperature=dist_temperature,
        )
        sample_scores[(variant_id, persona_id)].append(
            {
                "sample_index": sample_index,
                "score": score,
                "distribution": distribution,
                "scoring_method": method,
                "reaction": reaction,
            }
        )
        variant_reactions[variant_id].append(reaction)
        reaction_texts_for_embeddings.append(reaction)
        reaction_embedding_keys.append(key)

    embeddings_by_key = {}
    if reaction_texts_for_embeddings:
        embeddings = get_embeddings_batch(reaction_texts_for_embeddings)
        embeddings_by_key = dict(zip(reaction_embedding_keys, embeddings))

    per_persona_variant: dict[str, dict] = {}
    for variant_id in variants:
        for persona_id in personas:
            group_key = (variant_id, persona_id)
            samples = sample_scores.get(group_key, [])
            if len(samples) != n_samples:
                continue

            scores = [sample["score"] for sample in samples]
            distributions = [sample["distribution"] for sample in samples]
            score_spread = max(scores) - min(scores)
            sample_embedding_keys = [
                (variant_id, persona_id, sample["sample_index"]) for sample in samples
            ]
            pairwise_distances = []
            for i, left_key in enumerate(sample_embedding_keys):
                for right_key in sample_embedding_keys[i + 1 :]:
                    left = embeddings_by_key.get(left_key)
                    right = embeddings_by_key.get(right_key)
                    if left is not None and right is not None:
                        pairwise_distances.append(1 - _cosine_similarity(left, right))
            embedding_distance = (
                sum(pairwise_distances) / len(pairwise_distances)
                if pairwise_distances
                else math.inf
            )

            if (
                score_spread < COLLAPSE_SCORE_SPREAD_FLOOR
                and embedding_distance < COLLAPSE_EMBEDDING_DISTANCE_FLOOR
            ):
                issues.append(
                    {
                        "code": "collapsed_samples",
                        "variant_id": variant_id,
                        "persona_id": persona_id,
                        "score_spread": score_spread,
                        "embedding_distance": embedding_distance,
                        "message": "samples are near-duplicates; re-run with independent draws",
                    }
                )

            per_persona_variant[f"{variant_id}:{persona_id}"] = {
                "variant_id": variant_id,
                "persona_id": persona_id,
                "score": sum(scores) / len(scores),
                "score_label": "directional",
                "distribution": _average_distributions(distributions),
                "sample_scores": scores,
                "score_spread": score_spread,
                "embedding_distance": embedding_distance,
            }

    per_variant = {}
    for variant_id, copy_text in variants.items():
        groups = [
            per_persona_variant.get(f"{variant_id}:{persona_id}")
            for persona_id in personas
        ]
        groups = [group for group in groups if group]
        if groups:
            variant_score = sum(group["score"] for group in groups) / len(groups)
            per_variant[variant_id] = {
                "variant_id": variant_id,
                "copy": copy_text,
                "score": variant_score,
                "score_label": "directional",
                "reasoning_excerpt": _reasoning_excerpt(variant_reactions[variant_id]),
                "persona_scores": {
                    group["persona_id"]: group["score"] for group in groups
                },
            }
        else:
            per_variant[variant_id] = {
                "variant_id": variant_id,
                "copy": copy_text,
                "score": None,
                "score_label": "directional",
                "reasoning_excerpt": "",
                "persona_scores": {},
            }

    ranking = [
        {"variant_id": variant_id, "score": data["score"], "rank": rank}
        for rank, (variant_id, data) in enumerate(
            sorted(
                (
                    (variant_id, data)
                    for variant_id, data in per_variant.items()
                    if data["score"] is not None
                ),
                key=lambda item: item[1]["score"],
                reverse=True,
            ),
            start=1,
        )
    ]

    verdict = "not_comparable" if issues else "comparable"
    verdict_message = (
        "Comparable directional ranking. Absolute scores are directional only."
        if verdict == "comparable"
        else "Not comparable; ranking is unreliable. Re-run with complete, independent draws."
    )

    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "stimulus_type": payload.get("stimulus_type", DEFAULT_STIMULUS_TYPE),
        "n_samples": n_samples,
        "dist_temperature": dist_temperature,
        "score_scale": "1-5 directional",
        "comparability": {
            "verdict": verdict,
            "message": verdict_message,
            "issues": issues,
        },
        "per_variant": per_variant,
        "per_persona_variant": per_persona_variant,
        "ranking": ranking,
    }


def _resolve_test_personas(
    personas_data: dict, requested_persona: str | None
) -> list[str]:
    """Resolve one requested persona or every persona in the loaded file."""
    return resolve_persona_names(personas_data, requested_persona)


# =============================================================================
# Single Variation Testing
# =============================================================================


def test_variation(
    client_info: tuple,
    persona_data: dict,
    stimulus_type: str,
    content: str,
    reference_statements: dict,
    n_samples: int = None,
    context: str = "",
) -> dict:
    """
    Test a single variation and return results.

    Per SSR paper methodology (arXiv:2510.08338), generates n response samples
    and averages scores/distributions for stability. Default n=2.

    Args:
        client_info: Tuple of (provider_name, client_instance)
        persona_data: Persona definition dictionary
        stimulus_type: Type of stimulus (headline, email, etc.)
        content: The content variation to test
        reference_statements: SSR reference anchor statements
        n_samples: Number of samples to average (default: SSR_N_SAMPLES env or 2)
        context: Optional background context for the stimulus prompt

    Returns:
        Dict with content, response, all_responses, score, distribution, scoring_method
    """
    if n_samples is None:
        n_samples = int(os.environ.get("SSR_N_SAMPLES", "2"))

    persona_prompt = build_persona_prompt(persona_data)
    stimulus_prompt = get_stimulus_prompt(stimulus_type, content, context)

    # Generate n samples and average (paper uses n=2)
    all_scores = []
    all_probs = []
    all_responses = []
    scoring_methods = []
    failed_samples = 0

    for _ in range(n_samples):
        response = generate_response(client_info, persona_prompt, stimulus_prompt)
        try:
            score, probs, method = calculate_ssr_score(response, reference_statements)
            all_scores.append(score)
            all_probs.append(probs)
            all_responses.append(response)
            scoring_methods.append(method)
        except SSRScoringError:
            failed_samples += 1
            all_responses.append(response)

    if not all_scores:
        raise SSRScoringError(
            f"All {n_samples} samples failed SSR scoring for: {content[:50]}..."
        )

    if failed_samples > 0:
        console.print(
            f"[yellow]{failed_samples}/{n_samples} samples failed SSR scoring, "
            f"averaging {len(all_scores)} successful samples[/yellow]"
        )

    # Average scores and distributions
    avg_score = sum(all_scores) / len(all_scores)
    avg_probs = [sum(p[i] for p in all_probs) / len(all_probs) for i in range(5)]

    # Scoring method: "ssr" if all succeeded, "ssr_partial" if some failed
    final_method = "ssr" if failed_samples == 0 else "ssr_partial"

    # Return first response for display (representative sample)
    return {
        "content": content,
        "response": all_responses[0],  # Primary response for display
        "all_responses": all_responses,  # All samples for storage
        "score": avg_score,
        "distribution": avg_probs,
        "scoring_method": final_method,
    }


async def test_variation_async(
    client_info: tuple,
    persona_data: dict,
    stimulus_type: str,
    content: str,
    reference_statements: dict,
    n_samples: int = None,
    semaphore: asyncio.Semaphore | None = None,
    context: str = "",
) -> dict:
    """
    Test a single variation asynchronously with concurrent sample generation.

    Per SSR paper methodology (arXiv:2510.08338), generates n response samples
    concurrently and averages scores/distributions for stability. Default n=2.

    Args:
        client_info: Tuple of (provider_name, async_client_instance)
        persona_data: Persona definition dictionary
        stimulus_type: Type of stimulus (headline, email, etc.)
        content: The content variation to test
        reference_statements: SSR reference anchor statements
        n_samples: Number of samples to average (default: SSR_N_SAMPLES env or 2)
        semaphore: Optional semaphore for rate limiting
        context: Optional background context for the stimulus prompt

    Returns:
        Dict with content, response, all_responses, score, distribution, scoring_method
    """
    if n_samples is None:
        n_samples = int(os.environ.get("SSR_N_SAMPLES", "2"))

    persona_prompt = build_persona_prompt(persona_data)
    stimulus_prompt = get_stimulus_prompt(stimulus_type, content, context)

    # Generate n samples concurrently
    async def generate_and_score():
        response = await generate_response_async(
            client_info, persona_prompt, stimulus_prompt, semaphore
        )
        # Note: calculate_ssr_score is CPU-bound, runs synchronously
        try:
            score, probs, method = calculate_ssr_score(response, reference_statements)
            return response, score, probs, method
        except SSRScoringError:
            return response, None, None, "failed"

    # Run all samples concurrently
    results = await asyncio.gather(*[generate_and_score() for _ in range(n_samples)])

    all_responses = [r[0] for r in results]
    successful = [(r[1], r[2], r[3]) for r in results if r[1] is not None]
    failed_samples = len(results) - len(successful)

    if not successful:
        raise SSRScoringError(
            f"All {n_samples} samples failed SSR scoring for: {content[:50]}..."
        )

    if failed_samples > 0:
        console.print(
            f"[yellow]{failed_samples}/{n_samples} samples failed SSR scoring, "
            f"averaging {len(successful)} successful samples[/yellow]"
        )

    all_scores = [s[0] for s in successful]
    all_probs = [s[1] for s in successful]

    # Average scores and distributions
    avg_score = sum(all_scores) / len(all_scores)
    avg_probs = [sum(p[i] for p in all_probs) / len(all_probs) for i in range(5)]

    final_method = "ssr" if failed_samples == 0 else "ssr_partial"

    return {
        "content": content,
        "response": all_responses[0],  # Primary response for display
        "all_responses": all_responses,  # All samples for storage
        "score": avg_score,
        "distribution": avg_probs,
        "scoring_method": final_method,
    }


# =============================================================================
# Batch API Support (50% cost reduction for large test runs)
# =============================================================================


def _check_batch_api_available() -> bool:
    """Check if Anthropic batch API is available (requires anthropic SDK)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic

        # Check SDK version supports batches (added in late 2024)
        return hasattr(anthropic.Anthropic().messages, "batches")
    except (ImportError, AttributeError):
        return False


def _create_batch_requests(
    personas_data: dict,
    test_personas: list[str],
    stimulus_type: str,
    variations: list[str],
    n_samples: int,
    skip_indices: dict[str, set[int]] | None = None,
    context: str = "",
) -> list[dict]:
    """
    Create batch request objects for all persona-variation-sample combinations.

    Args:
        personas_data: Loaded personas configuration
        test_personas: List of persona names to test
        stimulus_type: Type of stimulus (headline, email, etc.)
        variations: List of content variations to test
        n_samples: Number of samples per variation
        skip_indices: Dict mapping persona name -> set of variation indices to skip (cached)
        context: Optional background context for the stimulus prompt

    Returns:
        List of dicts with custom_id and params for batch API
    """
    model = os.environ.get("SSR_MODEL", "agent-provided")
    requests = []
    skip_indices = skip_indices or {}

    for persona_name in test_personas:
        persona_data = personas_data[persona_name]
        persona_prompt = build_persona_prompt(persona_data)
        persona_skip = skip_indices.get(persona_name, set())

        for var_idx, variation in enumerate(variations):
            # Skip cached variations
            if var_idx in persona_skip:
                continue

            stimulus_prompt = get_stimulus_prompt(stimulus_type, variation, context)
            full_prompt = f"{persona_prompt}\n\n---\n\n{stimulus_prompt}"

            for sample_idx in range(n_samples):
                # custom_id must match ^[a-zA-Z0-9_-]{1,64}$
                requests.append(
                    {
                        "custom_id": f"{persona_name}_v{var_idx}_s{sample_idx}",
                        "params": {
                            "model": model,
                            "max_tokens": 1024,
                            "messages": [{"role": "user", "content": full_prompt}],
                        },
                    }
                )

    return requests


def _submit_batch(requests: list[dict]) -> tuple[str, "anthropic.types.MessageBatch"]:
    """
    Submit batch to Anthropic API (non-blocking).

    Args:
        requests: List of batch request objects

    Returns:
        Tuple of (batch_id, batch_object)
    """
    import anthropic

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    return batch.id, batch


async def _poll_batch(
    batch_id: str, poll_interval: int = 30, quiet: bool = False
) -> "anthropic.types.MessageBatch":
    """
    Poll a batch until completion.

    Args:
        batch_id: The batch ID to poll
        poll_interval: Seconds between status checks (default: 30)
        quiet: If True, suppress progress output

    Returns:
        Completed batch object with results_url
    """
    import anthropic

    client = anthropic.Anthropic()

    while True:
        batch = client.messages.batches.retrieve(batch_id)

        if not quiet:
            counts = batch.request_counts
            console.print(
                f"[dim]{batch_id[:20]}... "
                f"P:{counts.processing} S:{counts.succeeded} E:{counts.errored}[/dim]",
                end="\r",
            )

        if batch.processing_status == "ended":
            break

        await asyncio.sleep(poll_interval)

    return batch


async def _submit_and_poll_batch(
    requests: list[dict], poll_interval: int = 30
) -> "anthropic.types.MessageBatch":
    """
    Submit batch to Anthropic API and poll until completion.

    Args:
        requests: List of batch request objects
        poll_interval: Seconds between status checks (default: 30)

    Returns:
        Completed batch object with results_url
    """
    # Submit batch
    console.print(f"[cyan]Submitting batch of {len(requests)} requests...[/cyan]")
    batch_id, batch = _submit_batch(requests)

    console.print(f"[green]Batch submitted: {batch_id}[/green]")
    console.print(f"[dim]Expires: {batch.expires_at}[/dim]")

    # Poll until complete
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing batch...", total=None)

        while True:
            batch = await _poll_batch(batch_id, poll_interval, quiet=True)

            counts = batch.request_counts
            status_msg = (
                f"Processing: {counts.processing} | "
                f"Succeeded: {counts.succeeded} | "
                f"Errored: {counts.errored}"
            )
            progress.update(task, description=status_msg)

            if batch.processing_status == "ended":
                break

            await asyncio.sleep(poll_interval)

    # Report final status
    console.print("\n[bold green]Batch complete![/bold green]")
    console.print(f"  Succeeded: {batch.request_counts.succeeded}")
    if batch.request_counts.errored > 0:
        console.print(f"  [red]Errored: {batch.request_counts.errored}[/red]")
    if batch.request_counts.expired > 0:
        console.print(f"  [yellow]Expired: {batch.request_counts.expired}[/yellow]")

    return batch


def _retrieve_batch_results(batch) -> dict[str, str]:
    """
    Retrieve results from completed batch using SDK's built-in streaming.

    Args:
        batch: Completed batch object

    Returns:
        Dict mapping custom_id to response text
    """
    import anthropic

    # Check batch has results
    if not batch.results_url:
        console.print("[red]Batch has no results URL - may have failed entirely[/red]")
        return {}

    client = anthropic.Anthropic()
    results = {}

    console.print("[dim]Retrieving results...[/dim]")

    # Use SDK's built-in streaming results method
    for result in client.messages.batches.results(batch.id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            message = result.result.message
            if message.content and message.content[0].type == "text":
                results[custom_id] = message.content[0].text
        elif result.result.type == "errored":
            error = result.result.error
            console.print(f"[yellow]Request {custom_id} errored: {error.type}[/yellow]")
        elif result.result.type == "expired":
            console.print(f"[yellow]Request {custom_id} expired[/yellow]")
        elif result.result.type == "canceled":
            console.print(f"[dim]Request {custom_id} canceled[/dim]")

    return results


def _process_batch_results(
    results: dict[str, str],
    personas_data: dict,
    test_personas: list[str],
    variations: list[str],
    reference_statements: dict,
    n_samples: int,
    cached_results: dict[str, dict[int, dict]] | None = None,
) -> dict[str, list[dict]]:
    """
    Process batch results into scored variation results by persona.

    Args:
        results: Dict mapping custom_id to response text
        personas_data: Full persona definitions
        test_personas: List of persona names tested
        variations: List of variation strings
        reference_statements: SSR reference statements
        n_samples: Number of samples per variation
        cached_results: Optional dict of persona -> var_idx -> cached result

    Returns:
        Dict mapping persona name to sorted list of result dicts
    """
    all_results = {}
    cached_results = cached_results or {}

    for persona_name in test_personas:
        persona_results = []
        persona_cached = cached_results.get(persona_name, {})

        for var_idx, variation in enumerate(variations):
            # Check if this variation was cached
            if var_idx in persona_cached:
                persona_results.append(persona_cached[var_idx])
                continue

            # Collect all samples for this variation from batch results
            samples = []
            for sample_idx in range(n_samples):
                custom_id = f"{persona_name}_v{var_idx}_s{sample_idx}"
                response_text = results.get(custom_id)
                if response_text:
                    samples.append(response_text)

            if not samples:
                # All samples failed for this variation
                console.print(
                    f"[red]No results for {persona_name}:{variation[:30]}...[/red]"
                )
                persona_results.append(
                    {
                        "content": variation,
                        "response": "Error: No batch results",
                        "all_responses": [],
                        "score": 0.0,
                        "distribution": [0.2] * 5,
                    }
                )
                continue

            # Calculate SSR scores for each sample
            scores = []
            distributions = []
            failed_samples = 0
            for response_text in samples:
                try:
                    score, probs, _method = calculate_ssr_score(
                        response_text, reference_statements
                    )
                    scores.append(score)
                    distributions.append(probs)
                except SSRScoringError:
                    failed_samples += 1

            if not scores:
                console.print(
                    f"[red]All samples failed SSR scoring for "
                    f"{persona_name}:{variation[:30]}...[/red]"
                )
                persona_results.append(
                    {
                        "content": variation,
                        "response": samples[0],
                        "all_responses": samples,
                        "score": 0.0,
                        "distribution": [0.2] * 5,
                        "scoring_method": "failed",
                    }
                )
                continue

            if failed_samples > 0:
                console.print(
                    f"[yellow]{failed_samples}/{len(samples)} samples failed SSR "
                    f"for {persona_name}:{variation[:30]}...[/yellow]"
                )

            # Average across samples
            avg_score = sum(scores) / len(scores)
            avg_probs = [
                sum(d[i] for d in distributions) / len(distributions) for i in range(5)
            ]

            method = "ssr" if failed_samples == 0 else "ssr_partial"
            persona_results.append(
                {
                    "content": variation,
                    "response": samples[0],  # Primary response for display
                    "all_responses": samples,
                    "score": avg_score,
                    "distribution": avg_probs,
                    "scoring_method": method,
                }
            )

        # Sort by score descending
        persona_results.sort(key=lambda x: x["score"], reverse=True)
        all_results[persona_name] = persona_results

    return all_results


def prepare_batch_context(
    test_config: dict,
    persona_filter: Optional[str] = None,
    n_samples: int = None,
    use_cache: bool = True,
) -> dict:
    """
    Prepare batch context for submission without actually submitting.

    Used by batch_runner.py to submit multiple batches in parallel.

    Args:
        test_config: Test configuration dictionary
        persona_filter: Override persona from config
        n_samples: Override n_samples (None = use env/default)
        use_cache: Whether to check cache (default: True)

    Returns:
        Dict with keys: requests, test_personas, stimulus_type, variations,
        reference_statements, n_samples, test_id, test_name, personas_data,
        cached_results, skip_indices, cache_hits
    """
    personas_data = load_personas()

    # Determine which personas to test
    requested_persona = persona_filter or test_config.get("persona")
    test_personas = _resolve_test_personas(personas_data, requested_persona)

    stimulus_type = test_config.get("stimulus_type", "headline")
    variations = test_config.get("variations", [])
    context = test_config.get("context", "")
    reference_statements = (
        test_config.get("reference_statements") or load_default_anchors()
    )

    test_id = (
        test_config.get("test_id")
        or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    test_name = test_config.get("name", "Unnamed Test")

    if n_samples is None:
        n_samples = int(os.environ.get("SSR_N_SAMPLES", "2"))

    # Check cache for each variation/persona combination
    cached_results: dict[str, dict[int, dict]] = {}
    skip_indices: dict[str, set[int]] = {}
    cache_hits = 0

    if use_cache:
        fingerprint = get_config_fingerprint()
        fingerprint_str = fingerprint.get("fingerprint", "")

        for persona_name in test_personas:
            cached_results[persona_name] = {}
            skip_indices[persona_name] = set()

            for var_idx, variation in enumerate(variations):
                cached = get_cached_result(
                    content=variation,
                    test_type=stimulus_type,
                    personas=[persona_name],
                    current_fingerprint=fingerprint_str,
                )
                if cached:
                    cached_results[persona_name][var_idx] = cached
                    skip_indices[persona_name].add(var_idx)
                    cache_hits += 1

    # Create batch requests (skipping cached variations)
    requests = _create_batch_requests(
        personas_data,
        test_personas,
        stimulus_type,
        variations,
        n_samples,
        skip_indices=skip_indices,
        context=context,
    )

    return {
        "requests": requests,
        "test_personas": test_personas,
        "stimulus_type": stimulus_type,
        "variations": variations,
        "context": context,
        "reference_statements": reference_statements,
        "n_samples": n_samples,
        "test_id": test_id,
        "test_name": test_name,
        "personas_data": personas_data,
        "cached_results": cached_results,
        "skip_indices": skip_indices,
        "cache_hits": cache_hits,
    }


# =============================================================================
# Full Test Execution
# =============================================================================


async def run_test_batch(
    test_config: dict,
    persona_filter: Optional[str] = None,
    track: bool = False,
    n_samples: int = None,
    poll_interval: int = 30,
    use_cache: bool = True,
) -> dict:
    """
    Run test using Anthropic Message Batches API (50% cost reduction).

    Similar to run_test_async but submits all requests as a single batch,
    polls for completion, then processes results. Better for large test runs
    where real-time feedback isn't needed.

    Args:
        test_config: Test configuration dictionary
        persona_filter: Override persona from config
        track: Whether to track predictions for calibration
        n_samples: Override n_samples (None = use env/default)
        poll_interval: Seconds between status checks (default: 30)
        use_cache: Whether to use cached results (default: True)

    Returns:
        Dict mapping persona name to sorted list of result dicts
    """
    if not _check_batch_api_available():
        console.print(
            "[red]Batch API requires Anthropic SDK with ANTHROPIC_API_KEY[/red]"
        )
        console.print("[yellow]Falling back to parallel async mode...[/yellow]")
        return await run_test_async(test_config, persona_filter, track)

    personas_data = load_personas()

    # Determine which personas to test
    requested_persona = persona_filter or test_config.get("persona")
    test_personas = _resolve_test_personas(personas_data, requested_persona)

    stimulus_type = test_config.get("stimulus_type", "headline")
    variations = test_config.get("variations", [])
    context = test_config.get("context", "")
    reference_statements = (
        test_config.get("reference_statements") or load_default_anchors()
    )

    # Generate test ID for tracking
    test_id = (
        test_config.get("test_id")
        or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    test_name = test_config.get("name", "Unnamed Test")

    if n_samples is None:
        n_samples = int(os.environ.get("SSR_N_SAMPLES", "2"))

    # Check cache for each variation/persona combination
    cached_results: dict[str, dict[int, dict]] = {}  # persona -> var_idx -> result
    skip_indices: dict[str, set[int]] = {}  # persona -> set of var_idx to skip
    cache_hits = 0

    if use_cache:
        fingerprint = get_config_fingerprint()
        fingerprint_str = fingerprint.get("fingerprint", "")

        for persona_name in test_personas:
            cached_results[persona_name] = {}
            skip_indices[persona_name] = set()

            for var_idx, variation in enumerate(variations):
                cached = get_cached_result(
                    content=variation,
                    test_type=stimulus_type,
                    personas=[persona_name],
                    current_fingerprint=fingerprint_str,
                )
                if cached:
                    cached_results[persona_name][var_idx] = cached
                    skip_indices[persona_name].add(var_idx)
                    cache_hits += 1

    # Calculate actual requests needed (excluding cached)
    total_possible = len(test_personas) * len(variations) * n_samples
    total_requests = sum(
        (len(variations) - len(skip_indices.get(p, set()))) * n_samples
        for p in test_personas
    )
    estimated_savings = total_requests * 0.005  # ~$0.01 per call, 50% off

    cache_info = (
        f"Cache: [green]{cache_hits} hits[/green], {total_requests // n_samples} API calls"
        if cache_hits > 0
        else ""
    )

    console.print(
        Panel(
            f"[bold]Batch Testing: {test_name}[/bold]\n"
            f"Test ID: {test_id}\n"
            f"Stimulus type: {stimulus_type}\n"
            f"Variations: {len(variations)}\n"
            f"Personas: {', '.join(test_personas)}\n"
            f"Samples per variation: {n_samples}\n"
            f"Total API requests: {total_requests} (of {total_possible} possible)\n"
            f"Mode: [green]BATCH[/green] (50% cost reduction)\n"
            f"Est. savings: ~${estimated_savings:.2f}\n"
            f"{cache_info}\n"
            f"Tracking: {'[green]enabled[/green]' if track else '[dim]disabled[/dim]'}",
            title="Batch Configuration",
        )
    )

    # If everything is cached, skip the batch API entirely
    if total_requests == 0:
        console.print("[green]All variations cached - no API calls needed![/green]")
        all_results = {}
        for persona_name in test_personas:
            results_list = []
            for var_idx in range(len(variations)):
                if var_idx in cached_results.get(persona_name, {}):
                    results_list.append(cached_results[persona_name][var_idx])
            # Sort by score descending
            results_list.sort(key=lambda x: x.get("score", 0), reverse=True)
            all_results[persona_name] = results_list

        # Display and return cached results
        for persona_name in test_personas:
            persona_data = personas_data[persona_name]
            display_results(
                persona_name, persona_data["name"], all_results[persona_name]
            )
        return all_results

    # Create batch requests (skipping cached variations)
    requests = _create_batch_requests(
        personas_data,
        test_personas,
        stimulus_type,
        variations,
        n_samples,
        skip_indices=skip_indices,
        context=context,
    )

    # Submit and poll
    batch = await _submit_and_poll_batch(requests, poll_interval)

    # Retrieve results
    results = _retrieve_batch_results(batch)

    all_results = _process_batch_results(
        results,
        personas_data,
        test_personas,
        variations,
        reference_statements,
        n_samples,
        cached_results=cached_results,
    )

    # Display results
    for persona_name in test_personas:
        persona_data = personas_data[persona_name]
        display_results(persona_name, persona_data["name"], all_results[persona_name])

    # Save responses
    save_responses(
        test_id=test_id,
        test_name=test_name,
        test_type=stimulus_type,
        all_results=all_results,
    )

    # Track predictions for calibration if requested
    if track:
        track_predictions(
            test_id=test_id,
            test_name=test_name,
            test_type=stimulus_type,
            personas_tested=test_personas,
            all_results=all_results,
        )

    return all_results


async def run_test_async(
    test_config: dict,
    persona_filter: Optional[str] = None,
    track: bool = False,
    max_concurrent: int = 10,
    n_samples: int = None,
    use_cache: bool = True,
) -> dict:
    """
    Run a full test suite asynchronously with concurrent variation testing.

    Args:
        test_config: Test configuration dictionary
        persona_filter: Override persona from config
        track: Whether to track predictions for calibration
        max_concurrent: Maximum concurrent API calls (default: 10)
        n_samples: Override n_samples (None = use env/default)
        use_cache: Use cached results when available (default: True)

    Returns:
        Dict mapping persona name to sorted list of result dicts
    """
    personas_data = load_personas()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Determine which personas to test
    requested_persona = persona_filter or test_config.get("persona")
    test_personas = _resolve_test_personas(personas_data, requested_persona)

    stimulus_type = test_config.get("stimulus_type", "headline")
    variations = test_config.get("variations", [])
    context = test_config.get("context", "")
    reference_statements = (
        test_config.get("reference_statements") or load_default_anchors()
    )

    # Generate test ID for tracking
    test_id = (
        test_config.get("test_id") or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    test_name = test_config.get("name", "Unnamed Test")

    # Get config fingerprint for cache lookup
    config_fp = get_config_fingerprint(test_personas)
    fingerprint_str = config_fp.get("fingerprint", "")

    n_samples_display = n_samples or int(os.environ.get("SSR_N_SAMPLES", "2"))
    cache_status = (
        "[green]enabled[/green]" if use_cache else "[yellow]disabled (--force)[/yellow]"
    )
    console.print(
        Panel(
            f"[bold]Testing: {test_name}[/bold]\n"
            f"Test ID: {test_id}\n"
            f"Stimulus type: {stimulus_type}\n"
            f"Variations: {len(variations)}\n"
            f"Personas: {', '.join(test_personas)}\n"
            f"Mode: [green]parallel[/green] (max {max_concurrent} concurrent)\n"
            f"Samples per variation: {n_samples_display}\n"
            f"Cache: {cache_status}\n"
            f"Tracking: {'[green]enabled[/green]' if track else '[dim]disabled[/dim]'}",
            title="Test Configuration",
        )
    )

    all_results = {}

    # Create async client with proper lifecycle management
    provider, client = get_async_llm_client()
    client_info = (provider, client)

    try:
        for persona_name in test_personas:
            if persona_name not in personas_data:
                console.print(f"[red]Unknown persona: {persona_name}[/red]")
                continue

            persona_data = personas_data[persona_name]
            console.print(
                f"\n[bold cyan]Testing with persona: {persona_data['name']}[/bold cyan]"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Testing {len(variations)} variations concurrently...", total=None
                )

                # Track cache hits for this persona
                cache_hits = 0
                cache_misses = 0

                # Run all variations concurrently with error handling
                async def test_with_index(idx: int, variation: str):
                    nonlocal cache_hits, cache_misses
                    try:
                        # Check cache first if enabled
                        if use_cache:
                            cached = get_cached_result(
                                content=variation,
                                test_type=stimulus_type,
                                personas=[persona_name],
                                current_fingerprint=fingerprint_str,
                            )
                            if cached:
                                cache_hits += 1
                                return idx, cached, None

                        # Cache miss - run actual LLM call
                        cache_misses += 1
                        result = await test_variation_async(
                            client_info,
                            persona_data,
                            stimulus_type,
                            variation,
                            reference_statements,
                            n_samples=n_samples,
                            semaphore=semaphore,
                            context=context,
                        )
                        return idx, result, None
                    except Exception as e:
                        return idx, None, e

                indexed_results = await asyncio.gather(
                    *[test_with_index(i, v) for i, v in enumerate(variations)]
                )

                # Show cache stats if any hits
                if cache_hits > 0:
                    console.print(
                        f"[dim]Cache: {cache_hits} hits, {cache_misses} API calls[/dim]"
                    )

                progress.update(task, description="Done!")

            # Process results, handling any errors
            results = []
            for idx, result, error in sorted(indexed_results, key=lambda x: x[0]):
                if error:
                    console.print(f"[red]Error testing variation {idx}: {error}[/red]")
                    # Create a placeholder result for failed variations
                    results.append(
                        {
                            "content": variations[idx],
                            "response": f"Error: {error}",
                            "score": 0.0,
                            "distribution": [0.2] * 5,
                            "scoring_method": "failed",
                        }
                    )
                else:
                    results.append(result)

            results.sort(key=lambda x: x["score"], reverse=True)
            all_results[persona_name] = results

            # Display results
            display_results(persona_name, persona_data["name"], results)
    finally:
        # Properly close async client to prevent resource leaks
        if "client" in locals() and hasattr(client, "close"):
            await client.close()

    # Always save responses (we're paying for them!)
    save_responses(
        test_id=test_id,
        test_name=test_name,
        test_type=stimulus_type,
        all_results=all_results,
    )

    # Track predictions for calibration if requested
    if track:
        track_predictions(
            test_id=test_id,
            test_name=test_name,
            test_type=stimulus_type,
            personas_tested=test_personas,
            all_results=all_results,
        )

    return all_results


def run_test(
    test_config: dict, persona_filter: Optional[str] = None, track: bool = False
):
    """
    Run a full test suite from config (sequential mode).

    Args:
        test_config: Test configuration dictionary
        persona_filter: Override persona from config
        track: Whether to track predictions for calibration

    Returns:
        Dict mapping persona name to sorted list of result dicts
    """
    personas_data = load_personas()
    client_info = get_llm_client()

    # Determine which personas to test
    requested_persona = persona_filter or test_config.get("persona")
    test_personas = _resolve_test_personas(personas_data, requested_persona)

    stimulus_type = test_config.get("stimulus_type", "headline")
    variations = test_config.get("variations", [])
    context = test_config.get("context", "")
    reference_statements = (
        test_config.get("reference_statements") or load_default_anchors()
    )

    # Generate test ID for tracking
    test_id = (
        test_config.get("test_id") or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    test_name = test_config.get("name", "Unnamed Test")

    console.print(
        Panel(
            f"[bold]Testing: {test_name}[/bold]\n"
            f"Test ID: {test_id}\n"
            f"Stimulus type: {stimulus_type}\n"
            f"Variations: {len(variations)}\n"
            f"Personas: {', '.join(test_personas)}\n"
            f"Tracking: {'[green]enabled[/green]' if track else '[dim]disabled[/dim]'}",
            title="Test Configuration",
        )
    )

    all_results = {}

    for persona_name in test_personas:
        if persona_name not in personas_data:
            console.print(f"[red]Unknown persona: {persona_name}[/red]")
            continue

        persona_data = personas_data[persona_name]
        console.print(
            f"\n[bold cyan]Testing with persona: {persona_data['name']}[/bold cyan]"
        )

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Testing variations...", total=len(variations))

            for variation in variations:
                progress.update(task, description=f"Testing: {variation[:50]}...")
                result = test_variation(
                    client_info,
                    persona_data,
                    stimulus_type,
                    variation,
                    reference_statements,
                    context=context,
                )
                results.append(result)
                progress.advance(task)

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        all_results[persona_name] = results

        # Display results
        display_results(persona_name, persona_data["name"], results)

    # Always save responses (we're paying for them!)
    save_responses(
        test_id=test_id,
        test_name=test_name,
        test_type=stimulus_type,
        all_results=all_results,
    )

    # Track predictions for calibration if requested
    if track:
        track_predictions(
            test_id=test_id,
            test_name=test_name,
            test_type=stimulus_type,
            personas_tested=test_personas,
            all_results=all_results,
        )

    return all_results


# =============================================================================
# Display and Storage
# =============================================================================


def display_results(persona_name: str, persona_full_name: str, results: list):
    """Display test results in a formatted table."""
    from rich.table import Table

    table = Table(title=f"Results for {persona_full_name}")
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Variation", style="white", max_width=50)
    table.add_column("Score", style="green", width=6)
    table.add_column("Confidence", style="yellow", width=12)

    for i, result in enumerate(results, 1):
        score = result["score"]
        # Visual confidence bar
        filled = int((score - 1) / 4 * 10)
        bar = "█" * filled + "░" * (10 - filled)

        # Truncate variation for display
        variation = result["content"]
        if len(variation) > 47:
            variation = variation[:47] + "..."

        table.add_row(str(i), variation, f"{score:.2f}", bar)

    console.print(table)

    # Show top performer rationale
    if results:
        top = results[0]
        console.print(
            Panel(
                top["response"][:500] + ("..." if len(top["response"]) > 500 else ""),
                title="[bold green]Top Performer Rationale[/bold green]",
                subtitle=f"Score: {top['score']:.2f}",
            )
        )


def save_responses(
    test_id: str, test_name: str, test_type: str, all_results: dict[str, list[dict]]
) -> None:
    """
    Save all persona responses to disk for future analysis and UI display.

    Stores each persona's response to each variation with full metadata.
    """
    responses = load_responses()
    timestamp = datetime.now().isoformat()

    for persona_name, results in all_results.items():
        for i, result in enumerate(results):
            # Generate content hash for linking to calibration records
            content = result["content"]
            content_hash = hex(hash(content) & 0xFFFFFFFFFFFFFFFF)[2:]

            response_record = {
                "id": str(uuid.uuid4())[:8],
                "test_id": test_id,
                "test_name": test_name,
                "test_type": test_type,
                "created_at": timestamp,
                "persona": persona_name,
                "variation_index": i,
                "variation_content": content,
                "responses": result.get("all_responses", [result.get("response", "")]),
                "score": result["score"],
                "distribution": result.get("distribution", []),
                "content_hash": content_hash,
            }
            responses.append(response_record)

    save_responses_to_disk(responses)
    console.print(
        f"[dim]Saved {sum(len(r) for r in all_results.values())} responses to {RESPONSES_PATH_GZ.name}[/dim]"
    )


def track_predictions(
    test_id: str,
    test_name: str,
    test_type: str,
    personas_tested: list[str],
    all_results: dict,
) -> None:
    """Record predictions to calibration store for later validation."""
    try:
        from calibration import CalibrationStore
    except ImportError:
        console.print("[yellow]Warning: calibration module not available[/yellow]")
        return

    store = CalibrationStore()

    # Generate config fingerprint for version tracking
    config_metadata = {"config": get_config_fingerprint(personas_tested)}

    # Aggregate scores across personas for each variation
    variation_scores: dict[str, list[float]] = {}
    variation_content: dict[str, str] = {}
    variation_distributions: dict[str, list[list[float]]] = {}

    for persona_name, results in all_results.items():
        for result in results:
            content = result["content"]
            if content not in variation_scores:
                variation_scores[content] = []
                variation_content[content] = content
                variation_distributions[content] = []

            variation_scores[content].append(result["score"])
            variation_distributions[content].append(
                result.get("distribution", [0.2] * 5)
            )

    # Compute average scores and prepare for recording
    aggregated_results = []
    for i, (content, scores) in enumerate(variation_scores.items()):
        avg_score = sum(scores) / len(scores)

        # Average distributions
        dists = variation_distributions[content]
        avg_dist = [sum(d[j] for d in dists) / len(dists) for j in range(5)]

        aggregated_results.append(
            {
                "variation_id": f"v{i + 1}",
                "content": content,
                "score": avg_score,
                "distribution": avg_dist,
            }
        )

    # Record to calibration store with config fingerprint
    records = store.record_test_predictions(
        test_id=test_id,
        test_name=test_name,
        test_type=test_type,
        personas_tested=personas_tested,
        results=aggregated_results,
        metadata=config_metadata,
    )

    # Check for duplicate content warnings and config changes
    duplicates = [r for r in records if r.metadata.get("prior_tests")]
    if duplicates:
        dup_lines = []
        config_changed = False
        current_fp = config_metadata.get("config", {}).get("fingerprint", "")

        for r in duplicates[:3]:  # Show first 3
            pt = r.metadata["prior_tests"]
            content_preview = (
                r.variation_content[:40] + "..."
                if len(r.variation_content) > 40
                else r.variation_content
            )
            dup_lines.append(f'  • "{content_preview}" ({pt["count"]} prior tests)')

            # Check if any prior tests used different config (via store lookup)
            for prior_test_id in pt.get("test_ids", [])[:1]:  # Check most recent
                prior_records = store.get_test_records(prior_test_id)
                if prior_records:
                    prior_config = prior_records[0].metadata.get("config", {})
                    prior_fp = prior_config.get("fingerprint", "")
                    if prior_fp and prior_fp != current_fp:
                        config_changed = True
                        break

        if len(duplicates) > 3:
            dup_lines.append(f"  ... and {len(duplicates) - 3} more")

        msg = "[yellow]Some variations were tested before:[/yellow]\n" + "\n".join(
            dup_lines
        )
        if config_changed:
            msg += "\n\n[red]⚠ Config has changed since prior tests![/red]\n"
            msg += "[dim]Prior scores may not be directly comparable.[/dim]"
        else:
            msg += "\n\n[dim]Consider using unique variations to maximize calibration data value.[/dim]"

        console.print()
        console.print(
            Panel(msg, title="Duplicate Content Detected", border_style="yellow")
        )

    # Display tracking confirmation with config info
    config_fp = config_metadata.get("config", {}).get("fingerprint", "unknown")
    console.print()
    console.print(
        Panel(
            f"[green]Recorded {len(records)} predictions[/green]\n"
            f"Test ID: [bold]{test_id}[/bold]\n"
            f"Config: [dim]{config_fp}[/dim]",
            title="Prediction Tracking",
        )
    )
