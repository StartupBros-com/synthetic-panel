import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from synthetic_panel import embeddings
from synthetic_panel.config import load_default_anchors, load_personas
from synthetic_panel.testing import emit_prompt_handoff, score_reaction_handoff


VARIANTS = {
    "weak": "A basic tool that might help you compare options.",
    "middle": "Compare options faster with clear tradeoffs and examples.",
    "strong": "Choose confidently today with clear tradeoffs, proof, and a simple next step.",
}


REACTIONS_BY_VARIANT = {
    "weak": [
        "As Alex, this feels too generic and low urgency. The message gives him little proof, no clear outcome, and he would probably leave without taking another step.",
        "Alex would hear this as vague and unfinished. It might be mildly relevant, but it does not address his concerns or make him interested enough to continue.",
    ],
    "middle": [
        "Alex would see some practical value here. The promise of faster comparison and clearer tradeoffs is relevant, though he would still want more evidence before acting.",
        "This would make Alex somewhat interested because it names a useful outcome. He is not fully convinced yet, but he would be open to reading the details.",
    ],
    "strong": [
        "Alex would find this compelling because it promises confidence, proof, and a simple next step. He would feel ready to continue and seriously consider acting.",
        "This strongly matches what Alex needs. The proof and clear next step reduce uncertainty, so he would be highly inclined to learn more or sign up soon.",
    ],
}


@pytest.fixture(autouse=True)
def local_embedding_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
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
    embeddings.EmbeddingCache._provider_cache = {}


@pytest.fixture
def prompt_handoff():
    personas = load_personas(Path("synthetic_panel/data/personas.example.yaml"))
    return emit_prompt_handoff(
        personas,
        VARIANTS,
        stimulus_type="headline",
        n_samples=2,
        persona_ids=["alex"],
    )


def _reaction_handoff(prompt_handoff, reactions_by_variant=None):
    payload = copy.deepcopy(prompt_handoff)
    reactions_by_variant = reactions_by_variant or REACTIONS_BY_VARIANT
    for record in payload["records"]:
        record["reaction"] = reactions_by_variant[record["variant_id"]][
            record["sample_index"]
        ]
    return payload


def test_emit_prompts_writes_one_record_per_sample_with_third_person_framing(
    prompt_handoff,
):
    assert prompt_handoff["schema_version"] == 1
    assert prompt_handoff["stimulus_type"] == "headline"
    assert prompt_handoff["n_samples"] == 2
    assert prompt_handoff["personas"] == ["alex"]
    assert len(prompt_handoff["records"]) == len(VARIANTS) * 2

    seen = set()
    for record in prompt_handoff["records"]:
        seen.add((record["variant_id"], record["persona_id"], record["sample_index"]))
        assert (
            "market research analyst simulating how Alex Chen would respond"
            in record["prompt"]
        )
        assert "RESPONDENT PROFILE: Alex Chen" in record["prompt"]
        assert "INTERVIEW SCENARIO:" in record["prompt"]
    assert len(seen) == len(VARIANTS) * 2


@pytest.mark.slow
@pytest.mark.model_dependent
def test_score_reactions_ranks_variants_and_returns_comparable_directional_scores(
    prompt_handoff, tmp_path
):
    reactions_path = tmp_path / "reactions.json"
    reactions_path.write_text(json.dumps(_reaction_handoff(prompt_handoff)))

    result = score_reaction_handoff(
        reactions_path,
        reference_statements=load_default_anchors(),
        dist_temperature=1.0,
    )

    assert result["comparability"]["verdict"] == "comparable"
    assert result["score_scale"] == "1-5 directional"
    ranked_ids = [item["variant_id"] for item in result["ranking"]]
    assert ranked_ids == ["strong", "middle", "weak"]
    assert all(
        variant["score_label"] == "directional"
        for variant in result["per_variant"].values()
    )
    assert result["per_variant"]["strong"]["reasoning_excerpt"]


@pytest.mark.slow
@pytest.mark.model_dependent
def test_score_reactions_identical_samples_return_not_comparable(
    prompt_handoff, tmp_path
):
    collapsed = {
        variant_id: [
            "Alex would give the same reaction here. It sounds acceptable but not distinctive, and he would need more proof before making any decision.",
            "Alex would give the same reaction here. It sounds acceptable but not distinctive, and he would need more proof before making any decision.",
        ]
        for variant_id in VARIANTS
    }
    reactions_path = tmp_path / "collapsed.json"
    reactions_path.write_text(json.dumps(_reaction_handoff(prompt_handoff, collapsed)))

    result = score_reaction_handoff(reactions_path, load_default_anchors())

    assert result["comparability"]["verdict"] == "not_comparable"
    assert any(
        issue["code"] == "collapsed_samples"
        for issue in result["comparability"]["issues"]
    )
    assert "unreliable" in result["comparability"]["message"]


@pytest.mark.slow
@pytest.mark.model_dependent
def test_score_reactions_missing_sample_returns_not_comparable(
    prompt_handoff, tmp_path
):
    payload = _reaction_handoff(prompt_handoff)
    payload["records"] = [
        record
        for record in payload["records"]
        if not (
            record["variant_id"] == "weak"
            and record["persona_id"] == "alex"
            and record["sample_index"] == 1
        )
    ]
    reactions_path = tmp_path / "missing.json"
    reactions_path.write_text(json.dumps(payload))

    result = score_reaction_handoff(reactions_path, load_default_anchors())

    assert result["comparability"]["verdict"] == "not_comparable"
    assert any(
        issue["code"] == "missing_sample" for issue in result["comparability"]["issues"]
    )


@pytest.mark.slow
@pytest.mark.model_dependent
def test_score_reactions_uses_local_embedding_provider_without_api_key(
    prompt_handoff, tmp_path
):
    assert not os.environ.get("OPENAI_API_KEY")
    assert not os.environ.get("GOOGLE_API_KEY")
    assert not os.environ.get("GEMINI_API_KEY")
    reactions_path = tmp_path / "reactions.json"
    reactions_path.write_text(json.dumps(_reaction_handoff(prompt_handoff)))

    result = score_reaction_handoff(reactions_path, load_default_anchors())

    assert result["comparability"]["verdict"] == "comparable"


@pytest.mark.slow
@pytest.mark.model_dependent
def test_score_reactions_json_cli_is_parseable_and_secret_free(
    prompt_handoff, tmp_path
):
    reactions_path = tmp_path / "reactions.json"
    reactions_path.write_text(json.dumps(_reaction_handoff(prompt_handoff)))

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "sk-test-token-identifiable-secret"
    env["SSR_EMBEDDING_PROVIDER"] = "local"
    env["SSR_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "synthetic_panel.cli",
            "score-reactions",
            str(reactions_path),
            "--json",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["comparability"]["verdict"] == "comparable"
    assert parsed["ranking"]
    assert "sk-test-token-identifiable-secret" not in completed.stdout
    assert "OPENAI_API_KEY" not in completed.stdout


def test_score_reactions_missing_file_cli_exits_one_line_without_traceback(tmp_path):
    missing_path = tmp_path / "missing-reactions.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "synthetic_panel.cli",
            "score-reactions",
            str(missing_path),
            "--json",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.count("\n") == 1
    assert "missing reactions file" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_emit_prompts_json_cli_is_parseable(tmp_path):
    persona_path = tmp_path / "personas.yaml"
    persona_path.write_text(
        yaml.safe_dump(
            {
                "casey": {
                    "name": "Casey Morgan",
                    "demographics": {
                        "age": 41,
                        "location": "Portland, Oregon",
                        "family": "Partnered",
                        "occupation": "Operations lead",
                        "income": "$110,000 annual income",
                    },
                }
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "synthetic_panel.cli",
            "emit-prompts",
            "--persona-file",
            str(persona_path),
            "--variant",
            "a=Compare options clearly",
            "--n-samples",
            "2",
            "--json",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["personas"] == ["casey"]
    assert len(parsed["records"]) == 2
