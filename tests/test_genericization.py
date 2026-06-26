import argparse
import importlib
from pathlib import Path

import pytest
import yaml

from synthetic_panel.config import (
    build_persona_prompt,
    get_persona_names,
    get_stimulus_prompt,
    load_default_anchors,
    load_personas,
    resolve_persona_names,
)


RELEASE_GATE_TERMS = [
    "side" + " " + "business",
    "course",
    "membership",
]


def test_custom_persona_names_load_and_validate_without_fixed_names(tmp_path):
    persona_file = tmp_path / "personas.yaml"
    persona_file.write_text(
        yaml.safe_dump(
            {
                "alex": {
                    "name": "Alex Rivera",
                    "demographics": {
                        "age": 37,
                        "location": "Denver, Colorado",
                        "family": "Married",
                        "occupation": "Operations consultant",
                        "income": "$120,000 annual income",
                    },
                },
                "priya": {
                    "name": "Priya Shah",
                    "demographics": {
                        "age": 31,
                        "location": "Seattle, Washington",
                        "family": "Single",
                        "occupation": "Product manager",
                        "income": "$135,000 annual income",
                    },
                },
            }
        )
    )

    personas = load_personas(persona_file)
    names = get_persona_names(personas)

    parser = argparse.ArgumentParser()
    parser.add_argument("--persona")
    args = parser.parse_args(["--persona", "priya"])

    assert names == ["alex", "priya"]
    assert args.persona in names
    assert resolve_persona_names(personas, "all") == ["alex", "priya"]


def test_headline_stimulus_prompt_is_neutral():
    prompt = get_stimulus_prompt("headline", "A clearer way to compare options")

    normalized = prompt.lower()
    assert "encounters this on a page" in normalized
    assert all(term not in normalized for term in RELEASE_GATE_TERMS)


def test_default_anchor_statements_are_neutral_and_not_prompt_templates():
    anchors = load_default_anchors()
    persona_prompt = build_persona_prompt(
        {
            "name": "Alex Rivera",
            "demographics": {
                "age": 37,
                "location": "Denver, Colorado",
                "family": "Married",
                "occupation": "Operations consultant",
                "income": "$120,000 annual income",
            },
        }
    )
    stimulus_prompt = get_stimulus_prompt(
        "headline", "A clearer way to compare options"
    )
    template_text = f"{persona_prompt}\n{stimulus_prompt}"

    for anchor_set in anchors.values():
        assert set(anchor_set.keys()) == {1, 2, 3, 4, 5}
        for statement in anchor_set.values():
            normalized = statement.lower()
            assert statement not in template_text
            assert all(term not in normalized for term in RELEASE_GATE_TERMS)


@pytest.mark.parametrize(
    "module_name",
    [
        "synthetic_panel",
        "synthetic_panel.config",
        "synthetic_panel.cli",
        "synthetic_panel.embeddings",
        "synthetic_panel.scoring",
        "synthetic_panel.testing",
    ],
)
def test_package_and_submodules_import(module_name):
    assert importlib.import_module(module_name)


def test_example_personas_are_minimal_and_generic():
    personas = load_personas(Path("synthetic_panel/data/personas.example.yaml"))

    assert set(personas) == {"alex", "priya", "jordan"}
    for persona in personas.values():
        assert set(persona) == {"name", "demographics"}
