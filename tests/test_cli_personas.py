"""Integration tests for persona YAML validation at the CLI boundary."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _emit(personas, output):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "synthetic_panel.cli",
            "emit-prompts",
            "--persona-file",
            str(personas),
            "--variant",
            "a=Example copy",
            "--n-samples",
            "1",
            "--output",
            str(output),
            "--json",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("- name: Example\n", "invalid persona file: {path} must contain a mapping"),
        ("not-a-mapping\n", "invalid persona file: {path} must contain a mapping"),
        ("unexpected: [\n", "invalid persona YAML: {path}"),
        (None, "missing persona file: {path}"),
        ("", "empty persona file: {path}"),
        ("{}\n", "empty persona file: {path}"),
        ("[]\n", "empty persona file: {path}"),
        ("reference_statements: {}\n", "empty persona file: {path}"),
    ],
    ids=["sequence", "scalar", "syntax", "missing", "empty", "empty-map",
         "empty-list", "metadata-only"],
)
def test_emit_persona_input_error_has_no_output(tmp_path, body, diagnostic):
    persona_path = tmp_path / "persona.yaml"
    output = tmp_path / "prompts.json"
    if body is not None:
        persona_path.write_text(body, encoding="utf-8")
    before = set(tmp_path.iterdir())
    completed = _emit(persona_path, output)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == f"error: {diagnostic.format(path=persona_path)}\n"
    assert not output.exists()
    assert set(tmp_path.iterdir()) == before


def test_emit_valid_persona_mapping_writes_and_prints_handoff(tmp_path):
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(
        "example:\n"
        "  name: Hypothetical Person\n"
        "  demographics:\n"
        "    age: 41\n"
        "    location: Example City\n"
        "    family: Single\n"
        "    occupation: Technician\n"
        "    income: 50000 annually\n",
        encoding="utf-8",
    )
    output = tmp_path / "prompts.json"

    completed = _emit(persona_path, output)

    assert completed.returncode == 0
    assert completed.stderr == ""
    handoff = json.loads(completed.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == handoff
    assert len(handoff["records"]) == 1
    assert handoff["records"][0]["persona_id"] == "example"
