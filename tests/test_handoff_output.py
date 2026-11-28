"""Pin prompt and score CLI handoff bytes without an embedding model."""

import json
from pathlib import Path

import pytest

from synthetic_panel import cli


@pytest.mark.parametrize("write_file, json_flag", [(False, False), (True, False), (True, True)])
def test_emit_prompt_output(tmp_path, capsys, write_file, json_flag):
    personas = Path(__file__).resolve().parents[1] / "synthetic_panel/data/personas.example.yaml"
    output = tmp_path / "prompts.json"
    argv = [
        "emit-prompts", "--persona-file", str(personas), "--persona", "alex",
        "--variant", "z=Try a café", "--variant", "a=Compare options", "--n-samples", "1",
    ]
    if write_file:
        argv.extend(["--output", str(output)])
    if json_flag:
        argv.append("--json")

    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    data = output.read_bytes() if write_file else captured.out.encode("ascii")[:-1]
    handoff = json.loads(data)
    assert handoff["variants"] == {"a": "Compare options", "z": "Try a café"}
    assert [record["variant_id"] for record in handoff["records"]] == ["z", "a"]
    assert rb"caf\u00e9" in data
    assert data == json.dumps(handoff, indent=2, sort_keys=True).encode("ascii")
    if not write_file or json_flag:
        assert captured.out.encode("ascii") == data + b"\n"
    else:
        assert captured.out.startswith("Wrote 2 prompts to ")
    assert output.exists() is write_file


@pytest.mark.parametrize(
    "write_file, json_flag", [(False, False), (False, True), (True, False), (True, True)]
)
def test_score_output(tmp_path, capsys, monkeypatch, write_file, json_flag):
    reactions = tmp_path / "reactions.json"
    reactions.write_text("{}")
    output = tmp_path / "results.json"
    fake_score = {
        "ranking": [],
        "comparability": {"verdict": "comparable", "message": "Same café"},
    }
    monkeypatch.setattr(cli, "load_default_anchors", lambda _: {})
    monkeypatch.setattr(cli, "score_reaction_handoff", lambda *args, **kwargs: fake_score)

    argv = ["score-reactions", str(reactions)]
    if write_file:
        argv.extend(["--output", str(output)])
    if json_flag:
        argv.append("--json")

    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    expected = (
        b'{\n  "comparability": {\n    "message": "Same caf\\u00e9",\n'
        b'    "verdict": "comparable"\n  },\n  "ranking": []\n}'
    )
    assert output.exists() is write_file
    if write_file:
        assert output.read_bytes() == expected
    if json_flag:
        assert captured.out.encode("ascii") == expected + b"\n"
    else:
        assert captured.out == "Comparability: comparable\nSame café\n"
