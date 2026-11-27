from io import StringIO

import pytest
from rich.console import Console

from synthetic_panel import cli, testing


@pytest.mark.parametrize("width", (75, 120))
@pytest.mark.parametrize("has_results", (False, True))
def test_legacy_results_match_without_mixing_consoles(monkeypatch, width, has_results):
    results = (
        [
            {"score": 3.5, "content": "A" * 47 + "not rendered", "response": "R" * 500 + "not rendered"},
            {"score": 1.0, "content": "Other variation", "response": "Other rationale"},
        ]
        if has_results
        else []
    )
    cli_text, runner_text = StringIO(), StringIO()
    monkeypatch.setattr(cli, "stdout_console", Console(file=cli_text, width=width, color_system=None))
    monkeypatch.setattr(testing, "console", Console(file=runner_text, width=width, color_system=None))

    cli.display_results("unused-name", "Fixture Persona", results)
    assert runner_text.getvalue() == ""
    testing.display_results("unused-name", "Fixture Persona", results)

    rendered = cli_text.getvalue()
    assert rendered == runner_text.getvalue()
    assert "Results for Fixture Persona" in rendered
    assert ("Top Performer Rationale" in rendered) is has_results
    if has_results:
        assert "██████░░░░" in rendered
        assert "3.50" in rendered
        assert "Other variation" in rendered
        assert "not rendered" not in rendered
