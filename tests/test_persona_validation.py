import pytest

from synthetic_panel.config import PersonaValidationError, build_persona_prompt


def test_minimal_persona_builds_prompt_with_third_person_framing():
    persona = {
        "name": "Casey Morgan",
        "demographics": {
            "age": 41,
            "location": "Portland, Oregon",
            "family": "Partnered",
            "occupation": "Operations lead",
            "income": "$110,000 annual income",
        },
    }

    prompt = build_persona_prompt(persona)

    assert "market research analyst simulating how Casey Morgan would respond" in prompt
    assert "RESPONDENT PROFILE: Casey Morgan" in prompt
    assert "DEMOGRAPHICS:" in prompt
    assert "SITUATION:" not in prompt
    assert "neutral researcher" in prompt


def test_rich_persona_missing_motivations_raises_actionable_validation_error():
    persona = {
        "name": "alex",
        "demographics": {
            "age": 37,
            "location": "Denver, Colorado",
            "family": "Married",
            "occupation": "Operations consultant",
            "income": "$120,000 annual income",
        },
        "situation": {
            "time_available": "5 hours per week",
            "capital_available": "$2,000",
            "risk_tolerance": "Moderate",
            "current_status": "Comparing software options",
        },
    }

    with pytest.raises(PersonaValidationError) as exc_info:
        build_persona_prompt(persona)

    assert str(exc_info.value) == (
        "persona 'alex' is missing required field 'motivations.primary'"
    )
