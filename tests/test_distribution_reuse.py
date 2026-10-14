"""Offline characterization of the existing five-bin distribution averages."""

import asyncio
import copy
import socket
import sys
from types import SimpleNamespace

import pytest

from synthetic_panel import config, embeddings, scoring, testing

DISTRIBUTIONS = [
    [0.6, 0.1, 0.2, 0.05, 0.05],
    [0.1, 0.2, 0.1, 0.3, 0.3],
    [0.0, 0.1, 0.2, 0.3, 0.4],
]


def forbidden(*_args, **_kwargs):
    raise AssertionError("unexpected network, generation, embedding, or storage access")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for module, names in [
        (embeddings, ["get_embedding", "get_embeddings_batch"]),
        (scoring, ["get_embedding", "get_embeddings_batch", "calculate_ssr_score"]),
        (
            config,
            ["load_responses", "save_responses_to_disk", "get_config_fingerprint"],
        ),
        (
            testing,
            [
                "get_llm_client",
                "get_async_llm_client",
                "generate_response",
                "generate_response_async",
                "calculate_ssr_score",
                "load_responses",
                "save_responses_to_disk",
                "get_config_fingerprint",
            ],
        ),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(testing, "build_persona_prompt", lambda _persona: "persona")
    monkeypatch.setattr(testing, "get_stimulus_prompt", lambda *_args: "stimulus")
    monkeypatch.setattr(
        testing, "console", SimpleNamespace(print=lambda *_a, **_k: None)
    )


def run_samples(monkeypatch, mode, outcomes):
    responses = [f"reaction-{i}" for i in range(len(outcomes))]
    iterator = iter(responses)

    def score(response, _anchors):
        outcome = outcomes[responses.index(response)]
        if outcome is None:
            raise testing.SSRScoringError("synthetic scoring failure")
        return outcome

    monkeypatch.setattr(testing, "calculate_ssr_score", score)
    if mode == "batch":
        return testing._process_batch_results(
            {f"persona_v0_s{i}": response for i, response in enumerate(responses)},
            {},
            ["persona"],
            ["copy"],
            {},
            len(responses),
        )["persona"][0]
    if mode == "sync":
        monkeypatch.setattr(testing, "generate_response", lambda *_args: next(iterator))
        return testing.test_variation(
            ("offline", None), {}, "headline", "copy", {}, len(responses)
        )

    async def generate(*_args):
        return next(iterator)

    monkeypatch.setattr(testing, "generate_response_async", generate)
    return asyncio.run(
        testing.test_variation_async(
            ("offline", None),
            {},
            "headline",
            "copy",
            {},
            len(responses),
        )
    )


@pytest.mark.parametrize("mode", ["sync", "async", "batch"])
@pytest.mark.parametrize("partial", [False, True])
def test_callers_average_only_successful_distributions(monkeypatch, mode, partial):
    outcomes = [(1.5 + i, list(dist), "ssr") for i, dist in enumerate(DISTRIBUTIONS)]
    if partial:
        outcomes.insert(0, None)
    before = copy.deepcopy(outcomes)
    result = run_samples(monkeypatch, mode, outcomes)
    assert result == {
        "content": "copy",
        "response": "reaction-0",
        "all_responses": [f"reaction-{i}" for i in range(len(outcomes))],
        "score": 2.5,
        "distribution": [sum(row[i] for row in DISTRIBUTIONS) / 3 for i in range(5)],
        "scoring_method": "ssr_partial" if partial else "ssr",
    }
    assert outcomes == before


@pytest.mark.parametrize("mode", ["sync", "async", "batch"])
def test_callers_keep_single_success_and_all_failure_contract(monkeypatch, mode):
    result = run_samples(monkeypatch, mode, [(0.0, DISTRIBUTIONS[0], "ssr")])
    assert result["score"] == 0.0
    assert result["distribution"] == DISTRIBUTIONS[0]
    if mode == "batch":
        result = run_samples(monkeypatch, mode, [None, None])
        assert result["scoring_method"] == "failed"
        assert result["distribution"] == [0.2] * 5
        assert result["all_responses"] == ["reaction-0", "reaction-1"]
    else:
        with pytest.raises(testing.SSRScoringError, match="All 2 samples failed"):
            run_samples(monkeypatch, mode, [None, None])


def test_batch_keeps_cache_identity_and_missing_sample_placeholder(monkeypatch):
    cached = {"content": "cached", "score": 4.0, "distribution": DISTRIBUTIONS[1]}
    result = testing._process_batch_results(
        {},
        {},
        ["persona"],
        ["cached", "missing"],
        {},
        2,
        cached_results={"persona": {0: cached}},
    )["persona"]
    assert result[0] is cached
    assert result[1] == {
        "content": "missing",
        "response": "Error: No batch results",
        "all_responses": [],
        "score": 0.0,
        "distribution": [0.2] * 5,
    }


def test_tracking_keeps_group_order_default_distribution_and_mean(monkeypatch):
    captured = []

    class Store:
        def record_test_predictions(self, **kwargs):
            captured.append(kwargs)
            return []

    monkeypatch.setitem(
        sys.modules, "calibration", SimpleNamespace(CalibrationStore=Store)
    )
    monkeypatch.setattr(
        testing, "get_config_fingerprint", lambda _personas: {"fingerprint": "fixture"}
    )
    results = {
        "first": [
            {"content": "one", "score": 2.0, "distribution": DISTRIBUTIONS[0]},
            {"content": "two", "score": 4.0, "distribution": DISTRIBUTIONS[1]},
        ],
        "second": [{"content": "one", "score": 4.0}],
    }
    before = copy.deepcopy(results)
    testing.track_predictions("id", "name", "headline", ["first", "second"], results)
    assert captured == [
        {
            "test_id": "id",
            "test_name": "name",
            "test_type": "headline",
            "personas_tested": ["first", "second"],
            "metadata": {"config": {"fingerprint": "fixture"}},
            "results": [
                {
                    "variation_id": "v1",
                    "content": "one",
                    "score": 3.0,
                    "distribution": [(DISTRIBUTIONS[0][i] + 0.2) / 2 for i in range(5)],
                },
                {
                    "variation_id": "v2",
                    "content": "two",
                    "score": 4.0,
                    "distribution": DISTRIBUTIONS[1],
                },
            ],
        }
    ]
    assert results == before


def test_shared_average_preserves_bin_width_arithmetic_order_and_errors():
    values = [[1e16] * 6, [1.0] * 6, [-1e16] * 6]
    assert testing._average_distributions(values) == [sum([1e16, 1.0, -1e16]) / 3] * 5
    assert testing._average_distributions([values[0]]) == [1e16] * 5
    with pytest.raises(ZeroDivisionError):
        testing._average_distributions([])
    with pytest.raises(IndexError):
        testing._average_distributions([[0.1] * 4])
