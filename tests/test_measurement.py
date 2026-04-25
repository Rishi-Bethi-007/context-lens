"""Tests for context_lens.engine.measurement — all LLM calls mocked."""
from unittest.mock import MagicMock, patch

import pytest

from context_lens.engine.measurement import (
    MeasurementResult,
    _validate_inputs,
    measure_context_health,
)
from context_lens.engine.probes import ProbeResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

HAYSTACK = (
    "The industrial revolution changed society in many ways. "
    "Factories replaced cottage industries and cities grew rapidly. "
    "Workers migrated to urban centres seeking new employment opportunities. "
    "Technology improved productivity across all sectors of the economy. "
) * 60  # ~1000 tokens

NEEDLE   = "The secret code for this measurement is: MEASVAL-007."
QUESTION = "What is the secret code for this measurement?"
EXPECTED = "MEASVAL-007"


def make_mock_anthropic(response_text: str = "MEASVAL-007"):
    """Return a mock anthropic.Anthropic class whose instances respond with response_text."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_resp

    mock_class = MagicMock(return_value=mock_client)
    return mock_class


# ── measure_context_health ────────────────────────────────────────────────────

def test_measure_returns_measurement_result():
    with patch("context_lens.engine.measurement.anthropic.Anthropic", make_mock_anthropic()):
        result = measure_context_health(
            "test-agent", HAYSTACK,
            needle=NEEDLE, question=QUESTION, expected=EXPECTED,
            positions=[0.1, 0.5], token_counts=[500],
            n_votes=1, api_key="fake-key",
        )
    assert isinstance(result, MeasurementResult)


def test_measure_result_has_correct_cell_count():
    positions = [0.1, 0.5, 0.9]
    token_counts = [500, 1000]
    with patch("context_lens.engine.measurement.anthropic.Anthropic", make_mock_anthropic()):
        result = measure_context_health(
            "test-agent", HAYSTACK,
            needle=NEEDLE, question=QUESTION, expected=EXPECTED,
            positions=positions, token_counts=token_counts,
            n_votes=1, api_key="fake-key",
        )
    assert len(result.probe_results) == len(positions) * len(token_counts)


def test_measure_agent_name_stored():
    with patch("context_lens.engine.measurement.anthropic.Anthropic", make_mock_anthropic()):
        result = measure_context_health(
            "my-cool-agent", HAYSTACK,
            needle=NEEDLE, question=QUESTION, expected=EXPECTED,
            positions=[0.5], token_counts=[500],
            n_votes=1, api_key="fake-key",
        )
    assert result.agent_name == "my-cool-agent"


def test_measure_token_counts_and_positions_stored():
    positions = [0.1, 0.9]
    token_counts = [500, 1000]
    with patch("context_lens.engine.measurement.anthropic.Anthropic", make_mock_anthropic()):
        result = measure_context_health(
            "test-agent", HAYSTACK,
            needle=NEEDLE, question=QUESTION, expected=EXPECTED,
            positions=positions, token_counts=token_counts,
            n_votes=1, api_key="fake-key",
        )
    assert result.positions_tested == positions
    assert result.token_counts_tested == token_counts


def test_measure_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Anthropic API key"):
        measure_context_health(
            "test-agent", HAYSTACK,
            needle=NEEDLE, question=QUESTION, expected=EXPECTED,
            positions=[0.5], token_counts=[500],
            n_votes=1,
            # no api_key kwarg, env var deleted
        )


def test_measure_uses_env_var_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
    with patch("context_lens.engine.measurement.anthropic.Anthropic", make_mock_anthropic()) as mock_cls:
        measure_context_health(
            "test-agent", HAYSTACK,
            needle=NEEDLE, question=QUESTION, expected=EXPECTED,
            positions=[0.5], token_counts=[500], n_votes=1,
        )
        mock_cls.assert_called_once_with(api_key="env-key-123")


# ── MeasurementResult aggregation ─────────────────────────────────────────────

def _make_result(corrects: list[bool], positions: list[float] | None = None) -> MeasurementResult:
    """Build a MeasurementResult directly from a list of correct flags."""
    positions = positions or [0.1] * len(corrects)
    token_counts = [500] * len(corrects)
    probe_results = [
        ProbeResult(
            position=p,
            token_count=500,
            target_token_count=tc,
            correct=c,
            response="resp",
            expected="X",
        )
        for p, tc, c in zip(positions, token_counts, corrects)
    ]
    return MeasurementResult(
        agent_name="test",
        probe_results=probe_results,
        token_counts_tested=list(set(token_counts)),
        positions_tested=list(set(positions)),
    )


def test_mean_accuracy_all_correct():
    result = _make_result([True, True, True])
    assert result.mean_accuracy() == 1.0


def test_mean_accuracy_all_wrong():
    result = _make_result([False, False, False])
    assert result.mean_accuracy() == 0.0


def test_mean_accuracy_mixed():
    result = _make_result([True, False, True, False])
    assert result.mean_accuracy() == pytest.approx(0.5)


def test_mean_accuracy_empty():
    result = MeasurementResult(
        agent_name="empty", probe_results=[],
        token_counts_tested=[], positions_tested=[],
    )
    assert result.mean_accuracy() == 0.0


def test_accuracy_by_position_groups_correctly():
    result = _make_result(
        corrects=[True, False, True, False],
        positions=[0.1, 0.1, 0.9, 0.9],
    )
    by_pos = result.accuracy_by_position()
    assert by_pos[0.1] == pytest.approx(0.5)
    assert by_pos[0.9] == pytest.approx(0.5)


def test_accuracy_by_token_count():
    probe_results = [
        ProbeResult(0.5, 490, 500,  True,  "r", "X"),
        ProbeResult(0.5, 490, 500,  False, "r", "X"),
        ProbeResult(0.5, 990, 1000, True,  "r", "X"),
        ProbeResult(0.5, 990, 1000, True,  "r", "X"),
    ]
    result = MeasurementResult(
        agent_name="t",
        probe_results=probe_results,
        token_counts_tested=[500, 1000],
        positions_tested=[0.5],
    )
    by_tc = result.accuracy_by_token_count()
    assert by_tc[500]  == pytest.approx(0.5)
    assert by_tc[1000] == pytest.approx(1.0)


# ── _validate_inputs ──────────────────────────────────────────────────────────

def test_validate_empty_agent_name():
    with pytest.raises(ValueError, match="agent_name must not be empty"):
        _validate_inputs("  ", HAYSTACK, NEEDLE, [0.5], [500], 3)


def test_validate_empty_haystack():
    with pytest.raises(ValueError, match="haystack must not be empty"):
        _validate_inputs("agent", "  ", NEEDLE, [0.5], [500], 3)


def test_validate_empty_needle():
    with pytest.raises(ValueError, match="needle must not be empty"):
        _validate_inputs("agent", HAYSTACK, "  ", [0.5], [500], 3)


def test_validate_empty_positions():
    with pytest.raises(ValueError, match="positions list must not be empty"):
        _validate_inputs("agent", HAYSTACK, NEEDLE, [], [500], 3)


def test_validate_bad_position():
    with pytest.raises(ValueError, match="out of range"):
        _validate_inputs("agent", HAYSTACK, NEEDLE, [1.5], [500], 3)


def test_validate_empty_token_counts():
    with pytest.raises(ValueError, match="token_counts list must not be empty"):
        _validate_inputs("agent", HAYSTACK, NEEDLE, [0.5], [], 3)


def test_validate_zero_token_count():
    with pytest.raises(ValueError, match="must be a positive integer"):
        _validate_inputs("agent", HAYSTACK, NEEDLE, [0.5], [0], 3)


def test_validate_n_votes_zero():
    with pytest.raises(ValueError, match="n_votes must be"):
        _validate_inputs("agent", HAYSTACK, NEEDLE, [0.5], [500], 0)
