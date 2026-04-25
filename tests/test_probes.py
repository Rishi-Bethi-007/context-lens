"""Tests for context_lens.engine.probes — no real API calls."""
from unittest.mock import MagicMock, patch

import pytest

from context_lens.engine.probes import (
    ProbeConfig,
    ProbeRunner,
    count_tokens,
    inject_needle,
    truncate_to_tokens,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

HAYSTACK = (
    "The industrial revolution changed society profoundly. "
    "Factories replaced cottage industries and cities grew quickly. "
    "Workers moved to urban centres in search of employment and opportunity. "
    "New technologies improved productivity across all sectors of the economy. "
) * 60  # ~1000 tokens


def make_config(**overrides) -> ProbeConfig:
    defaults = dict(
        needle="The secret identifier for this test is: TESTVAL-001.",
        question="What is the secret identifier for this test?",
        expected="TESTVAL-001",
        haystack=HAYSTACK,
        positions=[0.1, 0.5, 0.9],
        token_counts=[500, 1000],
        n_votes=3,
        sleep_between_calls=0.0,
    )
    defaults.update(overrides)
    return ProbeConfig(**defaults)


def make_mock_client(response_text: str = "TESTVAL-001") -> MagicMock:
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = mock_resp
    return client


# ── count_tokens ──────────────────────────────────────────────────────────────

def test_count_tokens_empty_string():
    assert count_tokens("") == 0


def test_count_tokens_known_text():
    # "hello" is a single token in cl100k_base
    assert count_tokens("hello") == 1


def test_count_tokens_increases_with_length():
    short = count_tokens("Hello world.")
    long = count_tokens("Hello world. " * 100)
    assert long > short


# ── truncate_to_tokens ────────────────────────────────────────────────────────

def test_truncate_to_tokens_exact():
    text = "alpha beta gamma delta"
    truncated = truncate_to_tokens(text, 2)
    assert count_tokens(truncated) <= 2


def test_truncate_to_tokens_no_op_when_short():
    text = "hi"
    assert truncate_to_tokens(text, 1000) == text


def test_truncate_to_tokens_reduces_length():
    long_text = "word " * 500
    short = truncate_to_tokens(long_text, 50)
    assert count_tokens(short) <= 50
    assert count_tokens(short) < count_tokens(long_text)


# ── inject_needle ─────────────────────────────────────────────────────────────

def test_inject_needle_contains_needle():
    result = inject_needle(HAYSTACK, "SECRET FACT HERE.", 0.5)
    assert "SECRET FACT HERE." in result


def test_inject_needle_position_zero():
    result = inject_needle(HAYSTACK, "NEEDLE.", 0.0)
    assert "NEEDLE." in result


def test_inject_needle_position_one():
    result = inject_needle(HAYSTACK, "NEEDLE.", 1.0)
    assert "NEEDLE." in result


def test_inject_needle_longer_than_original():
    needle = "This is the injected needle sentence."
    result = inject_needle(HAYSTACK, needle, 0.3)
    assert count_tokens(result) > count_tokens(HAYSTACK)


def test_inject_needle_invalid_position_below():
    with pytest.raises(ValueError, match="position must be in"):
        inject_needle(HAYSTACK, "NEEDLE.", -0.1)


def test_inject_needle_invalid_position_above():
    with pytest.raises(ValueError, match="position must be in"):
        inject_needle(HAYSTACK, "NEEDLE.", 1.1)


def test_inject_needle_different_positions_differ():
    r1 = inject_needle(HAYSTACK, "FACT.", 0.1)
    r5 = inject_needle(HAYSTACK, "FACT.", 0.5)
    r9 = inject_needle(HAYSTACK, "FACT.", 0.9)
    # The needle appears at different byte offsets
    assert r1.index("FACT.") < r5.index("FACT.") < r9.index("FACT.")


# ── ProbeConfig validation ────────────────────────────────────────────────────

def test_probe_config_empty_needle_raises():
    with pytest.raises(ValueError, match="needle must not be empty"):
        make_config(needle="   ")


def test_probe_config_empty_haystack_raises():
    with pytest.raises(ValueError, match="haystack must not be empty"):
        make_config(haystack="  ")


def test_probe_config_empty_positions_raises():
    with pytest.raises(ValueError, match="positions list must not be empty"):
        make_config(positions=[])


def test_probe_config_bad_position_raises():
    with pytest.raises(ValueError, match="out of range"):
        make_config(positions=[0.5, 1.5])


def test_probe_config_zero_token_count_raises():
    with pytest.raises(ValueError, match="must be positive"):
        make_config(token_counts=[0])


def test_probe_config_n_votes_zero_raises():
    with pytest.raises(ValueError, match="n_votes must be"):
        make_config(n_votes=0)


# ── ProbeRunner ───────────────────────────────────────────────────────────────

def test_probe_runner_correct_answer_returns_correct():
    client = make_mock_client(response_text="TESTVAL-001")
    config = make_config(n_votes=3)
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert all(r.correct for r in results)


def test_probe_runner_wrong_answer_returns_incorrect():
    client = make_mock_client(response_text="WRONG-999")
    config = make_config(n_votes=3)
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert all(not r.correct for r in results)


def test_probe_runner_majority_vote_two_of_three_wins():
    """2 correct + 1 wrong across 3 votes should yield correct=True."""
    responses = ["TESTVAL-001", "WRONG", "TESTVAL-001"]
    call_count = 0

    client = MagicMock()

    def side_effect(**kwargs):
        nonlocal call_count
        text = responses[call_count % len(responses)]
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=text)]
        return mock_resp

    client.messages.create.side_effect = side_effect

    config = make_config(n_votes=3, positions=[0.5], token_counts=[500])
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert len(results) == 1
    assert results[0].correct is True


def test_probe_runner_one_vote_of_three_wrong_loses():
    """1 correct + 2 wrong = correct=False."""
    responses = ["TESTVAL-001", "WRONG", "WRONG"]
    call_count = 0

    client = MagicMock()

    def side_effect(**kwargs):
        nonlocal call_count
        text = responses[call_count % len(responses)]
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=text)]
        return mock_resp

    client.messages.create.side_effect = side_effect

    config = make_config(n_votes=3, positions=[0.5], token_counts=[500])
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert results[0].correct is False


def test_probe_runner_result_count_equals_cells():
    client = make_mock_client()
    config = make_config(positions=[0.1, 0.5, 0.9], token_counts=[500, 1000])
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert len(results) == 3 * 2  # 3 positions x 2 token counts


def test_probe_runner_stores_response_text():
    client = make_mock_client(response_text="TESTVAL-001 found here")
    config = make_config(positions=[0.5], token_counts=[500], n_votes=1)
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert results[0].response == "TESTVAL-001 found here"


def test_probe_runner_stores_expected():
    client = make_mock_client()
    config = make_config(positions=[0.5], token_counts=[500], n_votes=1)
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert results[0].expected == "TESTVAL-001"


def test_probe_runner_target_token_count_recorded():
    client = make_mock_client()
    config = make_config(positions=[0.5], token_counts=[500], n_votes=1)
    runner = ProbeRunner(client=client, config=config)
    results = runner.run()
    assert results[0].target_token_count == 500


def test_probe_runner_needle_in_haystack_raises():
    """If the expected answer already exists in haystack, raise ValueError."""
    config = make_config(
        expected="industrial",   # "industrial" appears in HAYSTACK
        needle="The industrial revolution is key.",
        question="What is key?",
    )
    client = make_mock_client()
    runner = ProbeRunner(client=client, config=config)
    with pytest.raises(ValueError, match="already appears in the haystack"):
        runner.run()


def test_probe_runner_token_count_too_small_raises():
    config = make_config(
        token_counts=[10],  # far too small for needle + overhead
    )
    client = make_mock_client()
    runner = ProbeRunner(client=client, config=config)
    with pytest.raises(ValueError, match="too small"):
        runner.run()
