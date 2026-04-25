"""Orchestrates the full context health measurement sweep."""
import logging
import os
from dataclasses import dataclass, field

import anthropic

from context_lens.engine.probes import (
    ProbeConfig,
    ProbeResult,
    ProbeRunner,
)

logger = logging.getLogger(__name__)

DEFAULT_POSITIONS: list[float] = [0.10, 0.30, 0.50, 0.70, 0.90]
DEFAULT_TOKEN_COUNTS: list[int] = [5_000, 10_000, 20_000, 30_000]
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class MeasurementConfig:
    """Full configuration for a context health measurement sweep."""

    agent_name: str
    haystack: str
    needle: str
    question: str
    expected: str
    positions: list[float] = field(default_factory=lambda: list(DEFAULT_POSITIONS))
    token_counts: list[int] = field(default_factory=lambda: list(DEFAULT_TOKEN_COUNTS))
    n_votes: int = 3
    model: str = DEFAULT_MODEL
    api_key: str | None = None   # falls back to ANTHROPIC_API_KEY env var


@dataclass
class MeasurementResult:
    """Results of a full context health measurement sweep."""

    agent_name: str
    probe_results: list[ProbeResult]
    token_counts_tested: list[int]
    positions_tested: list[float]

    def mean_accuracy(self) -> float:
        """Return overall mean accuracy across all probe cells."""
        if not self.probe_results:
            return 0.0
        return sum(r.correct for r in self.probe_results) / len(self.probe_results)

    def accuracy_by_position(self) -> dict[float, float]:
        """Return mean accuracy keyed by position, averaged across token counts."""
        by_pos: dict[float, list[bool]] = {}
        for r in self.probe_results:
            by_pos.setdefault(r.position, []).append(r.correct)
        return {pos: sum(vals) / len(vals) for pos, vals in by_pos.items()}

    def accuracy_by_token_count(self) -> dict[int, float]:
        """Return mean accuracy keyed by target token count, averaged across positions."""
        by_tc: dict[int, list[bool]] = {}
        for r in self.probe_results:
            by_tc.setdefault(r.target_token_count, []).append(r.correct)
        return {tc: sum(vals) / len(vals) for tc, vals in by_tc.items()}


# ── Public API ────────────────────────────────────────────────────────────────

def measure_context_health(
    agent_name: str,
    haystack: str,
    *,
    needle: str,
    question: str,
    expected: str,
    positions: list[float] | None = None,
    token_counts: list[int] | None = None,
    n_votes: int = 3,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> MeasurementResult:
    """Run a full context health sweep and return a MeasurementResult.

    Plants needle at each position inside contexts of each token count, asks
    question, and checks whether expected appears in the response.

    Args:
        agent_name:   Human-readable name for the agent being measured.
        haystack:     Background text to fill the context window.
        needle:       Fact sentence to plant at each test position.
        question:     Question to ask about the planted fact.
        expected:     Substring that must appear in a correct response.
        positions:    Positions to test (default: [0.10, 0.30, 0.50, 0.70, 0.90]).
        token_counts: Context sizes to test (default: [5K, 10K, 20K, 30K]).
        n_votes:      API calls per cell for majority voting (default: 3).
        model:        Claude model to use (default: claude-haiku-4-5-20251001).
        api_key:      Anthropic API key; falls back to ANTHROPIC_API_KEY env var.

    Returns:
        MeasurementResult with all ProbeResult objects and helper aggregation methods.

    Raises:
        ValueError: If configuration is invalid or no API key is available.
    """
    positions = positions or list(DEFAULT_POSITIONS)
    token_counts = token_counts or list(DEFAULT_TOKEN_COUNTS)

    _validate_inputs(agent_name, haystack, needle, positions, token_counts, n_votes)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "No Anthropic API key provided. "
            "Pass api_key= or set the ANTHROPIC_API_KEY environment variable."
        )

    client = anthropic.Anthropic(api_key=key)
    config = ProbeConfig(
        needle=needle,
        question=question,
        expected=expected,
        haystack=haystack,
        positions=positions,
        token_counts=token_counts,
        n_votes=n_votes,
        model=model,
    )

    logger.info(
        "Starting measurement for %r: %d cells (%d token_counts x %d positions)",
        agent_name, len(token_counts) * len(positions), len(token_counts), len(positions),
    )

    runner = ProbeRunner(client=client, config=config)
    probe_results = runner.run()

    return MeasurementResult(
        agent_name=agent_name,
        probe_results=probe_results,
        token_counts_tested=token_counts,
        positions_tested=positions,
    )


# ── Internal validation ───────────────────────────────────────────────────────

def _validate_inputs(
    agent_name: str,
    haystack: str,
    needle: str,
    positions: list[float],
    token_counts: list[int],
    n_votes: int,
) -> None:
    """Raise ValueError for any invalid measurement inputs."""
    if not agent_name.strip():
        raise ValueError("agent_name must not be empty")
    if not haystack.strip():
        raise ValueError("haystack must not be empty")
    if not needle.strip():
        raise ValueError("needle must not be empty")
    if not positions:
        raise ValueError("positions list must not be empty")
    if not token_counts:
        raise ValueError("token_counts list must not be empty")
    for p in positions:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Position {p} is out of range [0.0, 1.0]")
    for tc in token_counts:
        if tc <= 0:
            raise ValueError(f"token_count {tc} must be a positive integer")
    if n_votes < 1:
        raise ValueError(f"n_votes must be >= 1, got {n_votes}")
