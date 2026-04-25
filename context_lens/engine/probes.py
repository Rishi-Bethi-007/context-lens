"""Probe generation, needle injection, and per-cell execution."""
import logging
import time
from dataclasses import dataclass

import anthropic
import tiktoken

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Result of a single probe cell (one position x one token count)."""

    position: float           # 0.0-1.0 where probe was planted in context
    token_count: int          # actual total tokens in the final prompt
    target_token_count: int   # requested token count (for grouping results)
    correct: bool             # majority-vote outcome
    response: str             # last raw model response
    expected: str             # expected answer substring


@dataclass
class ProbeConfig:
    """Full configuration for a probe sweep."""

    needle: str                        # fact sentence to plant in context
    question: str                      # question to ask about the needle
    expected: str                      # substring that must appear in a correct response
    haystack: str                      # background text used to fill context
    positions: list[float]             # context positions to sweep, e.g. [0.1, 0.5, 0.9]
    token_counts: list[int]            # context sizes to sweep, e.g. [5000, 10000]
    n_votes: int = 3                   # API calls per cell; result is majority vote
    sleep_between_calls: float = 0.3   # seconds to sleep between API calls
    model: str = "claude-haiku-4-5-20251001"

    def __post_init__(self) -> None:
        if not self.needle.strip():
            raise ValueError("needle must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.expected.strip():
            raise ValueError("expected must not be empty")
        if not self.haystack.strip():
            raise ValueError("haystack must not be empty")
        if not self.positions:
            raise ValueError("positions list must not be empty")
        if not self.token_counts:
            raise ValueError("token_counts list must not be empty")
        for p in self.positions:
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"position {p} is out of range [0.0, 1.0]")
        for tc in self.token_counts:
            if tc <= 0:
                raise ValueError(f"token_count {tc} must be positive")
        if self.n_votes < 1:
            raise ValueError(f"n_votes must be >= 1, got {self.n_votes}")


# ── Runner ───────────────────────────────────────────────────────────────────

class ProbeRunner:
    """Runs NIAH probes against an Anthropic client and returns ProbeResult objects."""

    _PROMPT_TEMPLATE = (
        "Below is a document. Read it carefully, then answer the question.\n\n"
        "<document>\n{context}\n</document>\n\n"
        "Question: {question}\n"
        "Answer concisely:"
    )
    _BUFFER_TOKENS = 80

    def __init__(self, client: anthropic.Anthropic, config: ProbeConfig) -> None:
        """Initialise with an injected Anthropic client and probe configuration."""
        self._client = client
        self._config = config

    def run(self) -> list[ProbeResult]:
        """Run the full sweep over all (token_count x position) combinations."""
        results: list[ProbeResult] = []
        for token_count in self._config.token_counts:
            chunk = self._build_haystack_chunk(token_count)
            for position in self._config.positions:
                result = self._run_cell(chunk, token_count, position)
                results.append(result)
                logger.debug(
                    "probe cell token_count=%d position=%.2f correct=%s",
                    token_count, position, result.correct,
                )
        return results

    def _run_cell(
        self, haystack_chunk: str, target_token_count: int, position: float
    ) -> ProbeResult:
        """Run one cell: inject needle, call Claude n_votes times, majority-vote."""
        self._assert_needle_absent(haystack_chunk)
        context = inject_needle(haystack_chunk, self._config.needle, position)
        prompt = self._PROMPT_TEMPLATE.format(
            context=context, question=self._config.question
        )
        actual_tokens = count_tokens(prompt)

        votes: list[bool] = []
        last_response = ""
        for _ in range(self._config.n_votes):
            resp = self._ask(prompt)
            votes.append(self._is_correct(resp))
            last_response = resp
            time.sleep(self._config.sleep_between_calls)

        correct = sum(votes) >= (self._config.n_votes // 2 + 1)
        return ProbeResult(
            position=position,
            token_count=actual_tokens,
            target_token_count=target_token_count,
            correct=correct,
            response=last_response,
            expected=self._config.expected,
        )

    def _build_haystack_chunk(self, token_count: int) -> str:
        """Truncate haystack to fit inside token_count minus needle + overhead."""
        overhead = count_tokens(
            self._PROMPT_TEMPLATE.format(context="", question=self._config.question)
        )
        needle_tokens = count_tokens(self._config.needle)
        budget = token_count - overhead - needle_tokens - self._BUFFER_TOKENS
        if budget <= 0:
            raise ValueError(
                f"token_count {token_count} is too small to fit needle + overhead. "
                f"Minimum required: {overhead + needle_tokens + self._BUFFER_TOKENS}"
            )
        return truncate_to_tokens(self._config.haystack, budget)

    def _assert_needle_absent(self, haystack: str) -> None:
        """Raise ValueError if expected answer already exists in haystack."""
        if self._config.expected.lower() in haystack.lower():
            raise ValueError(
                f'Expected answer "{self._config.expected}" already appears in the '
                "haystack. Choose a needle whose answer is absent from the background text."
            )

    def _ask(self, prompt: str) -> str:
        """Call Claude and return the stripped text response."""
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _is_correct(self, response: str) -> bool:
        """Return True if the expected answer substring appears in the response."""
        return self._config.expected.lower() in response.lower()


# ── Public helpers (reused by other engine modules) ──────────────────────────

def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding (approximation for Claude)."""
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens, returning the decoded string."""
    return _enc.decode(_enc.encode(text)[:max_tokens])


def inject_needle(haystack: str, needle: str, position: float) -> str:
    """
    Insert needle into haystack at position fraction of total tokens.

    Snaps the insertion point forward to the next sentence-ending period so the
    needle is always placed between sentences, never mid-sentence.

    Args:
        haystack: Background text to receive the needle.
        needle:   Fact sentence to plant.
        position: Float in [0.0, 1.0] indicating depth into the haystack.

    Returns:
        New string with needle inserted at the computed sentence boundary.

    Raises:
        ValueError: If position is outside [0.0, 1.0].
    """
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be in [0.0, 1.0], got {position}")

    h_tokens = _enc.encode(haystack)
    ins_tokens = _enc.encode(" " + needle + " ")
    idx = int(len(h_tokens) * position)

    # Snap forward to the next period to avoid splitting mid-sentence
    period = _enc.encode(".")[0]
    for i in range(idx, min(idx + 200, len(h_tokens))):
        if h_tokens[i] == period:
            idx = i + 1
            break

    return _enc.decode(h_tokens[:idx] + ins_tokens + h_tokens[idx:])
