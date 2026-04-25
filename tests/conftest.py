"""Shared pytest fixtures for the context-lens test suite."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_haystack() -> str:
    """A long plain-text haystack with no needle content."""
    paragraph = (
        "The industrial revolution transformed European society in profound ways. "
        "Factories replaced cottage industries and urbanization accelerated rapidly. "
        "Workers moved from rural areas to cities seeking employment in the new mills. "
        "Social structures changed as a new middle class emerged alongside the working class. "
        "Technological innovation drove productivity gains across many sectors of the economy. "
    )
    return (paragraph * 120).strip()  # ~4000 tokens, enough for small test contexts


@pytest.fixture
def make_mock_client():
    """Factory fixture: returns a function that creates a mock Anthropic client."""

    def _make(response_text: str = "correct answer") -> MagicMock:
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=response_text)]
        client.messages.create.return_value = mock_response
        return client

    return _make
