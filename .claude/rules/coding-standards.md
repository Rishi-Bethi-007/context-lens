# Coding Standards

## Always
- Type hints on every function
- Docstring on every public function (one line minimum)
- Return dataclasses, not raw dicts
- Raise specific exceptions with helpful messages
- Use `logging` at DEBUG level inside library — never `print()`

## Never
- Hardcode API keys — always `os.environ.get()` or dotenv
- Make real API/network calls in tests — mock with `pytest-mock`
- Modify the user's agent during instrumentation — read-only
- Make network calls on `import context_lens`
- Use bare `except Exception` — catch specific exceptions

## Testing Pattern
```python
# Mock all LLM calls
@pytest.fixture
def mock_anthropic(mocker):
    return mocker.patch("context_lens.engine.probes.anthropic.Anthropic")

def test_probe_injection(mock_anthropic):
    mock_anthropic.return_value.messages.create.return_value = ...
```

## File Naming
- Snake case: `lost_in_middle.py` not `LostInMiddle.py`
- Tests mirror source: `tests/test_probes.py` ↔ `context_lens/engine/probes.py`

## pyproject.toml (key fields)
```toml
[project]
name = "context-lens"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.52.0", "openai>=1.84.0", "tiktoken>=0.9.0",
    "plotly>=5.0.0", "jinja2>=3.0.0", "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
langgraph = ["langgraph>=0.2.0", "langchain-core>=0.3.0"]
dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0", "pytest-mock>=3.0.0"]
```
