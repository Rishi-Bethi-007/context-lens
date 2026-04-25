# context-lens

Python library that diagnoses context degradation in LLM agents. Tells you exactly where your agent fails, at what token count, and how to fix it.

```bash
pip install context-lens
```
```python
from context_lens import diagnose
report = diagnose(agent=my_agent, test_inputs=["..."])
report.show()  # opens HTML diagnostic report
```

## Architecture

```
context_lens/
├── engine/          # probe injection + measurement
├── classifiers/     # 6 failure pattern detectors
├── instrumentation/ # LangGraph + OpenAI + Anthropic wrappers
├── report/          # HTML report generator (Plotly + Jinja2)
└── recommendations/ # pattern → fix mapper
```

See `.claude/rules/` for detailed specs per module.

## Dev Commands

```powershell
# Windows PowerShell — never use `export`
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Setup
uv venv && .venv\Scripts\activate
uv pip install -e ".[dev]"

# Test
pytest tests/ -v --cov=context_lens

# Build + publish
uv build && uv publish
```

## Current Phase

**PHASE 1 — Research** (Weeks 1–2)

Chroma's repo is at `C:\Users\rishi\LU\AI Engg\context-rot\context-rot` — learn from it, don't modify it.

Next step: run their `repeated_words` experiment with Anthropic + Claude Haiku, then write our own NIAH probe in `notebooks/01_research_experiments.ipynb`.

## Key Constraints

- **Windows**: use `$env:VAR` not `export VAR`
- **Cost**: use `claude-haiku-4-5-20251001` always during dev. Sonnet only for final demo.
- **No network calls in tests** — mock all LLM calls with pytest-mock
- **Never modify user's agent** — instrumentation is purely observational
- **No database, no Docker, no cloud infra** — local Python library only

## Stack

| Tool | Purpose |
|------|---------|
| anthropic + openai | LLM API calls |
| tiktoken | token counting |
| langgraph + langchain-core | agent instrumentation hooks |
| plotly | charts in HTML report |
| jinja2 | report templating |
| pytest + pytest-cov | testing |
| uv | package management |

## Demo Target

Run context-lens against ReguliQ (production LangGraph compliance agent) at:
`C:\Users\rishi\LU\AI Engg\PROJECTS\eu-regulatory-intelligence-agent`

The README story: "I ran this on my production agent. Found it degrades at 31K tokens. Here's the report."

