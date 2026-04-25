# context-lens

Tells you **where** your LLM agent's memory breaks, **at what token count**, and **how to fix it**.

```
pip install reguliq-diagnostics
```

---

## Quickstart

```python
from context_lens.engine.measurement import measure_context_health
from context_lens.reporter import Reporter

# 1. Probe your agent's context window
result = measure_context_health(
    agent_name="my-rag-agent",
    haystack=my_background_text,
    needle="The Q3 revenue was $4.2M",
    question="What was Q3 revenue?",
    expected="4.2M",
)

# 2. Run all 6 classifiers
report = Reporter().run(result)

# 3. View results
report.summary()          # terminal output
report.save("report.html")  # open in browser
```

---

## What it finds

| Pattern | What it means | Severity |
|---|---|---|
| `beginning_anchored` | Model retrieves facts only from the first 15% of context | HIGH |
| `cliff_detector` | Accuracy drops >20% between adjacent token counts | HIGH |
| `distractor_confusion` | Near-miss facts in context cause wrong answers | HIGH |
| `tool_burial` | Accuracy collapses after 3rd+ sequential tool call | MEDIUM |
| `instruction_drift` | System-prompt constraints weaken over conversation turns | MEDIUM |
| `recency_bias` | Model ignores everything except the last 20% of context | MEDIUM |

---

## Demo

### ReguliQ (production LangGraph agent) — healthy

> Instrumented with real LangGraph callbacks. Peak context: 965 tokens.
> At that scale, Claude retrieves with 100% accuracy.

```
context-lens: ReguliQ
  score: A  |  mean accuracy: 100.0%  |  5 classifiers run
  no patterns detected — context health looks good
```

[View reguliq_report.html](examples/reguliq_report.html)

### Synthetic unhealthy agent — context degradation detected

> Beginning-anchored retrieval + cliff at 30K tokens.

```
context-lens: my-rag-agent (synthetic)
  score: F  |  mean accuracy: 35.0%  |  5 classifiers run
  cliff: 30,000 tokens
  4 pattern(s) detected:
    [MEDIUM] beginning_anchored  conf=0.50
    [MEDIUM] cliff_detector      conf=0.58
    [HIGH  ] tool_burial         conf=0.62
    [HIGH  ] instruction_drift   conf=0.62
```

[View unhealthy_report.html](examples/unhealthy_report.html)

---

## Architecture

```
context_lens/
├── engine/
│   ├── probes.py          # NIAH probe injection + needle-in-haystack runs
│   ├── measurement.py     # sweeps positions × token counts, returns MeasurementResult
│   └── snapshots.py       # ContextSnapshot capture for live agents
│
├── classifiers/           # 6 pattern detectors (detect() + recommend())
│   ├── beginning_anchored.py
│   ├── cliff_detector.py
│   ├── distractor_confusion.py
│   ├── tool_burial.py
│   ├── instruction_drift.py
│   └── recency_bias.py
│
├── instrumentation/
│   └── langgraph.py       # LangGraphInstrumentor — wraps any compiled graph
│
├── reporter.py            # Reporter.run() → ReportData (score + recommendations)
│
└── report/
    ├── renderer.py        # renders ReportData → self-contained HTML (no CDN)
    └── template.html      # dark terminal theme, SVG charts, zero dependencies
```

### How it works

```
your agent          context-lens
──────────          ────────────────────────────────────
LangGraph    ──►  LangGraphInstrumentor
   graph           │  captures token counts per node
                   ▼
             measure_context_health()
                   │  plants NIAH probes at each
                   │  position × token count cell
                   ▼
             MeasurementResult
                   │  accuracy_by_position()
                   │  accuracy_by_token_count()
                   ▼
             Reporter.run()
                   │  runs all 6 classifiers
                   │  computes A-F grade
                   ▼
             ReportData.save("report.html")
```

---

## Installation

```bash
# Core (probing + classifiers + HTML report)
pip install reguliq-diagnostics

# LangGraph instrumentation
pip install "reguliq-diagnostics[langgraph]"

# Development
pip install "reguliq-diagnostics[dev]"
```

> Import name is unchanged: `import context_lens`

---

## Running the demos

```bash
# Unhealthy agent (synthetic — no API key needed)
python examples/unhealthy_agent_demo.py

# ReguliQ (requires API keys + ReguliQ repo)
python examples/reguliq_demo.py

# ReguliQ with Phase 3 baseline only (no API calls)
python examples/reguliq_demo.py --synthetic
```

---

## Dev

```powershell
# Setup (Windows)
uv venv && .venv\Scripts\activate
uv pip install -e ".[dev,langgraph]"

# Test
pytest tests/ -v --cov=context_lens

# Build
uv build
```

![209 tests passing](https://img.shields.io/badge/tests-209%20passing-brightgreen)

---

## License

MIT
