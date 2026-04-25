# Mistakes & Design Decisions

## Decisions Made
- **Haiku for all dev, Sonnet for demo only** — 20x cost difference
- **No database** — library runs locally, no persistence needed
- **Single HTML file report** — must work offline, be emailable
- **LangSmith callbacks for instrumentation** — non-invasive, already in LangGraph
- **OpenAI + Anthropic only** — skip Google (too complex setup for users)
- **uv over pip/poetry** — modern, fast, signals current tooling knowledge

## Known Gotchas
- **Windows**: `$env:KEY = "val"` not `export KEY=val`
- **Chroma's data**: requires Google Drive download — generate our own haystack instead
- **tiktoken models**: use `cl100k_base` encoding for Claude token counting approximation
- **LangGraph callbacks**: use `on_chain_start`/`on_chain_end` not node-level hooks for compatibility

## What NOT to Build
These exist — do not rebuild them:
- LLM monitoring dashboard → Langfuse, Arize Phoenix
- LLM eval framework → DeepEval, RAGAS
- Agent framework → LangGraph, CrewAI
- Prompt management → Langfuse, PromptLayer

## Mistakes to Avoid
- Do not try to instrument CrewAI in v1 — focus on LangGraph only
- Do not generate haystacks with Lorem Ipsum — use real text (PG essays, Wikipedia)
- Do not run full test suite against real API — always mock in tests
- Do not measure position accuracy with < 10 probes per position — too noisy

## Cost Decisions (learned from Phase 1)
- **Option A (50K-200K token cliff detection) costs $2.40 minimum** — defer to after we have more credits. The 200K context row alone is 3M input tokens × 15 calls.
- **Option B (semantic needle + distractors at 5K-30K) costs $0.24** and is sufficient for Phase 1. It already proves distractor confusion and position bias with real numbers.
- We will return to Option A once credits are topped up. For now Option B proves the core concept cheaply and we move to Phase 2.

## ReguliQ Demo Strategy (learned from Phase 3)
- **ReguliQ peaks at ~900 tokens** — well below context rot territory. Demo story: show context-lens on ReguliQ (healthy result) then show synthetic unhealthy agent for comparison.
- **ReguliQ has a DB serialization bug at researcher node** — "object supporting buffer API required" — unrelated to context-lens; do not fix.

## Probe Design Rules (learned from Phase 1)
- **Probes must require comprehension, not pattern matching.** A unique string like "ZEPHYR-4829" is trivially found by the model regardless of token count or position — it degrades like ctrl+F. Use semantically embedded facts.
- **Needles must be semantically similar to distractors.** Without near-miss wrong answers in the haystack, the model has no reason to confuse or skip the needle. Add 5+ distractors with the same structure but wrong values.
- **Start token count experiments at 50K minimum for modern models.** Claude Haiku handles 30K tokens with 100% accuracy on simple retrieval. Context degradation for Haiku shows up at 50K–200K. At 30K you are not even testing the model's limits.
- **Distractor insertion order matters.** Insert distractors before the needle (back-to-front by position) so the needle's final position is predictable relative to the clean haystack.
