# Build Phases

## Phase 1 — Research (Weeks 1–2) ✓ COMPLETE
Goal: reproduce context rot with our own code, get real numbers.

Real finding: NOT a U-shaped lost-in-middle pattern. Model reads full context
but fails disambiguation when distractors present. Only position 10–15% is
reliable (83% accuracy). All other positions ≤25%. Named this:
**beginning-anchored retrieval**. This changes classifier design for Phase 4 —
distractor and lost_in_middle classifiers need different detection logic than
originally planned. 90% (recency) position fails just as badly as 50%.

Results saved in results/option_b_results.json and option_b_heatmap.png.
Option A (50K–200K cliff detection) deferred until credits topped up.

## Phase 2 — Core Engine (Weeks 3–4) ✓ COMPLETE
Build: `engine/probes.py`, `engine/snapshots.py`, `engine/measurement.py`
97/97 tests passing, 98% coverage.

## Phase 3 — LangGraph Instrumentation (Weeks 5–6) ✓ COMPLETE
Built: `instrumentation/langgraph.py` using LangChain BaseCallbackHandler hooks.
29/29 tests passing, 97% coverage.
Tested against ReguliQ (real LLM calls):
  - Node name extracted from metadata["langgraph_node"] (not serialized["name"])
  - Q1 GDPR: risk_classifier(264) → planner(566) → researcher(839) tokens, peak=839
  - Q2 EU AI Act: risk_classifier(294) → planner(634) → researcher(965) tokens, peak=965
  - Both queries: MINIMAL_RISK, graph stopped at researcher due to ReguliQ DB bug
Gate: zero modifications to ReguliQ code — confirmed.

## Phase 4 — Classifiers (Weeks 7–8) ← CURRENT
Build all 6 classifiers + `recommendations/engine.py`
Each classifier needs: detect(), recommend(), synthetic test data
Gate: all classifiers detect their pattern in synthetic data

## Phase 5 — HTML Report (Week 9)
Build: Plotly heatmap + degradation curve + Jinja2 template
Single self-contained .html file. Dark theme. Terminal aesthetic.
Gate: screenshot looks good enough to post on Twitter without explanation

## Phase 6 — Launch (Week 10)
PyPI publish → HN post → LangChain Discord → Twitter @atroyn @hwchase17
README must show real ReguliQ diagnostic numbers in first scroll
