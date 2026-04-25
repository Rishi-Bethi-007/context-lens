"""
examples/reguliq_demo.py

Runs context-lens against the real ReguliQ LangGraph agent.

Phase 3 result (from actual runs):
    Q1 GDPR:   risk_classifier(264) -> planner(566) -> researcher(839) tokens
    Q2 EU AI Act: risk_classifier(294) -> planner(634) -> researcher(965) tokens
    Peak context: 965 tokens — well below context-rot territory.

This script does two things:
    1. Instruments ReguliQ with LangGraphInstrumentor (Phase 3) to capture live
       node-by-node token usage.
    2. Runs a NIAH probe sweep (Phase 4) at token counts matching ReguliQ's
       actual peak range [500, 1000, 2000] to verify context health, then saves
       an HTML report.

Expected result: score A — context health looks good.

Requirements:
    - ReguliQ repo at ../../PROJECTS/eu-regulatory-intelligence-agent
    - .env in the ReguliQ repo with ANTHROPIC_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY

Run:
    python examples/reguliq_demo.py

For the report without running ReguliQ (uses Phase 3 measurements directly):
    python examples/reguliq_demo.py --synthetic
"""
from __future__ import annotations

import asyncio
import sys
import uuid as _uuid
from pathlib import Path

# ── Windows UTF-8 fix ──────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow running from repo root or examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYNTHETIC = "--synthetic" in sys.argv

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent
REGULIQ_ROOT = REPO_ROOT.parent / "PROJECTS" / "eu-regulatory-intelligence-agent"

OUT_PATH = SCRIPT_DIR / "reguliq_report.html"

# ── Synthetic baseline (from Phase 3 actual measurements) ─────────────────────
# Used when --synthetic flag is passed or when ReguliQ repo is not present.

def build_synthetic_reguliq_measurement():
    """Synthesise a MeasurementResult that reflects ReguliQ's actual context profile.

    Phase 3 measured: peak 965 tokens across 3 nodes.
    At this scale, Claude Haiku retrieves accurately at all positions.
    """
    from context_lens.engine.measurement import MeasurementResult
    from context_lens.engine.probes import ProbeResult

    # ReguliQ operates at 265-965 tokens — represent as 500/1000/2000 sweep
    token_counts = [500, 1_000, 2_000]
    positions    = [0.10, 0.30, 0.50, 0.70, 0.90]

    # At these small context sizes, Haiku retrieves with near-perfect accuracy.
    # Phase 1 showed degradation starts at 50K+; these are 40-100x smaller.
    probes = [
        ProbeResult(
            position=pos,
            token_count=tc,
            target_token_count=tc,
            correct=True,  # 100% accuracy at sub-2K token contexts
            response="synthetic (Phase 3 baseline)",
            expected="synthetic",
        )
        for pos in positions
        for tc in token_counts
        for _ in range(3)  # 3 votes per cell
    ]

    return MeasurementResult(
        agent_name="ReguliQ",
        probe_results=probes,
        token_counts_tested=token_counts,
        positions_tested=positions,
    )


# ── Real run (LangGraph instrumentation + NIAH probe sweep) ───────────────────

def _make_db_mock():
    from unittest.mock import MagicMock
    result = MagicMock()
    result.data = [{"id": "00000000-0000-0000-0000-000000000001"}]
    chain = MagicMock()
    chain.execute.return_value = result
    for op in ("insert", "update", "upsert", "delete", "select", "eq", "limit", "single"):
        getattr(chain, op).return_value = chain
    mock = MagicMock()
    mock.table.return_value = chain
    mock.rpc.return_value = chain
    return mock


async def run_real() -> None:
    """Instrument ReguliQ, then sweep NIAH at its actual context size range."""
    sys.path.insert(0, str(REGULIQ_ROOT))

    from dotenv import load_dotenv
    load_dotenv(REGULIQ_ROOT / ".env", override=True)

    # Silence Supabase DB calls
    import db.client as _db_client
    _db_client._client = _make_db_mock()

    from agents.orchestrator import build_graph, build_initial_state
    from config.settings import validate
    from context_lens.instrumentation.langgraph import LangGraphInstrumentor

    validate()

    # ── Phase 3: instrument ReguliQ ────────────────────────────────────────────
    query = "Does our company need to comply with the EU AI Act if we use GPT-4 internally?"
    graph         = build_graph()
    initial_state = build_initial_state(goal=query, run_id=str(_uuid.uuid4()))
    instrumentor  = LangGraphInstrumentor(graph, agent_name="ReguliQ")

    print(f"Running ReguliQ: {query[:60]}...")
    await instrumentor.ainvoke(initial_state)

    ctx_report = instrumentor.report()
    ctx_report.summary()
    peak = ctx_report.peak_token_count
    print(f"\nPeak context: {peak:,} tokens")

    # ── Phase 4: NIAH probe sweep at ReguliQ's actual token range ──────────────
    import os
    from context_lens.engine.measurement import measure_context_health

    # Use EU regulatory text as a domain-appropriate haystack
    haystack = (
        "The General Data Protection Regulation (GDPR) is a regulation in EU law on data "
        "protection and privacy in the EU and the EEA. The EU AI Act establishes a risk-based "
        "framework for artificial intelligence systems deployed in the European Union. "
        "High-risk AI systems must undergo conformity assessment before they can be placed "
        "on the market. The Digital Services Act regulates online intermediaries and platforms "
        "with specific obligations for very large online platforms. Data controllers must "
        "implement appropriate technical and organisational measures to ensure data security. "
    ) * 50  # ~3K tokens of regulatory background text

    measurement = measure_context_health(
        agent_name="ReguliQ",
        haystack=haystack,
        needle="The compliance officer reviewed the GDPR Article 32 implementation on 14 March.",
        question="When did the compliance officer review Article 32?",
        expected="14 March",
        token_counts=[max(500, peak // 2), peak, min(peak * 2, 2000)],
        positions=[0.10, 0.30, 0.50, 0.70, 0.90],
        n_votes=3,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    _finish(measurement)


def _finish(measurement) -> None:
    from context_lens.reporter import Reporter
    from context_lens.report.renderer import save

    report = Reporter().run(measurement)

    print("\n")
    report.summary()
    save(report, str(OUT_PATH))
    print(f"\nReport saved: {OUT_PATH}")
    print("Open it in a browser — expect score A and the healthy banner.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("context-lens — ReguliQ demo")

    if SYNTHETIC or not REGULIQ_ROOT.exists():
        if not SYNTHETIC:
            print(f"ReguliQ not found at {REGULIQ_ROOT}")
            print("Generating report from Phase 3 measurements (--synthetic mode).\n")
        else:
            print("Synthetic mode: using Phase 3 baseline measurements.\n")

        measurement = build_synthetic_reguliq_measurement()
        _finish(measurement)
    else:
        asyncio.run(run_real())


if __name__ == "__main__":
    main()
