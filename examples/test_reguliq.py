"""
examples/test_reguliq.py

Tests context-lens instrumentation against a real ReguliQ run.
This script makes REAL LLM API calls (Anthropic + OpenAI + Tavily).
Supabase DB calls are silenced with a MagicMock so no live DB is needed.

Run from the context-lens project root:
    python examples/test_reguliq.py

Or with ReguliQ's venv (if context-lens is not installed there):
    cd .../eu-regulatory-intelligence-agent
    .venv/Scripts/python .../context-lens/examples/test_reguliq.py

Requires:
    - ReguliQ repo at ../../PROJECTS/eu-regulatory-intelligence-agent
    - .env in the ReguliQ repo with ANTHROPIC_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY set
"""
from __future__ import annotations

import asyncio
import sys
import uuid as _uuid

# Windows terminals default to cp1252; ReguliQ prints emoji — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from unittest.mock import MagicMock

# ── Resolve paths ──────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent
REGULIQ_ROOT = REPO_ROOT.parent / "PROJECTS" / "eu-regulatory-intelligence-agent"

if not REGULIQ_ROOT.exists():
    sys.exit(f"ERROR: ReguliQ not found at {REGULIQ_ROOT}")

# Add both repos to sys.path before any imports
sys.path.insert(0, str(REPO_ROOT))      # context-lens package
sys.path.insert(0, str(REGULIQ_ROOT))   # ReguliQ agents, config, etc.

# Load ReguliQ's .env (LLM API keys)
from dotenv import load_dotenv
load_dotenv(REGULIQ_ROOT / ".env", override=True)

# ── Silence Supabase DB calls ─────────────────────────────────────────────────
# ReguliQ uses Supabase only for logging; inject a mock client before the first
# DB call so that all db.client functions (which call get_client() lazily) use it.

def _make_db_mock() -> MagicMock:
    """Build a MagicMock Supabase client where all call chains end in valid .data."""
    _result = MagicMock()
    _result.data = [{"id": "00000000-0000-0000-0000-000000000001"}]

    # A single chain object: every table op returns itself so .execute() always works
    _chain = MagicMock()
    _chain.execute.return_value = _result
    for _op in ("insert", "update", "upsert", "delete", "select", "eq", "limit", "single"):
        getattr(_chain, _op).return_value = _chain

    _mock = MagicMock()
    _mock.table.return_value = _chain
    _mock.rpc.return_value = _chain
    return _mock

# ── context-lens imports ───────────────────────────────────────────────────────

from context_lens.instrumentation.langgraph import LangGraphInstrumentor

# ── ReguliQ imports (after sys.path and env are configured) ───────────────────

from agents.orchestrator import build_graph, build_initial_state
from config.settings import validate

# Inject the mock Supabase client before any graph run so all db.client
# functions (which call get_client() at call-time) use it.
import db.client as _db_client
_db_client._client = _make_db_mock()  # pre-populate singleton → no real HTTP calls

# ── Test queries ───────────────────────────────────────────────────────────────

QUERIES = [
    "What are the GDPR requirements for data retention in Sweden?",
    "Does our company need to comply with the EU AI Act if we use GPT-4 internally?",
]


# ── Runner ─────────────────────────────────────────────────────────────────────

async def run_with_instrumentation(goal: str, query_num: int):
    print(f"\n{'=' * 70}")
    print(f"QUERY {query_num}: {goal}")
    print(f"{'=' * 70}")

    run_id        = str(_uuid.uuid4())  # skip DB insert — ID is all we need
    graph         = build_graph()
    initial_state = build_initial_state(goal=goal, run_id=run_id)

    # Wrap with context-lens — zero changes to graph or state
    instrumentor = LangGraphInstrumentor(graph, agent_name=f"ReguliQ-Q{query_num}")

    final_state = await instrumentor.ainvoke(initial_state)

    # ── Print ReguliQ result ───────────────────────────────────────────────────
    output = final_state.get("final_output", "")
    print(f"\n--- ReguliQ output (first 500 chars) ---")
    print(output[:500] + ("..." if len(output) > 500 else ""))

    print(f"\nRisk level : {final_state.get('risk_level', 'N/A')}")
    print(f"Tokens used: {final_state.get('tokens_used', 0):,}")
    print(f"Cost (USD) : ${final_state.get('cost_usd', 0):.6f}")

    # ── Print context-lens report ──────────────────────────────────────────────
    report = instrumentor.report()
    report.summary()

    by_node = report.token_counts_by_node()
    if by_node:
        print("\nNodes visited (in order):")
        for node, tokens in by_node.items():
            print(f"  {node:<22} {tokens:>10,} tokens out")

    print(f"\nPeak context size : {report.peak_token_count:,} tokens")
    print(f"Nodes captured    : {len(report.node_snapshots)}")

    return report


async def main() -> None:
    validate()

    reports = []
    for i, goal in enumerate(QUERIES, start=1):
        report = await run_with_instrumentation(goal, i)
        reports.append(report)

    # ── Cross-query summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-QUERY CONTEXT SUMMARY")
    print(f"{'=' * 70}")
    for i, report in enumerate(reports, start=1):
        print(f"Query {i}: peak={report.peak_token_count:,} tokens  "
              f"nodes={len(report.node_snapshots)}")


if __name__ == "__main__":
    asyncio.run(main())
