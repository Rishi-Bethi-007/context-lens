"""Tests for context_lens.instrumentation.langgraph — no real API calls."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from context_lens.instrumentation.langgraph import (
    ContextLensCallback,
    ContextReport,
    LangGraphInstrumentor,
    NodeSnapshot,
    _count_state_tokens,
)


# ── _count_state_tokens ───────────────────────────────────────────────────────

def test_count_state_tokens_none():
    assert _count_state_tokens(None) == 0


def test_count_state_tokens_empty_dict():
    # {} serialises to the string "{}" which is 1 token
    assert _count_state_tokens({}) == 1


def test_count_state_tokens_string():
    # "hello" serializes to '"hello"' in JSON
    result = _count_state_tokens("hello")
    assert result > 0


def test_count_state_tokens_increases_with_content():
    small = _count_state_tokens({"key": "value"})
    large = _count_state_tokens({"key": "value " * 200})
    assert large > small


def test_count_state_tokens_nested_dict():
    state = {"goal": "What is GDPR?", "output": "GDPR is ...", "tokens_used": 100}
    assert _count_state_tokens(state) > 0


# ── NodeSnapshot ──────────────────────────────────────────────────────────────

def test_node_snapshot_token_delta_positive():
    snap = NodeSnapshot(
        node_name="planner", run_id="abc", tokens_in=100, tokens_out=150,
        duration_ms=50.0,
    )
    assert snap.token_delta == 50


def test_node_snapshot_token_delta_negative():
    snap = NodeSnapshot(
        node_name="summarizer", run_id="abc", tokens_in=500, tokens_out=200,
        duration_ms=30.0,
    )
    assert snap.token_delta == -300


# ── ContextReport ─────────────────────────────────────────────────────────────

def _make_report(nodes: list[tuple[str, int, int]]) -> ContextReport:
    """Build a ContextReport from (name, tokens_in, tokens_out) tuples."""
    snaps = [
        NodeSnapshot(node_name=n, run_id="r", tokens_in=ti, tokens_out=to, duration_ms=1.0)
        for n, ti, to in nodes
    ]
    return ContextReport(agent_name="test", node_snapshots=snaps)


def test_report_token_counts_by_node():
    report = _make_report([("planner", 100, 200), ("researcher", 200, 800)])
    by_node = report.token_counts_by_node()
    assert by_node["planner"] == 200
    assert by_node["researcher"] == 800


def test_report_peak_token_count():
    report = _make_report([("a", 100, 500), ("b", 500, 300)])
    assert report.peak_token_count == 500


def test_report_peak_token_count_empty():
    report = ContextReport(agent_name="empty")
    assert report.peak_token_count == 0


def test_report_summary_no_crash_empty(capsys):
    ContextReport(agent_name="empty").summary()
    captured = capsys.readouterr()
    assert "empty" in captured.out


def test_report_summary_prints_node_names(capsys):
    report = _make_report([("planner", 100, 200), ("researcher", 200, 800)])
    report.summary()
    captured = capsys.readouterr()
    assert "planner" in captured.out
    assert "researcher" in captured.out


def test_report_summary_prints_peak(capsys):
    report = _make_report([("a", 0, 999)])
    report.summary()
    captured = capsys.readouterr()
    assert "999" in captured.out


# ── ContextLensCallback — hierarchy tracking ───────────────────────────────────

def _fire_chain_start(
    cb: ContextLensCallback,
    name: str,
    inputs: dict,
    run_id: UUID,
    parent_run_id: UUID | None,
):
    cb.on_chain_start(
        serialized={"name": name},
        inputs=inputs,
        run_id=run_id,
        parent_run_id=parent_run_id,
    )


def _fire_chain_end(
    cb: ContextLensCallback,
    outputs: dict,
    run_id: UUID,
    parent_run_id: UUID | None,
):
    cb.on_chain_end(outputs=outputs, run_id=run_id, parent_run_id=parent_run_id)


def test_callback_captures_node_after_graph_start():
    cb = ContextLensCallback(agent_name="test")
    graph_id = uuid4()
    node_id = uuid4()

    _fire_chain_start(cb, "LangGraph", {"goal": "hello"}, graph_id, None)
    _fire_chain_start(cb, "planner", {"goal": "hello"}, node_id, graph_id)
    _fire_chain_end(cb, {"goal": "hello", "plan": "x"}, node_id, graph_id)

    assert len(cb.node_snapshots) == 1
    assert cb.node_snapshots[0].node_name == "planner"


def test_callback_ignores_graph_level_end():
    cb = ContextLensCallback(agent_name="test")
    graph_id = uuid4()

    _fire_chain_start(cb, "LangGraph", {}, graph_id, None)
    _fire_chain_end(cb, {"result": "done"}, graph_id, None)

    assert len(cb.node_snapshots) == 0


def test_callback_ignores_deeply_nested_chains():
    cb = ContextLensCallback(agent_name="test")
    graph_id = uuid4()
    node_id = uuid4()
    llm_id = uuid4()

    _fire_chain_start(cb, "LangGraph", {}, graph_id, None)
    _fire_chain_start(cb, "planner", {}, node_id, graph_id)
    # LLM call inside the node — parent is node, not graph
    _fire_chain_start(cb, "ChatAnthropic", {}, llm_id, node_id)
    _fire_chain_end(cb, {"text": "response"}, llm_id, node_id)
    _fire_chain_end(cb, {"plan": "done"}, node_id, graph_id)

    assert len(cb.node_snapshots) == 1
    assert cb.node_snapshots[0].node_name == "planner"


def test_callback_captures_multiple_nodes_in_order():
    cb = ContextLensCallback(agent_name="test")
    graph_id = uuid4()
    node_ids = [uuid4() for _ in range(3)]
    node_names = ["planner", "researcher", "synthesizer"]

    _fire_chain_start(cb, "LangGraph", {}, graph_id, None)
    for nid, name in zip(node_ids, node_names):
        _fire_chain_start(cb, name, {"step": name}, nid, graph_id)
        _fire_chain_end(cb, {"step": name, "done": True}, nid, graph_id)

    assert [s.node_name for s in cb.node_snapshots] == node_names


def test_callback_records_token_counts():
    cb = ContextLensCallback(agent_name="test")
    graph_id = uuid4()
    node_id = uuid4()

    small_state = {"goal": "test"}
    large_state = {"goal": "test", "output": "x " * 500}

    _fire_chain_start(cb, "LangGraph", {}, graph_id, None)
    _fire_chain_start(cb, "synthesizer", small_state, node_id, graph_id)
    _fire_chain_end(cb, large_state, node_id, graph_id)

    snap = cb.node_snapshots[0]
    assert snap.tokens_in < snap.tokens_out
    assert snap.tokens_out > 0


def test_callback_error_clears_pending():
    cb = ContextLensCallback(agent_name="test")
    graph_id = uuid4()
    node_id = uuid4()

    _fire_chain_start(cb, "LangGraph", {}, graph_id, None)
    _fire_chain_start(cb, "planner", {}, node_id, graph_id)
    cb.on_chain_error(ValueError("oops"), run_id=node_id, parent_run_id=graph_id)

    assert node_id.__str__() not in cb._pending
    assert len(cb.node_snapshots) == 0


def test_callback_report_returns_context_report():
    cb = ContextLensCallback(agent_name="my-agent")
    report = cb.report()
    assert isinstance(report, ContextReport)
    assert report.agent_name == "my-agent"


def test_callback_no_graph_start_ignores_nodes():
    """If on_chain_start for graph never fires, node events are ignored."""
    cb = ContextLensCallback(agent_name="test")
    node_id = uuid4()
    orphan_id = uuid4()

    # Fire a node start with some random parent (not graph_run_id, which is None)
    _fire_chain_start(cb, "planner", {}, node_id, orphan_id)
    _fire_chain_end(cb, {"x": 1}, node_id, orphan_id)

    assert len(cb.node_snapshots) == 0


# ── LangGraphInstrumentor ─────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_instrumentor_ainvoke_returns_graph_result():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"final_output": "report text"}

    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test-agent")
    result = await instrumentor.ainvoke({"goal": "test"})

    assert result == {"final_output": "report text"}


@pytest.mark.anyio
async def test_instrumentor_ainvoke_injects_callback():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {}

    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test-agent")
    await instrumentor.ainvoke({"goal": "test"})

    _, kwargs = mock_graph.ainvoke.call_args
    config = kwargs.get("config", {})
    assert "callbacks" in config
    assert any(isinstance(cb, ContextLensCallback) for cb in config["callbacks"])


@pytest.mark.anyio
async def test_instrumentor_ainvoke_preserves_existing_callbacks():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {}
    existing_cb = MagicMock()

    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test-agent")
    await instrumentor.ainvoke({"goal": "test"}, config={"callbacks": [existing_cb]})

    _, kwargs = mock_graph.ainvoke.call_args
    cbs = kwargs["config"]["callbacks"]
    assert existing_cb in cbs
    assert any(isinstance(cb, ContextLensCallback) for cb in cbs)


@pytest.mark.anyio
async def test_instrumentor_ainvoke_does_not_mutate_original_config():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {}
    original_config = {"callbacks": [], "tags": ["v1"]}

    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test-agent")
    await instrumentor.ainvoke({}, config=original_config)

    # Original config should be untouched
    assert original_config["callbacks"] == []


def test_instrumentor_invoke_returns_graph_result():
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"final_output": "sync result"}

    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test-agent")
    result = instrumentor.invoke({"goal": "test"})

    assert result == {"final_output": "sync result"}


def test_instrumentor_invoke_injects_callback():
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {}

    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test-agent")
    instrumentor.invoke({"goal": "test"})

    _, kwargs = mock_graph.invoke.call_args
    config = kwargs.get("config", {})
    assert any(isinstance(cb, ContextLensCallback) for cb in config.get("callbacks", []))


def test_instrumentor_node_snapshots_empty_before_invoke():
    mock_graph = MagicMock()
    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test")
    assert instrumentor.node_snapshots == []


def test_instrumentor_report_empty_before_invoke():
    mock_graph = MagicMock()
    instrumentor = LangGraphInstrumentor(mock_graph, agent_name="test")
    report = instrumentor.report()
    assert isinstance(report, ContextReport)
    assert report.node_snapshots == []
