"""LangGraph instrumentation using LangChain/LangSmith callback hooks.

Non-invasive: wraps graph.ainvoke/invoke to inject a callback; zero changes
to the user's agent code.

Captures token counts at each LangGraph node boundary by tracking the
parent_run_id hierarchy:
    graph (parent_run_id=None) → nodes (parent_run_id=graph_run_id) → ...
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangGraph instrumentation requires langchain-core. "
        "Install with: pip install 'context-lens[langgraph]'"
    ) from exc

import tiktoken

logger = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")


# ── Token counting ─────────────────────────────────────────────────────────────

def _count_state_tokens(state: Any) -> int:
    """Estimate the token count of a LangGraph state by serialising to JSON."""
    if state is None:
        return 0
    try:
        text = json.dumps(state, default=str, ensure_ascii=False)
        return len(_enc.encode(text))
    except Exception:
        return 0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class NodeSnapshot:
    """Context data captured at a single LangGraph node boundary."""

    node_name: str
    run_id: str
    tokens_in: int
    tokens_out: int
    duration_ms: float
    state_in: dict = field(repr=False, default_factory=dict)
    state_out: dict = field(repr=False, default_factory=dict)

    @property
    def token_delta(self) -> int:
        """tokens_out minus tokens_in."""
        return self.tokens_out - self.tokens_in


@dataclass
class ContextReport:
    """Aggregated context health report for one full graph run."""

    agent_name: str
    node_snapshots: list[NodeSnapshot] = field(default_factory=list)

    def token_counts_by_node(self) -> dict[str, int]:
        """Return {node_name: tokens_out} for each visited node in order."""
        return {s.node_name: s.tokens_out for s in self.node_snapshots}

    @property
    def peak_token_count(self) -> int:
        """Maximum token count seen at any node boundary."""
        if not self.node_snapshots:
            return 0
        return max(max(s.tokens_in, s.tokens_out) for s in self.node_snapshots)

    def summary(self) -> None:
        """Print a formatted node-by-node table to stdout."""
        print(f"\n=== Context-Lens Report: {self.agent_name} ===")
        if not self.node_snapshots:
            print("  (no node snapshots captured)")
            return
        print(f"{'Node':<22} {'Tokens In':>10} {'Tokens Out':>11} {'Delta':>8} {'ms':>8}")
        print("-" * 63)
        for s in self.node_snapshots:
            sign = "+" if s.token_delta >= 0 else ""
            print(
                f"{s.node_name:<22} {s.tokens_in:>10,} {s.tokens_out:>11,} "
                f"{sign}{s.token_delta:>7,} {s.duration_ms:>8.0f}"
            )
        print("-" * 63)
        print(f"{'Peak tokens:':<22} {self.peak_token_count:>10,}")
        print(f"{'Nodes visited:':<22} {len(self.node_snapshots):>10}")


# ── Callback handler ───────────────────────────────────────────────────────────

class ContextLensCallback(BaseCallbackHandler):
    """
    LangChain callback that records state token counts at each LangGraph node.

    LangGraph fires on_chain_start/on_chain_end for both the graph and each
    individual node.  The graph call has parent_run_id=None; node calls have
    parent_run_id equal to the graph's run_id.  We use this hierarchy to
    capture only node-level events.
    """

    raise_error: bool = False  # don't let callback errors kill the agent

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name
        self._graph_run_id: str | None = None
        self._pending: dict[str, dict[str, Any]] = {}  # run_id → pending node data
        self.node_snapshots: list[NodeSnapshot] = []

    # ── LangChain callback interface ───────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        name = (serialized or {}).get("name", "") or ""

        if parent_run_id is None:
            # Top-level graph invocation — record graph run_id, do not snapshot
            self._graph_run_id = run_id_str
            logger.debug("Graph started: run_id=%s name=%r", run_id_str, name)
            return

        parent_id_str = str(parent_run_id)
        if parent_id_str != self._graph_run_id:
            # Deeper nesting (e.g. LLM calls inside a node) — ignore
            return

        # Direct child of graph = a LangGraph node.
        # LangGraph (0.2+) stores the node label in metadata["langgraph_node"];
        # fall back to serialized["name"] for older versions.
        node_name = (metadata or {}).get("langgraph_node") or name

        tokens_in = _count_state_tokens(inputs)
        self._pending[run_id_str] = {
            "name": node_name,
            "tokens_in": tokens_in,
            "start_time": time.monotonic(),
            "state_in": inputs if isinstance(inputs, dict) else {},
        }
        logger.debug("Node started: %r  tokens_in=%d", node_name, tokens_in)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        pending = self._pending.pop(run_id_str, None)
        if pending is None:
            return  # graph-level end or deeper nesting — skip

        tokens_out = _count_state_tokens(outputs)
        duration_ms = (time.monotonic() - pending["start_time"]) * 1000.0

        snapshot = NodeSnapshot(
            node_name=pending["name"],
            run_id=run_id_str,
            tokens_in=pending["tokens_in"],
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            state_in=pending["state_in"],
            state_out=outputs if isinstance(outputs, dict) else {},
        )
        self.node_snapshots.append(snapshot)
        logger.debug(
            "Node ended: %r  in=%d  out=%d  delta=%+d  (%.0f ms)",
            pending["name"], pending["tokens_in"], tokens_out,
            tokens_out - pending["tokens_in"], duration_ms,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        self._pending.pop(run_id_str, None)
        logger.warning("Node error (run_id=%s): %s", run_id_str, error)

    # ── Report factory ─────────────────────────────────────────────────────────

    def report(self) -> ContextReport:
        """Build a ContextReport from all captured node snapshots."""
        return ContextReport(
            agent_name=self.agent_name,
            node_snapshots=list(self.node_snapshots),
        )


# ── Instrumentor ───────────────────────────────────────────────────────────────

class LangGraphInstrumentor:
    """
    Wraps a compiled LangGraph and injects context-lens callbacks transparently.

    The wrapped graph returns exactly the same output as an un-instrumented
    graph — no state is modified.

    Usage::

        graph = build_graph()
        instrumentor = LangGraphInstrumentor(graph, agent_name="my-agent")

        result = await instrumentor.ainvoke(initial_state)

        instrumentor.report().summary()
        by_node = instrumentor.report().token_counts_by_node()
    """

    def __init__(self, graph: Any, *, agent_name: str) -> None:
        """
        Args:
            graph: A compiled LangGraph (result of StateGraph.compile()).
            agent_name: Human-readable name used in reports.
        """
        self._graph = graph
        self.agent_name = agent_name
        self._last_callback: ContextLensCallback | None = None

    # ── Async invoke ───────────────────────────────────────────────────────────

    async def ainvoke(
        self, state: Any, config: dict | None = None, **kwargs: Any
    ) -> Any:
        """Async graph invocation with context-lens instrumentation.

        Identical to graph.ainvoke — same arguments, same return value.
        """
        callback = ContextLensCallback(agent_name=self.agent_name)
        merged_config = self._inject_callback(config, callback)
        result = await self._graph.ainvoke(state, config=merged_config, **kwargs)
        self._last_callback = callback
        return result

    # ── Sync invoke ────────────────────────────────────────────────────────────

    def invoke(
        self, state: Any, config: dict | None = None, **kwargs: Any
    ) -> Any:
        """Sync graph invocation with context-lens instrumentation.

        Identical to graph.invoke — same arguments, same return value.
        """
        callback = ContextLensCallback(agent_name=self.agent_name)
        merged_config = self._inject_callback(config, callback)
        result = self._graph.invoke(state, config=merged_config, **kwargs)
        self._last_callback = callback
        return result

    # ── Report accessors ───────────────────────────────────────────────────────

    @property
    def node_snapshots(self) -> list[NodeSnapshot]:
        """Node snapshots from the most recent invoke/ainvoke call."""
        if self._last_callback is None:
            return []
        return list(self._last_callback.node_snapshots)

    def report(self) -> ContextReport:
        """Return a ContextReport for the most recent invoke/ainvoke call."""
        if self._last_callback is None:
            return ContextReport(agent_name=self.agent_name)
        return self._last_callback.report()

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _inject_callback(
        config: dict | None, callback: ContextLensCallback
    ) -> dict:
        """Return a config dict with callback appended to the callbacks list."""
        merged = dict(config or {})
        existing = list(merged.get("callbacks", []))
        merged["callbacks"] = existing + [callback]
        return merged
