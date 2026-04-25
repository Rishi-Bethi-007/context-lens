"""
examples/unhealthy_agent_demo.py

Demonstrates what context-lens reports look like for a degraded agent.

Synthesises probe results that exhibit two patterns:
  - beginning_anchored (HIGH): model only retrieves facts from the first 15%
  - cliff_detector     (HIGH): accuracy collapses at 30K tokens

No real API calls — this is a controlled illustration.

Run:
    python examples/unhealthy_agent_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from either repo root or examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_lens.engine.measurement import MeasurementResult
from context_lens.engine.probes import ProbeResult
from context_lens.reporter import Reporter
from context_lens.report.renderer import save


# ── Synthetic probe data ───────────────────────────────────────────────────────
# Represents a RAG agent tested at four context sizes.
# Pattern 1 – beginning_anchored: position 0.10 is 90%, rest ≤20%
# Pattern 2 – cliff_detector: accuracy collapses between 20K and 30K tokens


def _probe(position: float, token_count: int, correct: bool) -> ProbeResult:
    return ProbeResult(
        position=position,
        token_count=token_count,
        target_token_count=token_count,
        correct=correct,
        response="synthetic",
        expected="synthetic",
    )


def build_unhealthy_measurement() -> MeasurementResult:
    """Build a MeasurementResult that exhibits beginning_anchored + cliff patterns."""
    specs: list[tuple[float, int, bool]] = []

    for tc in [5_000, 10_000, 20_000]:
        # Position 0.10 — beginning of context: mostly correct
        specs += [(0.10, tc, True), (0.10, tc, True), (0.10, tc, True)]   # 100%
        # Positions 0.30–0.90 — buried in context: mostly wrong
        for pos in [0.30, 0.50, 0.70, 0.90]:
            specs += [(pos, tc, False), (pos, tc, False), (pos, tc, True)] # 33%

    # Cliff at 30K: everything collapses
    for pos in [0.10, 0.30, 0.50, 0.70, 0.90]:
        specs += [(pos, 30_000, False), (pos, 30_000, False), (pos, 30_000, False)]  # 0%

    probes = [_probe(p, tc, c) for p, tc, c in specs]
    positions = sorted({p for p, _, _ in specs})
    token_counts = sorted({tc for _, tc, _ in specs})
    return MeasurementResult(
        agent_name="my-rag-agent (synthetic)",
        probe_results=probes,
        token_counts_tested=token_counts,
        positions_tested=positions,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("context-lens — unhealthy agent demo")
    print("Generating synthetic probe data ...\n")

    measurement = build_unhealthy_measurement()
    report = Reporter().run(measurement)

    report.summary()

    out_path = Path(__file__).parent / "unhealthy_report.html"
    save(report, str(out_path))
    print(f"\nReport saved: {out_path}")
    print("Open it in a browser to see the degradation patterns.")


if __name__ == "__main__":
    main()
