"""Renders a ReportData to a self-contained HTML file.

No external dependencies: uses only Python stdlib (json, pathlib, tempfile,
webbrowser). The template is a static HTML file in the same directory;
the renderer injects a single __DATA_JSON__ block and returns the result.
"""
from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "template.html"
_PLACEHOLDER = "__DATA_JSON__"


def render(data: "ReportData") -> str:  # type: ignore[name-defined]
    """Render a ReportData to a self-contained HTML string.

    Args:
        data: ReportData produced by Reporter.run().

    Returns:
        A complete HTML string with all data embedded as JSON.

    Raises:
        FileNotFoundError: If the HTML template is missing.
        ValueError: If the template does not contain the expected placeholder.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if _PLACEHOLDER not in template:
        raise ValueError(
            f"Template at {_TEMPLATE_PATH} does not contain placeholder {_PLACEHOLDER!r}"
        )
    payload = _to_payload(data)
    return template.replace(_PLACEHOLDER, json.dumps(payload, ensure_ascii=False))


def save(data: "ReportData", path: str) -> None:  # type: ignore[name-defined]
    """Render and write the HTML report to path.

    Args:
        data: ReportData produced by Reporter.run().
        path: Destination file path (will be created or overwritten).
    """
    html = render(data)
    Path(path).write_text(html, encoding="utf-8")


def render_to_tempfile(data: "ReportData") -> str:  # type: ignore[name-defined]
    """Render the report to a temp file and open it in the default browser.

    Args:
        data: ReportData produced by Reporter.run().

    Returns:
        Path of the temporary HTML file.
    """
    html = render(data)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(html)
        path = f.name
    webbrowser.open(f"file://{path}")
    return path


# ── Internal serialization ─────────────────────────────────────────────────────

def _to_payload(data: "ReportData") -> dict:  # type: ignore[name-defined]
    """Convert ReportData to a JSON-safe dict for the HTML template."""
    m = data.measurement

    # Position accuracy: [[pos, acc], ...]
    pos_acc = sorted(m.accuracy_by_position().items())

    # Token count accuracy: [[tc, acc], ...]  (sorted ascending)
    tc_acc = sorted(m.accuracy_by_token_count().items())

    # Heatmap: [{position, token_count, accuracy}, ...]
    # Group probe results by (position, target_token_count) and compute cell accuracy
    cells: dict[tuple[float, int], list[bool]] = {}
    for pr in m.probe_results:
        key = (pr.position, pr.target_token_count)
        cells.setdefault(key, []).append(pr.correct)
    heatmap = [
        {
            "position": pos,
            "token_count": tc,
            "accuracy": round(sum(votes) / len(votes), 3),
        }
        for (pos, tc), votes in sorted(cells.items())
    ]

    return {
        "agent_name": data.agent_name,
        "timestamp": data.timestamp,
        "overall_score": data.overall_score,
        "mean_accuracy": round(m.mean_accuracy(), 3),
        "degradation_cliff_tokens": data.degradation_cliff_tokens,
        "position_accuracies": [[p, round(a, 3)] for p, a in pos_acc],
        "token_count_accuracies": [[tc, round(a, 3)] for tc, a in tc_acc],
        "heatmap": heatmap,
        "patterns": [
            {
                "name": p.pattern_name,
                "detected": p.detected,
                "severity": p.severity.value,
                "confidence": p.confidence,
                "evidence": _safe_evidence(p.evidence),
            }
            for p in data.patterns
        ],
        "recommendations": [
            {
                "pattern_name": rec.pattern_name,
                "description": rec.description,
                "code_before": rec.code_before,
                "code_after": rec.code_after,
                "estimated_recovery": rec.estimated_recovery,
            }
            for rec in data.recommendations
        ],
    }


def _safe_evidence(evidence: dict) -> dict:
    """Strip non-JSON-serializable values from evidence dicts."""
    safe: dict = {}
    for k, v in evidence.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, list):
            safe[k] = v
        elif isinstance(v, dict):
            safe[k] = _safe_evidence(v)
        else:
            safe[k] = str(v)
    return safe
