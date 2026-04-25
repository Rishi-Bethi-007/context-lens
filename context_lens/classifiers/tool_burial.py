"""Classifier: tool burial.

Detects accuracy collapse that begins after the 3rd+ sequential tool call.
When agents make many tool calls in sequence, the tool results stack on top of
the original instruction, pushing it down in the context. The model starts
ignoring the original task and instead responds to the most recent tool output.

Detection: accuracy at positions representing early tool calls (≤0.25) is
substantially higher than at positions representing the 3rd+ tool call (≥0.40).

By convention, the MeasurementResult passed here should be produced with
positions representing normalized tool-call depth (e.g. 0.1 = 1st call in a
10-call chain, 0.3 = 3rd call, etc.).
"""
import logging

from context_lens.classifiers import ClassifierResult, Recommendation, Severity
from context_lens.engine.measurement import MeasurementResult

logger = logging.getLogger(__name__)

_PATTERN_NAME = "tool_burial"

# Positions ≤ this are "before burial" (tool calls 1–2)
_EARLY_CUTOFF = 0.25
# Positions ≥ this are "buried" (tool call 3 onwards)
_LATE_CUTOFF = 0.40
_DETECT_DROP = 0.20     # absolute accuracy drop that triggers detection
_HIGH_DROP = 0.50       # drop above which severity is HIGH


def detect(measurement: MeasurementResult) -> ClassifierResult:
    """Detect tool burial from a MeasurementResult keyed by tool-call depth.

    Returns detected=True when early tool-call accuracy exceeds late
    tool-call accuracy by at least 20 percentage points.
    """
    by_pos = measurement.accuracy_by_position()

    early_positions = sorted(p for p in by_pos if p <= _EARLY_CUTOFF)
    late_positions = sorted(p for p in by_pos if p >= _LATE_CUTOFF)

    if not early_positions or not late_positions:
        logger.debug(
            "tool_burial: insufficient coverage — need positions ≤%.2f and ≥%.2f",
            _EARLY_CUTOFF,
            _LATE_CUTOFF,
        )
        return ClassifierResult(
            pattern_name=_PATTERN_NAME,
            detected=False,
            severity=Severity.LOW,
            confidence=0.0,
            evidence={"reason": "insufficient tool-call depth coverage"},
        )

    early_acc = sum(by_pos[p] for p in early_positions) / len(early_positions)
    late_acc = sum(by_pos[p] for p in late_positions) / len(late_positions)
    drop = early_acc - late_acc

    detected = drop >= _DETECT_DROP

    if drop >= _HIGH_DROP:
        severity = Severity.HIGH
    elif drop >= _DETECT_DROP:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    # Find the first "buried" position (first late position where accuracy drops sharply)
    burial_depth: float | None = None
    for p in sorted(by_pos):
        if p >= _LATE_CUTOFF and by_pos[p] < early_acc - _DETECT_DROP:
            burial_depth = p
            break

    confidence = round(min(1.0, drop / 0.8), 3)

    logger.debug(
        "tool_burial: early_acc=%.3f late_acc=%.3f drop=%.3f detected=%s burial_depth=%s",
        early_acc,
        late_acc,
        drop,
        detected,
        burial_depth,
    )

    return ClassifierResult(
        pattern_name=_PATTERN_NAME,
        detected=detected,
        severity=severity,
        confidence=confidence,
        evidence={
            "early_accuracy": round(early_acc, 3),
            "late_accuracy": round(late_acc, 3),
            "accuracy_drop": round(drop, 3),
            "burial_depth_position": burial_depth,
            "early_positions": early_positions,
            "late_positions": late_positions,
        },
    )


def recommend(result: ClassifierResult) -> Recommendation:
    """Return an actionable fix recommendation for tool burial."""
    if not result.detected:
        return Recommendation(
            pattern_name=_PATTERN_NAME,
            description="No tool burial detected.",
            code_before="",
            code_after="",
            estimated_recovery="N/A",
        )

    burial_depth = result.evidence.get("burial_depth_position")
    depth_str = f" (burial starts at ~{burial_depth:.0%} of the tool-call chain)" if burial_depth else ""

    return Recommendation(
        pattern_name=_PATTERN_NAME,
        description=(
            f"Your agent shows tool burial{depth_str}: accuracy drops after the "
            "3rd+ sequential tool call. Accumulated tool results push the original "
            "instruction far down in the context window, causing the model to lose "
            "track of the task goal."
        ),
        code_before=(
            "# Accumulating all tool results in the message history\n"
            "state['messages'].append(tool_result_1)\n"
            "state['messages'].append(tool_result_2)\n"
            "state['messages'].append(tool_result_3)  # original instruction now buried"
        ),
        code_after=(
            "# Summarize intermediate tool results instead of appending raw output\n"
            "if len(tool_results) >= 2:\n"
            "    summary = llm.invoke(f'Summarize these findings in 2 sentences: {tool_results}')\n"
            "    state['messages'] = [state['messages'][0], summary]  # keep original + summary\n"
            "# Or: use a scratchpad pattern — write to scratch, not to main context"
        ),
        estimated_recovery="30–50% accuracy improvement by keeping tool-call depth ≤ 2",
    )
