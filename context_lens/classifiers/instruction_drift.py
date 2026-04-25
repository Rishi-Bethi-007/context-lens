"""Classifier: instruction drift.

Detects gradual weakening of system-prompt constraints over the course of a
long conversation. Unlike tool_burial (which happens after 3 sequential tool
calls within one run), instruction drift is a turn-by-turn phenomenon: the
model honors early instructions reliably but progressively ignores them as the
conversation grows.

Detection: accuracy at early turns (positions ≤0.20) is substantially higher
than accuracy at late turns (positions ≥0.80), indicating the model has drifted
from the original instructions by the end of the conversation.

By convention, positions here represent normalized turn depth in a conversation
(0.1 = turn 1 of 10, 0.9 = turn 9 of 10).
"""
import logging

from context_lens.classifiers import ClassifierResult, Recommendation, Severity
from context_lens.engine.measurement import MeasurementResult

logger = logging.getLogger(__name__)

_PATTERN_NAME = "instruction_drift"

# First 20% of the conversation — instructions freshly given
_EARLY_CUTOFF = 0.20
# Last 20% of the conversation — far from original instructions
_LATE_CUTOFF = 0.80
_DETECT_DROP = 0.25     # larger threshold than tool_burial — drift is a stronger signal
_HIGH_DROP = 0.50       # drop above which severity is HIGH


def detect(measurement: MeasurementResult) -> ClassifierResult:
    """Detect instruction drift from a MeasurementResult keyed by turn depth.

    Returns detected=True when early-turn accuracy exceeds late-turn accuracy
    by at least 25 percentage points.
    """
    by_pos = measurement.accuracy_by_position()

    early_positions = sorted(p for p in by_pos if p <= _EARLY_CUTOFF)
    late_positions = sorted(p for p in by_pos if p >= _LATE_CUTOFF)

    if not early_positions or not late_positions:
        logger.debug(
            "instruction_drift: insufficient coverage — need positions ≤%.2f and ≥%.2f",
            _EARLY_CUTOFF,
            _LATE_CUTOFF,
        )
        return ClassifierResult(
            pattern_name=_PATTERN_NAME,
            detected=False,
            severity=Severity.LOW,
            confidence=0.0,
            evidence={"reason": "insufficient turn coverage"},
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

    # Mid-conversation accuracy helps characterise whether it's a cliff or gradual drift
    mid_positions = sorted(p for p in by_pos if _EARLY_CUTOFF < p < _LATE_CUTOFF)
    mid_acc: float | None = None
    if mid_positions:
        mid_acc = round(
            sum(by_pos[p] for p in mid_positions) / len(mid_positions), 3
        )

    confidence = round(min(1.0, drop / 0.8), 3)

    logger.debug(
        "instruction_drift: early_acc=%.3f late_acc=%.3f drop=%.3f detected=%s",
        early_acc,
        late_acc,
        drop,
        detected,
    )

    return ClassifierResult(
        pattern_name=_PATTERN_NAME,
        detected=detected,
        severity=severity,
        confidence=confidence,
        evidence={
            "early_accuracy": round(early_acc, 3),
            "late_accuracy": round(late_acc, 3),
            "mid_accuracy": mid_acc,
            "accuracy_drop": round(drop, 3),
            "early_positions": early_positions,
            "late_positions": late_positions,
        },
    )


def recommend(result: ClassifierResult) -> Recommendation:
    """Return an actionable fix recommendation for instruction drift."""
    if not result.detected:
        return Recommendation(
            pattern_name=_PATTERN_NAME,
            description="No instruction drift detected.",
            code_before="",
            code_after="",
            estimated_recovery="N/A",
        )

    drop_pct = int(result.evidence.get("accuracy_drop", 0) * 100)

    return Recommendation(
        pattern_name=_PATTERN_NAME,
        description=(
            f"Your agent shows instruction drift: accuracy drops {drop_pct}% from "
            "early to late conversation turns. The model stops honoring the original "
            "system-prompt constraints as the conversation grows — later messages "
            "dominate attention and the initial instructions fade."
        ),
        code_before=(
            "# System prompt set once at the start; never reinforced\n"
            "messages = [{\"role\": \"system\", \"content\": system_prompt}]\n"
            "for turn in conversation:\n"
            "    messages.append(turn)  # system prompt now buried under many turns"
        ),
        code_after=(
            "# Re-inject key constraints periodically in long conversations\n"
            "REINJECT_EVERY = 5  # re-state constraints every N turns\n"
            "if len(messages) % REINJECT_EVERY == 0:\n"
            "    messages.append({\n"
            "        \"role\": \"user\",\n"
            "        \"content\": f\"Reminder: {constraint_summary}\",\n"
            "    })\n"
            "# Or: use a persistent system prompt slot if your framework supports it"
        ),
        estimated_recovery="20–40% accuracy improvement at late conversation turns",
    )
