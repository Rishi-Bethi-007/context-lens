"""Classifier: beginning-anchored retrieval.

Phase 1 finding: the model reliably retrieves facts only from the first ~15% of
the context window. This is NOT the classic U-shaped lost-in-middle pattern —
recency (90%) fails just as badly as middle (50%). The primacy effect dominates
because semantic distractors confuse disambiguation everywhere else.

Detection: accuracy at positions ≤0.15 is ≥2× higher than at positions ≥0.30.
"""
import logging

from context_lens.classifiers import ClassifierResult, Recommendation, Severity
from context_lens.engine.measurement import MeasurementResult

logger = logging.getLogger(__name__)

_PATTERN_NAME = "beginning_anchored"

# Thresholds
_BEGINNING_CUTOFF = 0.15   # positions at or below this count as "beginning"
_OTHER_CUTOFF = 0.30       # positions at or above this count as "other"
_DETECT_RATIO = 2.0        # beginning_acc must be ≥ this × other_acc
_DETECT_MIN_ACC = 0.40     # beginning accuracy must exceed this to rule out uniform failure
_HIGH_RATIO = 3.5          # ratio above which severity is HIGH


def detect(measurement: MeasurementResult) -> ClassifierResult:
    """Detect beginning-anchored retrieval from a MeasurementResult.

    Returns detected=True when accuracy at the start of the context is
    substantially higher than at middle and end positions.
    """
    by_pos = measurement.accuracy_by_position()

    beginning_positions = sorted(p for p in by_pos if p <= _BEGINNING_CUTOFF)
    other_positions = sorted(p for p in by_pos if p >= _OTHER_CUTOFF)

    if not beginning_positions or not other_positions:
        logger.debug(
            "beginning_anchored: insufficient positions — need at least one ≤%.2f and one ≥%.2f",
            _BEGINNING_CUTOFF,
            _OTHER_CUTOFF,
        )
        return ClassifierResult(
            pattern_name=_PATTERN_NAME,
            detected=False,
            severity=Severity.LOW,
            confidence=0.0,
            evidence={"reason": "insufficient position coverage"},
        )

    beginning_acc = sum(by_pos[p] for p in beginning_positions) / len(beginning_positions)
    other_acc = sum(by_pos[p] for p in other_positions) / len(other_positions)

    # Avoid divide-by-zero: if other_acc is 0, ratio is infinite → always detected
    ratio = beginning_acc / other_acc if other_acc > 0.0 else float("inf")

    detected = ratio >= _DETECT_RATIO and beginning_acc >= _DETECT_MIN_ACC

    if ratio >= _HIGH_RATIO:
        severity = Severity.HIGH
    elif ratio >= _DETECT_RATIO:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    # Confidence scales from 0 at ratio=1 to 1 at ratio=5+
    raw_conf = (ratio - 1.0) / 4.0 if ratio > 1.0 else 0.0
    confidence = round(min(1.0, raw_conf), 3)

    logger.debug(
        "beginning_anchored: beginning_acc=%.3f other_acc=%.3f ratio=%.2f detected=%s",
        beginning_acc,
        other_acc,
        ratio,
        detected,
    )

    return ClassifierResult(
        pattern_name=_PATTERN_NAME,
        detected=detected,
        severity=severity,
        confidence=confidence,
        evidence={
            "beginning_accuracy": round(beginning_acc, 3),
            "other_accuracy": round(other_acc, 3),
            "ratio": round(ratio, 3) if ratio != float("inf") else "inf",
            "beginning_positions": beginning_positions,
            "other_positions": other_positions,
        },
    )


def recommend(result: ClassifierResult) -> Recommendation:
    """Return an actionable fix recommendation for beginning-anchored retrieval."""
    if not result.detected:
        return Recommendation(
            pattern_name=_PATTERN_NAME,
            description="No beginning-anchored retrieval detected.",
            code_before="",
            code_after="",
            estimated_recovery="N/A",
        )

    return Recommendation(
        pattern_name=_PATTERN_NAME,
        description=(
            "Your agent shows beginning-anchored retrieval: the model reliably "
            "retrieves facts only from the first ~15% of the context window. "
            "This is driven by semantic disambiguation failure — distractors in the "
            "context confuse the model everywhere except the beginning where "
            "primacy effects dominate. Recency (end-of-context) fails equally badly."
        ),
        code_before=(
            "# Appending retrieved chunks at the end of a long context\n"
            "context = background_text + retrieved_chunks\n"
            "response = llm.invoke(context + question)"
        ),
        code_after=(
            "# Place critical facts at the beginning; use RAG to reduce context size\n"
            "relevant_chunks = retriever.get_relevant_documents(question)  # top-k only\n"
            "context = relevant_chunks + background_text  # key facts first\n"
            "response = llm.invoke(context + question)\n"
            "# Better: trim context to ≤20K tokens to stay within reliable retrieval range"
        ),
        estimated_recovery="30–60% accuracy improvement at non-beginning positions",
    )
