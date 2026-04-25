"""Classifier: recency bias.

The mirror image of beginning-anchored retrieval. Detects when the model
ignores everything except the most recent messages: accuracy is high only at
late context positions (≥0.80) and low everywhere else (≤0.50).

Common cause: the model has been fine-tuned or prompted in a way that over-
weights recent tokens — e.g. heavy instruction-following RLHF can create
a model that anchors to the last user message and treats earlier context
as irrelevant background.

Detection: accuracy at positions ≥0.80 is ≥2× higher than at positions ≤0.50,
AND late accuracy exceeds a minimum threshold (rules out uniform failure).
"""
import logging

from context_lens.classifiers import ClassifierResult, Recommendation, Severity
from context_lens.engine.measurement import MeasurementResult

logger = logging.getLogger(__name__)

_PATTERN_NAME = "recency_bias"

# Positions ≥ this are "recent" (model pays attention here)
_RECENT_CUTOFF = 0.80
# Positions ≤ this are "non-recent" (model ignores here)
_NON_RECENT_CUTOFF = 0.50
_DETECT_RATIO = 2.0        # recent_acc must be ≥ this × non_recent_acc
_DETECT_MIN_ACC = 0.40     # recent accuracy must exceed this (rules out uniform failure)
_HIGH_RATIO = 3.5          # ratio above which severity is HIGH


def detect(measurement: MeasurementResult) -> ClassifierResult:
    """Detect recency bias from a MeasurementResult.

    Returns detected=True when accuracy is concentrated at late positions and
    low everywhere else — the opposite of beginning-anchored retrieval.
    """
    by_pos = measurement.accuracy_by_position()

    recent_positions = sorted(p for p in by_pos if p >= _RECENT_CUTOFF)
    non_recent_positions = sorted(p for p in by_pos if p <= _NON_RECENT_CUTOFF)

    if not recent_positions or not non_recent_positions:
        logger.debug(
            "recency_bias: insufficient coverage — need positions ≥%.2f and ≤%.2f",
            _RECENT_CUTOFF,
            _NON_RECENT_CUTOFF,
        )
        return ClassifierResult(
            pattern_name=_PATTERN_NAME,
            detected=False,
            severity=Severity.LOW,
            confidence=0.0,
            evidence={"reason": "insufficient position coverage"},
        )

    recent_acc = sum(by_pos[p] for p in recent_positions) / len(recent_positions)
    non_recent_acc = (
        sum(by_pos[p] for p in non_recent_positions) / len(non_recent_positions)
    )

    ratio = recent_acc / non_recent_acc if non_recent_acc > 0.0 else float("inf")

    detected = ratio >= _DETECT_RATIO and recent_acc >= _DETECT_MIN_ACC

    if ratio >= _HIGH_RATIO:
        severity = Severity.HIGH
    elif ratio >= _DETECT_RATIO:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    # Confidence mirrors beginning_anchored: scales from 0 at ratio=1 to 1 at ratio=5+
    raw_conf = (ratio - 1.0) / 4.0 if ratio > 1.0 else 0.0
    confidence = round(min(1.0, raw_conf), 3)

    logger.debug(
        "recency_bias: recent_acc=%.3f non_recent_acc=%.3f ratio=%.2f detected=%s",
        recent_acc,
        non_recent_acc,
        ratio,
        detected,
    )

    return ClassifierResult(
        pattern_name=_PATTERN_NAME,
        detected=detected,
        severity=severity,
        confidence=confidence,
        evidence={
            "recent_accuracy": round(recent_acc, 3),
            "non_recent_accuracy": round(non_recent_acc, 3),
            "ratio": round(ratio, 3) if ratio != float("inf") else "inf",
            "recent_positions": recent_positions,
            "non_recent_positions": non_recent_positions,
        },
    )


def recommend(result: ClassifierResult) -> Recommendation:
    """Return an actionable fix recommendation for recency bias."""
    if not result.detected:
        return Recommendation(
            pattern_name=_PATTERN_NAME,
            description="No recency bias detected.",
            code_before="",
            code_after="",
            estimated_recovery="N/A",
        )

    return Recommendation(
        pattern_name=_PATTERN_NAME,
        description=(
            "Your agent shows recency bias: the model reliably retrieves facts "
            "only from the last ~20% of the context window, ignoring earlier content. "
            "Critical information placed early in a long context will be missed. "
            "Move key facts to the end of the context, or use structured retrieval "
            "to surface them independently of position."
        ),
        code_before=(
            "# Important instructions placed at the top of a long context\n"
            "context = important_instructions + long_background_text\n"
            "response = llm.invoke(context + question)  # instructions get ignored"
        ),
        code_after=(
            "# Place critical instructions at the END of the context (just before question)\n"
            "context = long_background_text + important_instructions\n"
            "response = llm.invoke(context + question)\n"
            "# Or: repeat key constraints immediately before the question\n"
            "response = llm.invoke(context + f'Remember: {key_constraint}\\n' + question)"
        ),
        estimated_recovery="30–55% accuracy improvement by repositioning critical facts",
    )
