"""Classifier: accuracy cliff detector.

Detects a sharp accuracy drop between adjacent token count measurements —
a "cliff" where performance falls off more than 20 percentage points in one
step. Identifies the exact token count where the cliff begins.
"""
import logging

from context_lens.classifiers import ClassifierResult, Recommendation, Severity
from context_lens.engine.measurement import MeasurementResult

logger = logging.getLogger(__name__)

_PATTERN_NAME = "cliff_detector"

# Thresholds
_DETECT_DROP = 0.20     # minimum absolute drop between adjacent counts to detect
_HIGH_DROP = 0.50       # drop above which severity is HIGH


def detect(measurement: MeasurementResult) -> ClassifierResult:
    """Detect a sharp accuracy cliff across adjacent token count measurements.

    Iterates through token counts in ascending order and finds the largest
    single-step accuracy drop. Returns detected=True if that drop exceeds 20%.
    """
    by_tc = measurement.accuracy_by_token_count()
    sorted_counts = sorted(by_tc)

    if len(sorted_counts) < 2:
        return ClassifierResult(
            pattern_name=_PATTERN_NAME,
            detected=False,
            severity=Severity.LOW,
            confidence=0.0,
            evidence={"reason": "need at least two token counts"},
        )

    max_drop = 0.0
    cliff_at: int | None = None
    cliff_before_acc = 0.0
    cliff_after_acc = 0.0

    for i in range(len(sorted_counts) - 1):
        tc_before = sorted_counts[i]
        tc_after = sorted_counts[i + 1]
        drop = by_tc[tc_before] - by_tc[tc_after]
        if drop > max_drop:
            max_drop = drop
            cliff_at = tc_after
            cliff_before_acc = by_tc[tc_before]
            cliff_after_acc = by_tc[tc_after]

    detected = max_drop >= _DETECT_DROP

    if max_drop >= _HIGH_DROP:
        severity = Severity.HIGH
    elif max_drop >= _DETECT_DROP:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    # Confidence scales from 0 at drop=0 to 1 at drop=0.8+
    confidence = round(min(1.0, max_drop / 0.8), 3)

    logger.debug(
        "cliff_detector: max_drop=%.3f cliff_at=%s detected=%s",
        max_drop,
        cliff_at,
        detected,
    )

    return ClassifierResult(
        pattern_name=_PATTERN_NAME,
        detected=detected,
        severity=severity,
        confidence=confidence,
        evidence={
            "cliff_token_count": cliff_at,
            "max_drop": round(max_drop, 3),
            "accuracy_before_cliff": round(cliff_before_acc, 3),
            "accuracy_after_cliff": round(cliff_after_acc, 3),
            "token_count_accuracies": {tc: round(by_tc[tc], 3) for tc in sorted_counts},
        },
    )


def recommend(result: ClassifierResult) -> Recommendation:
    """Return an actionable fix recommendation for a detected accuracy cliff."""
    if not result.detected:
        return Recommendation(
            pattern_name=_PATTERN_NAME,
            description="No accuracy cliff detected.",
            code_before="",
            code_after="",
            estimated_recovery="N/A",
        )

    cliff_tc = result.evidence.get("cliff_token_count")
    drop_pct = int(result.evidence.get("max_drop", 0) * 100)
    cliff_str = f"{cliff_tc:,}" if cliff_tc else "unknown"

    return Recommendation(
        pattern_name=_PATTERN_NAME,
        description=(
            f"Your agent hits a performance cliff at {cliff_str} tokens: "
            f"accuracy drops {drop_pct}% in a single step. "
            "Keep context windows below this threshold or use compression "
            "to avoid entering the degraded zone."
        ),
        code_before=(
            "# No context size guard — context can grow unbounded\n"
            "for doc in retrieved_docs:\n"
            "    context += doc.page_content  # may exceed cliff token count"
        ),
        code_after=(
            f"# Cap context at the safe limit (below {cliff_str} tokens)\n"
            "from context_lens.engine.probes import count_tokens, truncate_to_tokens\n"
            f"MAX_SAFE_TOKENS = {(cliff_tc or 0) - 2000 if cliff_tc else 'CLIFF_TOKENS - 2000'}\n"
            "context = truncate_to_tokens(context, MAX_SAFE_TOKENS)\n"
            "# Or: use summarization to compress old messages before they exceed the cliff"
        ),
        estimated_recovery=f"Full accuracy recovery by staying under {cliff_str} tokens",
    )
