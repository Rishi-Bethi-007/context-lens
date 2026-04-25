"""Classifier: distractor confusion.

Detects accuracy collapse caused by semantically similar distractors. Requires
two MeasurementResults with identical probe configuration: one run without
distractors (baseline) and one with distractors inserted into the haystack.

Phase 1 finding: model reads the full context but fails disambiguation when
near-miss wrong answers are present — the accuracy drop is the signal.
"""
import logging

from context_lens.classifiers import ClassifierResult, Recommendation, Severity
from context_lens.engine.measurement import MeasurementResult

logger = logging.getLogger(__name__)

_PATTERN_NAME = "distractor_confusion"

# Thresholds
_DETECT_DROP = 0.20     # absolute accuracy drop that triggers detection
_HIGH_DROP = 0.40       # drop above which severity is HIGH


def detect(
    baseline: MeasurementResult,
    with_distractors: MeasurementResult,
) -> ClassifierResult:
    """Detect distractor confusion by comparing two MeasurementResults.

    Args:
        baseline:         Results from probing without distractors in the haystack.
        with_distractors: Results from probing with semantically similar distractors added.

    Returns:
        ClassifierResult with detected=True if adding distractors causes a
        meaningful accuracy drop (≥20 percentage points overall).
    """
    baseline_acc = baseline.mean_accuracy()
    distractor_acc = with_distractors.mean_accuracy()
    drop = baseline_acc - distractor_acc

    detected = drop >= _DETECT_DROP

    if drop >= _HIGH_DROP:
        severity = Severity.HIGH
    elif drop >= _DETECT_DROP:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    # Confidence scales linearly: 0 at drop=0, 1 at drop=0.6+
    confidence = round(min(1.0, drop / 0.6), 3) if drop > 0 else 0.0

    # Per-position breakdown for evidence
    baseline_by_pos = baseline.accuracy_by_position()
    distractor_by_pos = with_distractors.accuracy_by_position()
    shared_positions = sorted(set(baseline_by_pos) & set(distractor_by_pos))
    position_drops = {
        pos: round(baseline_by_pos[pos] - distractor_by_pos[pos], 3)
        for pos in shared_positions
    }

    logger.debug(
        "distractor_confusion: baseline_acc=%.3f distractor_acc=%.3f drop=%.3f detected=%s",
        baseline_acc,
        distractor_acc,
        drop,
        detected,
    )

    return ClassifierResult(
        pattern_name=_PATTERN_NAME,
        detected=detected,
        severity=severity,
        confidence=confidence,
        evidence={
            "baseline_accuracy": round(baseline_acc, 3),
            "distractor_accuracy": round(distractor_acc, 3),
            "accuracy_drop": round(drop, 3),
            "position_drops": position_drops,
        },
    )


def recommend(result: ClassifierResult) -> Recommendation:
    """Return an actionable fix recommendation for distractor confusion."""
    if not result.detected:
        return Recommendation(
            pattern_name=_PATTERN_NAME,
            description="No distractor confusion detected.",
            code_before="",
            code_after="",
            estimated_recovery="N/A",
        )

    drop_pct = int(result.evidence.get("accuracy_drop", 0) * 100)

    return Recommendation(
        pattern_name=_PATTERN_NAME,
        description=(
            f"Your agent shows distractor confusion: accuracy drops {drop_pct}% "
            "when semantically similar but incorrect facts are present in context. "
            "The model retrieves information by semantic similarity, not position — "
            "near-miss distractors beat the correct answer in embedding space."
        ),
        code_before=(
            "# Dumping all retrieved chunks into context, including near-miss results\n"
            "chunks = retriever.get_relevant_documents(query, k=20)\n"
            "context = '\\n'.join(c.page_content for c in chunks)"
        ),
        code_after=(
            "# Re-rank retrieved chunks to push near-miss distractors to the top of context\n"
            "from langchain.retrievers import ContextualCompressionRetriever\n"
            "from langchain.retrievers.document_compressors import CrossEncoderReranker\n"
            "reranker = CrossEncoderReranker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', top_n=5)\n"
            "retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)\n"
            "# Or: reduce k; fewer chunks means fewer distractors"
        ),
        estimated_recovery="20–50% accuracy improvement by filtering distractor chunks",
    )
