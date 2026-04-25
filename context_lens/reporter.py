"""Reporter: orchestrates all classifiers, computes a letter grade, and produces ReportData."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from context_lens.classifiers import ClassifierResult, Recommendation, Severity
from context_lens.classifiers import (
    beginning_anchored,
    cliff_detector,
    distractor_confusion,
    instruction_drift,
    recency_bias,
    tool_burial,
)
from context_lens.engine.measurement import MeasurementResult

logger = logging.getLogger(__name__)


@dataclass
class ReportData:
    """All data needed to render a diagnostic report."""

    agent_name: str
    timestamp: str
    measurement: MeasurementResult
    patterns: list[ClassifierResult]
    recommendations: list[Recommendation]
    overall_score: str              # A / B / C / D / F
    degradation_cliff_tokens: int | None

    def show(self) -> None:
        """Render HTML report and open in the default browser."""
        from context_lens.report.renderer import render_to_tempfile
        render_to_tempfile(self)

    def save(self, path: str) -> None:
        """Render HTML report and write to path."""
        from context_lens.report.renderer import save
        save(self, path)

    def summary(self) -> None:
        """Print a compact text summary to stdout."""
        detected = [p for p in self.patterns if p.detected]
        acc = self.measurement.mean_accuracy()
        print(f"context-lens: {self.agent_name}")
        print(f"  score: {self.overall_score}  |  mean accuracy: {acc:.1%}  |  {len(self.patterns)} classifiers run")
        if self.degradation_cliff_tokens:
            print(f"  cliff: {self.degradation_cliff_tokens:,} tokens")
        if not detected:
            print("  no patterns detected — context health looks good")
            return
        print(f"  {len(detected)} pattern(s) detected:")
        for p in detected:
            print(f"    [{p.severity.value:6}] {p.pattern_name}  conf={p.confidence:.2f}")


class Reporter:
    """Runs all classifiers against a MeasurementResult and returns a ReportData."""

    def run(
        self,
        measurement: MeasurementResult,
        distractor_measurement: MeasurementResult | None = None,
    ) -> ReportData:
        """Run all classifiers and return a complete ReportData.

        Args:
            measurement:           Results from the NIAH probe sweep.
            distractor_measurement: Optional second sweep run with semantically
                                   similar distractors in the haystack. When
                                   provided, distractor_confusion is also run.

        Returns:
            ReportData with all classifier results, recommendations, and grade.
        """
        patterns: list[ClassifierResult] = [
            beginning_anchored.detect(measurement),
            cliff_detector.detect(measurement),
            tool_burial.detect(measurement),
            instruction_drift.detect(measurement),
            recency_bias.detect(measurement),
        ]

        if distractor_measurement is not None:
            patterns.append(
                distractor_confusion.detect(measurement, distractor_measurement)
            )

        recommendations = [_recommend(p) for p in patterns if p.detected]

        score = _compute_score(patterns)

        cliff_result = next(
            (p for p in patterns if p.pattern_name == "cliff_detector" and p.detected),
            None,
        )
        cliff_tokens: int | None = (
            cliff_result.evidence.get("cliff_token_count") if cliff_result else None
        )

        logger.info(
            "report %r: score=%s detected=%d/%d",
            measurement.agent_name,
            score,
            sum(1 for p in patterns if p.detected),
            len(patterns),
        )

        return ReportData(
            agent_name=measurement.agent_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            measurement=measurement,
            patterns=patterns,
            recommendations=recommendations,
            overall_score=score,
            degradation_cliff_tokens=cliff_tokens,
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

_RECOMMEND_DISPATCH = {
    "beginning_anchored": beginning_anchored.recommend,
    "cliff_detector": cliff_detector.recommend,
    "distractor_confusion": distractor_confusion.recommend,
    "instruction_drift": instruction_drift.recommend,
    "recency_bias": recency_bias.recommend,
    "tool_burial": tool_burial.recommend,
}


def _recommend(pattern: ClassifierResult) -> Recommendation:
    """Route a ClassifierResult to its module's recommend() function."""
    fn = _RECOMMEND_DISPATCH.get(pattern.pattern_name)
    if fn is None:
        raise ValueError(f"No recommend() registered for pattern {pattern.pattern_name!r}")
    return fn(pattern)


def _compute_score(patterns: list[ClassifierResult]) -> str:
    """Compute an A–F grade from classifier results.

    A: nothing detected
    B: one or more LOW severity patterns, no MEDIUM or HIGH
    C: one or more MEDIUM severity patterns, no HIGH
    D: exactly one HIGH severity pattern
    F: two or more HIGH severity patterns
    """
    detected = [p for p in patterns if p.detected]
    if not detected:
        return "A"
    high = sum(1 for p in detected if p.severity == Severity.HIGH)
    medium = sum(1 for p in detected if p.severity == Severity.MEDIUM)
    if high >= 2:
        return "F"
    if high == 1:
        return "D"
    if medium >= 1:
        return "C"
    return "B"
