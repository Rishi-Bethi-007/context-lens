"""Tests for context_lens.reporter (Phase 5)."""
import pytest

from context_lens.classifiers import Severity
from context_lens.classifiers import ClassifierResult
from context_lens.engine.measurement import MeasurementResult
from context_lens.engine.probes import ProbeResult
from context_lens.reporter import ReportData, Reporter, _compute_score


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_probe(position: float, token_count: int, correct: bool) -> ProbeResult:
    return ProbeResult(
        position=position,
        token_count=token_count,
        target_token_count=token_count,
        correct=correct,
        response="r",
        expected="e",
    )


def _make_measurement(
    specs: list[tuple[float, int, bool]],
    agent_name: str = "test-agent",
) -> MeasurementResult:
    probes = [_make_probe(p, tc, c) for p, tc, c in specs]
    positions = sorted({p for p, _, _ in specs})
    tcs = sorted({tc for _, tc, _ in specs})
    return MeasurementResult(
        agent_name=agent_name,
        probe_results=probes,
        token_counts_tested=tcs,
        positions_tested=positions,
    )


def _uniform_healthy(agent_name: str = "test-agent") -> MeasurementResult:
    """All probes correct at all positions and token counts — no pattern should detect."""
    specs = [
        (pos, tc, True)
        for pos in [0.10, 0.30, 0.50, 0.70, 0.90]
        for tc in [5_000, 10_000, 20_000]
        for _ in range(3)
    ]
    return _make_measurement(specs, agent_name)


def _cliff_measurement() -> MeasurementResult:
    """All correct at 5K, all wrong at 20K — cliff_detector should fire HIGH."""
    specs = (
        [(pos, 5_000, True)  for pos in [0.10, 0.50, 0.90] for _ in range(3)]
        + [(pos, 20_000, False) for pos in [0.10, 0.50, 0.90] for _ in range(3)]
    )
    return _make_measurement(specs)


def _beginning_anchored_measurement() -> MeasurementResult:
    """High at start, low everywhere else — beginning_anchored fires."""
    specs = (
        [(0.10, 10_000, True)  for _ in range(3)]
        + [(pos, 10_000, False) for pos in [0.50, 0.70, 0.90] for _ in range(3)]
    )
    return _make_measurement(specs)


# ── Reporter.run() ─────────────────────────────────────────────────────────────

class TestReporterRun:
    def test_returns_report_data(self):
        r = Reporter()
        data = r.run(_uniform_healthy())
        assert isinstance(data, ReportData)

    def test_agent_name_preserved(self):
        r = Reporter()
        data = r.run(_uniform_healthy(agent_name="ReguliQ"))
        assert data.agent_name == "ReguliQ"

    def test_timestamp_is_set(self):
        r = Reporter()
        data = r.run(_uniform_healthy())
        assert data.timestamp != ""
        assert "T" in data.timestamp  # ISO 8601

    def test_five_classifiers_without_distractor(self):
        r = Reporter()
        data = r.run(_uniform_healthy())
        assert len(data.patterns) == 5

    def test_six_classifiers_with_distractor(self):
        r = Reporter()
        m = _uniform_healthy()
        data = r.run(m, distractor_measurement=m)
        assert len(data.patterns) == 6

    def test_distractor_classifier_name_present_when_passed(self):
        r = Reporter()
        m = _uniform_healthy()
        data = r.run(m, distractor_measurement=m)
        names = [p.pattern_name for p in data.patterns]
        assert "distractor_confusion" in names

    def test_distractor_classifier_absent_when_not_passed(self):
        r = Reporter()
        data = r.run(_uniform_healthy())
        names = [p.pattern_name for p in data.patterns]
        assert "distractor_confusion" not in names

    def test_healthy_agent_gets_score_a(self):
        r = Reporter()
        data = r.run(_uniform_healthy())
        assert data.overall_score == "A"

    def test_no_recommendations_for_healthy_agent(self):
        r = Reporter()
        data = r.run(_uniform_healthy())
        assert data.recommendations == []

    def test_cliff_tokens_populated_when_cliff_detected(self):
        r = Reporter()
        data = r.run(_cliff_measurement())
        cliff_p = next(p for p in data.patterns if p.pattern_name == "cliff_detector")
        if cliff_p.detected:
            assert data.degradation_cliff_tokens is not None
            assert data.degradation_cliff_tokens == 20_000

    def test_cliff_tokens_none_when_no_cliff(self):
        r = Reporter()
        data = r.run(_uniform_healthy())
        assert data.degradation_cliff_tokens is None

    def test_recommendations_only_for_detected_patterns(self):
        r = Reporter()
        data = r.run(_cliff_measurement())
        detected_names = {p.pattern_name for p in data.patterns if p.detected}
        rec_names = {rec.pattern_name for rec in data.recommendations}
        assert rec_names <= detected_names  # recs ⊆ detected patterns

    def test_measurement_stored_on_report_data(self):
        r = Reporter()
        m = _uniform_healthy()
        data = r.run(m)
        assert data.measurement is m


# ── _compute_score ─────────────────────────────────────────────────────────────

def _make_result(pattern_name: str, detected: bool, severity: Severity) -> ClassifierResult:
    return ClassifierResult(
        pattern_name=pattern_name,
        detected=detected,
        severity=severity,
        confidence=0.9,
        evidence={},
    )


class TestComputeScore:
    def test_a_when_nothing_detected(self):
        patterns = [
            _make_result("x", False, Severity.LOW),
            _make_result("y", False, Severity.MEDIUM),
        ]
        assert _compute_score(patterns) == "A"

    def test_a_when_empty(self):
        assert _compute_score([]) == "A"

    def test_b_when_only_low(self):
        patterns = [
            _make_result("x", True, Severity.LOW),
            _make_result("y", True, Severity.LOW),
        ]
        assert _compute_score(patterns) == "B"

    def test_c_when_medium_no_high(self):
        patterns = [
            _make_result("x", True, Severity.MEDIUM),
            _make_result("y", True, Severity.LOW),
        ]
        assert _compute_score(patterns) == "C"

    def test_d_when_one_high(self):
        patterns = [
            _make_result("x", True, Severity.HIGH),
            _make_result("y", True, Severity.LOW),
        ]
        assert _compute_score(patterns) == "D"

    def test_f_when_two_high(self):
        patterns = [
            _make_result("x", True, Severity.HIGH),
            _make_result("y", True, Severity.HIGH),
        ]
        assert _compute_score(patterns) == "F"

    def test_d_when_exactly_one_high_no_medium(self):
        patterns = [_make_result("x", True, Severity.HIGH)]
        assert _compute_score(patterns) == "D"

    def test_undetected_patterns_ignored_for_scoring(self):
        # Two HIGH patterns but both undetected — still A
        patterns = [
            _make_result("x", False, Severity.HIGH),
            _make_result("y", False, Severity.HIGH),
        ]
        assert _compute_score(patterns) == "A"


# ── ReportData.summary() ───────────────────────────────────────────────────────

class TestReportDataSummary:
    def test_summary_runs_without_error(self, capsys):
        r = Reporter()
        data = r.run(_uniform_healthy(agent_name="ReguliQ"))
        data.summary()
        captured = capsys.readouterr()
        assert "ReguliQ" in captured.out

    def test_summary_shows_score(self, capsys):
        r = Reporter()
        data = r.run(_uniform_healthy())
        data.summary()
        captured = capsys.readouterr()
        assert data.overall_score in captured.out

    def test_summary_healthy_message(self, capsys):
        r = Reporter()
        data = r.run(_uniform_healthy())
        data.summary()
        captured = capsys.readouterr()
        assert "no patterns detected" in captured.out.lower()

    def test_summary_lists_detected_patterns(self, capsys):
        r = Reporter()
        data = r.run(_cliff_measurement())
        data.summary()
        captured = capsys.readouterr()
        # If any pattern was detected, it should appear in the output
        detected = [p for p in data.patterns if p.detected]
        if detected:
            assert detected[0].pattern_name in captured.out
