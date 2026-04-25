"""Tests for context_lens.report.renderer (Phase 5)."""
import json
import os
import tempfile

import pytest

import context_lens.report.renderer as renderer_module
from context_lens.engine.measurement import MeasurementResult
from context_lens.engine.probes import ProbeResult
from context_lens.reporter import Reporter, ReportData
from context_lens.report.renderer import render, save, _to_payload, render_to_tempfile, _safe_evidence


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


def _healthy_report(agent_name: str = "TestAgent") -> ReportData:
    """Healthy agent: all probes correct, score A."""
    specs = [
        (pos, tc, True)
        for pos in [0.10, 0.30, 0.50, 0.70, 0.90]
        for tc in [5_000, 10_000, 20_000]
        for _ in range(2)
    ]
    m = _make_measurement(specs, agent_name)
    return Reporter().run(m)


def _degraded_report() -> ReportData:
    """Agent with a cliff: all correct at 5K, all wrong at 20K → cliff HIGH."""
    specs = (
        [(pos, 5_000, True)   for pos in [0.10, 0.50, 0.90] for _ in range(3)]
        + [(pos, 20_000, False) for pos in [0.10, 0.50, 0.90] for _ in range(3)]
    )
    m = _make_measurement(specs, "DegradedAgent")
    return Reporter().run(m)


# ── render() ──────────────────────────────────────────────────────────────────

class TestRender:
    def test_returns_string(self):
        html = render(_healthy_report())
        assert isinstance(html, str)

    def test_has_doctype(self):
        html = render(_healthy_report())
        assert "<!DOCTYPE html>" in html

    def test_agent_name_in_output(self):
        html = render(_healthy_report("ReguliQ"))
        assert "ReguliQ" in html

    def test_score_in_output(self):
        data = _healthy_report()
        html = render(data)
        assert data.overall_score in html

    def test_pattern_names_in_output(self):
        html = render(_healthy_report())
        assert "beginning_anchored" in html
        assert "cliff_detector" in html

    def test_data_json_placeholder_replaced(self):
        html = render(_healthy_report())
        assert "__DATA_JSON__" not in html

    def test_output_is_self_contained(self):
        html = render(_healthy_report())
        # No links to external resources
        assert "cdn.jsdelivr.net" not in html
        assert "unpkg.com" not in html
        assert "googleapis.com" not in html
        assert "cloudflare.com" not in html

    def test_healthy_agent_renders_without_recommendations(self):
        data = _healthy_report()
        assert data.overall_score == "A"
        html = render(data)
        # Healthy banner text should appear
        assert "healthy" in html.lower() or "no patterns" in html.lower() or "looks good" in html.lower()

    def test_degraded_agent_renders_cliff_token(self):
        data = _degraded_report()
        if data.degradation_cliff_tokens:
            html = render(data)
            assert str(data.degradation_cliff_tokens) in html

    def test_embedded_json_is_valid(self):
        html = render(_healthy_report())
        # Extract the JSON between window.REPORT_DATA = and the next ;
        marker = "window.REPORT_DATA = "
        start = html.index(marker) + len(marker)
        # Find the closing </script> and back up to the semicolon
        end = html.index(";</script>", start)
        json_str = html[start:end]
        payload = json.loads(json_str)
        assert "agent_name" in payload
        assert "overall_score" in payload
        assert "patterns" in payload

    def test_embedded_json_has_required_keys(self):
        html = render(_healthy_report())
        marker = "window.REPORT_DATA = "
        start = html.index(marker) + len(marker)
        end = html.index(";</script>", start)
        payload = json.loads(html[start:end])
        required = {
            "agent_name", "timestamp", "overall_score", "mean_accuracy",
            "position_accuracies", "token_count_accuracies", "heatmap",
            "patterns", "recommendations",
        }
        assert required <= set(payload.keys())


# ── save() ────────────────────────────────────────────────────────────────────

class TestSave:
    def test_writes_file(self, tmp_path):
        path = str(tmp_path / "report.html")
        save(_healthy_report(), path)
        assert os.path.exists(path)

    def test_file_is_nonempty(self, tmp_path):
        path = str(tmp_path / "report.html")
        save(_healthy_report(), path)
        assert os.path.getsize(path) > 1000

    def test_file_content_matches_render(self, tmp_path):
        data = _healthy_report("MatchTest")
        path = str(tmp_path / "report.html")
        save(data, path)
        with open(path, encoding="utf-8") as f:
            on_disk = f.read()
        assert on_disk == render(data)

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "report.html")
        # Write once with agent "First"
        save(_healthy_report("First"), path)
        # Write again with agent "Second"
        save(_healthy_report("Second"), path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "Second" in content


# ── _to_payload() ─────────────────────────────────────────────────────────────

class TestToPayload:
    def test_position_accuracies_sorted(self):
        data = _healthy_report()
        payload = _to_payload(data)
        positions = [row[0] for row in payload["position_accuracies"]]
        assert positions == sorted(positions)

    def test_token_count_accuracies_sorted(self):
        data = _healthy_report()
        payload = _to_payload(data)
        tcs = [row[0] for row in payload["token_count_accuracies"]]
        assert tcs == sorted(tcs)

    def test_heatmap_has_position_and_token_count(self):
        data = _healthy_report()
        payload = _to_payload(data)
        assert len(payload["heatmap"]) > 0
        cell = payload["heatmap"][0]
        assert "position" in cell
        assert "token_count" in cell
        assert "accuracy" in cell

    def test_heatmap_accuracy_in_range(self):
        data = _healthy_report()
        payload = _to_payload(data)
        for cell in payload["heatmap"]:
            assert 0.0 <= cell["accuracy"] <= 1.0

    def test_mean_accuracy_is_float(self):
        data = _healthy_report()
        payload = _to_payload(data)
        assert isinstance(payload["mean_accuracy"], float)

    def test_patterns_have_required_fields(self):
        data = _healthy_report()
        payload = _to_payload(data)
        for p in payload["patterns"]:
            assert "name" in p
            assert "detected" in p
            assert "severity" in p
            assert "confidence" in p

    def test_degradation_cliff_tokens_null_when_none(self):
        data = _healthy_report()
        payload = _to_payload(data)
        assert payload["degradation_cliff_tokens"] is None

    def test_degradation_cliff_tokens_set_when_detected(self):
        data = _degraded_report()
        payload = _to_payload(data)
        if data.degradation_cliff_tokens:
            assert payload["degradation_cliff_tokens"] == data.degradation_cliff_tokens


# ── previously uncovered branches ─────────────────────────────────────────────

class TestUncoveredBranches:
    def test_render_raises_when_placeholder_missing(self, tmp_path, monkeypatch):
        """ValueError branch: template exists but has no __DATA_JSON__ marker."""
        fake = tmp_path / "bad_template.html"
        fake.write_text("<!DOCTYPE html><html><body>no marker here</body></html>", encoding="utf-8")
        monkeypatch.setattr(renderer_module, "_TEMPLATE_PATH", fake)
        with pytest.raises(ValueError, match="does not contain placeholder"):
            render(_healthy_report())

    def test_render_to_tempfile_opens_browser(self, mocker):
        """render_to_tempfile writes a file and calls webbrowser.open."""
        mock_open = mocker.patch("webbrowser.open")
        path = render_to_tempfile(_healthy_report())
        assert mock_open.call_count == 1
        assert path.endswith(".html")
        assert os.path.exists(path)
        os.unlink(path)  # clean up temp file

    def test_safe_evidence_stringifies_unknown_type(self):
        """_safe_evidence else-branch: non-JSON-serialisable values become str."""
        class Opaque:
            def __str__(self) -> str:
                return "opaque-value"

        result = _safe_evidence({"key": Opaque()})
        assert result["key"] == "opaque-value"
