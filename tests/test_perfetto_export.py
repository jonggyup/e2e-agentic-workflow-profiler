"""Tests for the Perfetto trace exporter."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_profiler.cli import app
from agent_profiler.exporter.perfetto import export_perfetto
from agent_profiler.schema.trace import load_trace

DEMOS = Path(__file__).parent.parent / "demos" / "output"

runner = CliRunner()


def _export(scenario: str, tmp_path: Path, **kwargs):
    """Load a demo trace and export to Perfetto JSON, return parsed output."""
    trace = load_trace(DEMOS / f"{scenario}.jsonl", strict=True)
    out = tmp_path / f"{scenario}.perfetto.json"
    export_perfetto(trace, out, **kwargs)
    return json.loads(out.read_text()), trace


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestBasicStructure:
    def test_output_is_valid_json_with_trace_events(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        assert "traceEvents" in data
        assert isinstance(data["traceEvents"], list)
        assert len(data["traceEvents"]) > 0

    def test_display_time_unit_is_ms(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        assert data["displayTimeUnit"] == "ms"

    def test_every_event_has_required_fields(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        for ev in data["traceEvents"]:
            assert "ph" in ev, f"Missing 'ph' in event: {ev}"
            # Metadata events have different required fields
            if ev["ph"] == "M":
                assert "name" in ev
                assert "pid" in ev
                assert "args" in ev
            else:
                assert "name" in ev, f"Missing 'name' in event: {ev}"
                assert "cat" in ev, f"Missing 'cat' in event: {ev}"
                assert "ts" in ev, f"Missing 'ts' in event: {ev}"
                assert "pid" in ev, f"Missing 'pid' in event: {ev}"
                assert "tid" in ev, f"Missing 'tid' in event: {ev}"


# ---------------------------------------------------------------------------
# Thread lane mapping
# ---------------------------------------------------------------------------


class TestThreadMapping:
    def test_tool_calls_on_tid_5(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        tool_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "tool" and e.get("ph") == "X"
        ]
        assert len(tool_events) > 0
        for ev in tool_events:
            assert ev["tid"] == 5

    def test_model_calls_on_tid_4(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        model_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "model" and e.get("ph") == "X"
        ]
        assert len(model_events) > 0
        for ev in model_events:
            assert ev["tid"] == 4

    def test_loop_iterations_on_tid_3(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        loop_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "loop" and e.get("ph") == "X"
        ]
        assert len(loop_events) > 0
        for ev in loop_events:
            assert ev["tid"] == 3

    def test_attempts_on_tid_2(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        attempt_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "attempt" and e.get("ph") == "X"
        ]
        assert len(attempt_events) > 0
        for ev in attempt_events:
            assert ev["tid"] == 2

    def test_run_on_tid_1(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        run_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "run" and e.get("ph") == "X"
        ]
        assert len(run_events) == 1
        assert run_events[0]["tid"] == 1


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_timestamps_are_in_microseconds(self, tmp_path: Path) -> None:
        data, trace = _export("happy_path", tmp_path)
        run_event = [
            e for e in data["traceEvents"]
            if e.get("cat") == "run" and e.get("ph") == "X"
        ][0]
        # Run starts at ts=0 (relative to itself)
        assert run_event["ts"] == 0.0
        # Duration should match (end_ns - start_ns) / 1000
        expected_dur = (trace.run.end_ns - trace.run.start_ns) / 1000.0
        assert abs(run_event["dur"] - expected_dur) < 0.1

    def test_all_timestamps_non_negative(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        for ev in data["traceEvents"]:
            if "ts" in ev:
                assert ev["ts"] >= 0, f"Negative timestamp in event: {ev['name']}"

    def test_all_durations_non_negative(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        for ev in data["traceEvents"]:
            if "dur" in ev:
                assert ev["dur"] >= 0, f"Negative duration in event: {ev['name']}"


# ---------------------------------------------------------------------------
# Failed attempts coloring
# ---------------------------------------------------------------------------


class TestFailedAttemptColoring:
    def test_failed_attempts_get_terrible_red(self, tmp_path: Path) -> None:
        data, _ = _export("wrong_tool_retry", tmp_path)
        attempt_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "attempt" and e.get("ph") == "X"
        ]
        failed = [e for e in attempt_events if "FAIL" in e["name"].upper() or "failure" in str(e.get("args", {}).get("outcome", ""))]
        assert len(failed) > 0
        for ev in failed:
            assert ev.get("cname") == "terrible_red"

    def test_successful_attempts_get_good_color(self, tmp_path: Path) -> None:
        data, _ = _export("wrong_tool_retry", tmp_path)
        attempt_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "attempt" and e.get("ph") == "X"
        ]
        success = [e for e in attempt_events if e.get("args", {}).get("outcome") == "success"]
        assert len(success) > 0
        for ev in success:
            assert ev.get("cname") == "good"

    def test_error_tool_calls_get_terrible_red(self, tmp_path: Path) -> None:
        data, _ = _export("wrong_tool_retry", tmp_path)
        tool_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "tool" and e.get("ph") == "X"
        ]
        errored = [e for e in tool_events if e.get("args", {}).get("outcome") == "error"]
        assert len(errored) > 0
        for ev in errored:
            assert ev.get("cname") == "terrible_red"


# ---------------------------------------------------------------------------
# Evaluations as instant events
# ---------------------------------------------------------------------------


class TestEvaluationInstants:
    def test_eval_instant_events_exist(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        eval_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "eval" and e.get("ph") == "i"
        ]
        assert len(eval_events) > 0

    def test_eval_name_contains_pass_or_fail(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        eval_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "eval" and e.get("ph") == "i"
        ]
        for ev in eval_events:
            assert "PASS" in ev["name"] or "FAIL" in ev["name"]

    def test_eval_on_attempt_tid(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        eval_events = [
            e for e in data["traceEvents"]
            if e.get("cat") == "eval" and e.get("ph") == "i"
        ]
        for ev in eval_events:
            assert ev["tid"] == 2


# ---------------------------------------------------------------------------
# System samples → counter events
# ---------------------------------------------------------------------------


class TestSystemSamples:
    def _make_samples(self, tmp_path: Path, base_ns: int) -> Path:
        """Create a minimal system samples file."""
        import orjson

        samples_path = tmp_path / "samples.jsonl"
        samples = [
            {
                "sample_type": "system",
                "timestamp_ns": base_ns + 100_000_000,
                "cpu_percent": 45.2,
                "memory_rss_mb": 128.5,
                "network_sent_bytes": 1000,
                "network_recv_bytes": 2000,
            },
            {
                "sample_type": "system",
                "timestamp_ns": base_ns + 600_000_000,
                "cpu_percent": 62.1,
                "memory_rss_mb": 135.0,
                "network_sent_bytes": 5000,
                "network_recv_bytes": 8000,
            },
            {
                "sample_type": "marker",
                "timestamp_ns": base_ns + 300_000_000,
                "event_name": "test_marker",
            },
        ]
        with samples_path.open("wb") as f:
            for s in samples:
                f.write(orjson.dumps(s) + b"\n")
        return samples_path

    def test_counter_events_present(self, tmp_path: Path) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        samples_path = self._make_samples(tmp_path, trace.run.start_ns)
        out = tmp_path / "out.perfetto.json"
        export_perfetto(trace, out, system_samples_path=samples_path)
        data = json.loads(out.read_text())
        counter_events = [
            e for e in data["traceEvents"] if e.get("ph") == "C"
        ]
        assert len(counter_events) > 0

    def test_cpu_counters_on_pid_2(self, tmp_path: Path) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        samples_path = self._make_samples(tmp_path, trace.run.start_ns)
        out = tmp_path / "out.perfetto.json"
        export_perfetto(trace, out, system_samples_path=samples_path)
        data = json.loads(out.read_text())
        cpu_counters = [
            e for e in data["traceEvents"]
            if e.get("ph") == "C" and e.get("name") == "CPU %"
        ]
        assert len(cpu_counters) == 2
        for ev in cpu_counters:
            assert ev["pid"] == 2

    def test_memory_counters_present(self, tmp_path: Path) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        samples_path = self._make_samples(tmp_path, trace.run.start_ns)
        out = tmp_path / "out.perfetto.json"
        export_perfetto(trace, out, system_samples_path=samples_path)
        data = json.loads(out.read_text())
        mem_counters = [
            e for e in data["traceEvents"]
            if e.get("ph") == "C" and e.get("name") == "Memory MB"
        ]
        assert len(mem_counters) == 2

    def test_network_rate_counters_present(self, tmp_path: Path) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        samples_path = self._make_samples(tmp_path, trace.run.start_ns)
        out = tmp_path / "out.perfetto.json"
        export_perfetto(trace, out, system_samples_path=samples_path)
        data = json.loads(out.read_text())
        net_counters = [
            e for e in data["traceEvents"]
            if e.get("ph") == "C" and e.get("name") == "Network KB/s"
        ]
        # First sample has no previous, so only 1 rate event
        assert len(net_counters) == 1

    def test_markers_are_filtered_out(self, tmp_path: Path) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        samples_path = self._make_samples(tmp_path, trace.run.start_ns)
        out = tmp_path / "out.perfetto.json"
        export_perfetto(trace, out, system_samples_path=samples_path)
        data = json.loads(out.read_text())
        # No event should reference the marker
        for ev in data["traceEvents"]:
            assert "test_marker" not in str(ev)


# ---------------------------------------------------------------------------
# Metadata events
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_process_name_events(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        proc_names = [
            e for e in data["traceEvents"]
            if e.get("ph") == "M" and e.get("name") == "process_name"
        ]
        names = {e["args"]["name"] for e in proc_names}
        assert "Agent" in names
        assert "System" in names

    def test_thread_name_events(self, tmp_path: Path) -> None:
        data, _ = _export("happy_path", tmp_path)
        thread_names = [
            e for e in data["traceEvents"]
            if e.get("ph") == "M" and e.get("name") == "thread_name"
        ]
        names = {e["args"]["name"] for e in thread_names}
        assert "Run" in names
        assert "Attempts" in names
        assert "Model Calls" in names
        assert "Tool Calls" in names


# ---------------------------------------------------------------------------
# CLI: export-perfetto command
# ---------------------------------------------------------------------------


class TestCLIExportPerfetto:
    def test_export_perfetto_exits_zero(self, tmp_path: Path) -> None:
        out = tmp_path / "out.perfetto.json"
        result = runner.invoke(
            app,
            ["export-perfetto", str(DEMOS / "happy_path.jsonl"), "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_export_perfetto_creates_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "out.perfetto.json"
        runner.invoke(
            app,
            ["export-perfetto", str(DEMOS / "happy_path.jsonl"), "-o", str(out)],
        )
        data = json.loads(out.read_text())
        assert "traceEvents" in data

    def test_export_perfetto_default_output_name(self, tmp_path: Path) -> None:
        # Copy the fixture to tmp so we can test default naming
        import shutil
        src = DEMOS / "happy_path.jsonl"
        dest = tmp_path / "happy_path.jsonl"
        shutil.copy(src, dest)
        result = runner.invoke(
            app,
            ["export-perfetto", str(dest)],
        )
        assert result.exit_code == 0, result.output
        default_out = tmp_path / "happy_path.perfetto.json"
        assert default_out.exists()

    def test_export_perfetto_missing_file(self) -> None:
        result = runner.invoke(
            app,
            ["export-perfetto", "/nonexistent/trace.jsonl"],
        )
        assert result.exit_code == 1

    def test_export_perfetto_prints_ui_url(self, tmp_path: Path) -> None:
        out = tmp_path / "out.perfetto.json"
        result = runner.invoke(
            app,
            ["export-perfetto", str(DEMOS / "happy_path.jsonl"), "-o", str(out)],
        )
        assert "ui.perfetto.dev" in result.output


# ---------------------------------------------------------------------------
# CLI: --export-perfetto flag on analyze
# ---------------------------------------------------------------------------


class TestAnalyzeExportPerfettoFlag:
    def test_analyze_with_export_perfetto(self, tmp_path: Path) -> None:
        import shutil
        src = DEMOS / "happy_path.jsonl"
        dest = tmp_path / "happy_path.jsonl"
        shutil.copy(src, dest)
        result = runner.invoke(
            app,
            ["analyze", str(dest), "--export-perfetto"],
        )
        assert result.exit_code == 0, result.output
        perfetto_out = tmp_path / "happy_path.perfetto.json"
        assert perfetto_out.exists()
        data = json.loads(perfetto_out.read_text())
        assert "traceEvents" in data
        assert "ui.perfetto.dev" in result.output


# ---------------------------------------------------------------------------
# Multiple scenarios produce valid exports
# ---------------------------------------------------------------------------


class TestAllScenarios:
    @pytest.mark.parametrize("scenario", [
        "happy_path",
        "wrong_tool_retry",
        "slow_program",
        "reasoning_heavy",
        "reasoning_loop",
        "transient_failure",
        "context_overflow",
        "hallucinated_tool",
    ])
    def test_scenario_exports_valid_perfetto(self, scenario: str, tmp_path: Path) -> None:
        data, _ = _export(scenario, tmp_path)
        assert "traceEvents" in data
        assert len(data["traceEvents"]) > 0
        # Every non-metadata event must have ts
        for ev in data["traceEvents"]:
            if ev["ph"] != "M":
                assert "ts" in ev
                assert ev["ts"] >= 0
