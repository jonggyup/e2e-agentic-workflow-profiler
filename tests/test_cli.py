"""Tests for the agent-profiler CLI commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_profiler.cli import app

DEMOS = Path(__file__).parent.parent / "demos" / "output"
FIXTURES = Path(__file__).parent / "fixtures"

runner = CliRunner()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    def test_exits_zero_on_valid_trace(self) -> None:
        result = runner.invoke(app, ["analyze", str(DEMOS / "happy_path.jsonl")])
        assert result.exit_code == 0, result.output

    def test_exits_zero_wrong_tool_retry(self) -> None:
        result = runner.invoke(app, ["analyze", str(DEMOS / "wrong_tool_retry.jsonl")])
        assert result.exit_code == 0, result.output

    def test_exits_zero_slow_program(self) -> None:
        result = runner.invoke(
            app,
            ["analyze", str(DEMOS / "slow_program.jsonl"), "--program-tool", "bash"],
        )
        assert result.exit_code == 0, result.output

    def test_output_contains_table_header(self) -> None:
        result = runner.invoke(app, ["analyze", str(DEMOS / "happy_path.jsonl")])
        assert "e2e_wall_ms" in result.output

    def test_output_contains_attempt_breakdown(self) -> None:
        result = runner.invoke(app, ["analyze", str(DEMOS / "wrong_tool_retry.jsonl")])
        assert "Attempt Breakdown" in result.output
        assert "FAIL" in result.output
        assert "SUCCESS" in result.output

    def test_output_contains_verdict(self) -> None:
        result = runner.invoke(app, ["analyze", str(DEMOS / "wrong_tool_retry.jsonl")])
        assert "Primary bottleneck" in result.output

    def test_json_flag_produces_valid_json(self) -> None:
        result = runner.invoke(
            app, ["analyze", str(DEMOS / "happy_path.jsonl"), "--json"]
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "e2e_wall_ms" in parsed
        assert "primary_bottleneck" in parsed
        assert "per_attempt_summary" in parsed

    def test_json_flag_slow_program_valid_json(self) -> None:
        result = runner.invoke(
            app, ["analyze", str(DEMOS / "slow_program.jsonl"), "--json"]
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["attempt_count"] == 1
        assert parsed["first_pass_success"] is True

    def test_json_contains_all_20_metrics(self) -> None:
        result = runner.invoke(
            app, ["analyze", str(DEMOS / "happy_path.jsonl"), "--json"]
        )
        parsed = json.loads(result.output)
        expected_keys = {
            "e2e_wall_ms",
            "total_model_time_ms",
            "total_tool_time_ms",
            "agent_overhead_ms",
            "tool_execution_ms",
            "program_runtime_ms",
            "retry_waste_ms",
            "gap_time_ms",
            "user_idle_ms",
            "active_wall_ms",
            "idle_percentage",
            "first_pass_success",
            "attempt_count",
            "failure_categories",
            "correctness_score",
            "total_input_tokens",
            "total_output_tokens",
            "estimated_cost_usd",
            "wasted_tokens",
            "primary_bottleneck",
        }
        assert expected_keys.issubset(parsed.keys())

    def test_missing_file_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["analyze", "/tmp/does_not_exist.jsonl"])
        assert result.exit_code != 0

    def test_attempt_breakdown_format_wrong_tool(self) -> None:
        result = runner.invoke(app, ["analyze", str(DEMOS / "wrong_tool_retry.jsonl")])
        assert "wrong_tool" in result.output
        assert "dominant:" in result.output


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_exits_zero_on_valid_trace(self) -> None:
        result = runner.invoke(app, ["validate", str(DEMOS / "happy_path.jsonl")])
        assert result.exit_code == 0, result.output

    def test_valid_message_shown(self) -> None:
        result = runner.invoke(app, ["validate", str(DEMOS / "happy_path.jsonl")])
        assert "Trace is valid" in result.output

    def test_event_count_shown(self) -> None:
        result = runner.invoke(app, ["validate", str(DEMOS / "happy_path.jsonl")])
        # happy_path has 1 run + 1 attempt + 3 iterations + 3 model_calls + 2 tool_calls + 1 eval
        assert "events" in result.output

    def test_all_demo_scenarios_valid(self) -> None:
        for jsonl in sorted(DEMOS.glob("*.jsonl")):
            result = runner.invoke(app, ["validate", str(jsonl)])
            assert result.exit_code == 0, f"{jsonl.name}: {result.output}"

    def test_missing_file_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["validate", "/tmp/does_not_exist.jsonl"])
        assert result.exit_code != 0

    def test_malformed_file_exits_nonzero(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not valid json at all\n{also bad}\n")
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code != 0

    def test_malformed_reports_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{completely: invalid}\n")
        result = runner.invoke(app, ["validate", str(bad)])
        # Either exit code 1 or error message; either is acceptable
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_empty_file_exits_nonzero(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(app, ["validate", str(empty)])
        assert result.exit_code != 0

    def test_minimal_valid_fixture(self) -> None:
        fixture = FIXTURES / "minimal_success.jsonl"
        if fixture.exists():
            result = runner.invoke(app, ["validate", str(fixture)])
            assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


class TestDemoCommand:
    def test_single_scenario_exits_zero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["demo", "happy_path"])
        assert result.exit_code == 0, result.output

    def test_single_scenario_prints_path(self) -> None:
        result = runner.invoke(app, ["demo", "happy_path"])
        assert "wrote" in result.output

    def test_all_scenarios_exit_zero(self) -> None:
        result = runner.invoke(app, ["demo", "all"])
        assert result.exit_code == 0, result.output

    def test_all_scenarios_writes_8_files(self) -> None:
        result = runner.invoke(app, ["demo", "all"])
        assert result.exit_code == 0, result.output
        assert "8 files" in result.output

    def test_unknown_scenario_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["demo", "no_such_scenario"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


class TestMonitorCommand:
    def test_list_sessions_exits_zero(self) -> None:
        result = runner.invoke(app, ["monitor", "--list-sessions"])
        assert result.exit_code == 0, result.output

    def test_manual_mode_accepted(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("agent_profiler.cli._find_latest_session", lambda d: None)
        result = runner.invoke(
            app,
            ["monitor", "--manual", "--output-dir", str(tmp_path)],
            input="\n",
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Per-tool resource breakdown in CLI output
# ---------------------------------------------------------------------------


class TestPerToolResourceInCLI:
    """Test that the per-tool resource breakdown table appears when resource data exists."""

    def _write_trace_and_samples(self, tmp_path: Path):
        """Create a trace file and matching system samples with tool resource data."""
        from uuid import uuid4
        import orjson

        NS = 1_000_000_000
        run_id = uuid4()
        attempt_id = uuid4()
        iter1_id = uuid4()
        iter2_id = uuid4()

        events = [
            {
                "event_type": "run",
                "run_id": str(run_id),
                "task_description": "test",
                "start_ns": 0,
                "end_ns": 10 * NS,
                "outcome": "success",
                "attempt_count": 1,
                "total_model_calls": 2,
                "total_tool_calls": 2,
                "model_provider": "test",
                "model_name": "test-model",
                "sandbox_mode": "off",
            },
            {
                "event_type": "attempt",
                "attempt_id": str(attempt_id),
                "run_id": str(run_id),
                "attempt_number": 1,
                "start_ns": 0,
                "end_ns": 10 * NS,
                "outcome": "success",
            },
            {
                "event_type": "loop_iteration",
                "iteration_id": str(iter1_id),
                "attempt_id": str(attempt_id),
                "iteration_number": 1,
                "start_ns": 0,
                "end_ns": 4 * NS,
                "has_tool_calls": True,
                "iteration_type": "reason_and_act",
            },
            {
                "event_type": "model_call",
                "model_call_id": str(uuid4()),
                "iteration_id": str(iter1_id),
                "model_provider": "test",
                "model_name": "test-model",
                "start_ns": 0,
                "end_ns": NS // 2,
                "input_tokens": 100,
                "output_tokens": 50,
            },
            {
                "event_type": "tool_call",
                "tool_call_id": str(uuid4()),
                "iteration_id": str(iter1_id),
                "tool_name": "bash",
                "tool_category": "shell",
                "start_ns": NS,
                "end_ns": 4 * NS,
                "outcome": "success",
            },
            {
                "event_type": "loop_iteration",
                "iteration_id": str(iter2_id),
                "attempt_id": str(attempt_id),
                "iteration_number": 2,
                "start_ns": 5 * NS,
                "end_ns": 9 * NS,
                "has_tool_calls": True,
                "iteration_type": "reason_and_act",
            },
            {
                "event_type": "model_call",
                "model_call_id": str(uuid4()),
                "iteration_id": str(iter2_id),
                "model_provider": "test",
                "model_name": "test-model",
                "start_ns": 5 * NS,
                "end_ns": 5 * NS + NS // 2,
                "input_tokens": 100,
                "output_tokens": 50,
            },
            {
                "event_type": "tool_call",
                "tool_call_id": str(uuid4()),
                "iteration_id": str(iter2_id),
                "tool_name": "browser",
                "tool_category": "browser",
                "start_ns": 6 * NS,
                "end_ns": 9 * NS,
                "outcome": "success",
            },
            {
                "event_type": "evaluation",
                "eval_id": str(uuid4()),
                "attempt_id": str(attempt_id),
                "evaluator": "heuristic",
                "passed": True,
                "score": 1.0,
            },
        ]

        trace_path = tmp_path / "trace.jsonl"
        with trace_path.open("wb") as f:
            for e in events:
                f.write(orjson.dumps(e) + b"\n")

        # System samples: bash period has high CPU, browser has high network
        samples = []
        for t in range(11):
            cpu = 90.0 if t <= 4 else 10.0
            net_sent = t * 5_000_000  # 5MB per second
            samples.append({
                "sample_type": "system",
                "timestamp_ns": t * NS,
                "cpu_percent": cpu,
                "cpu_per_core": [cpu],
                "memory_rss_mb": 100.0,
                "memory_system_percent": 50.0,
                "network_bytes_sent": net_sent,
                "network_bytes_recv": net_sent,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "active_threads": 4,
                "open_files": 10,
            })

        samples_path = tmp_path / "system_samples.jsonl"
        with samples_path.open("wb") as f:
            for s in samples:
                f.write(orjson.dumps(s) + b"\n")

        return trace_path, samples_path

    def test_per_tool_table_shown_with_resource_data(self, tmp_path: Path) -> None:
        trace_path, samples_path = self._write_trace_and_samples(tmp_path)
        result = runner.invoke(
            app,
            ["analyze", str(trace_path), "--system-samples", str(samples_path)],
        )
        assert result.exit_code == 0, result.output
        assert "Per-Tool Resource Breakdown" in result.output
        assert "bash" in result.output
        assert "browser" in result.output

    def test_per_tool_table_not_shown_without_resource_data(self) -> None:
        result = runner.invoke(app, ["analyze", str(DEMOS / "happy_path.jsonl")])
        assert result.exit_code == 0, result.output
        assert "Per-Tool Resource Breakdown" not in result.output

    def test_verdict_includes_resource_detail(self, tmp_path: Path) -> None:
        trace_path, samples_path = self._write_trace_and_samples(tmp_path)
        result = runner.invoke(
            app,
            ["analyze", str(trace_path), "--system-samples", str(samples_path)],
        )
        assert result.exit_code == 0, result.output
        # Verdict should contain resource detail (cpu-bound or network-bound)
        assert "Primary bottleneck" in result.output
        # At least one of the resource-bound indicators should be present
        output_lower = result.output.lower()
        assert any(
            kw in output_lower
            for kw in ["cpu-bound", "network-bound", "memory-bound", "disk-bound"]
        ), f"Expected resource detail in verdict, got: {result.output}"
