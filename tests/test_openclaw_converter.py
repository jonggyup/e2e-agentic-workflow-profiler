"""Tests for the OpenClaw → RunTrace converter."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_profiler.collector.openclaw_converter import (
    _detect_attempt_boundaries,
    _group_into_iterations,
    _has_restart_phrase,
    _parse_messages,
    convert_openclaw_session,
    write_trace,
)
from agent_profiler.schema.trace import RunTrace, load_trace

FIXTURES = Path(__file__).parent / "fixtures"
HAPPY = FIXTURES / "openclaw_happy.jsonl"
RETRY = FIXTURES / "openclaw_retry.jsonl"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _iterations_from_fixture(path: Path):
    messages = _parse_messages(path)
    return _group_into_iterations(messages)


# ---------------------------------------------------------------------------
# Unit tests — content helpers
# ---------------------------------------------------------------------------


def test_has_restart_phrase_positive():
    assert _has_restart_phrase("That didn't work, let me try again.")
    assert _has_restart_phrase("Let me try a different approach here.")
    assert _has_restart_phrase("Starting over with a fresh strategy.")


def test_has_restart_phrase_negative():
    assert not _has_restart_phrase("I'll use bash to list the files.")
    assert not _has_restart_phrase("Task complete. The README contains...")


# ---------------------------------------------------------------------------
# Unit tests — metadata lines skipped gracefully
# ---------------------------------------------------------------------------


def test_parse_messages_skips_non_message_types():
    """session, model_change, thinking_level_change, custom lines are ignored."""
    messages = _parse_messages(HAPPY)
    # All returned events must be type="message"
    for event in messages:
        assert event.get("type") == "message"


def test_parse_messages_skips_non_message_types_retry():
    """Retry fixture has a custom metadata line that must be silently skipped."""
    messages = _parse_messages(RETRY)
    for event in messages:
        assert event.get("type") == "message"


def test_metadata_only_file_raises(tmp_path):
    """A file with only metadata events (no messages) raises ValueError."""
    meta_only = tmp_path / "meta_only.jsonl"
    meta_only.write_bytes(
        b'{"type": "session", "id": "s1", "version": "1.0", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/"}\n'
        b'{"type": "model_change", "provider": "anthropic", "modelId": "claude-opus-4-5"}\n'
        b'{"type": "custom", "data": {}}\n'
    )
    with pytest.raises(ValueError, match="No assistant messages found"):
        convert_openclaw_session(meta_only, task_description="test")


# ---------------------------------------------------------------------------
# Unit tests — iteration grouping
# ---------------------------------------------------------------------------


def test_happy_path_iteration_count():
    iterations = _iterations_from_fixture(HAPPY)
    # user msg skipped; 2 assistant+tool pairs + 1 final assistant = 3 iterations
    assert len(iterations) == 3


def test_happy_path_iteration_types():
    iterations = _iterations_from_fixture(HAPPY)
    assert iterations[0].iteration_type == "reason_and_act"
    assert iterations[1].iteration_type == "reason_and_act"
    assert iterations[2].iteration_type == "reason_only"


def test_happy_path_tool_names():
    iterations = _iterations_from_fixture(HAPPY)
    assert iterations[0].tool_uses[0]["name"] == "bash"
    assert iterations[1].tool_uses[0]["name"] == "file_read"


def test_happy_path_tool_results_linked():
    iterations = _iterations_from_fixture(HAPPY)
    # First iteration: bash result is not an error
    assert len(iterations[0].tool_results) == 1
    assert not iterations[0].tool_results[0].is_error
    assert iterations[0].tool_results[0].tool_name == "bash"


def test_retry_iteration_count():
    iterations = _iterations_from_fixture(RETRY)
    # user msg skipped; 2 assistant+tool pairs + 1 final assistant = 3 iterations
    assert len(iterations) == 3


def test_retry_first_iteration_has_error():
    iterations = _iterations_from_fixture(RETRY)
    assert iterations[0].tool_results[0].is_error


def test_retry_second_iteration_has_restart_phrase():
    iterations = _iterations_from_fixture(RETRY)
    assert _has_restart_phrase(iterations[1].text)


# ---------------------------------------------------------------------------
# Unit tests — real timestamps propagated
# ---------------------------------------------------------------------------


def test_happy_path_iterations_have_real_timestamps():
    iterations = _iterations_from_fixture(HAPPY)
    # Timestamps come from message.timestamp (Unix ms) converted to ns
    # First iteration: model_start from user msg ts=1000ms, model_end from asst ts=2500ms
    assert iterations[0].model_start_ns == 1000 * 1_000_000
    assert iterations[0].model_end_ns == 2500 * 1_000_000
    assert iterations[0].iter_end_ns == 3000 * 1_000_000  # tool result user msg ts


def test_happy_path_token_counts_propagated():
    iterations = _iterations_from_fixture(HAPPY)
    assert iterations[0].input_tokens == 100
    assert iterations[0].output_tokens == 50
    assert iterations[1].input_tokens == 200
    assert iterations[2].output_tokens == 30


# ---------------------------------------------------------------------------
# Unit tests — attempt boundary detection
# ---------------------------------------------------------------------------


def test_happy_path_one_attempt():
    iterations = _iterations_from_fixture(HAPPY)
    boundaries = _detect_attempt_boundaries(iterations)
    assert boundaries == [0]


def test_retry_two_attempts():
    iterations = _iterations_from_fixture(RETRY)
    boundaries = _detect_attempt_boundaries(iterations)
    # Restart phrase in iteration 1 creates a boundary there
    assert 0 in boundaries
    assert 1 in boundaries
    assert len(boundaries) == 2


def test_rule2_boundary_after_three_consecutive_errors():
    """Rule 2: 3+ consecutive tool errors from the same tool, then a different tool."""
    from agent_profiler.collector.openclaw_converter import _ToolResult, _IterData

    def make_iter(tool_name: str, is_error: bool, text: str = "") -> _IterData:
        tu = {"type": "tool_use", "id": f"id_{tool_name}_{is_error}", "name": tool_name, "input": {}}
        tr = _ToolResult(tool_use_id=tu["id"], tool_name=tool_name, text="", is_error=is_error)
        return _IterData(text=text, tool_uses=[tu], tool_results=[tr])

    iterations = [
        make_iter("bash", True),   # idx 0 — attempt 1 start
        make_iter("bash", True),   # idx 1
        make_iter("bash", True),   # idx 2 — 3rd consecutive bash error
        make_iter("browser", False),  # idx 3 — different tool → new attempt
        _IterData(text="Done.", tool_uses=[], tool_results=[]),  # final answer
    ]

    boundaries = _detect_attempt_boundaries(iterations)
    assert 0 in boundaries
    assert 3 in boundaries


# ---------------------------------------------------------------------------
# Integration tests — convert_openclaw_session
# ---------------------------------------------------------------------------


def test_happy_path_produces_valid_run_trace():
    trace = convert_openclaw_session(HAPPY, task_description="List files and read README")
    assert isinstance(trace, RunTrace)


def test_happy_path_single_attempt():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert len(trace.attempts) == 1
    assert trace.run.attempt_count == 1


def test_happy_path_attempt_succeeds():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert trace.attempts[0].outcome == "success"
    assert trace.attempts[0].failure_category is None


def test_happy_path_three_iterations():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert len(trace.loop_iterations) == 3


def test_happy_path_two_tool_calls():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert len(trace.tool_calls) == 2
    assert trace.run.total_tool_calls == 2


def test_happy_path_three_model_calls():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert len(trace.model_calls) == 3
    assert trace.run.total_model_calls == 3


def test_happy_path_has_evaluation():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert len(trace.evaluations) == 1
    ev = trace.evaluations[0]
    assert ev.evaluator == "heuristic"
    assert ev.passed is True
    assert ev.score == 1.0


def test_happy_path_run_outcome_success():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert trace.run.outcome == "success"


def test_happy_path_no_estimated_timing_prefix():
    """Real timestamps: task description should not be prefixed with [estimated_timing]."""
    trace = convert_openclaw_session(HAPPY)
    assert "[estimated_timing]" not in trace.run.task_description


def test_happy_path_task_from_user_message():
    # No task_description → extracted from first user message
    trace = convert_openclaw_session(HAPPY)
    assert "List the files" in trace.run.task_description


def test_happy_path_tool_categories():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    tool_by_name = {tc.tool_name: tc for tc in trace.tool_calls}
    assert tool_by_name["bash"].tool_category == "shell"
    assert tool_by_name["file_read"].tool_category == "filesystem"


def test_happy_path_timestamps_monotonic():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    # Run
    assert trace.run.end_ns > trace.run.start_ns
    # All attempts
    for a in trace.attempts:
        assert a.end_ns >= a.start_ns
    # All loop iterations
    for li in trace.loop_iterations:
        assert li.end_ns > li.start_ns
    # All tool calls
    for tc in trace.tool_calls:
        assert tc.end_ns > tc.start_ns


def test_happy_path_real_timestamps_from_messages():
    """Run start/end are derived from real message timestamps, not synthetic values."""
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    # First user msg ts=1000ms → run_start_ns = 1_000_000_000 ns
    assert trace.run.start_ns == 1000 * 1_000_000
    # Last iteration iter_end_ns = tool_result user ts=5000ms + EPSILON (reason_only has no tool result)
    # Actually the final assistant (reason_only) has iter_end = asst_ts + EPSILON = 6500ms + 1ms
    assert trace.run.end_ns > 6500 * 1_000_000


def test_happy_path_model_info_extracted():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    assert trace.run.model_provider == "anthropic"
    assert trace.run.model_name == "claude-opus-4-5"
    for mc in trace.model_calls:
        assert mc.model_provider == "anthropic"
        assert mc.model_name == "claude-opus-4-5"


def test_happy_path_token_counts_in_model_calls():
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    # Check first model call has tokens from fixture
    first_mc = trace.model_calls[0]
    assert first_mc.input_tokens == 100
    assert first_mc.output_tokens == 50


def test_happy_path_program_tool_flag():
    trace = convert_openclaw_session(
        HAPPY, task_description="List files", program_tool_name="bash"
    )
    bash_calls = [tc for tc in trace.tool_calls if tc.tool_name == "bash"]
    assert all(tc.is_program_under_test for tc in bash_calls)

    file_read_calls = [tc for tc in trace.tool_calls if tc.tool_name == "file_read"]
    assert all(not tc.is_program_under_test for tc in file_read_calls)


# ---------------------------------------------------------------------------
# Retry fixture tests
# ---------------------------------------------------------------------------


def test_retry_produces_valid_run_trace():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    assert isinstance(trace, RunTrace)


def test_retry_two_attempts_detected():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    assert len(trace.attempts) == 2
    assert trace.run.attempt_count == 2


def test_retry_first_attempt_fails():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    first = trace.attempts[0]
    assert first.outcome == "failure"
    assert first.failure_category is not None


def test_retry_second_attempt_succeeds():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    second = trace.attempts[1]
    assert second.outcome == "success"
    assert second.failure_category is None


def test_retry_run_outcome_success():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    assert trace.run.outcome == "success"


def test_retry_evaluation_passed():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    assert len(trace.evaluations) == 1
    assert trace.evaluations[0].passed is True


def test_retry_three_loop_iterations():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    # iteration 0 → attempt 1, iterations 1-2 → attempt 2
    assert len(trace.loop_iterations) == 3


def test_retry_tool_calls_correct_iterations():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    bash_calls = [tc for tc in trace.tool_calls if tc.tool_name == "bash"]
    browser_calls = [tc for tc in trace.tool_calls if tc.tool_name == "browser"]
    assert len(bash_calls) == 1
    assert len(browser_calls) == 1
    assert bash_calls[0].outcome == "error"
    assert browser_calls[0].outcome == "success"


def test_retry_model_info_extracted():
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    assert trace.run.model_provider == "anthropic"
    assert trace.run.model_name == "claude-opus-4-5"


# ---------------------------------------------------------------------------
# write_trace round-trip
# ---------------------------------------------------------------------------


def test_write_trace_roundtrip_happy(tmp_path):
    trace = convert_openclaw_session(HAPPY, task_description="List files")
    out = tmp_path / "out.profiler.jsonl"
    write_trace(trace, out)
    assert out.exists()

    reloaded = load_trace(out, strict=True)
    assert reloaded.run.run_id == trace.run.run_id
    assert len(reloaded.attempts) == len(trace.attempts)
    assert len(reloaded.tool_calls) == len(trace.tool_calls)


# ---------------------------------------------------------------------------
# Timestamp conversion regression
# ---------------------------------------------------------------------------


def test_timestamp_conversion_60s(tmp_path):
    """Unix-ms timestamps 60 s apart must yield e2e_wall_ms ≈ 60,000 (not 60 or 60,000,000)."""
    start_ms = 1_742_726_400_000  # real-world Unix ms (2025)
    end_ms = start_ms + 60_000    # 60 seconds later

    fixture = tmp_path / "ts_60s.jsonl"
    fixture.write_bytes(
        b'{"type": "model_change", "provider": "test", "modelId": "test-model"}\n'
        + f'{{"type": "message", "id": "u1", "message": {{"role": "user", "content": [{{"type": "text", "text": "go"}}], "timestamp": {start_ms}}}}}\n'.encode()
        + f'{{"type": "message", "id": "a1", "message": {{"role": "assistant", "content": [{{"type": "text", "text": "Done."}}], "timestamp": {end_ms}}}}}\n'.encode()
    )

    trace = convert_openclaw_session(fixture, task_description="timing test")
    e2e_wall_ms = (trace.run.end_ns - trace.run.start_ns) / 1_000_000
    # Should be ≈ 60_000 ms (±10 ms for EPSILON).  Wrong unit conversions would
    # produce 60 (seconds) or 60_000_000 (microseconds treated as ms).
    assert abs(e2e_wall_ms - 60_000) < 10


# ---------------------------------------------------------------------------
# camelCase / toolResult role coverage
# ---------------------------------------------------------------------------


def test_happy_camelcase_tool_call_extracted():
    """toolCall blocks (iteration 2 of happy fixture) are extracted correctly."""
    iterations = _iterations_from_fixture(HAPPY)
    assert iterations[1].tool_uses[0]["name"] == "file_read"
    assert iterations[1].tool_uses[0]["input"] == {"path": "README.md"}


def test_happy_toolresult_role_linked():
    """toolResult-role messages (iteration 2 of happy fixture) are linked to results."""
    iterations = _iterations_from_fixture(HAPPY)
    assert len(iterations[1].tool_results) == 1
    assert iterations[1].tool_results[0].tool_name == "file_read"
    assert not iterations[1].tool_results[0].is_error


def test_retry_toolresult_role_linked():
    """toolResult-role messages (retry iteration 1) are linked to results."""
    iterations = _iterations_from_fixture(RETRY)
    assert len(iterations[1].tool_results) == 1
    assert iterations[1].tool_results[0].tool_name == "browser"
    assert not iterations[1].tool_results[0].is_error


def test_write_trace_roundtrip_retry(tmp_path):
    trace = convert_openclaw_session(RETRY, task_description="Fetch API data")
    out = tmp_path / "retry.profiler.jsonl"
    write_trace(trace, out)
    reloaded = load_trace(out, strict=True)
    assert len(reloaded.attempts) == 2
    assert reloaded.run.outcome == "success"
