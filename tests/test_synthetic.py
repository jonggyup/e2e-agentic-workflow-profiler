"""Tests for the synthetic trace generator.

Validates that every scenario:
1. Generates a .jsonl file that passes load_trace() without error.
2. Has correct top-level shape (attempt count, outcome, etc.).
3. Has internally consistent timestamps (monotonically increasing per event stream).
4. Meets the scenario-specific semantic requirements from docs/v0-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the src tree is importable when run via uv run pytest
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_profiler.schema.trace import RunTrace, load_trace

# Pull the generator so tests don't depend on pre-written output files
sys.path.insert(0, str(Path(__file__).parent.parent / "demos"))
from synthetic_run import generate, ALL_SCENARIOS

OUTPUT_DIR = Path(__file__).parent.parent / "demos" / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(scenario: str) -> RunTrace:
    """Generate (or re-generate) a scenario and load it."""
    path = generate(scenario)
    return load_trace(path, strict=True)


def _all_start_end_ns(trace: RunTrace) -> list[tuple[int, int]]:
    intervals = []
    for ev in (
        list(trace.attempts)
        + list(trace.loop_iterations)
        + list(trace.model_calls)
        + list(trace.tool_calls)
    ):
        intervals.append((ev.start_ns, ev.end_ns))
    return intervals


# ---------------------------------------------------------------------------
# Parametrised: every scenario loads cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_generates_valid_trace(scenario: str) -> None:
    """load_trace() must succeed (strict=True) for every scenario."""
    trace = _load(scenario)
    assert isinstance(trace, RunTrace)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_timestamps_are_monotonic(scenario: str) -> None:
    """Every timed event must have end_ns >= start_ns."""
    trace = _load(scenario)
    for start, end in _all_start_end_ns(trace):
        assert end >= start, f"end_ns {end} < start_ns {start}"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_run_start_before_end(scenario: str) -> None:
    trace = _load(scenario)
    assert trace.run.end_ns > trace.run.start_ns


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_attempt_count_matches(scenario: str) -> None:
    """run.attempt_count must equal the actual number of AttemptEvents."""
    trace = _load(scenario)
    assert trace.run.attempt_count == len(trace.attempts)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_model_call_count_matches(scenario: str) -> None:
    """run.total_model_calls must equal the actual number of ModelCallEvents."""
    trace = _load(scenario)
    assert trace.run.total_model_calls == len(trace.model_calls)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_tool_call_count_matches(scenario: str) -> None:
    """run.total_tool_calls must equal the actual number of ToolCallEvents."""
    trace = _load(scenario)
    assert trace.run.total_tool_calls == len(trace.tool_calls)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_has_evaluation(scenario: str) -> None:
    """Every scenario must have at least one EvaluationEvent on the final attempt."""
    trace = _load(scenario)
    final_attempt_id = trace.attempts[-1].attempt_id
    evals_for_final = [e for e in trace.evaluations if e.attempt_id == final_attempt_id]
    assert evals_for_final, "Final attempt must have an evaluation"
    assert evals_for_final[0].passed is True, "Final attempt evaluation must have passed=True"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_run_outcome_matches_last_attempt(scenario: str) -> None:
    """run.outcome must reflect the last attempt's outcome."""
    trace = _load(scenario)
    last = trace.attempts[-1].outcome
    assert trace.run.outcome == last


# ---------------------------------------------------------------------------
# Scenario-specific semantic tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_attempt_success(self) -> None:
        trace = _load("happy_path")
        assert len(trace.attempts) == 1
        assert trace.attempts[0].outcome == "success"
        assert trace.attempts[0].failure_category is None

    def test_three_loop_iterations(self) -> None:
        trace = _load("happy_path")
        assert len(trace.loop_iterations) == 3

    def test_final_iteration_is_reason_only(self) -> None:
        trace = _load("happy_path")
        last = trace.loop_iterations[-1]
        assert last.iteration_type == "reason_only"
        assert last.has_tool_calls is False

    def test_no_failed_tool_calls(self) -> None:
        trace = _load("happy_path")
        assert all(tc.outcome == "success" for tc in trace.tool_calls)

    def test_no_warnings(self) -> None:
        trace = _load("happy_path")
        assert trace.warnings == []


class TestWrongToolRetry:
    def test_two_attempts(self) -> None:
        trace = _load("wrong_tool_retry")
        assert len(trace.attempts) == 2

    def test_first_attempt_fails_wrong_tool(self) -> None:
        trace = _load("wrong_tool_retry")
        a1 = trace.attempts[0]
        assert a1.outcome == "failure"
        assert a1.failure_category == "wrong_tool"
        assert a1.failure_reason is not None

    def test_second_attempt_succeeds(self) -> None:
        trace = _load("wrong_tool_retry")
        assert trace.attempts[1].outcome == "success"

    def test_first_attempt_has_failed_tool_call(self) -> None:
        trace = _load("wrong_tool_retry")
        a1_id = trace.attempts[0].attempt_id
        a1_iter_ids = {
            it.iteration_id
            for it in trace.loop_iterations
            if it.attempt_id == a1_id
        }
        a1_tool_calls = [tc for tc in trace.tool_calls if tc.iteration_id in a1_iter_ids]
        assert any(tc.outcome == "error" for tc in a1_tool_calls)

    def test_second_attempt_uses_browser(self) -> None:
        trace = _load("wrong_tool_retry")
        a2_id = trace.attempts[1].attempt_id
        a2_iter_ids = {
            it.iteration_id
            for it in trace.loop_iterations
            if it.attempt_id == a2_id
        }
        a2_tool_calls = [tc for tc in trace.tool_calls if tc.iteration_id in a2_iter_ids]
        assert any(tc.tool_name == "browser" for tc in a2_tool_calls)


class TestSlowProgram:
    def test_single_attempt(self) -> None:
        trace = _load("slow_program")
        assert len(trace.attempts) == 1
        assert trace.attempts[0].outcome == "success"

    def test_has_program_under_test(self) -> None:
        trace = _load("slow_program")
        put_calls = [tc for tc in trace.tool_calls if tc.is_program_under_test]
        assert len(put_calls) >= 1

    def test_program_under_test_is_slow(self) -> None:
        """The make build call should be >= 20 seconds."""
        trace = _load("slow_program")
        put_calls = [tc for tc in trace.tool_calls if tc.is_program_under_test]
        durations_ms = [(tc.end_ns - tc.start_ns) / 1_000_000 for tc in put_calls]
        assert max(durations_ms) >= 20_000

    def test_program_runtime_dominates(self) -> None:
        """program_runtime_ms must be > 50% of e2e wall time."""
        trace = _load("slow_program")
        e2e_ms = (trace.run.end_ns - trace.run.start_ns) / 1_000_000
        put_ms = sum(
            (tc.end_ns - tc.start_ns) / 1_000_000
            for tc in trace.tool_calls
            if tc.is_program_under_test
        )
        assert put_ms > 0.50 * e2e_ms


class TestReasoningHeavy:
    def test_single_attempt(self) -> None:
        trace = _load("reasoning_heavy")
        assert len(trace.attempts) == 1
        assert trace.attempts[0].outcome == "success"

    def test_eight_iterations(self) -> None:
        trace = _load("reasoning_heavy")
        assert len(trace.loop_iterations) == 8

    def test_only_two_iterations_have_tool_calls(self) -> None:
        trace = _load("reasoning_heavy")
        with_tools = [it for it in trace.loop_iterations if it.has_tool_calls]
        assert len(with_tools) == 2

    def test_model_time_dominates(self) -> None:
        """Total model call time must be > 60% of wall time."""
        trace = _load("reasoning_heavy")
        e2e_ms = (trace.run.end_ns - trace.run.start_ns) / 1_000_000
        model_ms = sum(
            (mc.end_ns - mc.start_ns) / 1_000_000 for mc in trace.model_calls
        )
        assert model_ms > 0.60 * e2e_ms


class TestReasoningLoop:
    def test_two_attempts(self) -> None:
        trace = _load("reasoning_loop")
        assert len(trace.attempts) == 2

    def test_first_attempt_fails_reasoning_loop(self) -> None:
        trace = _load("reasoning_loop")
        a1 = trace.attempts[0]
        assert a1.outcome == "failure"
        assert a1.failure_category == "reasoning_loop"

    def test_same_command_repeated(self) -> None:
        """Attempt 1 must have the same bash command repeated ≥ 3 times."""
        trace = _load("reasoning_loop")
        a1_id = trace.attempts[0].attempt_id
        a1_iter_ids = {
            it.iteration_id
            for it in trace.loop_iterations
            if it.attempt_id == a1_id
        }
        a1_tool_calls = [tc for tc in trace.tool_calls if tc.iteration_id in a1_iter_ids]
        commands = [tc.tool_params.get("command", "") for tc in a1_tool_calls if tc.tool_name == "bash"]
        # All npm install calls should be identical
        assert len(commands) >= 3
        assert len(set(commands)) == 1, "Same command should be repeated"

    def test_second_attempt_succeeds_different_tool(self) -> None:
        trace = _load("reasoning_loop")
        a2 = trace.attempts[1]
        assert a2.outcome == "success"
        a2_iter_ids = {
            it.iteration_id
            for it in trace.loop_iterations
            if it.attempt_id == a2.attempt_id
        }
        a2_tools = [tc for tc in trace.tool_calls if tc.iteration_id in a2_iter_ids]
        commands = [tc.tool_params.get("command", "") for tc in a2_tools if tc.tool_name == "bash"]
        assert any("yarn" in cmd for cmd in commands)


class TestTransientFailure:
    def test_single_attempt(self) -> None:
        trace = _load("transient_failure")
        assert len(trace.attempts) == 1
        assert trace.attempts[0].outcome == "success"

    def test_has_one_failed_tool_call(self) -> None:
        """Exactly one tool call must have outcome='error' (the 503)."""
        trace = _load("transient_failure")
        errors = [tc for tc in trace.tool_calls if tc.outcome == "error"]
        assert len(errors) == 1
        assert "503" in (errors[0].error_message or "")

    def test_retry_succeeds(self) -> None:
        """The same tool called after the 503 must succeed."""
        trace = _load("transient_failure")
        errors = [tc for tc in trace.tool_calls if tc.outcome == "error"]
        assert errors
        failed_tool = errors[0].tool_name
        success_calls = [
            tc for tc in trace.tool_calls
            if tc.tool_name == failed_tool and tc.outcome == "success"
        ]
        assert success_calls, f"No successful {failed_tool} call found after 503"

    def test_first_pass_success(self) -> None:
        trace = _load("transient_failure")
        assert trace.attempts[0].outcome == "success"
        assert trace.attempts[0].failure_category is None


class TestContextOverflow:
    def test_two_attempts(self) -> None:
        trace = _load("context_overflow")
        assert len(trace.attempts) == 2

    def test_first_attempt_fails_context_overflow(self) -> None:
        trace = _load("context_overflow")
        a1 = trace.attempts[0]
        assert a1.outcome == "failure"
        assert a1.failure_category == "context_overflow"

    def test_first_attempt_has_many_iterations(self) -> None:
        trace = _load("context_overflow")
        a1_id = trace.attempts[0].attempt_id
        a1_iters = [it for it in trace.loop_iterations if it.attempt_id == a1_id]
        assert len(a1_iters) >= 15

    def test_second_attempt_succeeds_few_iterations(self) -> None:
        trace = _load("context_overflow")
        a2 = trace.attempts[1]
        assert a2.outcome == "success"
        a2_iters = [it for it in trace.loop_iterations if it.attempt_id == a2.attempt_id]
        assert len(a2_iters) <= 5

    def test_tokens_grow_in_first_attempt(self) -> None:
        """Input token counts in attempt 1 should grow across model calls."""
        trace = _load("context_overflow")
        a1_id = trace.attempts[0].attempt_id
        a1_iter_ids = {
            it.iteration_id
            for it in trace.loop_iterations
            if it.attempt_id == a1_id
        }
        a1_model_calls = [mc for mc in trace.model_calls if mc.iteration_id in a1_iter_ids]
        tokens = [mc.input_tokens for mc in a1_model_calls if mc.input_tokens is not None]
        assert tokens[-1] > tokens[0], "Input tokens must grow across attempt 1"


class TestHallucinatedTool:
    def test_two_attempts(self) -> None:
        trace = _load("hallucinated_tool")
        assert len(trace.attempts) == 2

    def test_first_attempt_fails_hallucinated_tool(self) -> None:
        trace = _load("hallucinated_tool")
        a1 = trace.attempts[0]
        assert a1.outcome == "failure"
        assert a1.failure_category == "hallucinated_tool"

    def test_nonexistent_tool_was_called(self) -> None:
        """Attempt 1 should have a tool call that errored with ToolNotFoundError."""
        trace = _load("hallucinated_tool")
        a1_id = trace.attempts[0].attempt_id
        a1_iter_ids = {
            it.iteration_id
            for it in trace.loop_iterations
            if it.attempt_id == a1_id
        }
        a1_tools = [tc for tc in trace.tool_calls if tc.iteration_id in a1_iter_ids]
        assert any(
            tc.outcome == "error" and "ToolNotFoundError" in (tc.error_message or "")
            for tc in a1_tools
        )

    def test_second_attempt_uses_bash(self) -> None:
        trace = _load("hallucinated_tool")
        a2_id = trace.attempts[1].attempt_id
        a2_iter_ids = {
            it.iteration_id
            for it in trace.loop_iterations
            if it.attempt_id == a2_id
        }
        a2_tools = [tc for tc in trace.tool_calls if tc.iteration_id in a2_iter_ids]
        assert any(tc.tool_name == "bash" for tc in a2_tools)

    def test_second_attempt_succeeds(self) -> None:
        trace = _load("hallucinated_tool")
        assert trace.attempts[1].outcome == "success"

    def test_hallucinated_tool_generates_warning(self) -> None:
        """The send_email hallucination should produce a trace warning."""
        trace = _load("hallucinated_tool")
        assert any("tools_called_in_response" in w.message for w in trace.warnings)
