"""Round-trip serialisation and validation tests for every event type."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

import orjson
import pytest

from agent_profiler.schema.events import (
    AttemptEvent,
    EvaluationCriteria,
    EvaluationEvent,
    LoopIterationEvent,
    ModelCallEvent,
    RunEvent,
    ToolCallEvent,
    _event_adapter,
    event_to_jsonl_bytes,
)
from agent_profiler.schema.trace import (
    RunTrace,
    TraceValidationError,
    load_trace,
)

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# UUIDs used throughout
# ---------------------------------------------------------------------------

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
ATTEMPT_ID = UUID("00000000-0000-0000-0000-000000000002")
ITER1_ID = UUID("00000000-0000-0000-0000-000000000003")
ITER2_ID = UUID("00000000-0000-0000-0000-000000000004")
MC1_ID = UUID("00000000-0000-0000-0000-000000000005")
TC1_ID = UUID("00000000-0000-0000-0000-000000000006")
MC2_ID = UUID("00000000-0000-0000-0000-000000000007")
EVAL_ID = UUID("00000000-0000-0000-0000-000000000008")

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_run(**kw) -> RunEvent:
    defaults = dict(
        event_type="run",
        run_id=RUN_ID,
        task_description="test task",
        start_ns=1_000_000_000,
        end_ns=2_500_000_000,
        outcome="success",
        attempt_count=1,
        total_model_calls=2,
        total_tool_calls=1,
        model_provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        sandbox_mode="off",
    )
    defaults.update(kw)
    return RunEvent(**defaults)


def make_attempt(**kw) -> AttemptEvent:
    defaults = dict(
        event_type="attempt",
        attempt_id=ATTEMPT_ID,
        run_id=RUN_ID,
        attempt_number=1,
        start_ns=1_000_000_000,
        end_ns=2_500_000_000,
        outcome="success",
    )
    defaults.update(kw)
    return AttemptEvent(**defaults)


def make_loop_iteration(
    iteration_id=ITER1_ID,
    iteration_number=1,
    has_tool_calls=True,
    iteration_type="reason_and_act",
    **kw,
) -> LoopIterationEvent:
    defaults = dict(
        event_type="loop_iteration",
        iteration_id=iteration_id,
        attempt_id=ATTEMPT_ID,
        iteration_number=iteration_number,
        start_ns=1_000_000_000,
        end_ns=1_800_000_000,
        has_tool_calls=has_tool_calls,
        iteration_type=iteration_type,
    )
    defaults.update(kw)
    return LoopIterationEvent(**defaults)


def make_model_call(model_call_id=MC1_ID, iteration_id=ITER1_ID, **kw) -> ModelCallEvent:
    defaults = dict(
        event_type="model_call",
        model_call_id=model_call_id,
        iteration_id=iteration_id,
        model_provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        start_ns=1_000_000_000,
        end_ns=1_500_000_000,
        input_tokens=1200,
        output_tokens=85,
        time_to_first_token_ms=320.0,
        requested_tools=["bash", "file_write"],
        tools_called_in_response=["file_write"],
    )
    defaults.update(kw)
    return ModelCallEvent(**defaults)


def make_tool_call(**kw) -> ToolCallEvent:
    defaults = dict(
        event_type="tool_call",
        tool_call_id=TC1_ID,
        iteration_id=ITER1_ID,
        tool_name="file_write",
        tool_category="filesystem",
        tool_params={"path": "/tmp/hello.txt", "content": "Hello"},
        start_ns=1_500_000_000,
        end_ns=1_800_000_000,
        outcome="success",
    )
    defaults.update(kw)
    return ToolCallEvent(**defaults)


def make_evaluation(**kw) -> EvaluationEvent:
    defaults = dict(
        event_type="evaluation",
        eval_id=EVAL_ID,
        attempt_id=ATTEMPT_ID,
        evaluator="heuristic",
        passed=True,
        score=1.0,
        reason="Task completed",
        criteria=EvaluationCriteria(
            task_completed=True,
            correct_result=True,
            no_side_effects=True,
            efficient_path=True,
        ),
    )
    defaults.update(kw)
    return EvaluationEvent(**defaults)


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------


def roundtrip(event):
    """Serialise to JSONL bytes, parse back, return reconstructed event."""
    raw = event_to_jsonl_bytes(event)
    data = orjson.loads(raw)
    return _event_adapter.validate_python(data)


def assert_roundtrip(event):
    restored = roundtrip(event)
    assert restored.model_dump() == event.model_dump()
    return restored


# ---------------------------------------------------------------------------
# RunEvent
# ---------------------------------------------------------------------------


class TestRunEvent:
    def test_round_trip(self):
        assert_roundtrip(make_run())

    def test_all_outcomes(self):
        for outcome in ("success", "failure", "timeout", "interrupted"):
            assert_roundtrip(make_run(outcome=outcome))

    def test_all_sandbox_modes(self):
        for mode in ("off", "docker", "openshell"):
            assert_roundtrip(make_run(sandbox_mode=mode))

    def test_invalid_end_before_start(self):
        with pytest.raises(Exception, match="end_ns"):
            make_run(start_ns=2_000_000_000, end_ns=1_000_000_000)

    def test_blank_task_description(self):
        with pytest.raises(Exception, match="task_description"):
            make_run(task_description="   ")

    def test_attempt_count_zero(self):
        with pytest.raises(Exception, match="attempt_count"):
            make_run(attempt_count=0)

    def test_negative_model_calls(self):
        with pytest.raises(Exception, match="total_model_calls"):
            make_run(total_model_calls=-1)

    def test_zero_end_equals_start(self):
        # end_ns == start_ns is valid (instantaneous)
        assert_roundtrip(make_run(start_ns=1_000_000_000, end_ns=1_000_000_000))


# ---------------------------------------------------------------------------
# AttemptEvent
# ---------------------------------------------------------------------------


class TestAttemptEvent:
    def test_round_trip_success(self):
        assert_roundtrip(make_attempt())

    def test_round_trip_failure(self):
        event = make_attempt(
            outcome="failure",
            failure_reason="Used wrong tool",
            failure_category="wrong_tool",
        )
        assert_roundtrip(event)

    def test_failure_requires_category(self):
        with pytest.raises(Exception, match="failure_category"):
            make_attempt(outcome="failure", failure_category=None)

    def test_success_rejects_failure_category(self):
        with pytest.raises(Exception, match="failure_category"):
            make_attempt(outcome="success", failure_category="wrong_tool")

    def test_success_rejects_failure_reason(self):
        with pytest.raises(Exception, match="failure_reason"):
            make_attempt(outcome="success", failure_reason="shouldn't be here")

    def test_all_failure_categories(self):
        for cat in (
            "wrong_tool", "bad_params", "tool_error", "transient", "timeout",
            "reasoning_loop", "hallucinated_tool", "context_overflow",
        ):
            event = make_attempt(outcome="failure", failure_category=cat)
            assert_roundtrip(event)

    def test_attempt_number_zero(self):
        with pytest.raises(Exception, match="attempt_number"):
            make_attempt(attempt_number=0)


# ---------------------------------------------------------------------------
# LoopIterationEvent
# ---------------------------------------------------------------------------


class TestLoopIterationEvent:
    def test_round_trip_reason_and_act(self):
        assert_roundtrip(make_loop_iteration())

    def test_round_trip_reason_only(self):
        assert_roundtrip(
            make_loop_iteration(
                iteration_id=ITER2_ID,
                has_tool_calls=False,
                iteration_type="reason_only",
            )
        )

    def test_round_trip_act_only(self):
        assert_roundtrip(
            make_loop_iteration(has_tool_calls=True, iteration_type="act_only")
        )

    def test_reason_only_with_tool_calls_invalid(self):
        with pytest.raises(Exception, match="has_tool_calls"):
            make_loop_iteration(has_tool_calls=True, iteration_type="reason_only")

    def test_reason_and_act_without_tool_calls_invalid(self):
        with pytest.raises(Exception, match="has_tool_calls"):
            make_loop_iteration(has_tool_calls=False, iteration_type="reason_and_act")

    def test_act_only_without_tool_calls_invalid(self):
        with pytest.raises(Exception, match="has_tool_calls"):
            make_loop_iteration(has_tool_calls=False, iteration_type="act_only")

    def test_iteration_number_zero(self):
        with pytest.raises(Exception, match="iteration_number"):
            make_loop_iteration(iteration_number=0)


# ---------------------------------------------------------------------------
# ModelCallEvent
# ---------------------------------------------------------------------------


class TestModelCallEvent:
    def test_round_trip_full(self):
        assert_roundtrip(make_model_call())

    def test_round_trip_optional_none(self):
        event = make_model_call(
            input_tokens=None,
            output_tokens=None,
            time_to_first_token_ms=None,
            requested_tools=[],
            tools_called_in_response=[],
        )
        assert_roundtrip(event)

    def test_negative_input_tokens(self):
        with pytest.raises(Exception, match="input_tokens"):
            make_model_call(input_tokens=-1)

    def test_negative_output_tokens(self):
        with pytest.raises(Exception, match="output_tokens"):
            make_model_call(output_tokens=-1)

    def test_negative_ttft(self):
        with pytest.raises(Exception, match="time_to_first_token_ms"):
            make_model_call(time_to_first_token_ms=-0.1)

    def test_blank_model_provider(self):
        with pytest.raises(Exception, match="model_provider"):
            make_model_call(model_provider="  ")

    def test_zero_tokens_allowed(self):
        assert_roundtrip(make_model_call(input_tokens=0, output_tokens=0))


# ---------------------------------------------------------------------------
# ToolCallEvent
# ---------------------------------------------------------------------------


class TestToolCallEvent:
    def test_round_trip_success(self):
        assert_roundtrip(make_tool_call())

    def test_round_trip_error(self):
        event = make_tool_call(outcome="error", error_message="Permission denied")
        assert_roundtrip(event)

    def test_round_trip_timeout(self):
        assert_roundtrip(make_tool_call(outcome="timeout"))

    def test_all_tool_categories(self):
        for cat in ("shell", "filesystem", "browser", "mcp", "canvas", "system"):
            assert_roundtrip(make_tool_call(tool_category=cat))

    def test_success_with_error_message_invalid(self):
        with pytest.raises(Exception, match="error_message"):
            make_tool_call(outcome="success", error_message="should not be here")

    def test_blank_tool_name(self):
        with pytest.raises(Exception, match="tool_name"):
            make_tool_call(tool_name="  ")

    def test_tool_params_empty(self):
        assert_roundtrip(make_tool_call(tool_params={}))

    def test_is_program_under_test_flag(self):
        event = make_tool_call(is_program_under_test=True)
        restored = assert_roundtrip(event)
        assert restored.is_program_under_test is True


# ---------------------------------------------------------------------------
# EvaluationEvent
# ---------------------------------------------------------------------------


class TestEvaluationEvent:
    def test_round_trip_full(self):
        assert_roundtrip(make_evaluation())

    def test_round_trip_no_score(self):
        assert_roundtrip(make_evaluation(score=None, reason=None))

    def test_score_boundary_zero(self):
        assert_roundtrip(make_evaluation(score=0.0))

    def test_score_boundary_one(self):
        assert_roundtrip(make_evaluation(score=1.0))

    def test_score_out_of_range_high(self):
        with pytest.raises(Exception, match="score"):
            make_evaluation(score=1.001)

    def test_score_out_of_range_low(self):
        with pytest.raises(Exception, match="score"):
            make_evaluation(score=-0.1)

    def test_all_evaluators(self):
        for ev in ("human", "llm_judge", "script", "heuristic"):
            assert_roundtrip(make_evaluation(evaluator=ev))

    def test_criteria_extra_keys(self):
        # extra="allow" on EvaluationCriteria should not raise
        criteria = EvaluationCriteria(task_completed=True, custom_check=True)
        event = make_evaluation(criteria=criteria)
        assert_roundtrip(event)

    def test_criteria_all_none(self):
        assert_roundtrip(make_evaluation(criteria=EvaluationCriteria()))

    def test_no_timestamps(self):
        ev = make_evaluation()
        assert not hasattr(ev, "start_ns")
        assert not hasattr(ev, "end_ns")


# ---------------------------------------------------------------------------
# Discriminated union dispatch
# ---------------------------------------------------------------------------


class TestAnyEvent:
    def test_dispatches_run(self):
        data = make_run().model_dump(mode="json")
        result = _event_adapter.validate_python(data)
        assert isinstance(result, RunEvent)

    def test_dispatches_attempt(self):
        data = make_attempt().model_dump(mode="json")
        result = _event_adapter.validate_python(data)
        assert isinstance(result, AttemptEvent)

    def test_dispatches_loop_iteration(self):
        data = make_loop_iteration().model_dump(mode="json")
        result = _event_adapter.validate_python(data)
        assert isinstance(result, LoopIterationEvent)

    def test_dispatches_model_call(self):
        data = make_model_call().model_dump(mode="json")
        result = _event_adapter.validate_python(data)
        assert isinstance(result, ModelCallEvent)

    def test_dispatches_tool_call(self):
        data = make_tool_call().model_dump(mode="json")
        result = _event_adapter.validate_python(data)
        assert isinstance(result, ToolCallEvent)

    def test_dispatches_evaluation(self):
        data = make_evaluation().model_dump(mode="json")
        result = _event_adapter.validate_python(data)
        assert isinstance(result, EvaluationEvent)

    def test_unknown_event_type_raises(self):
        with pytest.raises(Exception):
            _event_adapter.validate_python({"event_type": "nonexistent"})


# ---------------------------------------------------------------------------
# load_trace — happy path with minimal_success.jsonl
# ---------------------------------------------------------------------------


class TestLoadTrace:
    def test_loads_without_error(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert isinstance(trace, RunTrace)

    def test_run_fields(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert trace.run.run_id == RUN_ID
        assert trace.run.outcome == "success"
        assert trace.run.attempt_count == 1
        assert trace.run.total_model_calls == 2
        assert trace.run.total_tool_calls == 1

    def test_attempt_count(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert len(trace.attempts) == 1
        assert trace.attempts[0].attempt_id == ATTEMPT_ID
        assert trace.attempts[0].outcome == "success"

    def test_loop_iterations(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert len(trace.loop_iterations) == 2
        iter1, iter2 = trace.loop_iterations
        assert iter1.iteration_number == 1
        assert iter1.has_tool_calls is True
        assert iter1.iteration_type == "reason_and_act"
        assert iter2.iteration_number == 2
        assert iter2.has_tool_calls is False
        assert iter2.iteration_type == "reason_only"

    def test_model_calls(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert len(trace.model_calls) == 2
        mc_ids = {mc.model_call_id for mc in trace.model_calls}
        assert MC1_ID in mc_ids
        assert MC2_ID in mc_ids

    def test_tool_calls(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert len(trace.tool_calls) == 1
        assert trace.tool_calls[0].tool_call_id == TC1_ID
        assert trace.tool_calls[0].tool_name == "file_write"

    def test_evaluations(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert len(trace.evaluations) == 1
        ev = trace.evaluations[0]
        assert ev.eval_id == EVAL_ID
        assert ev.passed is True
        assert ev.score == 1.0

    def test_no_warnings(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        assert trace.warnings == []

    def test_attempts_sorted(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        numbers = [a.attempt_number for a in trace.attempts]
        assert numbers == sorted(numbers)

    def test_iterations_sorted_within_attempt(self):
        trace = load_trace(FIXTURES / "minimal_success.jsonl")
        nums = [i.iteration_number for i in trace.loop_iterations]
        assert nums == sorted(nums)


# ---------------------------------------------------------------------------
# load_trace — error handling
# ---------------------------------------------------------------------------


class TestLoadTraceErrors:
    def _write_trace(self, lines: list[str]) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, dir=tempfile.gettempdir()
        )
        tmp.write("\n".join(lines) + "\n")
        tmp.close()
        return Path(tmp.name)

    def _minimal_lines(self) -> list[str]:
        """Return 8 valid JSONL lines matching minimal_success.jsonl."""
        return (FIXTURES / "minimal_success.jsonl").read_text().splitlines()

    def test_strict_bad_json_raises(self):
        path = self._write_trace(["not json"])
        with pytest.raises(TraceValidationError):
            load_trace(path, strict=True)

    def test_lenient_bad_json_continues(self):
        lines = self._minimal_lines()
        lines.insert(0, "not json")
        path = self._write_trace(lines)
        trace = load_trace(path, strict=False)
        assert isinstance(trace, RunTrace)

    def test_strict_no_run_event_raises(self):
        lines = [
            line for line in self._minimal_lines()
            if '"event_type":"run"' not in line
        ]
        path = self._write_trace(lines)
        with pytest.raises(TraceValidationError, match="RunEvent"):
            load_trace(path, strict=True)

    def test_strict_wrong_attempt_run_id_raises(self):
        run = make_run()
        other_run_id = UUID("99999999-9999-9999-9999-999999999999")
        attempt = make_attempt(run_id=other_run_id)
        iter1 = make_loop_iteration()
        mc1 = make_model_call()
        tc1 = make_tool_call()
        iter2 = make_loop_iteration(
            iteration_id=ITER2_ID,
            iteration_number=2,
            has_tool_calls=False,
            iteration_type="reason_only",
            start_ns=1_800_000_000,
            end_ns=2_500_000_000,
        )
        mc2 = make_model_call(
            model_call_id=MC2_ID,
            iteration_id=ITER2_ID,
            start_ns=1_800_000_000,
            end_ns=2_500_000_000,
            tools_called_in_response=[],
        )
        ev = make_evaluation()
        lines = [
            event_to_jsonl_bytes(e).decode()
            for e in [run, attempt, iter1, mc1, tc1, iter2, mc2, ev]
        ]
        path = self._write_trace(lines)
        with pytest.raises(TraceValidationError, match="run_id"):
            load_trace(path, strict=True)

    def test_strict_unknown_iteration_id_raises(self):
        run = make_run()
        attempt = make_attempt()
        iter1 = make_loop_iteration()
        bogus_iter_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        mc1 = make_model_call(iteration_id=bogus_iter_id)
        lines = [
            event_to_jsonl_bytes(e).decode()
            for e in [run, attempt, iter1, mc1]
        ]
        path = self._write_trace(lines)
        with pytest.raises(TraceValidationError, match="iteration_id"):
            load_trace(path, strict=True)

    def test_blank_lines_skipped(self):
        lines = self._minimal_lines()
        lines_with_blanks = []
        for line in lines:
            lines_with_blanks.append(line)
            lines_with_blanks.append("")
        path = self._write_trace(lines_with_blanks)
        trace = load_trace(path, strict=True)
        assert isinstance(trace, RunTrace)


# ---------------------------------------------------------------------------
# RunTrace — count mismatch produces warnings, not errors
# ---------------------------------------------------------------------------


class TestRunTraceWarnings:
    def test_count_mismatch_generates_warning(self):
        # Declare 3 model calls but only provide 2
        run = make_run(total_model_calls=3, total_tool_calls=1)
        attempt = make_attempt()
        iter1 = make_loop_iteration()
        mc1 = make_model_call()
        tc1 = make_tool_call()
        iter2 = make_loop_iteration(
            iteration_id=ITER2_ID,
            iteration_number=2,
            has_tool_calls=False,
            iteration_type="reason_only",
            start_ns=1_800_000_000,
            end_ns=2_500_000_000,
        )
        mc2 = make_model_call(
            model_call_id=MC2_ID,
            iteration_id=ITER2_ID,
            start_ns=1_800_000_000,
            end_ns=2_500_000_000,
            tools_called_in_response=[],
        )
        ev = make_evaluation()
        trace = RunTrace(
            run=run,
            attempts=[attempt],
            loop_iterations=[iter1, iter2],
            model_calls=[mc1, mc2],
            tool_calls=[tc1],
            evaluations=[ev],
        )
        warning_msgs = [w.message for w in trace.warnings]
        assert any("total_model_calls" in m for m in warning_msgs)

    def test_extra_tool_in_response_generates_warning(self):
        run = make_run()
        attempt = make_attempt()
        iter1 = make_loop_iteration()
        # tools_called_in_response contains a tool not in requested_tools
        mc1 = make_model_call(
            requested_tools=["bash"],
            tools_called_in_response=["bash", "unknown_tool"],
        )
        tc1 = make_tool_call()
        iter2 = make_loop_iteration(
            iteration_id=ITER2_ID,
            iteration_number=2,
            has_tool_calls=False,
            iteration_type="reason_only",
            start_ns=1_800_000_000,
            end_ns=2_500_000_000,
        )
        mc2 = make_model_call(
            model_call_id=MC2_ID,
            iteration_id=ITER2_ID,
            start_ns=1_800_000_000,
            end_ns=2_500_000_000,
            tools_called_in_response=[],
        )
        ev = make_evaluation()
        trace = RunTrace(
            run=run,
            attempts=[attempt],
            loop_iterations=[iter1, iter2],
            model_calls=[mc1, mc2],
            tool_calls=[tc1],
            evaluations=[ev],
        )
        assert any("tools_called_in_response" in w.message for w in trace.warnings)
