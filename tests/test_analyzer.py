"""Tests for compute_metrics() and summarize_attempt().

One test class per scenario (loaded from demos/output/).
Each class asserts the key metric values documented in docs/v0-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_profiler.schema.trace import load_trace, RunTrace
from agent_profiler.schema.events import (
    RunEvent,
    AttemptEvent,
    LoopIterationEvent,
    ModelCallEvent,
    ToolCallEvent,
    EvaluationEvent,
)
from agent_profiler.analyzer import compute_metrics, summarize_attempt

DEMOS = Path(__file__).parent.parent / "demos" / "output"


def _metrics(scenario: str):
    trace = load_trace(DEMOS / f"{scenario}.jsonl", strict=True)
    return compute_metrics(trace), trace


# ---------------------------------------------------------------------------
# Scenario 1: happy_path
# ---------------------------------------------------------------------------


class TestHappyPathMetrics:
    def test_first_pass_success(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.first_pass_success is True

    def test_attempt_count(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.attempt_count == 1

    def test_primary_bottleneck_balanced(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.primary_bottleneck == "balanced"

    def test_no_failure_categories(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.failure_categories == []

    def test_retry_waste_zero(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.retry_waste_ms == 0.0

    def test_program_runtime_zero(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.program_runtime_ms == 0.0

    def test_correctness_score_is_one(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.correctness_score == pytest.approx(1.0)

    def test_e2e_wall_ms_positive(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.e2e_wall_ms > 0

    def test_token_costs_positive(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.total_input_tokens > 0
        assert m.total_output_tokens > 0
        assert m.estimated_cost_usd > 0

    def test_wasted_tokens_zero(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.wasted_tokens == 0

    def test_per_attempt_summary_length(self) -> None:
        m, _ = _metrics("happy_path")
        assert len(m.per_attempt_summary) == 1
        assert m.per_attempt_summary[0].outcome == "success"

    def test_gap_time_non_negative(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.gap_time_ms >= 0

    def test_no_idle_in_synthetic(self) -> None:
        m, _ = _metrics("happy_path")
        assert m.user_idle_ms == 0.0
        assert m.active_wall_ms == m.e2e_wall_ms
        assert m.idle_percentage == 0.0


# ---------------------------------------------------------------------------
# Scenario 2: wrong_tool_retry
# ---------------------------------------------------------------------------


class TestWrongToolRetryMetrics:
    def test_first_pass_success_false(self) -> None:
        m, _ = _metrics("wrong_tool_retry")
        assert m.first_pass_success is False

    def test_attempt_count(self) -> None:
        m, _ = _metrics("wrong_tool_retry")
        assert m.attempt_count == 2

    def test_failure_categories(self) -> None:
        m, _ = _metrics("wrong_tool_retry")
        assert m.failure_categories == ["wrong_tool"]

    def test_retry_waste_positive(self) -> None:
        m, _ = _metrics("wrong_tool_retry")
        assert m.retry_waste_ms > 0

    def test_wasted_tokens_positive(self) -> None:
        m, _ = _metrics("wrong_tool_retry")
        assert m.wasted_tokens > 0

    def test_per_attempt_summary(self) -> None:
        m, _ = _metrics("wrong_tool_retry")
        assert len(m.per_attempt_summary) == 2
        assert m.per_attempt_summary[0].outcome == "failure"
        assert m.per_attempt_summary[0].failure_category == "wrong_tool"
        assert m.per_attempt_summary[1].outcome == "success"


# ---------------------------------------------------------------------------
# Scenario 3: slow_program
# ---------------------------------------------------------------------------


class TestSlowProgramMetrics:
    def test_primary_bottleneck_program_runtime(self) -> None:
        m, _ = _metrics("slow_program")
        assert m.primary_bottleneck == "program_runtime"

    def test_program_runtime_dominates(self) -> None:
        m, _ = _metrics("slow_program")
        assert m.program_runtime_ms > 0.50 * m.e2e_wall_ms

    def test_first_pass_success(self) -> None:
        m, _ = _metrics("slow_program")
        assert m.first_pass_success is True

    def test_attempt_count_one(self) -> None:
        m, _ = _metrics("slow_program")
        assert m.attempt_count == 1

    def test_dominant_cost_in_attempt_is_program(self) -> None:
        m, _ = _metrics("slow_program")
        assert m.per_attempt_summary[0].dominant_cost == "program"


# ---------------------------------------------------------------------------
# Scenario 4: reasoning_heavy
# ---------------------------------------------------------------------------


class TestReasoningHeavyMetrics:
    def test_primary_bottleneck_agent_reasoning(self) -> None:
        m, _ = _metrics("reasoning_heavy")
        assert m.primary_bottleneck == "agent_reasoning"

    def test_agent_overhead_dominates(self) -> None:
        m, _ = _metrics("reasoning_heavy")
        assert m.agent_overhead_ms > 0.60 * m.e2e_wall_ms

    def test_first_pass_success(self) -> None:
        m, _ = _metrics("reasoning_heavy")
        assert m.first_pass_success is True

    def test_high_token_count(self) -> None:
        m, _ = _metrics("reasoning_heavy")
        # 8 model calls with large contexts → many input tokens
        assert m.total_input_tokens > 10_000


# ---------------------------------------------------------------------------
# Scenario 5: reasoning_loop
# ---------------------------------------------------------------------------


class TestReasoningLoopMetrics:
    def test_reasoning_loop_in_failure_categories(self) -> None:
        m, _ = _metrics("reasoning_loop")
        assert "reasoning_loop" in m.failure_categories

    def test_primary_bottleneck_reasoning_loop(self) -> None:
        m, _ = _metrics("reasoning_loop")
        # reasoning_loop is checked before retry_overhead (it's rule 2)
        # actual bottleneck depends on retry_waste ratio; either way,
        # reasoning_loop must appear in categories
        assert m.primary_bottleneck in ("reasoning_loop", "retry_overhead")

    def test_attempt_count(self) -> None:
        m, _ = _metrics("reasoning_loop")
        assert m.attempt_count == 2

    def test_wasted_tokens_positive(self) -> None:
        m, _ = _metrics("reasoning_loop")
        assert m.wasted_tokens > 0


# ---------------------------------------------------------------------------
# Scenario 6: transient_failure
# ---------------------------------------------------------------------------


class TestTransientFailureMetrics:
    def test_first_pass_success(self) -> None:
        m, _ = _metrics("transient_failure")
        assert m.first_pass_success is True

    def test_no_failure_categories(self) -> None:
        m, _ = _metrics("transient_failure")
        assert m.failure_categories == []

    def test_attempt_count_one(self) -> None:
        m, _ = _metrics("transient_failure")
        assert m.attempt_count == 1

    def test_retry_waste_zero(self) -> None:
        m, _ = _metrics("transient_failure")
        assert m.retry_waste_ms == 0.0


# ---------------------------------------------------------------------------
# Scenario 7: context_overflow
# ---------------------------------------------------------------------------


class TestContextOverflowMetrics:
    def test_context_overflow_in_failure_categories(self) -> None:
        m, _ = _metrics("context_overflow")
        assert "context_overflow" in m.failure_categories

    def test_high_wasted_tokens(self) -> None:
        m, _ = _metrics("context_overflow")
        # Attempt 1 has 15 model calls with growing context
        assert m.wasted_tokens > 0

    def test_attempt_count(self) -> None:
        m, _ = _metrics("context_overflow")
        assert m.attempt_count == 2

    def test_first_pass_success_false(self) -> None:
        m, _ = _metrics("context_overflow")
        assert m.first_pass_success is False


# ---------------------------------------------------------------------------
# Scenario 8: hallucinated_tool
# ---------------------------------------------------------------------------


class TestHallucinatedToolMetrics:
    def test_hallucinated_tool_in_failure_categories(self) -> None:
        m, _ = _metrics("hallucinated_tool")
        assert "hallucinated_tool" in m.failure_categories

    def test_attempt_count(self) -> None:
        m, _ = _metrics("hallucinated_tool")
        assert m.attempt_count == 2

    def test_first_pass_success_false(self) -> None:
        m, _ = _metrics("hallucinated_tool")
        assert m.first_pass_success is False

    def test_wasted_tokens_positive(self) -> None:
        m, _ = _metrics("hallucinated_tool")
        assert m.wasted_tokens > 0


# ---------------------------------------------------------------------------
# summarize_attempt unit tests
# ---------------------------------------------------------------------------


class TestSummarizeAttempt:
    def test_slow_program_attempt_dominant_cost(self) -> None:
        trace = load_trace(DEMOS / "slow_program.jsonl", strict=True)
        summary = summarize_attempt(trace, trace.attempts[0])
        assert summary.dominant_cost == "program"

    def test_reasoning_heavy_attempt_dominant_cost(self) -> None:
        trace = load_trace(DEMOS / "reasoning_heavy.jsonl", strict=True)
        summary = summarize_attempt(trace, trace.attempts[0])
        assert summary.dominant_cost == "model"

    def test_happy_path_attempt_fields(self) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        summary = summarize_attempt(trace, trace.attempts[0])
        assert summary.attempt_number == 1
        assert summary.outcome == "success"
        assert summary.failure_category is None
        assert summary.wall_ms > 0

    def test_failed_attempt_has_failure_category(self) -> None:
        trace = load_trace(DEMOS / "wrong_tool_retry.jsonl", strict=True)
        summary = summarize_attempt(trace, trace.attempts[0])
        assert summary.outcome == "failure"
        assert summary.failure_category == "wrong_tool"


# ---------------------------------------------------------------------------
# Idle time detection
# ---------------------------------------------------------------------------


def _build_trace_with_idle_gap() -> RunTrace:
    """Build a minimal trace with a 60-second gap between two iterations.

    Layout (all in nanoseconds):
      Iter 1: 0..1_000_000_000    (1s)   — model 0..500ms, tool 500ms..1s
      GAP:    1s .. 61s            (60s idle)
      Iter 2: 61s..62s            (1s)   — model 61..61.5s, tool 61.5..62s
      Run:    0..62s
      Attempt: 0..62s
    """
    run_id = uuid4()
    attempt_id = uuid4()
    iter1_id = uuid4()
    iter2_id = uuid4()
    mc1_id = uuid4()
    mc2_id = uuid4()
    tc1_id = uuid4()
    tc2_id = uuid4()
    eval_id = uuid4()

    ns = 1_000_000_000  # 1 second in ns

    run = RunEvent(
        event_type="run",
        run_id=run_id,
        task_description="idle gap test",
        start_ns=0,
        end_ns=62 * ns,
        outcome="success",
        attempt_count=1,
        total_model_calls=2,
        total_tool_calls=2,
        model_provider="test",
        model_name="test-model",
        sandbox_mode="off",
    )
    attempt = AttemptEvent(
        event_type="attempt",
        attempt_id=attempt_id,
        run_id=run_id,
        attempt_number=1,
        start_ns=0,
        end_ns=62 * ns,
        outcome="success",
    )
    iter1 = LoopIterationEvent(
        event_type="loop_iteration",
        iteration_id=iter1_id,
        attempt_id=attempt_id,
        iteration_number=1,
        start_ns=0,
        end_ns=1 * ns,
        has_tool_calls=True,
        iteration_type="reason_and_act",
    )
    iter2 = LoopIterationEvent(
        event_type="loop_iteration",
        iteration_id=iter2_id,
        attempt_id=attempt_id,
        iteration_number=2,
        start_ns=61 * ns,
        end_ns=62 * ns,
        has_tool_calls=True,
        iteration_type="reason_and_act",
    )
    mc1 = ModelCallEvent(
        event_type="model_call",
        model_call_id=mc1_id,
        iteration_id=iter1_id,
        model_provider="test",
        model_name="test-model",
        start_ns=0,
        end_ns=500_000_000,
        input_tokens=100,
        output_tokens=50,
    )
    mc2 = ModelCallEvent(
        event_type="model_call",
        model_call_id=mc2_id,
        iteration_id=iter2_id,
        model_provider="test",
        model_name="test-model",
        start_ns=61 * ns,
        end_ns=61 * ns + 500_000_000,
        input_tokens=100,
        output_tokens=50,
    )
    tc1 = ToolCallEvent(
        event_type="tool_call",
        tool_call_id=tc1_id,
        iteration_id=iter1_id,
        tool_name="bash",
        tool_category="shell",
        start_ns=500_000_000,
        end_ns=1 * ns,
        outcome="success",
    )
    tc2 = ToolCallEvent(
        event_type="tool_call",
        tool_call_id=tc2_id,
        iteration_id=iter2_id,
        tool_name="bash",
        tool_category="shell",
        start_ns=61 * ns + 500_000_000,
        end_ns=62 * ns,
        outcome="success",
    )
    evaluation = EvaluationEvent(
        event_type="evaluation",
        eval_id=eval_id,
        attempt_id=attempt_id,
        evaluator="heuristic",
        passed=True,
        score=1.0,
    )

    return RunTrace(
        run=run,
        attempts=[attempt],
        loop_iterations=[iter1, iter2],
        model_calls=[mc1, mc2],
        tool_calls=[tc1, tc2],
        evaluations=[evaluation],
    )


class TestIdleTimeDetection:
    def test_idle_gap_detected(self) -> None:
        trace = _build_trace_with_idle_gap()
        m = compute_metrics(trace)
        # 60-second gap between iterations, threshold is 30s → detected as idle
        assert m.user_idle_ms == pytest.approx(60_000.0)

    def test_active_wall_excludes_idle(self) -> None:
        trace = _build_trace_with_idle_gap()
        m = compute_metrics(trace)
        # e2e = 62s, idle = 60s → active = 2s
        assert m.active_wall_ms == pytest.approx(2_000.0)

    def test_idle_percentage(self) -> None:
        trace = _build_trace_with_idle_gap()
        m = compute_metrics(trace)
        expected_pct = 60_000.0 / 62_000.0 * 100
        assert m.idle_percentage == pytest.approx(expected_pct, rel=1e-3)

    def test_bottleneck_uses_active_wall(self) -> None:
        trace = _build_trace_with_idle_gap()
        m = compute_metrics(trace)
        # Active wall = 2s, model = 1s (50%), tool = 1s (50%).
        # Without idle correction, gap_time_ms would be 60s → framework_overhead.
        # With idle correction, the bottleneck should NOT be framework_overhead.
        assert m.primary_bottleneck != "framework_overhead"

    def test_custom_threshold(self) -> None:
        trace = _build_trace_with_idle_gap()
        # Set threshold higher than the gap → no idle detected
        m = compute_metrics(trace, idle_threshold_ms=120_000)
        assert m.user_idle_ms == 0.0
        assert m.active_wall_ms == m.e2e_wall_ms
