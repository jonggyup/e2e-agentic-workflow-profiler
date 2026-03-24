"""Per-attempt summary computation."""

from __future__ import annotations


from pydantic import BaseModel

from agent_profiler.schema.trace import (
    RunTrace,
    iterations_for_attempt,
    model_call_for_iteration,
    tool_calls_for_iteration,
)
from agent_profiler.schema.events import AttemptEvent


class AttemptSummary(BaseModel):
    attempt_number: int
    outcome: str
    wall_ms: float
    failure_category: str | None
    dominant_cost: str  # "model" | "tool" | "program" | "balanced"


def summarize_attempt(trace: RunTrace, attempt: AttemptEvent) -> AttemptSummary:
    """Compute a summary for a single attempt within a trace."""
    wall_ms = (attempt.end_ns - attempt.start_ns) / 1_000_000

    iterations = iterations_for_attempt(trace, attempt.attempt_id)

    model_time_ms = 0.0
    tool_time_ms = 0.0
    program_time_ms = 0.0

    for it in iterations:
        mc = model_call_for_iteration(trace, it.iteration_id)
        if mc is not None:
            model_time_ms += (mc.end_ns - mc.start_ns) / 1_000_000

        for tc in tool_calls_for_iteration(trace, it.iteration_id):
            dur = (tc.end_ns - tc.start_ns) / 1_000_000
            tool_time_ms += dur
            if tc.is_program_under_test:
                program_time_ms += dur

    dominant_cost = _dominant_cost(wall_ms, model_time_ms, tool_time_ms, program_time_ms)

    return AttemptSummary(
        attempt_number=attempt.attempt_number,
        outcome=attempt.outcome,
        wall_ms=wall_ms,
        failure_category=attempt.failure_category,
        dominant_cost=dominant_cost,
    )


def _dominant_cost(
    wall_ms: float,
    model_time_ms: float,
    tool_time_ms: float,
    program_time_ms: float,
) -> str:
    if wall_ms <= 0:
        return "balanced"
    if program_time_ms > 0.50 * wall_ms:
        return "program"
    if model_time_ms > 0.60 * wall_ms:
        return "model"
    if tool_time_ms > 0.50 * wall_ms:
        return "tool"
    return "balanced"
