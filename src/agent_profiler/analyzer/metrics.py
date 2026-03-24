"""Core metrics computation for a completed agent run.

compute_metrics(trace) → ProfileMetrics

All timing metrics are in milliseconds.  Token counts are raw integers.
Cost uses configurable per-token pricing (defaults: $3/M input, $15/M output).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from agent_profiler.schema.trace import (
    RunTrace,
    evaluation_for_attempt,
    iterations_for_attempt,
    model_call_for_iteration,
)
from .per_attempt import AttemptSummary, summarize_attempt
from .resource_analyzer import ResourceProfile, analyze_resources

# Default pricing: Anthropic Claude Sonnet approximate rates
_DEFAULT_INPUT_COST_PER_M = 3.0   # USD per million input tokens
_DEFAULT_OUTPUT_COST_PER_M = 15.0  # USD per million output tokens

# Idle detection: gaps between consecutive iterations above this threshold
# are classified as user idle time.
IDLE_THRESHOLD_MS: float = 30_000  # 30 seconds


class ProfileMetrics(BaseModel):
    # --- Timing (ms) ---
    e2e_wall_ms: float
    total_model_time_ms: float
    total_tool_time_ms: float
    agent_overhead_ms: float      # == total_model_time_ms
    tool_execution_ms: float      # == total_tool_time_ms
    program_runtime_ms: float
    retry_waste_ms: float
    gap_time_ms: float

    # --- Idle / active (ms) ---
    user_idle_ms: float
    active_wall_ms: float         # e2e_wall_ms - user_idle_ms
    idle_percentage: float        # user_idle_ms / e2e_wall_ms * 100

    # --- Correctness ---
    first_pass_success: bool
    attempt_count: int
    failure_categories: list[str]
    correctness_score: float | None

    # --- Cost ---
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    wasted_tokens: int

    # --- Token tracking (Phase 4) ---
    tokens_per_attempt: list[dict] | None = None
    tokens_per_iteration: list[dict] | None = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    context_efficiency: float | None = None  # output / input ratio

    # --- Derived ---
    primary_bottleneck: str  # see _primary_bottleneck() for ordered rules

    # --- Per-attempt ---
    per_attempt_summary: list[AttemptSummary]

    # --- Resource profile (Phase 2-3) ---
    resource_profile: ResourceProfile | None = None


def compute_metrics(
    trace: RunTrace,
    *,
    program_tool_name: str | None = None,
    input_cost_per_m: float = _DEFAULT_INPUT_COST_PER_M,
    output_cost_per_m: float = _DEFAULT_OUTPUT_COST_PER_M,
    idle_threshold_ms: float = IDLE_THRESHOLD_MS,
    system_samples_path: Path | str | None = None,
) -> ProfileMetrics:
    """Compute all profiling metrics for a completed RunTrace.

    Parameters
    ----------
    trace:
        A validated RunTrace (from load_trace).
    program_tool_name:
        If provided, any ToolCallEvent with this tool_name is treated as
        is_program_under_test=True in addition to the flag on the event itself.
    input_cost_per_m / output_cost_per_m:
        Token pricing in USD per million tokens.
    system_samples_path:
        Optional path to a system-samples JSONL file from SystemSampler.
        If provided, resource analysis is performed and attached.
    """
    run = trace.run

    # --- e2e wall time ---
    e2e_wall_ms = (run.end_ns - run.start_ns) / 1_000_000

    # --- Aggregate timing across all model/tool calls ---
    total_model_time_ms = sum(
        (mc.end_ns - mc.start_ns) / 1_000_000 for mc in trace.model_calls
    )
    total_tool_time_ms = sum(
        (tc.end_ns - tc.start_ns) / 1_000_000 for tc in trace.tool_calls
    )
    program_runtime_ms = sum(
        (tc.end_ns - tc.start_ns) / 1_000_000
        for tc in trace.tool_calls
        if tc.is_program_under_test
        or (program_tool_name is not None and tc.tool_name == program_tool_name)
    )

    agent_overhead_ms = total_model_time_ms
    tool_execution_ms = total_tool_time_ms
    gap_time_ms = e2e_wall_ms - total_model_time_ms - total_tool_time_ms

    # --- Idle detection ---
    user_idle_ms = _detect_idle_ms(trace, idle_threshold_ms=idle_threshold_ms)
    active_wall_ms = e2e_wall_ms - user_idle_ms
    idle_percentage = (user_idle_ms / e2e_wall_ms * 100) if e2e_wall_ms > 0 else 0.0

    # --- Retry waste: wall time of all failed attempts ---
    failed_attempts = [a for a in trace.attempts if a.outcome == "failure"]
    retry_waste_ms = sum(
        (a.end_ns - a.start_ns) / 1_000_000 for a in failed_attempts
    )

    # --- Correctness ---
    first_pass_success = trace.attempts[0].outcome == "success"
    attempt_count = len(trace.attempts)
    failure_categories: list[str] = [
        a.failure_category
        for a in trace.attempts
        if a.outcome == "failure" and a.failure_category is not None
    ]

    # correctness_score: from evaluation of first successful attempt, or last
    correctness_score = _best_correctness_score(trace)

    # --- Cost: tokens ---
    total_input_tokens = sum(
        mc.input_tokens for mc in trace.model_calls if mc.input_tokens is not None
    )
    total_output_tokens = sum(
        mc.output_tokens for mc in trace.model_calls if mc.output_tokens is not None
    )
    estimated_cost_usd = (
        (total_input_tokens / 1_000_000) * input_cost_per_m
        + (total_output_tokens / 1_000_000) * output_cost_per_m
    )

    # wasted_tokens: tokens consumed in failed attempts
    failed_attempt_ids = {a.attempt_id for a in failed_attempts}
    failed_iter_ids = {
        it.iteration_id
        for it in trace.loop_iterations
        if it.attempt_id in failed_attempt_ids
    }
    wasted_tokens = sum(
        (mc.input_tokens or 0) + (mc.output_tokens or 0)
        for mc in trace.model_calls
        if mc.iteration_id in failed_iter_ids
    )

    # --- Derived: primary bottleneck (uses active wall time) ---
    primary_bottleneck = _primary_bottleneck(
        active_wall_ms=active_wall_ms,
        retry_waste_ms=retry_waste_ms,
        failure_categories=failure_categories,
        agent_overhead_ms=agent_overhead_ms,
        program_runtime_ms=program_runtime_ms,
        tool_execution_ms=tool_execution_ms,
        gap_time_ms=gap_time_ms,
        user_idle_ms=user_idle_ms,
    )

    # --- Per-attempt summaries ---
    per_attempt_summary = [
        summarize_attempt(trace, attempt) for attempt in trace.attempts
    ]

    # --- Resource profile (optional) ---
    resource_profile: ResourceProfile | None = None
    if system_samples_path is not None:
        samples_path = Path(system_samples_path)
        if samples_path.exists():
            resource_profile = analyze_resources(samples_path, trace)

    # --- Token tracking (Phase 4) ---
    tokens_per_attempt = _compute_tokens_per_attempt(trace, input_cost_per_m, output_cost_per_m)
    tokens_per_iteration = _compute_tokens_per_iteration(trace)
    cache_read_tokens, cache_write_tokens = _compute_cache_tokens(trace)
    context_efficiency = (
        total_output_tokens / total_input_tokens
        if total_input_tokens > 0
        else None
    )

    return ProfileMetrics(
        e2e_wall_ms=e2e_wall_ms,
        total_model_time_ms=total_model_time_ms,
        total_tool_time_ms=total_tool_time_ms,
        agent_overhead_ms=agent_overhead_ms,
        tool_execution_ms=tool_execution_ms,
        program_runtime_ms=program_runtime_ms,
        retry_waste_ms=retry_waste_ms,
        gap_time_ms=gap_time_ms,
        user_idle_ms=user_idle_ms,
        active_wall_ms=active_wall_ms,
        idle_percentage=idle_percentage,
        first_pass_success=first_pass_success,
        attempt_count=attempt_count,
        failure_categories=failure_categories,
        correctness_score=correctness_score,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        estimated_cost_usd=estimated_cost_usd,
        wasted_tokens=wasted_tokens,
        tokens_per_attempt=tokens_per_attempt,
        tokens_per_iteration=tokens_per_iteration,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        context_efficiency=context_efficiency,
        primary_bottleneck=primary_bottleneck,
        per_attempt_summary=per_attempt_summary,
        resource_profile=resource_profile,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_correctness_score(trace: RunTrace) -> float | None:
    """Return evaluation.score from the first successful attempt, else last attempt."""
    for attempt in trace.attempts:
        if attempt.outcome == "success":
            ev = evaluation_for_attempt(trace, attempt.attempt_id)
            if ev is not None:
                return ev.score
    # Fallback: last attempt's evaluation
    if trace.attempts:
        ev = evaluation_for_attempt(trace, trace.attempts[-1].attempt_id)
        if ev is not None:
            return ev.score
    return None


def _detect_idle_ms(
    trace: RunTrace,
    *,
    idle_threshold_ms: float = IDLE_THRESHOLD_MS,
) -> float:
    """Sum gaps between consecutive loop iterations that exceed the idle threshold."""
    iters = sorted(trace.loop_iterations, key=lambda i: i.start_ns)
    if len(iters) < 2:
        return 0.0

    threshold_ns = idle_threshold_ms * 1_000_000
    total_idle_ns = 0
    for i in range(len(iters) - 1):
        gap_ns = iters[i + 1].start_ns - iters[i].end_ns
        if gap_ns > threshold_ns:
            total_idle_ns += gap_ns

    return total_idle_ns / 1_000_000


def _primary_bottleneck(
    *,
    active_wall_ms: float,
    retry_waste_ms: float,
    failure_categories: list[str],
    agent_overhead_ms: float,
    program_runtime_ms: float,
    tool_execution_ms: float,
    gap_time_ms: float,
    user_idle_ms: float,
) -> str:
    """Determine the primary bottleneck using the ordered rules from v0-design.md.

    All percentage thresholds use active_wall_ms (e2e minus user idle) so
    that long idle sessions are analyzed on their active portion only.

    Rules (first match wins):
    1. retry_waste > 50% of active_wall      → "retry_overhead"
    2. any "reasoning_loop" in failures       → "reasoning_loop"
    3. agent_overhead > 60% of active_wall    → "agent_reasoning"
    4. program_runtime > 50% of active_wall   → "program_runtime"
    5. tool_execution > 50% of active_wall    → "tool_slowness"
    6. active_gap > 30% of active_wall        → "framework_overhead"
    7. else                                   → "balanced"
    """
    if active_wall_ms <= 0:
        return "balanced"

    if retry_waste_ms > 0.50 * active_wall_ms:
        return "retry_overhead"

    if "reasoning_loop" in failure_categories:
        return "reasoning_loop"

    if agent_overhead_ms > 0.60 * active_wall_ms:
        return "agent_reasoning"

    if program_runtime_ms > 0.50 * active_wall_ms:
        return "program_runtime"

    if tool_execution_ms > 0.50 * active_wall_ms:
        return "tool_slowness"

    # Subtract idle time from gap so user idle doesn't trigger framework_overhead
    active_gap_ms = gap_time_ms - user_idle_ms
    if active_gap_ms > 0.30 * active_wall_ms:
        return "framework_overhead"

    return "balanced"


# ---------------------------------------------------------------------------
# Token tracking helpers (Phase 4)
# ---------------------------------------------------------------------------


def _compute_tokens_per_attempt(
    trace: RunTrace,
    input_cost_per_m: float,
    output_cost_per_m: float,
) -> list[dict]:
    """Compute token usage per attempt with cost breakdown."""
    result: list[dict] = []
    for attempt in trace.attempts:
        iters = iterations_for_attempt(trace, attempt.attempt_id)
        iter_ids = {it.iteration_id for it in iters}
        attempt_mcs = [mc for mc in trace.model_calls if mc.iteration_id in iter_ids]

        input_t = sum(mc.input_tokens or 0 for mc in attempt_mcs)
        output_t = sum(mc.output_tokens or 0 for mc in attempt_mcs)
        cost = (
            (input_t / 1_000_000) * input_cost_per_m
            + (output_t / 1_000_000) * output_cost_per_m
        )
        result.append({
            "attempt_number": attempt.attempt_number,
            "input_tokens": input_t,
            "output_tokens": output_t,
            "cost_usd": round(cost, 6),
        })
    return result


def _compute_tokens_per_iteration(trace: RunTrace) -> list[dict]:
    """Compute cumulative token growth over iterations."""
    iters = sorted(trace.loop_iterations, key=lambda i: i.start_ns)
    cumulative_input = 0
    cumulative_output = 0
    result: list[dict] = []
    for it in iters:
        mc = model_call_for_iteration(trace, it.iteration_id)
        if mc is not None:
            cumulative_input += mc.input_tokens or 0
            cumulative_output += mc.output_tokens or 0
        result.append({
            "iteration_number": it.iteration_number,
            "cumulative_input_tokens": cumulative_input,
            "cumulative_output_tokens": cumulative_output,
        })
    return result


def _compute_cache_tokens(trace: RunTrace) -> tuple[int, int]:
    """Extract cache_read and cache_write tokens from model calls.

    These fields are set by the OpenClaw converter when usage data
    contains cacheRead/cacheWrite.
    """
    cache_read = sum(
        getattr(mc, "cache_read_tokens", 0) or 0 for mc in trace.model_calls
    )
    cache_write = sum(
        getattr(mc, "cache_write_tokens", 0) or 0 for mc in trace.model_calls
    )
    return cache_read, cache_write
