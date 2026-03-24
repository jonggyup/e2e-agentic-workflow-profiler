"""Multi-run comparison for agent profiling metrics.

compare_runs(baseline, current) → RunComparison

Answers the question: "Did the change make things better or worse?"
"""

from __future__ import annotations

from pydantic import BaseModel

from .metrics import ProfileMetrics


class RunComparison(BaseModel):
    baseline_label: str
    current_label: str

    # Timing deltas (ms) — negative = faster
    e2e_delta_ms: float
    e2e_delta_percent: float
    active_wall_delta_ms: float
    active_wall_delta_percent: float
    model_time_delta_ms: float
    tool_time_delta_ms: float
    retry_waste_delta_ms: float

    # Cost deltas
    token_delta: int
    cost_delta_usd: float

    # Correctness deltas
    attempt_count_delta: int
    first_pass_improved: bool | None  # None if same, True if baseline failed & current succeeded

    # Bottleneck
    bottleneck_changed: bool
    baseline_bottleneck: str
    current_bottleneck: str

    # Human-readable summary
    summary: str


def _delta_pct(current: float, baseline: float) -> float:
    """Compute percentage change: (current - baseline) / baseline * 100."""
    if baseline == 0:
        return 0.0 if current == 0 else 100.0
    return (current - baseline) / abs(baseline) * 100


def _fmt_delta_ms(delta_ms: float) -> str:
    """Format a millisecond delta for the summary sentence."""
    abs_ms = abs(delta_ms)
    if abs_ms >= 1000:
        return f"{abs_ms / 1000:.1f}s"
    return f"{abs_ms:,.0f}ms"


def _generate_summary(
    *,
    baseline_label: str,
    current_label: str,
    active_wall_delta_ms: float,
    active_wall_delta_pct: float,
    baseline_active: float,
    current_active: float,
    bottleneck_changed: bool,
    baseline_bottleneck: str,
    current_bottleneck: str,
    attempt_count_delta: int,
    baseline_attempts: int,
    current_attempts: int,
    token_delta: int,
    baseline_tokens: int,
    cost_delta_usd: float,
    first_pass_improved: bool | None,
) -> str:
    """Generate a 1-2 sentence plain English summary of the comparison."""
    parts: list[str] = []

    # Speed assessment
    if abs(active_wall_delta_pct) < 2:
        parts.append("No significant performance change")
    elif active_wall_delta_ms < 0:
        parts.append(
            f"{current_label} run is {abs(active_wall_delta_pct):.0f}% faster "
            f"({_fmt_delta_ms(current_active)} vs {_fmt_delta_ms(baseline_active)} active time)"
        )
    else:
        parts.append(
            f"{current_label} run is {abs(active_wall_delta_pct):.0f}% slower"
        )

    # Bottleneck shift
    if bottleneck_changed:
        parts.append(
            f"Bottleneck shifted from {baseline_bottleneck} to {current_bottleneck}"
        )

    # Attempt count change
    if attempt_count_delta > 0:
        parts.append(
            f"Agent now needs {current_attempts} attempts instead of {baseline_attempts}"
        )
    elif attempt_count_delta < 0:
        parts.append(
            f"Agent now needs {current_attempts} attempt(s) instead of {baseline_attempts}"
        )

    # First-pass improvement
    if first_pass_improved is True:
        parts.append("First-pass correctness improved")
    elif first_pass_improved is False:
        parts.append("First-pass correctness regressed")

    # Token change
    if baseline_tokens > 0 and abs(token_delta) > 0:
        token_pct = abs(token_delta) / baseline_tokens * 100
        if token_pct >= 5:
            direction = "decreased" if token_delta < 0 else "increased"
            parts.append(f"Token usage {direction} by {token_pct:.0f}%")

    # Cost change
    if abs(cost_delta_usd) >= 0.005:
        sign = "-" if cost_delta_usd < 0 else "+"
        parts.append(f"Cost {sign}${abs(cost_delta_usd):.2f}")

    # If nothing changed at all
    if len(parts) == 1 and parts[0] == "No significant performance change":
        # Check if anything else is different
        if (
            not bottleneck_changed
            and attempt_count_delta == 0
            and first_pass_improved is None
            and abs(token_delta) == 0
        ):
            return "No significant change between runs."

    return ". ".join(parts) + "."


def compare_runs(
    baseline: ProfileMetrics,
    current: ProfileMetrics,
    *,
    baseline_label: str = "baseline",
    current_label: str = "current",
) -> RunComparison:
    """Compare two ProfileMetrics and produce a RunComparison.

    Parameters
    ----------
    baseline:
        The reference run metrics (before the change).
    current:
        The new run metrics (after the change).
    baseline_label / current_label:
        Human-readable labels for the summary text.
    """
    e2e_delta_ms = current.e2e_wall_ms - baseline.e2e_wall_ms
    active_wall_delta_ms = current.active_wall_ms - baseline.active_wall_ms
    model_time_delta_ms = current.total_model_time_ms - baseline.total_model_time_ms
    tool_time_delta_ms = current.total_tool_time_ms - baseline.total_tool_time_ms
    retry_waste_delta_ms = current.retry_waste_ms - baseline.retry_waste_ms

    baseline_total_tokens = baseline.total_input_tokens + baseline.total_output_tokens
    current_total_tokens = current.total_input_tokens + current.total_output_tokens
    token_delta = current_total_tokens - baseline_total_tokens
    cost_delta_usd = current.estimated_cost_usd - baseline.estimated_cost_usd

    attempt_count_delta = current.attempt_count - baseline.attempt_count

    # first_pass_improved: None if same, True if baseline failed & current succeeded
    if baseline.first_pass_success == current.first_pass_success:
        first_pass_improved = None
    else:
        first_pass_improved = current.first_pass_success

    bottleneck_changed = baseline.primary_bottleneck != current.primary_bottleneck

    summary = _generate_summary(
        baseline_label=baseline_label,
        current_label=current_label,
        active_wall_delta_ms=active_wall_delta_ms,
        active_wall_delta_pct=_delta_pct(current.active_wall_ms, baseline.active_wall_ms),
        baseline_active=baseline.active_wall_ms,
        current_active=current.active_wall_ms,
        bottleneck_changed=bottleneck_changed,
        baseline_bottleneck=baseline.primary_bottleneck,
        current_bottleneck=current.primary_bottleneck,
        attempt_count_delta=attempt_count_delta,
        baseline_attempts=baseline.attempt_count,
        current_attempts=current.attempt_count,
        token_delta=token_delta,
        baseline_tokens=baseline_total_tokens,
        cost_delta_usd=cost_delta_usd,
        first_pass_improved=first_pass_improved,
    )

    return RunComparison(
        baseline_label=baseline_label,
        current_label=current_label,
        e2e_delta_ms=e2e_delta_ms,
        e2e_delta_percent=_delta_pct(current.e2e_wall_ms, baseline.e2e_wall_ms),
        active_wall_delta_ms=active_wall_delta_ms,
        active_wall_delta_percent=_delta_pct(current.active_wall_ms, baseline.active_wall_ms),
        model_time_delta_ms=model_time_delta_ms,
        tool_time_delta_ms=tool_time_delta_ms,
        retry_waste_delta_ms=retry_waste_delta_ms,
        token_delta=token_delta,
        cost_delta_usd=cost_delta_usd,
        attempt_count_delta=attempt_count_delta,
        first_pass_improved=first_pass_improved,
        bottleneck_changed=bottleneck_changed,
        baseline_bottleneck=baseline.primary_bottleneck,
        current_bottleneck=current.primary_bottleneck,
        summary=summary,
    )
