"""RunTrace container and load_trace() loader.

RunTrace holds all events for one completed agent run in flat lists.
Cross-event invariants are enforced by a model_validator; invariant
violations that are hard errors raise TraceValidationError; soft
mismatches (e.g. run.attempt_count != len(attempts)) become TraceWarning
entries attached to the trace.

load_trace(path, strict=True) reads a .jsonl file and returns a validated
RunTrace.  In strict mode (default) any parse or validation failure raises
TraceValidationError immediately.  In lenient mode the loader accumulates
errors and returns the best-effort trace; callers should inspect
trace.warnings for soft issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from .events import (
    _event_adapter,
    AttemptEvent,
    EvaluationEvent,
    LoopIterationEvent,
    ModelCallEvent,
    RunEvent,
    ToolCallEvent,
)

# ---------------------------------------------------------------------------
# Error / warning types
# ---------------------------------------------------------------------------


@dataclass
class TraceError:
    line_number: int
    raw_line: str
    message: str

    def __str__(self) -> str:
        loc = f"line {self.line_number}" if self.line_number else "post-parse"
        return f"{loc}: {self.message}"


class TraceWarning(BaseModel):
    message: str
    context: str = ""


class TraceValidationError(Exception):
    def __init__(self, errors: list[TraceError]) -> None:
        self.errors = errors
        detail = "\n".join(f"  {e}" for e in errors)
        super().__init__(
            f"Trace validation failed with {len(errors)} error(s):\n{detail}"
        )


# ---------------------------------------------------------------------------
# RunTrace
# ---------------------------------------------------------------------------


class RunTrace(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    run: RunEvent
    attempts: list[AttemptEvent]
    loop_iterations: list[LoopIterationEvent]
    model_calls: list[ModelCallEvent]
    tool_calls: list[ToolCallEvent]
    evaluations: list[EvaluationEvent]
    warnings: list[TraceWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_event_invariants(self) -> Self:
        collected: list[TraceWarning] = []
        run_id: UUID = self.run.run_id

        # Hard: all attempt.run_id must match
        for a in self.attempts:
            if a.run_id != run_id:
                raise ValueError(
                    f"Attempt {a.attempt_id} has run_id={a.run_id}, "
                    f"expected {run_id}"
                )

        attempt_ids = {a.attempt_id for a in self.attempts}
        iteration_ids = {i.iteration_id for i in self.loop_iterations}

        # Hard: all iterations must reference a valid attempt
        for it in self.loop_iterations:
            if it.attempt_id not in attempt_ids:
                raise ValueError(
                    f"LoopIteration {it.iteration_id} references "
                    f"unknown attempt_id={it.attempt_id}"
                )

        # Hard: all model_calls must reference a valid iteration
        for mc in self.model_calls:
            if mc.iteration_id not in iteration_ids:
                raise ValueError(
                    f"ModelCall {mc.model_call_id} references "
                    f"unknown iteration_id={mc.iteration_id}"
                )

        # Hard: all tool_calls must reference a valid iteration
        for tc in self.tool_calls:
            if tc.iteration_id not in iteration_ids:
                raise ValueError(
                    f"ToolCall {tc.tool_call_id} references "
                    f"unknown iteration_id={tc.iteration_id}"
                )

        # Hard: all evaluations must reference a valid attempt
        for ev in self.evaluations:
            if ev.attempt_id not in attempt_ids:
                raise ValueError(
                    f"Evaluation {ev.eval_id} references "
                    f"unknown attempt_id={ev.attempt_id}"
                )

        # Soft: run summary counts match actual counts
        if self.run.attempt_count != len(self.attempts):
            collected.append(TraceWarning(
                message=(
                    f"run.attempt_count={self.run.attempt_count} "
                    f"but found {len(self.attempts)} attempts"
                ),
                context=f"run_id={run_id}",
            ))

        if self.run.total_model_calls != len(self.model_calls):
            collected.append(TraceWarning(
                message=(
                    f"run.total_model_calls={self.run.total_model_calls} "
                    f"but found {len(self.model_calls)} model_calls"
                ),
                context=f"run_id={run_id}",
            ))

        if self.run.total_tool_calls != len(self.tool_calls):
            collected.append(TraceWarning(
                message=(
                    f"run.total_tool_calls={self.run.total_tool_calls} "
                    f"but found {len(self.tool_calls)} tool_calls"
                ),
                context=f"run_id={run_id}",
            ))

        # Soft: each attempt has at most one evaluation
        eval_counts: dict[UUID, int] = {}
        for ev in self.evaluations:
            eval_counts[ev.attempt_id] = eval_counts.get(ev.attempt_id, 0) + 1
        for aid, count in eval_counts.items():
            if count > 1:
                collected.append(TraceWarning(
                    message=f"attempt has {count} evaluations (expected ≤1)",
                    context=f"attempt_id={aid}",
                ))

        # Soft: has_tool_calls=True but no ToolCall found for that iteration
        tool_iteration_ids = {tc.iteration_id for tc in self.tool_calls}
        for it in self.loop_iterations:
            if it.has_tool_calls and it.iteration_id not in tool_iteration_ids:
                collected.append(TraceWarning(
                    message="has_tool_calls=True but no ToolCall events found",
                    context=f"iteration_id={it.iteration_id}",
                ))

        # Soft: tools_called_in_response ⊆ requested_tools
        for mc in self.model_calls:
            extra = set(mc.tools_called_in_response) - set(mc.requested_tools)
            if extra:
                collected.append(TraceWarning(
                    message=(
                        f"tools_called_in_response contains tools not in "
                        f"requested_tools: {sorted(extra)}"
                    ),
                    context=f"model_call_id={mc.model_call_id}",
                ))

        self.warnings = collected
        return self


# ---------------------------------------------------------------------------
# Index helpers (free functions — no persistent state)
# ---------------------------------------------------------------------------


def iterations_for_attempt(
    trace: RunTrace, attempt_id: UUID
) -> list[LoopIterationEvent]:
    return [i for i in trace.loop_iterations if i.attempt_id == attempt_id]


def model_call_for_iteration(
    trace: RunTrace, iteration_id: UUID
) -> ModelCallEvent | None:
    for mc in trace.model_calls:
        if mc.iteration_id == iteration_id:
            return mc
    return None


def tool_calls_for_iteration(
    trace: RunTrace, iteration_id: UUID
) -> list[ToolCallEvent]:
    return [tc for tc in trace.tool_calls if tc.iteration_id == iteration_id]


def evaluation_for_attempt(
    trace: RunTrace, attempt_id: UUID
) -> EvaluationEvent | None:
    for ev in trace.evaluations:
        if ev.attempt_id == attempt_id:
            return ev
    return None


# ---------------------------------------------------------------------------
# load_trace
# ---------------------------------------------------------------------------


def load_trace(path: "Path | str", *, strict: bool = True) -> RunTrace:
    """Read a .jsonl profiler trace and return a validated RunTrace.

    Parameters
    ----------
    path:
        Path to the .jsonl file.
    strict:
        When True (default), any parse or validation failure raises
        TraceValidationError immediately.  When False, errors are collected
        and the loader proceeds; a TraceValidationError is raised only if the
        trace cannot be built at all (no RunEvent, or no AttemptEvent).

    Returns
    -------
    RunTrace
        Validated trace.  Check ``trace.warnings`` for soft issues.
    """
    path = Path(path)

    runs: list[RunEvent] = []
    attempts: list[AttemptEvent] = []
    loop_iterations: list[LoopIterationEvent] = []
    model_calls: list[ModelCallEvent] = []
    tool_calls: list[ToolCallEvent] = []
    evaluations: list[EvaluationEvent] = []
    errors: list[TraceError] = []

    with path.open("rb") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                data = orjson.loads(stripped)
                event = _event_adapter.validate_python(data)
            except Exception as exc:
                err = TraceError(
                    line_number=line_number,
                    raw_line=stripped.decode("utf-8", errors="replace"),
                    message=str(exc),
                )
                if strict:
                    raise TraceValidationError([err]) from exc
                errors.append(err)
                continue

            if isinstance(event, RunEvent):
                runs.append(event)
            elif isinstance(event, AttemptEvent):
                attempts.append(event)
            elif isinstance(event, LoopIterationEvent):
                loop_iterations.append(event)
            elif isinstance(event, ModelCallEvent):
                model_calls.append(event)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(event)
            elif isinstance(event, EvaluationEvent):
                evaluations.append(event)

    # Must have exactly one RunEvent
    if len(runs) != 1:
        err = TraceError(
            line_number=0,
            raw_line="",
            message=f"Expected exactly 1 RunEvent, found {len(runs)}",
        )
        errors.append(err)
        if strict or not runs:
            raise TraceValidationError(errors)

    # Must have at least one AttemptEvent
    if not attempts:
        err = TraceError(
            line_number=0,
            raw_line="",
            message="Expected at least 1 AttemptEvent, found 0",
        )
        errors.append(err)
        if strict:
            raise TraceValidationError(errors)

    if errors and strict:
        raise TraceValidationError(errors)

    # Sort into natural order before constructing RunTrace
    attempts.sort(key=lambda a: a.attempt_number)
    loop_iterations.sort(key=lambda i: (str(i.attempt_id), i.iteration_number))

    try:
        trace = RunTrace(
            run=runs[0],
            attempts=attempts,
            loop_iterations=loop_iterations,
            model_calls=model_calls,
            tool_calls=tool_calls,
            evaluations=evaluations,
        )
    except ValidationError as exc:
        err = TraceError(line_number=0, raw_line="", message=str(exc))
        raise TraceValidationError([err]) from exc

    return trace
