"""Pydantic models for every profiler event type.

Five timed event types (Run, Attempt, LoopIteration, ModelCall, ToolCall) share
a TimedEvent base that enforces end_ns >= start_ns.  EvaluationEvent has no
timestamps and inherits only the shared model_config.

AnyEvent is a discriminated union keyed on ``event_type`` used by load_trace
to dispatch incoming dicts to the correct model.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union
from uuid import UUID

import orjson
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from typing_extensions import Self

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

_SHARED_CONFIG = ConfigDict(
    populate_by_name=True,
    use_enum_values=True,
)


class BaseEvent(BaseModel):
    model_config = _SHARED_CONFIG


class TimedEvent(BaseEvent):
    """Base for all events that have start/end timestamps."""

    start_ns: int
    end_ns: int

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end_ns < self.start_ns:
            raise ValueError(
                f"end_ns ({self.end_ns}) must be >= start_ns ({self.start_ns})"
            )
        return self


# ---------------------------------------------------------------------------
# FailureCategory literal type
# ---------------------------------------------------------------------------

FailureCategory = Literal[
    "wrong_tool",
    "bad_params",
    "tool_error",
    "transient",
    "timeout",
    "reasoning_loop",
    "hallucinated_tool",
    "context_overflow",
]

# ---------------------------------------------------------------------------
# Event models
# ---------------------------------------------------------------------------


class RunEvent(TimedEvent):
    event_type: Literal["run"]
    run_id: UUID
    task_description: str
    outcome: Literal["success", "failure", "timeout", "interrupted"]
    attempt_count: int
    total_model_calls: int
    total_tool_calls: int
    model_provider: str
    model_name: str
    sandbox_mode: Literal["off", "docker", "openshell"]

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        if not self.task_description.strip():
            raise ValueError("task_description must not be blank")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be >= 1")
        if self.total_model_calls < 0:
            raise ValueError("total_model_calls must be >= 0")
        if self.total_tool_calls < 0:
            raise ValueError("total_tool_calls must be >= 0")
        if not self.model_provider.strip():
            raise ValueError("model_provider must not be blank")
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        return self


class AttemptEvent(TimedEvent):
    event_type: Literal["attempt"]
    attempt_id: UUID
    run_id: UUID
    attempt_number: int
    outcome: Literal["success", "failure"]
    failure_reason: str | None = None
    failure_category: FailureCategory | None = None  # type: ignore[valid-type]

    @model_validator(mode="after")
    def _validate_attempt(self) -> Self:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be >= 1")
        if self.outcome == "success":
            if self.failure_category is not None:
                raise ValueError(
                    "failure_category must be None when outcome is 'success'"
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "failure_reason must be None when outcome is 'success'"
                )
        if self.outcome == "failure" and self.failure_category is None:
            raise ValueError(
                "failure_category is required when outcome is 'failure'"
            )
        return self


class LoopIterationEvent(TimedEvent):
    event_type: Literal["loop_iteration"]
    iteration_id: UUID
    attempt_id: UUID
    iteration_number: int
    has_tool_calls: bool
    iteration_type: Literal["reason_and_act", "reason_only", "act_only"]

    @model_validator(mode="after")
    def _validate_loop_iteration(self) -> Self:
        if self.iteration_number < 1:
            raise ValueError("iteration_number must be >= 1")
        if self.iteration_type == "reason_only" and self.has_tool_calls:
            raise ValueError(
                "has_tool_calls must be False when iteration_type is 'reason_only'"
            )
        if self.iteration_type in ("reason_and_act", "act_only") and not self.has_tool_calls:
            raise ValueError(
                f"has_tool_calls must be True when iteration_type is '{self.iteration_type}'"
            )
        return self


class ModelCallEvent(TimedEvent):
    event_type: Literal["model_call"]
    model_call_id: UUID
    iteration_id: UUID
    model_provider: str
    model_name: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    time_to_first_token_ms: float | None = None
    requested_tools: list[str] = Field(default_factory=list)
    tools_called_in_response: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_model_call(self) -> Self:
        if not self.model_provider.strip():
            raise ValueError("model_provider must not be blank")
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")
        if self.time_to_first_token_ms is not None and self.time_to_first_token_ms < 0:
            raise ValueError("time_to_first_token_ms must be >= 0")
        # tools_called_in_response ⊆ requested_tools is a warning-level check
        # handled at the RunTrace level, not raised here.
        return self


class ToolCallEvent(TimedEvent):
    event_type: Literal["tool_call"]
    tool_call_id: UUID
    iteration_id: UUID
    tool_name: str
    tool_category: Literal["shell", "filesystem", "browser", "mcp", "canvas", "system"]
    tool_params: dict[str, Any] = Field(default_factory=dict)
    outcome: Literal["success", "error", "timeout"]
    error_message: str | None = None
    sandbox_used: bool = False
    is_program_under_test: bool = False

    @model_validator(mode="after")
    def _validate_tool_call(self) -> Self:
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be blank")
        if self.outcome == "success" and self.error_message is not None:
            raise ValueError("error_message must be None when outcome is 'success'")
        return self


class EvaluationCriteria(BaseModel):
    """Named criteria booleans; extra keys allowed for forward-compatibility."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task_completed: bool | None = None
    correct_result: bool | None = None
    no_side_effects: bool | None = None
    efficient_path: bool | None = None


class EvaluationEvent(BaseEvent):
    event_type: Literal["evaluation"]
    eval_id: UUID
    attempt_id: UUID
    evaluator: Literal["human", "llm_judge", "script", "heuristic"]
    passed: bool
    score: float | None = None
    reason: str | None = None
    criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)

    @model_validator(mode="after")
    def _validate_evaluation(self) -> Self:
        if self.score is not None and not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score must be in [0.0, 1.0], got {self.score}"
            )
        return self


# ---------------------------------------------------------------------------
# Discriminated union + adapter
# ---------------------------------------------------------------------------

AnyEvent = Annotated[
    Union[
        RunEvent,
        AttemptEvent,
        LoopIterationEvent,
        ModelCallEvent,
        ToolCallEvent,
        EvaluationEvent,
    ],
    Field(discriminator="event_type"),
]

_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def event_to_jsonl_bytes(event: BaseEvent) -> bytes:
    """Serialise a single event to a compact JSON bytes line (no trailing newline)."""
    return orjson.dumps(event.model_dump(mode="json"))
