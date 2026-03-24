"""agent_profiler.schema — Pydantic event models and RunTrace loader."""

from .events import (
    AnyEvent,
    AttemptEvent,
    BaseEvent,
    EvaluationCriteria,
    EvaluationEvent,
    FailureCategory,
    LoopIterationEvent,
    ModelCallEvent,
    RunEvent,
    TimedEvent,
    ToolCallEvent,
    event_to_jsonl_bytes,
)
from .trace import (
    RunTrace,
    TraceError,
    TraceValidationError,
    TraceWarning,
    evaluation_for_attempt,
    iterations_for_attempt,
    load_trace,
    model_call_for_iteration,
    tool_calls_for_iteration,
)

__all__ = [
    # events
    "AnyEvent",
    "AttemptEvent",
    "BaseEvent",
    "EvaluationCriteria",
    "EvaluationEvent",
    "FailureCategory",
    "LoopIterationEvent",
    "ModelCallEvent",
    "RunEvent",
    "TimedEvent",
    "ToolCallEvent",
    "event_to_jsonl_bytes",
    # trace
    "RunTrace",
    "TraceError",
    "TraceValidationError",
    "TraceWarning",
    "evaluation_for_attempt",
    "iterations_for_attempt",
    "load_trace",
    "model_call_for_iteration",
    "tool_calls_for_iteration",
]
