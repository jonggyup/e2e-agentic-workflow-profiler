"""OpenClaw session transcript → RunTrace converter.

Reads the JSONL transcript that OpenClaw writes to
  ~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl
and produces a validated RunTrace in the profiler's event schema.

OpenClaw session JSONL format (one object per line):
  - type: "session"  — header: id, version, timestamp (ISO 8601), cwd
  - type: "model_change" — provider, modelId
  - type: "thinking_level_change" — thinkingLevel
  - type: "custom" — metadata snapshots
  - type: "message" — conversation turns:
      id, parentId, timestamp (ISO 8601)
      message.role ("user" | "assistant")
      message.content (array of typed blocks)
      message.timestamp (Unix ms)
      usage.{input, output, cacheRead, cacheWrite, totalTokens, cost.total}
      stopReason
    Assistant content: text, thinking, tool_use blocks.
    Tool results arrive as tool_result blocks inside the *next* user message.

Real timestamps (message.timestamp Unix ms) are converted to nanoseconds.
"""
from __future__ import annotations

import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from agent_profiler.schema.events import (
    AttemptEvent,
    EvaluationCriteria,
    EvaluationEvent,
    LoopIterationEvent,
    ModelCallEvent,
    RunEvent,
    ToolCallEvent,
    event_to_jsonl_bytes,
)
from agent_profiler.schema.trace import RunTrace

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

_NS_PER_MS = 1_000_000
_EPSILON_NS = 1_000_000  # 1 ms minimum gap to ensure end_ns > start_ns

# ---------------------------------------------------------------------------
# Attempt boundary detection — restart phrases (Rule 1)
# ---------------------------------------------------------------------------

_RESTART_PHRASES = (
    "that didn't work",
    "let me try a different approach",
    "let me try a different",
    "i'll try another way",
    "starting over",
    "let me start over",
    "let me try again",
    "a different approach",
)

# ---------------------------------------------------------------------------
# Tool name → category mapping
# ---------------------------------------------------------------------------

_TOOL_CATEGORIES: dict[str, str] = {
    "bash": "shell",
    "exec": "shell",
    "file_read": "filesystem",
    "file_write": "filesystem",
    "file_edit": "filesystem",
    "apply_patch": "filesystem",
    "browser": "browser",
    "canvas": "canvas",
    "cron": "system",
    "config": "system",
    "gateway": "system",
}
_DEFAULT_CATEGORY = "mcp"


def _tool_category(name: str) -> str:
    return _TOOL_CATEGORIES.get(name.lower(), _DEFAULT_CATEGORY)


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def _extract_text(content: Any) -> str:
    """Return concatenated plain text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _extract_tool_uses(content: Any) -> list[dict[str, Any]]:
    """Return tool_use/toolCall blocks from an assistant message content array.

    Normalises real OpenClaw ``toolCall`` blocks (which use ``arguments``)
    to always expose an ``input`` key so downstream code is format-agnostic.
    """
    if not isinstance(content, list):
        return []
    blocks = []
    for b in content:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "tool_use":
            blocks.append(b)
        elif btype == "toolCall":
            # Real OpenClaw uses camelCase and "arguments" instead of "input".
            # Produce a normalised copy with an "input" key.
            normalised = dict(b)
            if "input" not in normalised:
                normalised["input"] = normalised.get("arguments") or {}
            blocks.append(normalised)
    return blocks


# ---------------------------------------------------------------------------
# Intermediate structures
# ---------------------------------------------------------------------------


class _ToolResult:
    __slots__ = ("tool_use_id", "tool_name", "text", "is_error")

    def __init__(
        self,
        tool_use_id: str,
        tool_name: str,
        text: str,
        is_error: bool,
    ) -> None:
        self.tool_use_id = tool_use_id
        self.tool_name = tool_name
        self.text = text
        self.is_error = is_error


class _IterData:
    """Raw data for one loop iteration before IDs are assigned."""

    __slots__ = (
        "text",
        "tool_uses",
        "tool_results",
        "model_start_ns",
        "model_end_ns",
        "iter_end_ns",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    )

    def __init__(
        self,
        text: str,
        tool_uses: list[dict[str, Any]],
        tool_results: list[_ToolResult],
        model_start_ns: int = 0,
        model_end_ns: int = 1,
        iter_end_ns: int = 2,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> None:
        self.text = text
        self.tool_uses = tool_uses
        self.tool_results = tool_results
        self.model_start_ns = model_start_ns
        self.model_end_ns = model_end_ns
        self.iter_end_ns = iter_end_ns
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_uses)

    @property
    def iteration_type(self) -> str:
        return "reason_and_act" if self.tool_uses else "reason_only"


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------


def _parse_messages(path: Path) -> list[dict[str, Any]]:
    """Parse JSONL and return only type='message' events.

    All other event types (session, model_change, thinking_level_change,
    custom, etc.) are silently skipped.  Malformed lines are also skipped.
    """
    messages: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = orjson.loads(stripped)
                if isinstance(obj, dict) and obj.get("type") == "message":
                    messages.append(obj)
            except Exception:
                continue
    return messages


def _extract_model_info(path: Path) -> tuple[str, str]:
    """Return (provider, model_name) from the last model_change event.

    Falls back to ('unknown', 'unknown') if no model_change event is found.
    """
    provider = "unknown"
    model_name = "unknown"
    with path.open("rb") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = orjson.loads(stripped)
                if isinstance(obj, dict) and obj.get("type") == "model_change":
                    provider = obj.get("provider", "unknown") or "unknown"
                    model_name = obj.get("modelId", "unknown") or "unknown"
            except Exception:
                continue
    return provider, model_name


# ---------------------------------------------------------------------------
# Iteration grouping
# ---------------------------------------------------------------------------


def _group_into_iterations(message_events: list[dict[str, Any]]) -> list[_IterData]:
    """Group message events into loop iterations.

    An iteration consists of one assistant message plus the immediately
    following user message *if* that user message contains tool_result blocks.

    Pure user prompt messages (no tool_result blocks) are skipped.
    The prior user message timestamp is used as the model-call start time.
    """
    iterations: list[_IterData] = []
    i = 0
    n = len(message_events)

    while i < n:
        event = message_events[i]
        inner = event.get("message", {})
        role = inner.get("role", "")

        if role != "assistant":
            i += 1
            continue

        # --- Assistant message → new iteration ---
        content = inner.get("content", [])
        text = _extract_text(content)
        tool_uses = _extract_tool_uses(content)

        # Real timestamps: message.timestamp is Unix ms
        asst_ts_ms: int = inner.get("timestamp", 0)
        asst_ts_ns: int = asst_ts_ms * _NS_PER_MS

        # Model-call start = the preceding message's timestamp (user or toolResult)
        model_start_ns: int = asst_ts_ns  # fallback
        if i > 0:
            prior_inner = message_events[i - 1].get("message", {})
            if prior_inner.get("role") in ("user", "toolResult"):
                prior_ts_ms: int = prior_inner.get("timestamp", 0)
                model_start_ns = prior_ts_ms * _NS_PER_MS

        # Guarantee model_start < model_end
        if model_start_ns >= asst_ts_ns:
            asst_ts_ns = model_start_ns + _EPSILON_NS

        # Token usage from the assistant event
        usage: dict[str, Any] = event.get("usage", {})
        input_tokens: int | None = usage.get("input")
        output_tokens: int | None = usage.get("output")
        cache_read_tokens: int | None = usage.get("cacheRead")
        cache_write_tokens: int | None = usage.get("cacheWrite")

        # Build lookup: tool_use_id → tool_use block (for name resolution)
        tu_by_id: dict[str, dict[str, Any]] = {
            tu.get("id", ""): tu for tu in tool_uses if tu.get("id")
        }

        # --- Consume following tool result message(s) ---
        #
        # Two formats are supported:
        #
        # 1. Real OpenClaw: one or more consecutive messages with
        #    role="toolResult", toolCallId, toolName, content=[text blocks].
        #
        # 2. Standard Anthropic: a single role="user" message whose content
        #    array contains tool_result / toolResult blocks.
        #
        tool_results: list[_ToolResult] = []
        iter_end_ns: int = asst_ts_ns + _EPSILON_NS  # default for reason_only
        j = i + 1

        while j < n:
            next_event = message_events[j]
            next_inner = next_event.get("message", {})
            next_role = next_inner.get("role", "")

            if next_role == "toolResult":
                # Real OpenClaw format: whole message is one tool result.
                tu_id: str = next_inner.get("toolCallId", "")
                tool_name: str = next_inner.get("toolName", "")
                if not tool_name and tu_id in tu_by_id:
                    tool_name = tu_by_id[tu_id].get("name", "") or ""
                tool_name = tool_name or "unknown"

                is_error: bool = bool(next_inner.get("is_error", False))

                raw_content = next_inner.get("content", [])
                if isinstance(raw_content, list):
                    text_parts = [
                        b.get("text", "")
                        for b in raw_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    result_text = " ".join(text_parts)
                else:
                    result_text = str(raw_content) if raw_content else ""

                user_ts_ms: int = next_inner.get("timestamp", 0)
                user_ts_ns: int = user_ts_ms * _NS_PER_MS
                iter_end_ns = max(user_ts_ns, asst_ts_ns + _EPSILON_NS)

                tool_results.append(
                    _ToolResult(
                        tool_use_id=tu_id,
                        tool_name=tool_name,
                        text=result_text,
                        is_error=is_error,
                    )
                )
                j += 1

            elif next_role == "user":
                # Standard Anthropic format: tool_result blocks inside user msg.
                next_content = next_inner.get("content", [])
                tr_blocks = [
                    b
                    for b in next_content
                    if isinstance(b, dict)
                    and b.get("type") in ("tool_result", "toolResult")
                ]
                if tr_blocks:
                    user_ts_ms = next_inner.get("timestamp", 0)
                    user_ts_ns = user_ts_ms * _NS_PER_MS
                    iter_end_ns = max(user_ts_ns, asst_ts_ns + _EPSILON_NS)

                    for tr_block in tr_blocks:
                        tu_id = tr_block.get("tool_use_id", "")
                        tool_name = ""
                        if tu_id in tu_by_id:
                            tool_name = tu_by_id[tu_id].get("name", "") or ""
                        tool_name = tool_name or "unknown"

                        is_error = bool(tr_block.get("is_error", False))

                        raw_content = tr_block.get("content", "")
                        if isinstance(raw_content, list):
                            text_parts = [
                                b.get("text", "")
                                for b in raw_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ]
                            result_text = " ".join(text_parts)
                        else:
                            result_text = str(raw_content) if raw_content else ""

                        tool_results.append(
                            _ToolResult(
                                tool_use_id=tu_id,
                                tool_name=tool_name,
                                text=result_text,
                                is_error=is_error,
                            )
                        )
                    j += 1
                break  # user message consumed (with or without tool results)

            else:
                break  # next message is assistant or unknown — stop

        iterations.append(
            _IterData(
                text=text,
                tool_uses=tool_uses,
                tool_results=tool_results,
                model_start_ns=model_start_ns,
                model_end_ns=asst_ts_ns,
                iter_end_ns=iter_end_ns,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
        )
        i = j

    return iterations


# ---------------------------------------------------------------------------
# Attempt boundary detection
# ---------------------------------------------------------------------------


def _has_restart_phrase(text: str) -> bool:
    """Rule 1: detect explicit restart language in assistant text."""
    lower = text.lower()
    return any(phrase in lower for phrase in _RESTART_PHRASES)


def _detect_attempt_boundaries(iterations: list[_IterData]) -> list[int]:
    """Return sorted list of iteration indices where new attempts begin.

    Always starts with index 0.  Additional boundaries are added by:
    - Rule 1: assistant text contains a restart phrase.
    - Rule 2: same tool fails 3+ consecutive times, then a different tool is called.
    """
    if not iterations:
        return [0]

    boundaries: set[int] = {0}

    # Rule 2 state
    consec_error_tool: str | None = None
    consec_error_count: int = 0

    for idx, it in enumerate(iterations):
        # Rule 1 (skip idx 0 — we never split at the very start)
        if idx > 0 and _has_restart_phrase(it.text):
            boundaries.add(idx)
            consec_error_tool = None
            consec_error_count = 0

        # Rule 2: different tool after 3+ consecutive errors
        if idx > 0 and consec_error_count >= 3 and it.tool_uses:
            first_tool = it.tool_uses[0].get("name", "")
            if first_tool and first_tool != consec_error_tool:
                boundaries.add(idx)
                consec_error_tool = None
                consec_error_count = 0

        # Update Rule 2 state from this iteration's tool results
        for tr in it.tool_results:
            if tr.is_error:
                if consec_error_tool == tr.tool_name:
                    consec_error_count += 1
                else:
                    consec_error_tool = tr.tool_name
                    consec_error_count = 1
            else:
                consec_error_tool = None
                consec_error_count = 0

    return sorted(boundaries)


def _map_iterations_to_attempts(
    n_iterations: int,
    boundaries: list[int],
) -> list[int]:
    """Return a list of 1-based attempt numbers, one per iteration index."""
    attempt_map: list[int] = []
    for idx in range(n_iterations):
        attempt_num = 1
        for k, boundary in enumerate(boundaries):
            if idx >= boundary:
                attempt_num = k + 1
        attempt_map.append(attempt_num)
    return attempt_map


# ---------------------------------------------------------------------------
# Failure category inference
# ---------------------------------------------------------------------------


def _infer_failure_category(
    iterations: list[_IterData],
    attempt_map: list[int],
    attempt_num: int,
) -> str:
    """Infer the failure category for a non-final failed attempt."""
    attempt_iters = [
        it for idx, it in enumerate(iterations) if attempt_map[idx] == attempt_num
    ]
    all_results = [tr for it in attempt_iters for tr in it.tool_results]
    errors = [tr for tr in all_results if tr.is_error]

    if not errors:
        return "tool_error"

    # reasoning_loop: same tool errors 3+ times
    error_counts = Counter(tr.tool_name for tr in errors)
    if any(count >= 3 for count in error_counts.values()):
        return "reasoning_loop"

    # transient: single error, multiple results (later ones succeeded)
    if len(errors) == 1 and len(all_results) > 1:
        return "transient"

    return "tool_error"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert_openclaw_session(
    session_path: Path,
    *,
    task_description: str = "",
    program_tool_name: str | None = None,
) -> RunTrace:
    """Read an OpenClaw session JSONL and return a validated RunTrace.

    Parameters
    ----------
    session_path:
        Path to the OpenClaw session JSONL file.
    task_description:
        Human-readable description of the task.  If empty, the content of
        the first user message is used; falls back to the file stem.
    program_tool_name:
        Tool name to mark as ``is_program_under_test`` (e.g. ``"bash"``).
    """
    path = Path(session_path)
    message_events = _parse_messages(path)
    model_provider, model_name = _extract_model_info(path)

    # Extract task description from first user message if not provided
    if not task_description:
        for event in message_events:
            inner = event.get("message", {})
            if inner.get("role") == "user":
                task_description = _extract_text(inner.get("content", []))
                break
    task_description = (task_description or path.stem or "unknown task").strip()

    iterations = _group_into_iterations(message_events)

    if not iterations:
        raise ValueError(
            f"No assistant messages found in {path}; "
            "cannot convert to RunTrace"
        )

    boundaries = _detect_attempt_boundaries(iterations)
    attempt_map = _map_iterations_to_attempts(len(iterations), boundaries)
    num_attempts = len(boundaries)

    # Pre-allocate IDs
    run_id = uuid.uuid4()
    attempt_ids: dict[int, uuid.UUID] = {
        n: uuid.uuid4() for n in range(1, num_attempts + 1)
    }

    # ---- Build events with real timestamps ----
    run_start_ns: int = iterations[0].model_start_ns

    attempt_start_ns: dict[int, int] = {}
    attempt_end_ns: dict[int, int] = {}
    iter_num_per_attempt: dict[int, int] = {n: 0 for n in range(1, num_attempts + 1)}

    loop_iteration_events: list[LoopIterationEvent] = []
    model_call_events: list[ModelCallEvent] = []
    tool_call_events: list[ToolCallEvent] = []

    for idx, it in enumerate(iterations):
        attempt_num = attempt_map[idx]
        attempt_id = attempt_ids[attempt_num]

        if attempt_num not in attempt_start_ns:
            attempt_start_ns[attempt_num] = it.model_start_ns

        iteration_id = uuid.uuid4()
        iter_num_per_attempt[attempt_num] += 1
        iter_num = iter_num_per_attempt[attempt_num]

        mc_start_ns = it.model_start_ns
        mc_end_ns = it.model_end_ns
        iter_start_ns = mc_start_ns
        iter_end_ns = it.iter_end_ns

        # Distribute tool-call slots evenly within [mc_end_ns, iter_end_ns]
        n_tools = len(it.tool_uses)
        if n_tools > 0 and iter_end_ns > mc_end_ns:
            slot_ns = (iter_end_ns - mc_end_ns) // n_tools
        else:
            slot_ns = _EPSILON_NS

        result_by_id: dict[str, _ToolResult] = {
            tr.tool_use_id: tr for tr in it.tool_results if tr.tool_use_id
        }

        tools_called: list[str] = []
        for tc_idx, tu in enumerate(it.tool_uses):
            tool_name = tu.get("name", "unknown") or "unknown"
            tools_called.append(tool_name)
            tu_id = tu.get("id", "")
            tr = result_by_id.get(tu_id)

            is_error = tr.is_error if tr is not None else False
            error_msg = tr.text if (tr is not None and tr.is_error) else None
            outcome: str = "error" if is_error else "success"

            tc_start_ns = mc_end_ns + slot_ns * tc_idx
            tc_end_ns = mc_end_ns + slot_ns * (tc_idx + 1)
            if tc_end_ns <= tc_start_ns:
                tc_end_ns = tc_start_ns + _EPSILON_NS

            tool_call_events.append(
                ToolCallEvent(
                    event_type="tool_call",
                    tool_call_id=uuid.uuid4(),
                    iteration_id=iteration_id,
                    tool_name=tool_name,
                    tool_category=_tool_category(tool_name),
                    tool_params=tu.get("input", {}),
                    start_ns=tc_start_ns,
                    end_ns=tc_end_ns,
                    outcome=outcome,
                    error_message=error_msg,
                    sandbox_used=False,
                    is_program_under_test=(
                        program_tool_name is not None
                        and tool_name == program_tool_name
                    ),
                )
            )

        attempt_end_ns[attempt_num] = iter_end_ns

        model_call_events.append(
            ModelCallEvent(
                event_type="model_call",
                model_call_id=uuid.uuid4(),
                iteration_id=iteration_id,
                model_provider=model_provider,
                model_name=model_name,
                start_ns=mc_start_ns,
                end_ns=mc_end_ns,
                requested_tools=tools_called,
                tools_called_in_response=tools_called,
                input_tokens=it.input_tokens,
                output_tokens=it.output_tokens,
                cache_read_tokens=it.cache_read_tokens,
                cache_write_tokens=it.cache_write_tokens,
            )
        )

        loop_iteration_events.append(
            LoopIterationEvent(
                event_type="loop_iteration",
                iteration_id=iteration_id,
                attempt_id=attempt_id,
                iteration_number=iter_num,
                start_ns=iter_start_ns,
                end_ns=iter_end_ns,
                has_tool_calls=it.has_tool_calls,
                iteration_type=it.iteration_type,
            )
        )

    run_end_ns: int = iterations[-1].iter_end_ns

    # ---- Determine attempt outcomes ----
    last_attempt_num = num_attempts
    last_iter = iterations[-1]
    last_attempt_succeeded = last_iter.iteration_type == "reason_only"

    attempt_events: list[AttemptEvent] = []
    evaluation_events: list[EvaluationEvent] = []

    for attempt_num in range(1, num_attempts + 1):
        attempt_id = attempt_ids[attempt_num]
        is_last = attempt_num == last_attempt_num

        if is_last:
            outcome = "success" if last_attempt_succeeded else "failure"
            failure_category = None if outcome == "success" else "tool_error"
            failure_reason = None
        else:
            outcome = "failure"
            failure_category = _infer_failure_category(
                iterations, attempt_map, attempt_num
            )
            failure_reason = "Heuristic: attempt ended without a final answer"

        attempt_events.append(
            AttemptEvent(
                event_type="attempt",
                attempt_id=attempt_id,
                run_id=run_id,
                attempt_number=attempt_num,
                start_ns=attempt_start_ns.get(attempt_num, run_start_ns),
                end_ns=attempt_end_ns.get(attempt_num, run_end_ns),
                outcome=outcome,
                failure_reason=failure_reason,
                failure_category=failure_category,
            )
        )

        if is_last:
            evaluation_events.append(
                EvaluationEvent(
                    event_type="evaluation",
                    eval_id=uuid.uuid4(),
                    attempt_id=attempt_id,
                    evaluator="heuristic",
                    passed=last_attempt_succeeded,
                    score=1.0 if last_attempt_succeeded else 0.0,
                    reason=(
                        "Agent produced a final text response (no tool calls)"
                        if last_attempt_succeeded
                        else "Agent did not produce a final text response"
                    ),
                    criteria=EvaluationCriteria(
                        task_completed=last_attempt_succeeded,
                        correct_result=None,
                        no_side_effects=None,
                        efficient_path=(
                            attempt_num == 1 and last_attempt_succeeded
                        ),
                    ),
                )
            )

    run_event = RunEvent(
        event_type="run",
        run_id=run_id,
        task_description=task_description,
        start_ns=run_start_ns,
        end_ns=run_end_ns,
        outcome="success" if last_attempt_succeeded else "failure",
        attempt_count=num_attempts,
        total_model_calls=len(model_call_events),
        total_tool_calls=len(tool_call_events),
        model_provider=model_provider,
        model_name=model_name,
        sandbox_mode="off",
    )

    return RunTrace(
        run=run_event,
        attempts=attempt_events,
        loop_iterations=loop_iteration_events,
        model_calls=model_call_events,
        tool_calls=tool_call_events,
        evaluations=evaluation_events,
    )


# ---------------------------------------------------------------------------
# Trace serialisation helper
# ---------------------------------------------------------------------------


def write_trace(trace: RunTrace, path: Path) -> None:
    """Serialise a RunTrace to a profiler JSONL file.

    Events are written in canonical order: run, attempts, loop_iterations,
    model_calls, tool_calls, evaluations.
    """
    with path.open("wb") as fh:
        for event in (
            [trace.run]
            + trace.attempts
            + trace.loop_iterations
            + trace.model_calls
            + trace.tool_calls
            + trace.evaluations
        ):
            fh.write(event_to_jsonl_bytes(event))
            fh.write(b"\n")
