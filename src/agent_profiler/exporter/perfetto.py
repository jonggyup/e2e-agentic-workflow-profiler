"""Export a RunTrace as a Perfetto-compatible Chrome Trace Event JSON file.

The output can be loaded directly into https://ui.perfetto.dev for
interactive timeline visualization. Uses the simpler JSON "Chrome Trace
Event Format" (catapult format) rather than Perfetto's native protobuf.

Thread lanes (tid values for visual separation):
  - tid=1: Run        — the overall run as one span
  - tid=2: Attempts   — each attempt as a span
  - tid=3: ReAct Loop — each loop iteration as a span
  - tid=4: Model Calls — each model call as a span
  - tid=5: Tool Calls  — each tool call as a span
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from agent_profiler.schema.events import (
    AttemptEvent,
    EvaluationEvent,
    LoopIterationEvent,
    ModelCallEvent,
    ToolCallEvent,
)
from agent_profiler.schema.trace import RunTrace, evaluation_for_attempt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PID_AGENT = 1
_PID_SYSTEM = 2

_TID_RUN = 1
_TID_ATTEMPTS = 2
_TID_LOOP = 3
_TID_MODEL = 4
_TID_TOOL = 5

_THREAD_NAMES: dict[int, str] = {
    _TID_RUN: "Run",
    _TID_ATTEMPTS: "Attempts",
    _TID_LOOP: "ReAct Loop",
    _TID_MODEL: "Model Calls",
    _TID_TOOL: "Tool Calls",
}

_SYSTEM_THREAD_NAMES: dict[int, str] = {
    1: "CPU %",
    2: "Memory MB",
    3: "Network KB/s",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns_to_us(ns: int, base_ns: int) -> float:
    """Convert monotonic nanoseconds to microseconds relative to run start."""
    return (ns - base_ns) / 1000.0


def _short_params(tool_params: dict[str, Any], max_len: int = 60) -> str:
    """Extract a short summary of the most relevant tool param value."""
    if not tool_params:
        return ""
    # Pick the first value that looks interesting
    for key in ("command", "query", "url", "path", "code", "content"):
        if key in tool_params:
            val = str(tool_params[key])
            return val[:max_len] if len(val) > max_len else val
    # Fall back to the first value
    first_val = str(next(iter(tool_params.values())))
    return first_val[:max_len] if len(first_val) > max_len else first_val


def _complete_event(
    name: str,
    cat: str,
    ts: float,
    dur: float,
    pid: int,
    tid: int,
    *,
    args: dict[str, Any] | None = None,
    cname: str | None = None,
) -> dict[str, Any]:
    """Build a ph='X' (complete) trace event."""
    ev: dict[str, Any] = {
        "name": name,
        "cat": cat,
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
    }
    if args:
        ev["args"] = args
    if cname:
        ev["cname"] = cname
    return ev


def _instant_event(
    name: str,
    cat: str,
    ts: float,
    pid: int,
    tid: int,
    *,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ph='i' (instant) trace event."""
    ev: dict[str, Any] = {
        "name": name,
        "cat": cat,
        "ph": "i",
        "ts": ts,
        "pid": pid,
        "tid": tid,
        "s": "t",  # scope: thread
    }
    if args:
        ev["args"] = args
    return ev


def _metadata_event(
    name: str, pid: int, args: dict[str, Any], *, tid: int | None = None
) -> dict[str, Any]:
    """Build a ph='M' (metadata) trace event."""
    ev: dict[str, Any] = {
        "ph": "M",
        "name": name,
        "pid": pid,
        "args": args,
    }
    if tid is not None:
        ev["tid"] = tid
    return ev


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def _convert_run(trace: RunTrace, base_ns: int) -> list[dict[str, Any]]:
    """Convert the RunEvent to a complete span."""
    run = trace.run
    ts = _ns_to_us(run.start_ns, base_ns)
    dur = _ns_to_us(run.end_ns, base_ns) - ts
    return [
        _complete_event(
            name=f"Run: {run.task_description}",
            cat="run",
            ts=ts,
            dur=dur,
            pid=_PID_AGENT,
            tid=_TID_RUN,
            args={
                "outcome": run.outcome,
                "attempt_count": run.attempt_count,
                "model_provider": run.model_provider,
                "model_name": run.model_name,
            },
        )
    ]


def _convert_attempt(
    attempt: AttemptEvent, base_ns: int, evaluation: EvaluationEvent | None
) -> list[dict[str, Any]]:
    """Convert an AttemptEvent (and its evaluation) to trace events."""
    ts = _ns_to_us(attempt.start_ns, base_ns)
    dur = _ns_to_us(attempt.end_ns, base_ns) - ts

    cname = None
    if attempt.outcome == "failure":
        cname = "terrible_red"
    elif attempt.outcome == "success":
        cname = "good"

    events: list[dict[str, Any]] = [
        _complete_event(
            name=f"Attempt {attempt.attempt_number}: {attempt.outcome}",
            cat="attempt",
            ts=ts,
            dur=dur,
            pid=_PID_AGENT,
            tid=_TID_ATTEMPTS,
            cname=cname,
            args={
                "outcome": attempt.outcome,
                "failure_reason": attempt.failure_reason,
                "failure_category": attempt.failure_category,
            },
        )
    ]

    # Evaluation instant event at attempt end time
    if evaluation is not None:
        score_str = f"{evaluation.score}" if evaluation.score is not None else "N/A"
        label = "PASS" if evaluation.passed else "FAIL"
        events.append(
            _instant_event(
                name=f"Eval: {label} ({score_str})",
                cat="eval",
                ts=_ns_to_us(attempt.end_ns, base_ns),
                pid=_PID_AGENT,
                tid=_TID_ATTEMPTS,
                args={
                    "passed": evaluation.passed,
                    "score": evaluation.score,
                    "reason": evaluation.reason,
                    "evaluator": evaluation.evaluator,
                },
            )
        )

    return events


def _convert_loop_iteration(
    iteration: LoopIterationEvent, base_ns: int
) -> list[dict[str, Any]]:
    ts = _ns_to_us(iteration.start_ns, base_ns)
    dur = _ns_to_us(iteration.end_ns, base_ns) - ts
    return [
        _complete_event(
            name=f"Iter {iteration.iteration_number}: {iteration.iteration_type}",
            cat="loop",
            ts=ts,
            dur=dur,
            pid=_PID_AGENT,
            tid=_TID_LOOP,
        )
    ]


def _convert_model_call(
    mc: ModelCallEvent, base_ns: int
) -> list[dict[str, Any]]:
    ts = _ns_to_us(mc.start_ns, base_ns)
    dur = _ns_to_us(mc.end_ns, base_ns) - ts
    return [
        _complete_event(
            name=f"LLM: {mc.model_name}",
            cat="model",
            ts=ts,
            dur=dur,
            pid=_PID_AGENT,
            tid=_TID_MODEL,
            args={
                "input_tokens": mc.input_tokens,
                "output_tokens": mc.output_tokens,
                "tools_called_in_response": mc.tools_called_in_response,
            },
        )
    ]


def _convert_tool_call(
    tc: ToolCallEvent, base_ns: int
) -> list[dict[str, Any]]:
    ts = _ns_to_us(tc.start_ns, base_ns)
    dur = _ns_to_us(tc.end_ns, base_ns) - ts

    short = _short_params(tc.tool_params)
    name = f"{tc.tool_name}: {short}" if short else tc.tool_name

    cname = "terrible_red" if tc.outcome == "error" else None

    return [
        _complete_event(
            name=name,
            cat="tool",
            ts=ts,
            dur=dur,
            pid=_PID_AGENT,
            tid=_TID_TOOL,
            cname=cname,
            args={
                "tool_params": tc.tool_params,
                "outcome": tc.outcome,
                "error_message": tc.error_message,
            },
        )
    ]


# ---------------------------------------------------------------------------
# System samples → counter tracks
# ---------------------------------------------------------------------------


def _load_system_samples(path: Path) -> list[dict[str, Any]]:
    """Load system sample JSONL, filtering to metric samples only."""
    samples: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            data = orjson.loads(stripped)
            if data.get("sample_type") == "system":
                samples.append(data)
    return samples


def _convert_system_samples(
    samples: list[dict[str, Any]], base_ns: int
) -> list[dict[str, Any]]:
    """Convert system samples to Perfetto counter (ph='C') events."""
    events: list[dict[str, Any]] = []
    prev_net: float | None = None
    prev_ts_ns: int | None = None

    for sample in samples:
        ts_ns = sample.get("timestamp_ns", 0)
        ts = _ns_to_us(ts_ns, base_ns)

        # CPU counter
        cpu = sample.get("cpu_percent", 0.0)
        events.append({
            "name": "CPU %",
            "cat": "system",
            "ph": "C",
            "ts": ts,
            "pid": _PID_SYSTEM,
            "tid": 1,
            "args": {"CPU %": cpu},
        })

        # Memory counter
        mem_mb = sample.get("memory_rss_mb", 0.0)
        events.append({
            "name": "Memory MB",
            "cat": "system",
            "ph": "C",
            "ts": ts,
            "pid": _PID_SYSTEM,
            "tid": 2,
            "args": {"Memory MB": mem_mb},
        })

        # Network rate counter
        net_sent = sample.get("network_sent_bytes", 0)
        net_recv = sample.get("network_recv_bytes", 0)
        net_total = net_sent + net_recv
        if prev_net is not None and prev_ts_ns is not None:
            dt_s = (ts_ns - prev_ts_ns) / 1e9
            if dt_s > 0:
                rate_kbs = (net_total - prev_net) / 1024.0 / dt_s
            else:
                rate_kbs = 0.0
            events.append({
                "name": "Network KB/s",
                "cat": "system",
                "ph": "C",
                "ts": ts,
                "pid": _PID_SYSTEM,
                "tid": 3,
                "args": {"Network KB/s": round(rate_kbs, 2)},
            })
        prev_net = net_total
        prev_ts_ns = ts_ns

    return events


# ---------------------------------------------------------------------------
# Metadata events
# ---------------------------------------------------------------------------


def _build_metadata() -> list[dict[str, Any]]:
    """Build process and thread name metadata events."""
    events: list[dict[str, Any]] = []

    # Process names
    events.append(_metadata_event("process_name", _PID_AGENT, {"name": "Agent"}))
    events.append(_metadata_event("process_name", _PID_SYSTEM, {"name": "System"}))

    # Agent thread names
    for tid, name in _THREAD_NAMES.items():
        events.append(
            _metadata_event("thread_name", _PID_AGENT, {"name": name}, tid=tid)
        )

    # System thread names
    for tid, name in _SYSTEM_THREAD_NAMES.items():
        events.append(
            _metadata_event("thread_name", _PID_SYSTEM, {"name": name}, tid=tid)
        )

    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_perfetto(
    trace: RunTrace,
    output_path: Path,
    *,
    system_samples_path: Path | None = None,
) -> None:
    """Export a RunTrace as a Perfetto-compatible JSON trace file.

    Parameters
    ----------
    trace:
        A validated RunTrace to export.
    output_path:
        Destination path for the JSON output file.
    system_samples_path:
        Optional path to a SystemSampler JSONL file. When provided,
        CPU/memory/network counter tracks are added to the trace.
    """
    base_ns = trace.run.start_ns
    trace_events: list[dict[str, Any]] = []

    # Metadata (process + thread names)
    trace_events.extend(_build_metadata())

    # Run span
    trace_events.extend(_convert_run(trace, base_ns))

    # Attempts + evaluations
    for attempt in trace.attempts:
        ev = evaluation_for_attempt(trace, attempt.attempt_id)
        trace_events.extend(_convert_attempt(attempt, base_ns, ev))

    # Loop iterations
    for iteration in trace.loop_iterations:
        trace_events.extend(_convert_loop_iteration(iteration, base_ns))

    # Model calls
    for mc in trace.model_calls:
        trace_events.extend(_convert_model_call(mc, base_ns))

    # Tool calls
    for tc in trace.tool_calls:
        trace_events.extend(_convert_tool_call(tc, base_ns))

    # System samples (counter tracks)
    if system_samples_path is not None and system_samples_path.exists():
        samples = _load_system_samples(system_samples_path)
        trace_events.extend(_convert_system_samples(samples, base_ns))

    # Write output
    output = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        orjson.dumps(output, option=orjson.OPT_INDENT_2)
    )
