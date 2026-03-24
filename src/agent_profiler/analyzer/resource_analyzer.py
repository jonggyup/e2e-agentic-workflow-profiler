"""Resource analysis — correlate system samples with agent tool calls.

Reads a system-samples JSONL file (produced by SystemSampler), matches
samples to tool-call time windows from a RunTrace, and produces a
ResourceProfile with per-tool resource summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel

from agent_profiler.schema.trace import RunTrace


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ToolResourceSummary(BaseModel):
    tool_name: str
    avg_cpu_during: float
    peak_cpu_during: float
    avg_memory_during: float
    peak_memory_during: float
    net_bytes_during: int
    disk_bytes_during: int
    resource_bottleneck: Literal["cpu", "memory", "network", "disk", "none"]


class AggregatedToolResource(BaseModel):
    tool_name: str
    call_count: int
    total_duration_ms: float
    avg_cpu: float
    peak_cpu: float
    total_network_bytes: int
    total_disk_bytes: int
    dominant_bottleneck: str


class ResourceProfile(BaseModel):
    peak_cpu_percent: float
    avg_cpu_percent: float
    peak_memory_mb: float
    avg_memory_mb: float
    total_network_sent_mb: float
    total_network_recv_mb: float
    total_disk_read_mb: float
    total_disk_write_mb: float
    per_tool_resources: list[ToolResourceSummary]


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _load_system_samples(path: Path) -> list[dict[str, Any]]:
    """Load system-type samples from a JSONL file (skip markers)."""
    samples: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = orjson.loads(stripped)
                if isinstance(obj, dict) and obj.get("sample_type") == "system":
                    samples.append(obj)
            except Exception:
                continue
    return samples


# ---------------------------------------------------------------------------
# Per-tool correlation
# ---------------------------------------------------------------------------


def _samples_in_window(
    samples: list[dict[str, Any]],
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    """Return samples whose timestamp_ns falls within [start_ns, end_ns]."""
    return [s for s in samples if start_ns <= s["timestamp_ns"] <= end_ns]


def _determine_bottleneck(
    avg_cpu: float,
    peak_memory_mb: float,
    memory_start_mb: float,
    net_bytes: int,
    disk_bytes: int,
    duration_ms: float,
) -> Literal["cpu", "memory", "network", "disk", "none"]:
    """Determine the resource bottleneck for a tool call.

    Rules (evaluated in order, first match wins):
    - "cpu" if avg_cpu > 80%
    - "memory" if peak_memory increased by >20% during the call
    - "network" if net_bytes > 1MB AND duration > 1s
    - "disk" if disk_bytes > 10MB
    - "none" otherwise
    """
    if avg_cpu > 80.0:
        return "cpu"

    if memory_start_mb > 0 and peak_memory_mb > memory_start_mb * 1.2:
        return "memory"

    if net_bytes > 1_000_000 and duration_ms > 1000:
        return "network"

    if disk_bytes > 10_000_000:
        return "disk"

    return "none"


def _summarize_tool(
    tool_name: str,
    window_samples: list[dict[str, Any]],
    duration_ms: float,
) -> ToolResourceSummary:
    """Build a ToolResourceSummary from system samples within a tool's window."""
    if not window_samples:
        return ToolResourceSummary(
            tool_name=tool_name,
            avg_cpu_during=0.0,
            peak_cpu_during=0.0,
            avg_memory_during=0.0,
            peak_memory_during=0.0,
            net_bytes_during=0,
            disk_bytes_during=0,
            resource_bottleneck="none",
        )

    cpus = [s.get("cpu_percent", 0.0) for s in window_samples]
    mems = [s.get("memory_rss_mb", 0.0) for s in window_samples]

    avg_cpu = sum(cpus) / len(cpus)
    peak_cpu = max(cpus)
    avg_mem = sum(mems) / len(mems)
    peak_mem = max(mems)

    # Network / disk deltas across the window
    first = window_samples[0]
    last = window_samples[-1]

    net_bytes = (
        (last.get("network_bytes_sent", 0) - first.get("network_bytes_sent", 0))
        + (last.get("network_bytes_recv", 0) - first.get("network_bytes_recv", 0))
    )
    net_bytes = max(net_bytes, 0)

    disk_bytes = (
        (last.get("disk_read_bytes", 0) - first.get("disk_read_bytes", 0))
        + (last.get("disk_write_bytes", 0) - first.get("disk_write_bytes", 0))
    )
    disk_bytes = max(disk_bytes, 0)

    memory_start = mems[0]

    bottleneck = _determine_bottleneck(
        avg_cpu=avg_cpu,
        peak_memory_mb=peak_mem,
        memory_start_mb=memory_start,
        net_bytes=net_bytes,
        disk_bytes=disk_bytes,
        duration_ms=duration_ms,
    )

    return ToolResourceSummary(
        tool_name=tool_name,
        avg_cpu_during=round(avg_cpu, 1),
        peak_cpu_during=round(peak_cpu, 1),
        avg_memory_during=round(avg_mem, 1),
        peak_memory_during=round(peak_mem, 1),
        net_bytes_during=net_bytes,
        disk_bytes_during=disk_bytes,
        resource_bottleneck=bottleneck,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_resources(
    system_samples_path: Path,
    trace: RunTrace,
) -> ResourceProfile:
    """Analyze system resource samples and correlate with agent tool calls.

    Parameters
    ----------
    system_samples_path:
        Path to the system-samples JSONL file from SystemSampler.
    trace:
        A validated RunTrace.

    Returns
    -------
    ResourceProfile
        Aggregate and per-tool resource summary.
    """
    samples = _load_system_samples(system_samples_path)

    # -- Global aggregates --
    if samples:
        cpus = [s.get("cpu_percent", 0.0) for s in samples]
        mems = [s.get("memory_rss_mb", 0.0) for s in samples]

        peak_cpu = max(cpus)
        avg_cpu = sum(cpus) / len(cpus)
        peak_mem = max(mems)
        avg_mem = sum(mems) / len(mems)

        last = samples[-1]
        total_net_sent = last.get("network_bytes_sent", 0) / (1024 * 1024)
        total_net_recv = last.get("network_bytes_recv", 0) / (1024 * 1024)
        total_disk_read = last.get("disk_read_bytes", 0) / (1024 * 1024)
        total_disk_write = last.get("disk_write_bytes", 0) / (1024 * 1024)
    else:
        peak_cpu = avg_cpu = peak_mem = avg_mem = 0.0
        total_net_sent = total_net_recv = 0.0
        total_disk_read = total_disk_write = 0.0

    # -- Per-tool correlation --
    per_tool: list[ToolResourceSummary] = []
    for tc in trace.tool_calls:
        window = _samples_in_window(samples, tc.start_ns, tc.end_ns)
        duration_ms = (tc.end_ns - tc.start_ns) / 1_000_000
        summary = _summarize_tool(tc.tool_name, window, duration_ms)
        per_tool.append(summary)

    return ResourceProfile(
        peak_cpu_percent=round(peak_cpu, 1),
        avg_cpu_percent=round(avg_cpu, 1),
        peak_memory_mb=round(peak_mem, 1),
        avg_memory_mb=round(avg_mem, 1),
        total_network_sent_mb=round(total_net_sent, 3),
        total_network_recv_mb=round(total_net_recv, 3),
        total_disk_read_mb=round(total_disk_read, 3),
        total_disk_write_mb=round(total_disk_write, 3),
        per_tool_resources=per_tool,
    )


# ---------------------------------------------------------------------------
# Aggregation by tool name
# ---------------------------------------------------------------------------


def aggregate_tool_resources(
    profile: ResourceProfile,
    trace: RunTrace,
) -> list[AggregatedToolResource]:
    """Aggregate per-call ToolResourceSummary entries by unique tool name.

    Returns one AggregatedToolResource per unique tool name, sorted by
    total_duration_ms descending.
    """
    from collections import Counter

    # Build a mapping from index → duration using the trace's tool_calls
    durations: list[float] = []
    for tc in trace.tool_calls:
        durations.append((tc.end_ns - tc.start_ns) / 1_000_000)

    # Group by tool name
    groups: dict[str, list[tuple[ToolResourceSummary, float]]] = {}
    for i, trs in enumerate(profile.per_tool_resources):
        dur = durations[i] if i < len(durations) else 0.0
        groups.setdefault(trs.tool_name, []).append((trs, dur))

    result: list[AggregatedToolResource] = []
    for tool_name, entries in groups.items():
        call_count = len(entries)
        total_dur = sum(d for _, d in entries)
        # Weighted average CPU (weighted by duration)
        total_weighted_cpu = sum(t.avg_cpu_during * d for t, d in entries)
        avg_cpu = round(total_weighted_cpu / total_dur, 1) if total_dur > 0 else 0.0
        peak_cpu = round(max(t.peak_cpu_during for t, _ in entries), 1)
        total_net = sum(t.net_bytes_during for t, _ in entries)
        total_disk = sum(t.disk_bytes_during for t, _ in entries)
        # Dominant bottleneck = most common across calls
        bottleneck_counts = Counter(t.resource_bottleneck for t, _ in entries)
        dominant = bottleneck_counts.most_common(1)[0][0]

        result.append(AggregatedToolResource(
            tool_name=tool_name,
            call_count=call_count,
            total_duration_ms=round(total_dur, 1),
            avg_cpu=avg_cpu,
            peak_cpu=peak_cpu,
            total_network_bytes=total_net,
            total_disk_bytes=total_disk,
            dominant_bottleneck=dominant,
        ))

    result.sort(key=lambda a: a.total_duration_ms, reverse=True)
    return result
