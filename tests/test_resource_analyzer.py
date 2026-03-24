"""Tests for resource_analyzer — fake samples + fake trace, per-tool correlation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import orjson

from agent_profiler.analyzer.resource_analyzer import (
    ResourceProfile,
    ToolResourceSummary,
    aggregate_tool_resources,
    analyze_resources,
    _determine_bottleneck,
)
from agent_profiler.schema.events import (
    AttemptEvent,
    EvaluationEvent,
    LoopIterationEvent,
    ModelCallEvent,
    RunEvent,
    ToolCallEvent,
)
from agent_profiler.schema.trace import RunTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NS = 1_000_000_000  # 1 second in ns


def _write_samples(path: Path, samples: list[dict]) -> None:
    with path.open("wb") as f:
        for s in samples:
            f.write(orjson.dumps(s) + b"\n")


def _build_trace_with_tools(
    tool_specs: list[tuple[str, int, int]],
) -> RunTrace:
    """Build a minimal RunTrace with tool calls at specified ns ranges.

    tool_specs: list of (tool_name, start_ns, end_ns)
    """
    run_id = uuid4()
    attempt_id = uuid4()

    run_start = min(s for _, s, _ in tool_specs)
    run_end = max(e for _, _, e in tool_specs)

    tool_calls = []
    loop_iters = []
    model_calls = []

    for i, (name, start, end) in enumerate(tool_specs):
        iter_id = uuid4()
        mc_start = max(start - NS // 10, run_start)
        mc_end = start

        loop_iters.append(
            LoopIterationEvent(
                event_type="loop_iteration",
                iteration_id=iter_id,
                attempt_id=attempt_id,
                iteration_number=i + 1,
                start_ns=mc_start,
                end_ns=end,
                has_tool_calls=True,
                iteration_type="reason_and_act",
            )
        )
        model_calls.append(
            ModelCallEvent(
                event_type="model_call",
                model_call_id=uuid4(),
                iteration_id=iter_id,
                model_provider="test",
                model_name="test-model",
                start_ns=mc_start,
                end_ns=mc_end,
                input_tokens=100,
                output_tokens=50,
            )
        )
        tool_calls.append(
            ToolCallEvent(
                event_type="tool_call",
                tool_call_id=uuid4(),
                iteration_id=iter_id,
                tool_name=name,
                tool_category="shell",
                start_ns=start,
                end_ns=end,
                outcome="success",
            )
        )

    return RunTrace(
        run=RunEvent(
            event_type="run",
            run_id=run_id,
            task_description="resource test",
            start_ns=run_start,
            end_ns=run_end,
            outcome="success",
            attempt_count=1,
            total_model_calls=len(model_calls),
            total_tool_calls=len(tool_calls),
            model_provider="test",
            model_name="test-model",
            sandbox_mode="off",
        ),
        attempts=[
            AttemptEvent(
                event_type="attempt",
                attempt_id=attempt_id,
                run_id=run_id,
                attempt_number=1,
                start_ns=run_start,
                end_ns=run_end,
                outcome="success",
            )
        ],
        loop_iterations=loop_iters,
        model_calls=model_calls,
        tool_calls=tool_calls,
        evaluations=[
            EvaluationEvent(
                event_type="evaluation",
                eval_id=uuid4(),
                attempt_id=attempt_id,
                evaluator="heuristic",
                passed=True,
                score=1.0,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnalyzeResources:
    def test_empty_samples(self, tmp_path: Path) -> None:
        samples_path = tmp_path / "empty.jsonl"
        samples_path.write_bytes(b"")

        trace = _build_trace_with_tools([("bash", 0, NS)])
        profile = analyze_resources(samples_path, trace)

        assert profile.peak_cpu_percent == 0.0
        assert profile.avg_cpu_percent == 0.0
        assert len(profile.per_tool_resources) == 1
        assert profile.per_tool_resources[0].resource_bottleneck == "none"

    def test_global_aggregates(self, tmp_path: Path) -> None:
        samples_path = tmp_path / "samples.jsonl"
        samples = [
            {
                "sample_type": "system",
                "timestamp_ns": 0,
                "cpu_percent": 40.0,
                "cpu_per_core": [40.0],
                "memory_rss_mb": 100.0,
                "memory_system_percent": 50.0,
                "network_bytes_sent": 0,
                "network_bytes_recv": 0,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "active_threads": 4,
                "open_files": 10,
            },
            {
                "sample_type": "system",
                "timestamp_ns": NS,
                "cpu_percent": 80.0,
                "cpu_per_core": [80.0],
                "memory_rss_mb": 200.0,
                "memory_system_percent": 60.0,
                "network_bytes_sent": 1024,
                "network_bytes_recv": 2048,
                "disk_read_bytes": 512,
                "disk_write_bytes": 256,
                "active_threads": 6,
                "open_files": 12,
            },
        ]
        _write_samples(samples_path, samples)
        trace = _build_trace_with_tools([("bash", 0, NS)])

        profile = analyze_resources(samples_path, trace)
        assert profile.peak_cpu_percent == 80.0
        assert profile.avg_cpu_percent == 60.0
        assert profile.peak_memory_mb == 200.0
        assert profile.avg_memory_mb == 150.0

    def test_per_tool_correlation(self, tmp_path: Path) -> None:
        """Samples are correctly attributed to tool windows."""
        samples_path = tmp_path / "samples.jsonl"
        # Tool 1: 0-2s, Tool 2: 4s-6s
        # Samples at t=0,1,2 are high CPU; t=3,4,5,6 are low CPU
        samples = []
        for t in range(7):
            samples.append({
                "sample_type": "system",
                "timestamp_ns": t * NS,
                "cpu_percent": 90.0 if t <= 2 else 20.0,
                "cpu_per_core": [90.0 if t <= 2 else 20.0],
                "memory_rss_mb": 100.0,
                "memory_system_percent": 50.0,
                "network_bytes_sent": t * 100,
                "network_bytes_recv": t * 200,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "active_threads": 4,
                "open_files": 10,
            })
        _write_samples(samples_path, samples)

        trace = _build_trace_with_tools([
            ("compile", 0, 2 * NS),
            ("curl", 4 * NS, 6 * NS),
        ])
        profile = analyze_resources(samples_path, trace)

        assert len(profile.per_tool_resources) == 2
        compile_res = profile.per_tool_resources[0]
        curl_res = profile.per_tool_resources[1]

        # compile runs during high-CPU period (samples at t=0,1,2 → all 90%)
        assert compile_res.tool_name == "compile"
        assert compile_res.avg_cpu_during == 90.0

        # curl runs during low-CPU period (samples at t=4,5,6 → all 20%)
        assert curl_res.tool_name == "curl"
        assert curl_res.avg_cpu_during == 20.0

    def test_markers_are_skipped(self, tmp_path: Path) -> None:
        samples_path = tmp_path / "samples.jsonl"
        samples = [
            {"sample_type": "marker", "timestamp_ns": 0, "event_name": "start"},
            {
                "sample_type": "system",
                "timestamp_ns": NS,
                "cpu_percent": 50.0,
                "cpu_per_core": [50.0],
                "memory_rss_mb": 100.0,
                "memory_system_percent": 50.0,
                "network_bytes_sent": 0,
                "network_bytes_recv": 0,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "active_threads": 4,
                "open_files": 10,
            },
        ]
        _write_samples(samples_path, samples)
        trace = _build_trace_with_tools([("bash", 0, NS)])

        profile = analyze_resources(samples_path, trace)
        # Only 1 system sample, so avg == peak
        assert profile.peak_cpu_percent == 50.0
        assert profile.avg_cpu_percent == 50.0


class TestDetermineBottleneck:
    def test_cpu_bottleneck(self) -> None:
        assert _determine_bottleneck(
            avg_cpu=85.0,
            peak_memory_mb=100.0,
            memory_start_mb=90.0,
            net_bytes=0,
            disk_bytes=0,
            duration_ms=1000,
        ) == "cpu"

    def test_memory_bottleneck(self) -> None:
        assert _determine_bottleneck(
            avg_cpu=50.0,
            peak_memory_mb=150.0,
            memory_start_mb=100.0,  # 50% increase > 20% threshold
            net_bytes=0,
            disk_bytes=0,
            duration_ms=1000,
        ) == "memory"

    def test_network_bottleneck(self) -> None:
        assert _determine_bottleneck(
            avg_cpu=50.0,
            peak_memory_mb=100.0,
            memory_start_mb=95.0,
            net_bytes=2_000_000,  # 2MB > 1MB threshold
            disk_bytes=0,
            duration_ms=2000,  # 2s > 1s threshold
        ) == "network"

    def test_disk_bottleneck(self) -> None:
        assert _determine_bottleneck(
            avg_cpu=50.0,
            peak_memory_mb=100.0,
            memory_start_mb=95.0,
            net_bytes=0,
            disk_bytes=15_000_000,  # 15MB > 10MB threshold
            duration_ms=1000,
        ) == "disk"

    def test_no_bottleneck(self) -> None:
        assert _determine_bottleneck(
            avg_cpu=50.0,
            peak_memory_mb=100.0,
            memory_start_mb=95.0,
            net_bytes=100,
            disk_bytes=100,
            duration_ms=1000,
        ) == "none"

    def test_priority_cpu_over_memory(self) -> None:
        """CPU is checked first even if memory also qualifies."""
        assert _determine_bottleneck(
            avg_cpu=90.0,
            peak_memory_mb=200.0,
            memory_start_mb=100.0,
            net_bytes=0,
            disk_bytes=0,
            duration_ms=1000,
        ) == "cpu"


class TestResourceProfileModel:
    def test_resource_profile_roundtrip(self) -> None:
        profile = ResourceProfile(
            peak_cpu_percent=95.0,
            avg_cpu_percent=60.0,
            peak_memory_mb=512.0,
            avg_memory_mb=256.0,
            total_network_sent_mb=1.5,
            total_network_recv_mb=3.0,
            total_disk_read_mb=0.0,
            total_disk_write_mb=0.5,
            per_tool_resources=[
                ToolResourceSummary(
                    tool_name="bash",
                    avg_cpu_during=85.0,
                    peak_cpu_during=95.0,
                    avg_memory_during=300.0,
                    peak_memory_during=512.0,
                    net_bytes_during=1000,
                    disk_bytes_during=0,
                    resource_bottleneck="cpu",
                )
            ],
        )
        data = profile.model_dump()
        restored = ResourceProfile.model_validate(data)
        assert restored.peak_cpu_percent == 95.0
        assert restored.per_tool_resources[0].resource_bottleneck == "cpu"


class TestAggregateToolResources:
    def test_aggregates_multiple_calls_to_same_tool(self, tmp_path: Path) -> None:
        """Two bash calls should be merged into one aggregated entry."""
        samples_path = tmp_path / "samples.jsonl"
        # 5 seconds of samples
        samples = []
        for t in range(6):
            samples.append({
                "sample_type": "system",
                "timestamp_ns": t * NS,
                "cpu_percent": 90.0 if t <= 2 else 20.0,
                "cpu_per_core": [90.0 if t <= 2 else 20.0],
                "memory_rss_mb": 100.0,
                "memory_system_percent": 50.0,
                "network_bytes_sent": t * 500_000,
                "network_bytes_recv": t * 500_000,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "active_threads": 4,
                "open_files": 10,
            })
        _write_samples(samples_path, samples)

        # Two bash calls: 0-2s (high CPU) and 3-5s (low CPU)
        trace = _build_trace_with_tools([
            ("bash", 0, 2 * NS),
            ("bash", 3 * NS, 5 * NS),
        ])
        profile = analyze_resources(samples_path, trace)
        aggregated = aggregate_tool_resources(profile, trace)

        assert len(aggregated) == 1
        agg = aggregated[0]
        assert agg.tool_name == "bash"
        assert agg.call_count == 2
        assert agg.total_duration_ms == 4000.0  # 2s + 2s
        assert agg.peak_cpu == 90.0  # max across both calls

    def test_multiple_tools_sorted_by_duration(self, tmp_path: Path) -> None:
        """Different tools should produce separate entries, sorted by duration desc."""
        samples_path = tmp_path / "samples.jsonl"
        samples = []
        for t in range(11):
            samples.append({
                "sample_type": "system",
                "timestamp_ns": t * NS,
                "cpu_percent": 30.0,
                "cpu_per_core": [30.0],
                "memory_rss_mb": 100.0,
                "memory_system_percent": 50.0,
                "network_bytes_sent": 0,
                "network_bytes_recv": 0,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "active_threads": 4,
                "open_files": 10,
            })
        _write_samples(samples_path, samples)

        # browser: 0-8s (long), bash: 9-10s (short)
        trace = _build_trace_with_tools([
            ("browser", 0, 8 * NS),
            ("bash", 9 * NS, 10 * NS),
        ])
        profile = analyze_resources(samples_path, trace)
        aggregated = aggregate_tool_resources(profile, trace)

        assert len(aggregated) == 2
        assert aggregated[0].tool_name == "browser"  # longest first
        assert aggregated[1].tool_name == "bash"

    def test_dominant_bottleneck_is_most_common(self, tmp_path: Path) -> None:
        """Dominant bottleneck should be the most frequent across calls."""
        samples_path = tmp_path / "samples.jsonl"
        # Create samples: first two calls high CPU, third call low CPU with network
        samples = []
        for t in range(10):
            cpu = 90.0 if t < 6 else 10.0
            samples.append({
                "sample_type": "system",
                "timestamp_ns": t * NS,
                "cpu_percent": cpu,
                "cpu_per_core": [cpu],
                "memory_rss_mb": 100.0,
                "memory_system_percent": 50.0,
                "network_bytes_sent": t * 100,
                "network_bytes_recv": t * 100,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "active_threads": 4,
                "open_files": 10,
            })
        _write_samples(samples_path, samples)

        # 3 bash calls: first two during high CPU, third during low
        trace = _build_trace_with_tools([
            ("bash", 0, 2 * NS),
            ("bash", 3 * NS, 5 * NS),
            ("bash", 7 * NS, 9 * NS),
        ])
        profile = analyze_resources(samples_path, trace)
        aggregated = aggregate_tool_resources(profile, trace)

        assert len(aggregated) == 1
        # Two calls have cpu bottleneck, one has none → dominant = cpu
        assert aggregated[0].dominant_bottleneck == "cpu"

    def test_empty_per_tool_resources(self) -> None:
        """An empty profile should return an empty list."""
        profile = ResourceProfile(
            peak_cpu_percent=0.0,
            avg_cpu_percent=0.0,
            peak_memory_mb=0.0,
            avg_memory_mb=0.0,
            total_network_sent_mb=0.0,
            total_network_recv_mb=0.0,
            total_disk_read_mb=0.0,
            total_disk_write_mb=0.0,
            per_tool_resources=[],
        )
        trace = _build_trace_with_tools([("bash", 0, NS)])
        aggregated = aggregate_tool_resources(profile, trace)
        assert aggregated == []
