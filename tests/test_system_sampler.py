"""Tests for SystemSampler — start/stop, sample writing, marker insertion."""

from __future__ import annotations

import time
from pathlib import Path

import orjson
import pytest

from agent_profiler.collector.system_sampler import SystemSampler


@pytest.fixture
def samples_path(tmp_path: Path) -> Path:
    return tmp_path / "system_samples.jsonl"


class TestSystemSamplerStartStop:
    def test_creates_output_file(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        time.sleep(0.3)
        sampler.stop()
        assert samples_path.exists()

    def test_writes_system_samples(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        time.sleep(0.5)
        sampler.stop()

        lines = [
            orjson.loads(line)
            for line in samples_path.read_bytes().strip().split(b"\n")
            if line.strip()
        ]
        system_samples = [s for s in lines if s.get("sample_type") == "system"]
        assert len(system_samples) >= 2

    def test_sample_has_expected_fields(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        time.sleep(0.3)
        sampler.stop()

        lines = [
            orjson.loads(line)
            for line in samples_path.read_bytes().strip().split(b"\n")
            if line.strip()
        ]
        system_samples = [s for s in lines if s.get("sample_type") == "system"]
        assert len(system_samples) >= 1

        sample = system_samples[0]
        expected_fields = {
            "sample_type",
            "timestamp_ns",
            "wall_time_iso",
            "cpu_percent",
            "cpu_per_core",
            "memory_rss_mb",
            "memory_system_percent",
            "network_bytes_sent",
            "network_bytes_recv",
            "disk_read_bytes",
            "disk_write_bytes",
            "active_threads",
            "open_files",
        }
        assert expected_fields.issubset(sample.keys())

    def test_cpu_per_core_is_list(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        time.sleep(0.3)
        sampler.stop()

        lines = [
            orjson.loads(line)
            for line in samples_path.read_bytes().strip().split(b"\n")
            if line.strip()
        ]
        system_samples = [s for s in lines if s.get("sample_type") == "system"]
        assert isinstance(system_samples[0]["cpu_per_core"], list)
        assert len(system_samples[0]["cpu_per_core"]) > 0

    def test_stop_is_idempotent(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        time.sleep(0.2)
        sampler.stop()
        sampler.stop()  # Should not raise

    def test_start_when_already_running(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        sampler.start()  # Should not create a second thread
        time.sleep(0.2)
        sampler.stop()


class TestSystemSamplerMarker:
    def test_marker_is_written(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        time.sleep(0.15)
        sampler.mark_event("tool_call_start:bash")
        time.sleep(0.15)
        sampler.stop()

        lines = [
            orjson.loads(line)
            for line in samples_path.read_bytes().strip().split(b"\n")
            if line.strip()
        ]
        markers = [s for s in lines if s.get("sample_type") == "marker"]
        assert len(markers) >= 1
        assert markers[0]["event_name"] == "tool_call_start:bash"
        assert "timestamp_ns" in markers[0]

    def test_marker_without_start(self, samples_path: Path) -> None:
        """Markers can be written even before start (output file opened on start)."""
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        sampler.mark_event("pre_event")
        sampler.stop()

        lines = [
            orjson.loads(line)
            for line in samples_path.read_bytes().strip().split(b"\n")
            if line.strip()
        ]
        markers = [s for s in lines if s.get("sample_type") == "marker"]
        assert any(m["event_name"] == "pre_event" for m in markers)


class TestSystemSamplerTimestamps:
    def test_timestamps_are_monotonic(self, samples_path: Path) -> None:
        sampler = SystemSampler(output_path=samples_path, interval_ms=100)
        sampler.start()
        time.sleep(0.5)
        sampler.stop()

        lines = [
            orjson.loads(line)
            for line in samples_path.read_bytes().strip().split(b"\n")
            if line.strip()
        ]
        timestamps = [s["timestamp_ns"] for s in lines if "timestamp_ns" in s]
        assert len(timestamps) >= 2
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]
