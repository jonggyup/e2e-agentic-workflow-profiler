"""Lightweight system metrics sampler for agent profiling.

Samples CPU, memory, network, disk, and thread/file-descriptor metrics at a
configurable interval and writes them as JSONL to a companion file.  Named
markers can be inserted for correlation with agent events.

Uses psutil for all metrics — works on macOS without root.
Target overhead: <1% CPU at 500 ms intervals.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import psutil


class SystemSampler:
    """Samples system metrics in a background thread and writes JSONL."""

    def __init__(
        self,
        output_path: Path,
        interval_ms: int = 500,
        pid: int | None = None,
    ) -> None:
        """Initialise the sampler.

        Parameters
        ----------
        output_path:
            Path for the output JSONL file.
        interval_ms:
            Sampling interval in milliseconds (default 500).
        pid:
            Process ID to monitor for per-process metrics. Defaults to the
            current process.
        """
        self._output_path = Path(output_path)
        self._interval_s = interval_ms / 1000.0
        self._pid = pid or psutil.Process().pid
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._lock = threading.Lock()

        # Baseline counters (set on start)
        self._net_base_sent: int = 0
        self._net_base_recv: int = 0
        self._disk_base_read: int = 0
        self._disk_base_write: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start sampling in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._output_path.open("ab")

        # Snapshot baseline counters
        net = psutil.net_io_counters()
        self._net_base_sent = net.bytes_sent
        self._net_base_recv = net.bytes_recv

        disk = psutil.disk_io_counters()
        if disk is not None:
            self._disk_base_read = disk.read_bytes
            self._disk_base_write = disk.write_bytes

        # Prime psutil's per-core CPU (first call returns 0.0)
        psutil.cpu_percent(percpu=True)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
            name="system-sampler",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and flush the output file."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def mark_event(self, event_name: str) -> None:
        """Insert a named marker at the current timestamp."""
        marker: dict[str, Any] = {
            "sample_type": "marker",
            "timestamp_ns": time.monotonic_ns(),
            "event_name": event_name,
        }
        self._write_line(marker)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """Run in background thread: collect samples until stopped."""
        while not self._stop_event.is_set():
            try:
                sample = self._collect_sample()
                self._write_line(sample)
            except Exception:
                pass  # Best-effort — don't crash the sampler
            self._stop_event.wait(self._interval_s)

    def _collect_sample(self) -> dict[str, Any]:
        """Collect one system metrics sample."""
        timestamp_ns = time.monotonic_ns()
        wall_time_iso = datetime.now(timezone.utc).isoformat()

        # CPU
        cpu_percent = psutil.cpu_percent()
        cpu_per_core = psutil.cpu_percent(percpu=True)

        # Memory — per-process and system
        memory_rss_mb = 0.0
        active_threads = 0
        open_files = 0
        try:
            proc = psutil.Process(self._pid)
            mem_info = proc.memory_info()
            memory_rss_mb = mem_info.rss / (1024 * 1024)
            active_threads = proc.num_threads()
            try:
                open_files = len(proc.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        vm = psutil.virtual_memory()
        memory_system_percent = vm.percent

        # Network (cumulative since start)
        net = psutil.net_io_counters()
        network_bytes_sent = net.bytes_sent - self._net_base_sent
        network_bytes_recv = net.bytes_recv - self._net_base_recv

        # Disk (cumulative since start)
        disk = psutil.disk_io_counters()
        if disk is not None:
            disk_read_bytes = disk.read_bytes - self._disk_base_read
            disk_write_bytes = disk.write_bytes - self._disk_base_write
        else:
            disk_read_bytes = 0
            disk_write_bytes = 0

        return {
            "sample_type": "system",
            "timestamp_ns": timestamp_ns,
            "wall_time_iso": wall_time_iso,
            "cpu_percent": cpu_percent,
            "cpu_per_core": cpu_per_core,
            "memory_rss_mb": round(memory_rss_mb, 1),
            "memory_system_percent": round(memory_system_percent, 1),
            "network_bytes_sent": network_bytes_sent,
            "network_bytes_recv": network_bytes_recv,
            "disk_read_bytes": disk_read_bytes,
            "disk_write_bytes": disk_write_bytes,
            "active_threads": active_threads,
            "open_files": open_files,
        }

    def _write_line(self, data: dict[str, Any]) -> None:
        """Thread-safe write of a single JSONL line."""
        line = orjson.dumps(data) + b"\n"
        with self._lock:
            if self._file is not None:
                self._file.write(line)
                self._file.flush()
