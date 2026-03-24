"""CLI entry point for agent-profiler.

Commands:
  agent-profiler analyze <trace_path> [--program-tool <name>] [--json]
  agent-profiler compare <baseline.jsonl> <current.jsonl> [--program-tool <name>]
  agent-profiler validate <trace_path>
  agent-profiler demo <scenario_name|all>
  agent-profiler monitor [command] [--manual] [--list-sessions] [--session <path>] [--task <desc>]
"""

from __future__ import annotations

import datetime
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console
from rich.table import Table
from rich import box

from agent_profiler.analyzer import ProfileMetrics, compute_metrics
from agent_profiler.analyzer.comparison import RunComparison, compare_runs
from agent_profiler.collector.openclaw_converter import convert_openclaw_session, write_trace
from agent_profiler.schema.trace import load_trace, TraceValidationError

app = typer.Typer(
    name="agent-profiler",
    help="Correctness-aware offline profiler for on-device agentic workflows.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DOMINANT_LABELS: dict[str, str] = {
    "model": "agent_reasoning",
    "tool": "tool_execution",
    "program": "program_runtime",
    "balanced": "balanced",
}

_BOTTLENECK_DESCRIPTIONS: dict[str, str] = {
    "retry_overhead": "of wall time was spent on failed attempts",
    "reasoning_loop": "of wall time on a looping reasoning failure",
    "agent_reasoning": "of wall time was model inference (reasoning-heavy)",
    "program_runtime": "of wall time was the program under test executing",
    "tool_slowness": "of wall time was spent in tool calls",
    "framework_overhead": "of wall time was unaccounted gap/framework overhead",
    "balanced": "no single phase dominated — workload is balanced",
}


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round(part / total * 100, 1)


def _fmt_ms(ms: float) -> str:
    return f"{ms:,.0f} ms"


def _bottleneck_pct(bottleneck: str, m) -> float:
    """Return the relevant percentage for the primary bottleneck (against active wall time)."""
    active = m.active_wall_ms
    mapping = {
        "retry_overhead": m.retry_waste_ms,
        "reasoning_loop": m.agent_overhead_ms,
        "agent_reasoning": m.agent_overhead_ms,
        "program_runtime": m.program_runtime_ms,
        "tool_slowness": m.tool_execution_ms,
        "framework_overhead": m.gap_time_ms - m.user_idle_ms,
        "balanced": 0.0,
    }
    return _pct(mapping.get(bottleneck, 0.0), active)


def _format_verdict(metrics, trace=None) -> str:
    """Build the primary bottleneck verdict line, optionally with resource detail."""
    from agent_profiler.analyzer.resource_analyzer import aggregate_tool_resources

    bn = metrics.primary_bottleneck
    pct = _bottleneck_pct(bn, metrics)
    desc = _BOTTLENECK_DESCRIPTIONS.get(bn, "")
    if metrics.user_idle_ms > 0:
        desc = desc.replace("of wall time", "of active wall time")

    if bn == "balanced":
        verdict = f"Primary bottleneck: {bn} — {desc}"
    else:
        verdict = f"Primary bottleneck: {bn} — {pct}% {desc}"

    # Append resource detail if resource data is available
    if metrics.resource_profile is not None and trace is not None:
        aggregated = aggregate_tool_resources(metrics.resource_profile, trace)
        # Find the dominant resource bottleneck across all tools (non-"none")
        resource_tools = [a for a in aggregated if a.dominant_bottleneck != "none"]
        if resource_tools:
            # Pick the tool with the most total duration among bottlenecked tools
            top = max(resource_tools, key=lambda a: a.total_duration_ms)
            detail_parts: list[str] = []
            if top.dominant_bottleneck == "network":
                net_mb = top.total_network_bytes / (1024 * 1024)
                detail_parts.append(f"network-bound: {net_mb:.1f} MB transferred")
            elif top.dominant_bottleneck == "cpu":
                detail_parts.append(f"cpu-bound: peak {top.peak_cpu:.1f}%")
            elif top.dominant_bottleneck == "memory":
                detail_parts.append("memory-bound")
            elif top.dominant_bottleneck == "disk":
                disk_mb = top.total_disk_bytes / (1024 * 1024)
                detail_parts.append(f"disk-bound: {disk_mb:.1f} MB I/O")
            if detail_parts:
                verdict += f" ({', '.join(detail_parts)})"

    return verdict


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    trace_path: Path = typer.Argument(..., help="Path to .jsonl trace file"),
    program_tool: Optional[str] = typer.Option(
        None, "--program-tool", help="Tool name to treat as program_under_test"
    ),
    system_samples: Optional[Path] = typer.Option(
        None, "--system-samples", help="Path to system-samples JSONL from SystemSampler"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Output ProfileMetrics as JSON instead of table"
    ),
) -> None:
    """Analyze a trace file and report profiling metrics."""
    if not trace_path.exists():
        err_console.print(f"[red]Error:[/red] File not found: {trace_path}")
        raise typer.Exit(1)

    try:
        trace = load_trace(trace_path, strict=True)
    except TraceValidationError as exc:
        err_console.print(f"[red]Trace validation failed:[/red]\n{exc}")
        raise typer.Exit(1)

    metrics = compute_metrics(
        trace,
        program_tool_name=program_tool,
        system_samples_path=system_samples,
    )

    if as_json:
        sys.stdout.buffer.write(
            orjson.dumps(metrics.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )
        sys.stdout.buffer.write(b"\n")
        return

    # ---- Metrics table ----
    table = Table(
        title=f"Profile: {trace_path.name}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Category", style="dim")

    rows = [
        # Timing
        ("e2e_wall_ms", _fmt_ms(metrics.e2e_wall_ms), "timing"),
        ("active_wall_ms", _fmt_ms(metrics.active_wall_ms), "timing"),
        ("user_idle_ms", _fmt_ms(metrics.user_idle_ms), "timing"),
        ("total_model_time_ms", _fmt_ms(metrics.total_model_time_ms), "timing"),
        ("total_tool_time_ms", _fmt_ms(metrics.total_tool_time_ms), "timing"),
        ("agent_overhead_ms", _fmt_ms(metrics.agent_overhead_ms), "timing"),
        ("tool_execution_ms", _fmt_ms(metrics.tool_execution_ms), "timing"),
        ("program_runtime_ms", _fmt_ms(metrics.program_runtime_ms), "timing"),
        ("retry_waste_ms", _fmt_ms(metrics.retry_waste_ms), "timing"),
        ("gap_time_ms", _fmt_ms(metrics.gap_time_ms), "timing"),
        # Correctness
        ("first_pass_success", str(metrics.first_pass_success), "correctness"),
        ("attempt_count", str(metrics.attempt_count), "correctness"),
        (
            "failure_categories",
            ", ".join(metrics.failure_categories) or "—",
            "correctness",
        ),
        (
            "correctness_score",
            f"{metrics.correctness_score:.2f}" if metrics.correctness_score is not None else "—",
            "correctness",
        ),
        # Cost
        ("total_input_tokens", f"{metrics.total_input_tokens:,}", "cost"),
        ("total_output_tokens", f"{metrics.total_output_tokens:,}", "cost"),
        ("estimated_cost_usd", f"${metrics.estimated_cost_usd:.4f}", "cost"),
        ("wasted_tokens", f"{metrics.wasted_tokens:,}", "cost"),
        # Derived
        ("primary_bottleneck", metrics.primary_bottleneck, "derived"),
    ]

    prev_category = None
    for name, value, category in rows:
        if category != prev_category and prev_category is not None:
            table.add_section()
        prev_category = category
        table.add_row(name, value, category)

    console.print(table)

    # ---- Idle note ----
    if metrics.idle_percentage > 50:
        console.print(
            f"\n[bold magenta]Note: {metrics.idle_percentage:.1f}% of wall time "
            f"was user idle — bottleneck analysis uses active time only "
            f"({_fmt_ms(metrics.active_wall_ms)})[/bold magenta]"
        )

    # ---- Attempt breakdown ----
    console.print("\n[bold]Attempt Breakdown[/bold]")
    for s in metrics.per_attempt_summary:
        outcome_str = "SUCCESS" if s.outcome == "success" else f"FAIL ({s.failure_category})"
        dominant = _DOMINANT_LABELS.get(s.dominant_cost, s.dominant_cost)
        line = (
            f"  {s.attempt_number}. {outcome_str} "
            f"— {s.wall_ms:,.0f} ms "
            f"— dominant: {dominant}"
        )
        color = "green" if s.outcome == "success" else "red"
        console.print(f"[{color}]{line}[/{color}]")

    # ---- Token summary (Phase 4) ----
    _print_token_summary(console, metrics)

    # ---- Resource Usage (Phase 3) ----
    if metrics.resource_profile is not None:
        _print_resource_summary(console, metrics.resource_profile, trace)

    # ---- Verdict ----
    verdict = _format_verdict(metrics, trace if metrics.resource_profile else None)
    console.print(f"\n[bold yellow]{verdict}[/bold yellow]")


# ---------------------------------------------------------------------------
# Resource / token display helpers
# ---------------------------------------------------------------------------


def _print_token_summary(console: Console, metrics) -> None:
    """Print token summary section."""
    if metrics.total_input_tokens == 0 and metrics.total_output_tokens == 0:
        return

    console.print("\n[bold]Token Summary[/bold]")
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Input tokens", f"{metrics.total_input_tokens:,}")
    table.add_row("Output tokens", f"{metrics.total_output_tokens:,}")
    if metrics.cache_read_tokens > 0:
        table.add_row("Cache read tokens", f"{metrics.cache_read_tokens:,}")
    if metrics.cache_write_tokens > 0:
        table.add_row("Cache write tokens", f"{metrics.cache_write_tokens:,}")
    table.add_row("Estimated cost", f"${metrics.estimated_cost_usd:.4f}")
    if metrics.context_efficiency is not None:
        table.add_row("Context efficiency", f"{metrics.context_efficiency:.3f}")
    if metrics.wasted_tokens > 0:
        total = metrics.total_input_tokens + metrics.total_output_tokens
        waste_pct = (metrics.wasted_tokens / total * 100) if total > 0 else 0
        table.add_row("Wasted tokens", f"{metrics.wasted_tokens:,} ({waste_pct:.0f}%)")
        if waste_pct > 20:
            console.print(
                f"  [bold red]Warning: {waste_pct:.0f}% of tokens were wasted "
                f"on failed attempts[/bold red]"
            )

    console.print(table)


def _print_resource_summary(console: Console, rp, trace=None) -> None:
    """Print resource usage and per-tool resource sections."""
    from agent_profiler.analyzer.resource_analyzer import aggregate_tool_resources

    console.print("\n[bold]Resource Usage[/bold]")
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Peak CPU", f"{rp.peak_cpu_percent:.1f}%")
    table.add_row("Avg CPU", f"{rp.avg_cpu_percent:.1f}%")
    table.add_row("Peak memory", f"{rp.peak_memory_mb:.1f} MB")
    table.add_row("Avg memory", f"{rp.avg_memory_mb:.1f} MB")
    table.add_row("Network sent", f"{rp.total_network_sent_mb:.3f} MB")
    table.add_row("Network recv", f"{rp.total_network_recv_mb:.3f} MB")
    table.add_row("Disk read", f"{rp.total_disk_read_mb:.3f} MB")
    table.add_row("Disk write", f"{rp.total_disk_write_mb:.3f} MB")
    console.print(table)

    if rp.per_tool_resources and trace is not None:
        aggregated = aggregate_tool_resources(rp, trace)
        if aggregated:
            console.print("\n[bold]Per-Tool Resource Breakdown[/bold]")
            tool_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
            tool_table.add_column("Tool")
            tool_table.add_column("Duration", justify="right")
            tool_table.add_column("Avg CPU", justify="right")
            tool_table.add_column("Peak CPU", justify="right")
            tool_table.add_column("Network", justify="right")
            tool_table.add_column("Bottleneck")

            for a in aggregated:
                net_mb = a.total_network_bytes / (1024 * 1024)
                tool_table.add_row(
                    a.tool_name,
                    _fmt_ms(a.total_duration_ms),
                    f"{a.avg_cpu:.1f}%",
                    f"{a.peak_cpu:.1f}%",
                    f"{net_mb:.1f} MB",
                    a.dominant_bottleneck,
                )
            console.print(tool_table)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _fmt_delta(value: float, unit: str, is_cost: bool = False) -> str:
    """Format a delta value with arrow indicator for the comparison table.

    Green ▼ = improvement (negative for time/cost/tokens, positive for correctness).
    Red ▲ = regression.
    """
    if value == 0:
        return "0"
    if is_cost:
        sign = "+" if value > 0 else "-"
        return f"{sign}${abs(value):.2f}"
    return f"{value:+,.0f}" if abs(value) >= 1 else f"{value:+.2f}"


def _delta_with_pct(delta: float, baseline: float, unit: str = "ms") -> str:
    """Format delta with percentage in parentheses, colored arrows."""
    if delta == 0 and baseline == 0:
        return "0"
    if baseline == 0:
        return f"{delta:+,.0f}"
    pct = delta / abs(baseline) * 100
    arrow = "[green]▼[/green]" if delta < 0 else "[red]▲[/red]" if delta > 0 else ""
    return f"{delta:+,.0f} ({arrow}{abs(pct):.0f}%)"


def _print_comparison_table(
    console: Console,
    comp: RunComparison,
    baseline_metrics: ProfileMetrics,
    current_metrics: ProfileMetrics,
) -> None:
    """Print the side-by-side comparison table."""
    table = Table(
        title="Run Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column(comp.baseline_label, justify="right")
    table.add_column(comp.current_label, justify="right")
    table.add_column("Delta", justify="right")

    bm = baseline_metrics
    cm = current_metrics

    rows = [
        (
            "active_wall_ms",
            _fmt_ms(bm.active_wall_ms),
            _fmt_ms(cm.active_wall_ms),
            _delta_with_pct(comp.active_wall_delta_ms, bm.active_wall_ms),
        ),
        (
            "total_model_time_ms",
            _fmt_ms(bm.total_model_time_ms),
            _fmt_ms(cm.total_model_time_ms),
            _delta_with_pct(comp.model_time_delta_ms, bm.total_model_time_ms),
        ),
        (
            "total_tool_time_ms",
            _fmt_ms(bm.total_tool_time_ms),
            _fmt_ms(cm.total_tool_time_ms),
            _delta_with_pct(comp.tool_time_delta_ms, bm.total_tool_time_ms),
        ),
        (
            "retry_waste_ms",
            _fmt_ms(bm.retry_waste_ms),
            _fmt_ms(cm.retry_waste_ms),
            _delta_with_pct(comp.retry_waste_delta_ms, bm.retry_waste_ms),
        ),
        (
            "attempt_count",
            str(bm.attempt_count),
            str(cm.attempt_count),
            f"{comp.attempt_count_delta:+d}" if comp.attempt_count_delta != 0 else "0",
        ),
        (
            "total_tokens",
            f"{bm.total_input_tokens + bm.total_output_tokens:,}",
            f"{cm.total_input_tokens + cm.total_output_tokens:,}",
            _delta_with_pct(
                comp.token_delta,
                bm.total_input_tokens + bm.total_output_tokens,
            ),
        ),
        (
            "estimated_cost_usd",
            f"${bm.estimated_cost_usd:.4f}",
            f"${cm.estimated_cost_usd:.4f}",
            _fmt_delta(comp.cost_delta_usd, "$", is_cost=True),
        ),
        (
            "primary_bottleneck",
            bm.primary_bottleneck,
            cm.primary_bottleneck,
            "changed" if comp.bottleneck_changed else "same",
        ),
    ]

    for name, bval, cval, delta in rows:
        table.add_row(name, bval, cval, delta)

    console.print(table)


@app.command()
def compare(
    baseline_path: Path = typer.Argument(..., help="Path to baseline trace .jsonl"),
    current_path: Path = typer.Argument(..., help="Path to current trace .jsonl"),
    program_tool: Optional[str] = typer.Option(
        None, "--program-tool", help="Tool name to treat as program_under_test"
    ),
) -> None:
    """Compare two trace files and show what changed."""
    for label, p in [("Baseline", baseline_path), ("Current", current_path)]:
        if not p.exists():
            err_console.print(f"[red]Error:[/red] {label} file not found: {p}")
            raise typer.Exit(1)

    try:
        baseline_trace = load_trace(baseline_path, strict=True)
    except TraceValidationError as exc:
        err_console.print(f"[red]Baseline trace validation failed:[/red]\n{exc}")
        raise typer.Exit(1)

    try:
        current_trace = load_trace(current_path, strict=True)
    except TraceValidationError as exc:
        err_console.print(f"[red]Current trace validation failed:[/red]\n{exc}")
        raise typer.Exit(1)

    baseline_metrics = compute_metrics(baseline_trace, program_tool_name=program_tool)
    current_metrics = compute_metrics(current_trace, program_tool_name=program_tool)

    comp = compare_runs(
        baseline_metrics,
        current_metrics,
        baseline_label=baseline_path.stem,
        current_label=current_path.stem,
    )

    _print_comparison_table(console, comp, baseline_metrics, current_metrics)
    console.print(f"\n[bold]{comp.summary}[/bold]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    trace_path: Path = typer.Argument(..., help="Path to .jsonl trace file"),
) -> None:
    """Validate a trace file and report any errors or warnings."""
    if not trace_path.exists():
        err_console.print(f"[red]Error:[/red] File not found: {trace_path}")
        raise typer.Exit(1)

    try:
        trace = load_trace(trace_path, strict=True)
    except TraceValidationError as exc:
        err_console.print("[red]Validation errors:[/red]")
        for e in exc.errors:
            err_console.print(f"  [red]✗[/red] {e}")
        raise typer.Exit(1)

    total_events = (
        1  # run
        + len(trace.attempts)
        + len(trace.loop_iterations)
        + len(trace.model_calls)
        + len(trace.tool_calls)
        + len(trace.evaluations)
    )
    console.print(f"[green]✓[/green] Trace is valid ({total_events} events)")

    if trace.warnings:
        console.print(f"\n[yellow]Warnings ({len(trace.warnings)}):[/yellow]")
        for w in trace.warnings:
            ctx = f" [{w.context}]" if w.context else ""
            console.print(f"  [yellow]⚠[/yellow] {w.message}{ctx}")


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def _load_synthetic_module():
    """Locate and load demos/synthetic_run.py via importlib."""
    candidates = [
        Path(__file__).parent.parent.parent / "demos" / "synthetic_run.py",
        Path.cwd() / "demos" / "synthetic_run.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("_synthetic_run", p)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise FileNotFoundError(
        "Cannot find demos/synthetic_run.py. Run agent-profiler from the project root."
    )


@app.command()
def demo(
    scenario: str = typer.Argument(
        ..., help="Scenario name or 'all'. Choices: happy_path, wrong_tool_retry, slow_program, "
        "reasoning_heavy, reasoning_loop, transient_failure, context_overflow, hallucinated_tool"
    ),
) -> None:
    """Generate synthetic trace file(s) for demo scenarios."""
    try:
        syn = _load_synthetic_module()
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if scenario == "all":
        console.print("[bold]Generating all scenarios...[/bold]")
        paths = []
        for name in syn.ALL_SCENARIOS:
            try:
                path = syn.generate(name)
                paths.append(path)
                console.print(f"  [green]✓[/green] wrote {path}")
            except Exception as exc:
                err_console.print(f"  [red]✗[/red] {name}: {exc}")
                raise typer.Exit(1)
        console.print(f"\n[bold]Done.[/bold] Wrote {len(paths)} files.")
    else:
        if scenario not in syn.ALL_SCENARIOS:
            valid = ", ".join(syn.ALL_SCENARIOS)
            err_console.print(
                f"[red]Error:[/red] Unknown scenario {scenario!r}. Valid: {valid}"
            )
            raise typer.Exit(1)
        try:
            path = syn.generate(scenario)
            console.print(f"[green]✓[/green] wrote {path}")
        except Exception as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)


# ---------------------------------------------------------------------------
# import-openclaw
# ---------------------------------------------------------------------------


@app.command(name="import-openclaw")
def import_openclaw(
    session_path: Path = typer.Argument(
        ..., help="Path to an OpenClaw session JSONL transcript"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Human-readable task description"
    ),
    program_tool: Optional[str] = typer.Option(
        None, "--program-tool", help="Tool name to treat as program_under_test"
    ),
) -> None:
    """Convert an OpenClaw session transcript to a profiler trace, then analyze it."""
    if not session_path.exists():
        err_console.print(f"[red]Error:[/red] File not found: {session_path}")
        raise typer.Exit(1)

    try:
        trace = convert_openclaw_session(
            session_path,
            task_description=task or "",
            program_tool_name=program_tool,
        )
    except Exception as exc:
        err_console.print(f"[red]Conversion failed:[/red] {exc}")
        raise typer.Exit(1)

    out_path = session_path.with_suffix(".profiler.jsonl")
    try:
        write_trace(trace, out_path)
    except Exception as exc:
        err_console.print(f"[red]Failed to write trace:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Converted → {out_path}")

    # Run analyze on the resulting trace
    metrics = compute_metrics(trace, program_tool_name=program_tool)

    table = Table(
        title=f"Profile: {out_path.name}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Category", style="dim")

    rows = [
        ("e2e_wall_ms", _fmt_ms(metrics.e2e_wall_ms), "timing"),
        ("active_wall_ms", _fmt_ms(metrics.active_wall_ms), "timing"),
        ("user_idle_ms", _fmt_ms(metrics.user_idle_ms), "timing"),
        ("total_model_time_ms", _fmt_ms(metrics.total_model_time_ms), "timing"),
        ("total_tool_time_ms", _fmt_ms(metrics.total_tool_time_ms), "timing"),
        ("retry_waste_ms", _fmt_ms(metrics.retry_waste_ms), "timing"),
        ("first_pass_success", str(metrics.first_pass_success), "correctness"),
        ("attempt_count", str(metrics.attempt_count), "correctness"),
        ("failure_categories", ", ".join(metrics.failure_categories) or "—", "correctness"),
        ("primary_bottleneck", metrics.primary_bottleneck, "derived"),
    ]

    prev_category = None
    for name, value, category in rows:
        if category != prev_category and prev_category is not None:
            table.add_section()
        prev_category = category
        table.add_row(name, value, category)

    console.print(table)

    if metrics.idle_percentage > 50:
        console.print(
            f"\n[bold magenta]Note: {metrics.idle_percentage:.1f}% of wall time "
            f"was user idle — bottleneck analysis uses active time only "
            f"({_fmt_ms(metrics.active_wall_ms)})[/bold magenta]"
        )

    verdict = _format_verdict(metrics)
    console.print(f"\n[bold yellow]{verdict}[/bold yellow]")


# ---------------------------------------------------------------------------
# monitor — helpers
# ---------------------------------------------------------------------------


_OPENCLAW_SESSIONS_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"


def _find_latest_session(sessions_dir: Path) -> Path | None:
    """Find the most recently modified .jsonl session file, excluding .profiler.jsonl."""
    if not sessions_dir.exists():
        return None
    sessions = sorted(
        (p for p in sessions_dir.glob("*.jsonl") if not p.name.endswith(".profiler.jsonl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return sessions[0] if sessions else None


def _list_recent_sessions(sessions_dir: Path, limit: int = 10) -> list[Path]:
    """Return the most recent session files, excluding .profiler.jsonl."""
    if not sessions_dir.exists():
        return []
    sessions = sorted(
        (p for p in sessions_dir.glob("*.jsonl") if not p.name.endswith(".profiler.jsonl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return sessions[:limit]


def _count_lines(path: Path) -> int:
    """Count lines in a file."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


@app.command()
def monitor(
    command: Optional[list[str]] = typer.Argument(
        default=None, help="Command to run and monitor"
    ),
    manual: bool = typer.Option(
        False, "--manual", help="Manual mode: wait for Enter instead of running a command"
    ),
    list_sessions: bool = typer.Option(
        False, "--list-sessions", help="List the 10 most recent OpenClaw sessions"
    ),
    session: Optional[Path] = typer.Option(
        None, "--session", help="Specific session file (skip auto-discovery)"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Task description"
    ),
    interval: int = typer.Option(
        500, "--interval", help="Sampling interval in milliseconds"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory for output files (default: cwd)"
    ),
    program_tool: Optional[str] = typer.Option(
        None, "--program-tool", help="Tool name to treat as program_under_test"
    ),
) -> None:
    """Run a command with system monitoring, then analyze the results.

    In --manual mode (or when no command is given), the sampler starts and
    you are prompted to perform your task in OpenClaw. Press Enter when done.
    """
    from agent_profiler.collector.system_sampler import SystemSampler

    # ---- List sessions mode (early exit) ----
    if list_sessions:
        sessions = _list_recent_sessions(_OPENCLAW_SESSIONS_DIR)
        if not sessions:
            console.print("[dim]No OpenClaw sessions found.[/dim]")
            return
        table = Table(
            title="Recent OpenClaw Sessions",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Session File", style="bold")
        table.add_column("Modified", style="dim")
        table.add_column("Lines", justify="right")
        for i, s in enumerate(sessions, 1):
            mtime = datetime.datetime.fromtimestamp(s.stat().st_mtime)
            table.add_row(
                str(i),
                s.name,
                mtime.strftime("%Y-%m-%d %H:%M:%S"),
                str(_count_lines(s)),
            )
        console.print(table)
        return

    # ---- Setup ----
    out_dir = output_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "system_samples.jsonl"
    sampler = SystemSampler(output_path=samples_path, interval_ms=interval)

    is_manual = manual or command is None

    # ---- Run (manual or command) ----
    if is_manual:
        console.print(f"[bold]System sampler started (sampling every {interval}ms)...[/bold]")
        console.print("[bold]Go run your task in OpenClaw. Press Enter when done.[/bold]")
        sampler.start()
        sampler.mark_event("manual_start")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        sampler.mark_event("manual_end")
        sampler.stop()
    else:
        console.print("[bold]Starting monitor...[/bold]")
        console.print(f"  Command: {' '.join(command)}")
        console.print(f"  Samples: {samples_path}")
        console.print(f"  Interval: {interval}ms")
        sampler.start()
        sampler.mark_event("command_start")
        try:
            result = subprocess.run(command, check=False)
            sampler.mark_event("command_end")
        except FileNotFoundError:
            err_console.print(f"[red]Error:[/red] Command not found: {command[0]}")
            raise typer.Exit(1)
        finally:
            sampler.stop()
        console.print(f"\n[bold]Command exited with code {result.returncode}[/bold]")

    console.print(f"[green]✓[/green] System samples written to {samples_path}")

    # ---- Discover session ----
    if session is not None:
        session_file: Path | None = session
        if not session_file.exists():
            err_console.print(f"[red]Error:[/red] Session file not found: {session_file}")
            raise typer.Exit(1)
    else:
        session_file = _find_latest_session(_OPENCLAW_SESSIONS_DIR)

    # ---- Analyze ----
    if session_file is not None:
        if session is None:
            mtime = datetime.datetime.fromtimestamp(session_file.stat().st_mtime)
            time_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
            console.print(
                f"\n[bold]Found session:[/bold] {session_file.name} (modified {time_str})"
            )
            confirm = input("Analyze this? [Y/n] ")
            if confirm.strip().lower() in ("n", "no"):
                console.print("[dim]Skipped analysis.[/dim]")
                return

        try:
            trace = convert_openclaw_session(
                session_file,
                task_description=task or "",
                program_tool_name=program_tool,
            )
            trace_path = out_dir / "trace.profiler.jsonl"
            write_trace(trace, trace_path)
            console.print(f"[green]✓[/green] Converted → {trace_path}")

            metrics = compute_metrics(
                trace,
                program_tool_name=program_tool,
                system_samples_path=samples_path,
            )

            console.print()
            _print_metrics_table(console, trace_path.name, metrics, trace)
        except Exception as exc:
            err_console.print(
                f"[yellow]Warning:[/yellow] Could not analyze OpenClaw session: {exc}"
            )
    else:
        console.print(
            "\n[dim]No OpenClaw session found. "
            "Use --system-samples with 'analyze' to correlate manually.[/dim]"
        )


def _print_metrics_table(console: Console, name: str, metrics, trace=None) -> None:
    """Print the standard metrics table (shared between analyze and monitor)."""
    table = Table(
        title=f"Profile: {name}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Category", style="dim")

    rows = [
        ("e2e_wall_ms", _fmt_ms(metrics.e2e_wall_ms), "timing"),
        ("active_wall_ms", _fmt_ms(metrics.active_wall_ms), "timing"),
        ("total_model_time_ms", _fmt_ms(metrics.total_model_time_ms), "timing"),
        ("total_tool_time_ms", _fmt_ms(metrics.total_tool_time_ms), "timing"),
        ("retry_waste_ms", _fmt_ms(metrics.retry_waste_ms), "timing"),
        ("first_pass_success", str(metrics.first_pass_success), "correctness"),
        ("attempt_count", str(metrics.attempt_count), "correctness"),
        ("failure_categories", ", ".join(metrics.failure_categories) or "—", "correctness"),
        ("primary_bottleneck", metrics.primary_bottleneck, "derived"),
    ]

    prev_category = None
    for name_col, value, category in rows:
        if category != prev_category and prev_category is not None:
            table.add_section()
        prev_category = category
        table.add_row(name_col, value, category)

    console.print(table)

    _print_token_summary(console, metrics)

    if metrics.resource_profile is not None:
        _print_resource_summary(console, metrics.resource_profile, trace)

    verdict = _format_verdict(metrics, trace if metrics.resource_profile else None)
    console.print(f"\n[bold yellow]{verdict}[/bold yellow]")
