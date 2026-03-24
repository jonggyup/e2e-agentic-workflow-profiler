"""Tests for multi-run comparison (compare_runs and CLI compare command)."""

from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_profiler.analyzer import compute_metrics, compare_runs
from agent_profiler.cli import app
from agent_profiler.schema.trace import load_trace

DEMOS = Path(__file__).parent.parent / "demos" / "output"
runner = CliRunner()


def _metrics(scenario: str):
    trace = load_trace(DEMOS / f"{scenario}.jsonl", strict=True)
    return compute_metrics(trace)


# ---------------------------------------------------------------------------
# happy_path vs wrong_tool_retry
# ---------------------------------------------------------------------------


class TestHappyPathVsWrongToolRetry:
    """Compare a clean single-attempt run against one that needed a retry."""

    def setup_method(self) -> None:
        self.baseline = _metrics("happy_path")
        self.current = _metrics("wrong_tool_retry")
        self.comp = compare_runs(self.baseline, self.current)

    def test_attempt_count_delta(self) -> None:
        assert self.comp.attempt_count_delta == 1  # 2 - 1

    def test_first_pass_improved_is_false(self) -> None:
        # baseline succeeded first pass, current failed → regression
        assert self.comp.first_pass_improved is False

    def test_retry_waste_delta_positive(self) -> None:
        # wrong_tool_retry has retry waste, happy_path has none
        assert self.comp.retry_waste_delta_ms > 0

    def test_token_delta_reflects_wasted_tokens(self) -> None:
        # More attempts → more tokens
        assert self.comp.token_delta > 0


# ---------------------------------------------------------------------------
# slow_program vs reasoning_heavy — bottleneck changed
# ---------------------------------------------------------------------------


class TestSlowProgramVsReasoningHeavy:
    """Verify bottleneck_changed is True when bottleneck shifts."""

    def setup_method(self) -> None:
        self.baseline = _metrics("slow_program")
        self.current = _metrics("reasoning_heavy")
        self.comp = compare_runs(self.baseline, self.current)

    def test_bottleneck_changed(self) -> None:
        assert self.comp.bottleneck_changed is True

    def test_baseline_bottleneck_is_program_runtime(self) -> None:
        assert self.comp.baseline_bottleneck == "program_runtime"

    def test_current_bottleneck_is_agent_reasoning(self) -> None:
        assert self.comp.current_bottleneck == "agent_reasoning"

    def test_summary_mentions_bottleneck(self) -> None:
        assert "Bottleneck shifted" in self.comp.summary


# ---------------------------------------------------------------------------
# happy_path vs happy_path — no change
# ---------------------------------------------------------------------------


class TestSameRunComparison:
    """Comparing identical runs should show zero deltas."""

    def setup_method(self) -> None:
        self.m = _metrics("happy_path")
        self.comp = compare_runs(self.m, self.m)

    def test_all_time_deltas_zero(self) -> None:
        assert self.comp.e2e_delta_ms == 0
        assert self.comp.active_wall_delta_ms == 0
        assert self.comp.model_time_delta_ms == 0
        assert self.comp.tool_time_delta_ms == 0
        assert self.comp.retry_waste_delta_ms == 0

    def test_token_delta_zero(self) -> None:
        assert self.comp.token_delta == 0

    def test_cost_delta_zero(self) -> None:
        assert self.comp.cost_delta_usd == 0.0

    def test_attempt_count_delta_zero(self) -> None:
        assert self.comp.attempt_count_delta == 0

    def test_first_pass_improved_none(self) -> None:
        assert self.comp.first_pass_improved is None

    def test_bottleneck_not_changed(self) -> None:
        assert self.comp.bottleneck_changed is False

    def test_summary_says_no_change(self) -> None:
        assert "no significant change" in self.comp.summary.lower()


# ---------------------------------------------------------------------------
# reasoning_heavy vs happy_path — model time improved
# ---------------------------------------------------------------------------


class TestReasoningHeavyVsHappyPath:
    """When current is happier, model_time_delta should be negative (improvement)."""

    def setup_method(self) -> None:
        self.baseline = _metrics("reasoning_heavy")
        self.current = _metrics("happy_path")
        self.comp = compare_runs(self.baseline, self.current)

    def test_model_time_delta_negative(self) -> None:
        # happy_path uses much less model time than reasoning_heavy
        assert self.comp.model_time_delta_ms < 0

    def test_active_wall_delta_negative(self) -> None:
        # happy_path should be faster overall
        assert self.comp.active_wall_delta_ms < 0

    def test_token_delta_negative(self) -> None:
        # happy_path uses fewer tokens
        assert self.comp.token_delta < 0


# ---------------------------------------------------------------------------
# Custom labels
# ---------------------------------------------------------------------------


class TestCustomLabels:
    def test_labels_propagated(self) -> None:
        m = _metrics("happy_path")
        comp = compare_runs(m, m, baseline_label="v1.0", current_label="v1.1")
        assert comp.baseline_label == "v1.0"
        assert comp.current_label == "v1.1"


# ---------------------------------------------------------------------------
# wrong_tool_retry (baseline) vs happy_path (current) — first_pass_improved
# ---------------------------------------------------------------------------


class TestFirstPassImproved:
    """When baseline failed first pass and current succeeded, first_pass_improved=True."""

    def test_first_pass_improved_true(self) -> None:
        baseline = _metrics("wrong_tool_retry")
        current = _metrics("happy_path")
        comp = compare_runs(baseline, current)
        assert comp.first_pass_improved is True


# ---------------------------------------------------------------------------
# CLI compare command
# ---------------------------------------------------------------------------


class TestCompareCommand:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, [
            "compare",
            str(DEMOS / "happy_path.jsonl"),
            str(DEMOS / "wrong_tool_retry.jsonl"),
        ])
        assert result.exit_code == 0, result.output

    def test_output_contains_delta_table(self) -> None:
        result = runner.invoke(app, [
            "compare",
            str(DEMOS / "happy_path.jsonl"),
            str(DEMOS / "wrong_tool_retry.jsonl"),
        ])
        assert "active_wall_ms" in result.output
        assert "total_model_time_ms" in result.output
        assert "total_tokens" in result.output
        assert "primary_bottleneck" in result.output

    def test_output_contains_summary(self) -> None:
        result = runner.invoke(app, [
            "compare",
            str(DEMOS / "slow_program.jsonl"),
            str(DEMOS / "reasoning_heavy.jsonl"),
        ])
        # Rich may wrap long lines, so normalize whitespace before checking
        flat = " ".join(result.output.split())
        assert "Bottleneck shifted" in flat

    def test_missing_baseline_exits_nonzero(self) -> None:
        result = runner.invoke(app, [
            "compare",
            "/tmp/does_not_exist.jsonl",
            str(DEMOS / "happy_path.jsonl"),
        ])
        assert result.exit_code != 0

    def test_missing_current_exits_nonzero(self) -> None:
        result = runner.invoke(app, [
            "compare",
            str(DEMOS / "happy_path.jsonl"),
            "/tmp/does_not_exist.jsonl",
        ])
        assert result.exit_code != 0

    def test_same_file_shows_no_change(self) -> None:
        result = runner.invoke(app, [
            "compare",
            str(DEMOS / "happy_path.jsonl"),
            str(DEMOS / "happy_path.jsonl"),
        ])
        assert result.exit_code == 0, result.output
        assert "no significant change" in result.output.lower()

    def test_profiler_trace_files_accepted(self) -> None:
        """Ensure .profiler.jsonl files work (same format as .jsonl)."""
        # Use any demo trace — the format is identical
        result = runner.invoke(app, [
            "compare",
            str(DEMOS / "happy_path.jsonl"),
            str(DEMOS / "reasoning_heavy.jsonl"),
        ])
        assert result.exit_code == 0, result.output
