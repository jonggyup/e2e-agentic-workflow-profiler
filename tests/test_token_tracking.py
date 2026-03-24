"""Tests for Phase 4 token tracking improvements."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_profiler.schema.trace import load_trace
from agent_profiler.analyzer import compute_metrics

DEMOS = Path(__file__).parent.parent / "demos" / "output"


class TestTokensPerAttempt:
    def test_happy_path_single_attempt(self) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(trace)
        assert m.tokens_per_attempt is not None
        assert len(m.tokens_per_attempt) == 1
        entry = m.tokens_per_attempt[0]
        assert entry["attempt_number"] == 1
        assert entry["input_tokens"] > 0
        assert entry["output_tokens"] > 0
        assert entry["cost_usd"] > 0

    def test_wrong_tool_retry_two_attempts(self) -> None:
        trace = load_trace(DEMOS / "wrong_tool_retry.jsonl", strict=True)
        m = compute_metrics(trace)
        assert m.tokens_per_attempt is not None
        assert len(m.tokens_per_attempt) == 2
        assert m.tokens_per_attempt[0]["attempt_number"] == 1
        assert m.tokens_per_attempt[1]["attempt_number"] == 2

    def test_tokens_per_attempt_sum_matches_total(self) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(trace)
        total_in = sum(e["input_tokens"] for e in m.tokens_per_attempt)
        total_out = sum(e["output_tokens"] for e in m.tokens_per_attempt)
        assert total_in == m.total_input_tokens
        assert total_out == m.total_output_tokens


class TestTokensPerIteration:
    def test_cumulative_growth(self) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(trace)
        assert m.tokens_per_iteration is not None
        assert len(m.tokens_per_iteration) > 0
        # Cumulative values should be non-decreasing
        for i in range(1, len(m.tokens_per_iteration)):
            prev = m.tokens_per_iteration[i - 1]
            curr = m.tokens_per_iteration[i]
            assert curr["cumulative_input_tokens"] >= prev["cumulative_input_tokens"]
            assert curr["cumulative_output_tokens"] >= prev["cumulative_output_tokens"]

    def test_final_cumulative_matches_total(self) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(trace)
        last = m.tokens_per_iteration[-1]
        assert last["cumulative_input_tokens"] == m.total_input_tokens
        assert last["cumulative_output_tokens"] == m.total_output_tokens


class TestContextEfficiency:
    def test_happy_path_has_efficiency(self) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(trace)
        assert m.context_efficiency is not None
        # output / input — should be a small fraction
        expected = m.total_output_tokens / m.total_input_tokens
        assert m.context_efficiency == pytest.approx(expected)

    def test_reasoning_heavy_lower_efficiency(self) -> None:
        """Reasoning-heavy runs have more input tokens relative to output."""
        trace_h = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        trace_r = load_trace(DEMOS / "reasoning_heavy.jsonl", strict=True)
        m_h = compute_metrics(trace_h)
        m_r = compute_metrics(trace_r)
        # Both should have context_efficiency defined
        assert m_h.context_efficiency is not None
        assert m_r.context_efficiency is not None


class TestCacheTokens:
    def test_default_zero_for_synthetic(self) -> None:
        """Synthetic traces don't have cache tokens — should default to 0."""
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(trace)
        assert m.cache_read_tokens == 0
        assert m.cache_write_tokens == 0


class TestResourceProfileIntegration:
    def test_none_when_no_samples_path(self) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(trace)
        assert m.resource_profile is None

    def test_none_when_path_doesnt_exist(self, tmp_path: Path) -> None:
        trace = load_trace(DEMOS / "happy_path.jsonl", strict=True)
        m = compute_metrics(
            trace,
            system_samples_path=tmp_path / "nonexistent.jsonl",
        )
        assert m.resource_profile is None
