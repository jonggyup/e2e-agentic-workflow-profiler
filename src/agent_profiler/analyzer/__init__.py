"""Analyzer package — compute_metrics, per-attempt summaries, resource analysis, and comparison."""

from .comparison import RunComparison, compare_runs
from .metrics import ProfileMetrics, compute_metrics
from .per_attempt import AttemptSummary, summarize_attempt
from .resource_analyzer import (
    AggregatedToolResource,
    ResourceProfile,
    ToolResourceSummary,
    aggregate_tool_resources,
    analyze_resources,
)

__all__ = [
    "AggregatedToolResource",
    "AttemptSummary",
    "ProfileMetrics",
    "ResourceProfile",
    "RunComparison",
    "ToolResourceSummary",
    "aggregate_tool_resources",
    "analyze_resources",
    "compare_runs",
    "compute_metrics",
    "summarize_attempt",
]
