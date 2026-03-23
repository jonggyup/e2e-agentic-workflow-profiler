# Agent Profiler — CLAUDE.md

## What this project is
A correctness-aware offline profiler for on-device agentic workflows.
First target: OpenClaw running locally on macOS (Apple Silicon M3 Pro).
It captures structured events from agent runs, then analyzes both
performance AND correctness to explain where time is wasted and why.

## Environment
- macOS (Apple Silicon M3 Pro) is the primary and only target
- Python 3.12+ with uv for package management
- Run tests: `uv run pytest`
- Run linter: `uv run ruff check .`
- Run CLI: `uv run agent-profiler <command>`

## Architecture rules
- The profiler is PASSIVE and OFFLINE. It never modifies agent behavior.
- Events are emitted as JSONL (one JSON object per line).
- Analysis happens after the run completes, not during.
- No databases, no servers, no web UIs in v0.
- No Perfetto or OpenTelemetry integration yet.
- Timestamps are monotonic nanoseconds (time.monotonic_ns()).
- Every metric must have a golden fixture and a regression test.

## Code rules
- Use Pydantic models for all data structures.
- Use orjson for JSON serialization (fast on Apple Silicon).
- Prefer simple functions over class hierarchies.
- Type hints on all public functions.
- No print() — use Rich library or structured logging.
- Never mix "add a metric" with "add an integration" in one change.

## Event model
Five event types: Run, Attempt, LoopIteration, ModelCall, ToolCall, Evaluation.
Read docs/v0-design.md for the full schema.
The key semantic question: "Was the agent WRONG or was the environment SLOW?"

## Do NOT
- Add macOS Instruments or signposts integration
- Add Perfetto or OpenTelemetry conversion
- Add OpenClaw integration (until synthetic path works fully)
- Add any web UI, dashboard, or database
- Add any self-optimization or agent-in-the-loop feedback
- Use LangChain, LlamaIndex, or any agent framework as a dependency
- Add NemoClaw or OpenShell support

