# Agent Profiler — CLAUDE.md

## What this project is
A correctness-aware offline profiler for on-device agentic workflows.
It captures structured events from agent runs, then analyzes both
performance AND correctness to explain where time is wasted and why.

## Environment
- macOS (Apple Silicon M3 Pro) is the primary and only target for now
- Python 3.12+ with uv for package management
- Tests: pytest (run with `uv run pytest`)
- Linting: ruff (run with `uv run ruff check .`)

## Architecture rules
- The profiler is PASSIVE and OFFLINE only. It never modifies agent behavior.
- Events are emitted as JSONL (one JSON object per line).
- Analysis happens after the run completes, not during.
- No databases, no servers, no web UIs in v0.
- No Perfetto integration yet — just JSONL in, metrics out.
- Canonical timestamps are monotonic nanoseconds (time.monotonic_ns()).
- Every metric must have a golden fixture and regression test.

## Code rules
- Analyzer code is Python. Use Pydantic models for all data structures.
- Use orjson for JSON parsing (faster on Apple Silicon).
- Prefer simple functions over class hierarchies.
- Never mix "add a new metric" work with "add a new integration" work.
- Type hints on all public functions.
- No print() for output — use the Rich library or structured logging.

## Event model (the core abstraction)
There are 5 event types: Run, Attempt, Step, ToolCall, Evaluation.
- Run: top-level container for one end-to-end agent task
- Attempt: one try at completing the task (retries create new attempts)
- Step: a reasoning/action unit within an attempt
- ToolCall: a specific tool invocation within a step
- Evaluation: a correctness judgment (pass/fail + reason) attached to
  an attempt or run

The semantic distinction that matters most:
"Was the agent WRONG (bad tool choice, bad params)
 or was the agent RIGHT but the tool was SLOW?"
This must be answerable from the event data.

## File layout

agent-profiler/
CLAUDE.md
pyproject.toml
src/
agent_profiler/
schema/        # Pydantic event models
collector/     # JSONL logger + shims
analyzer/      # Metric computation
verifier/      # Correctness evaluation
reporter/      # CLI output (Rich tables)
tests/
fixtures/        # Golden JSONL traces
test_schema.py
test_analyzer.py
test_verifier.py
demos/
synthetic_run.py # Generates fake but realistic traces

## What NOT to do
- Do not add macOS Instruments/signposts integration yet.
- Do not add Perfetto conversion yet.
- Do not add OpenClaw integration yet.
- Do not add any web UI or dashboard.
- Do not add any database.
- Do not add any self-optimization or agent-in-the-loop feedback.
- Do not use LangChain, LlamaIndex, or any agent framework as a dependency.
