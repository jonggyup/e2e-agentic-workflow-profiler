#!/bin/bash
# bootstrap.sh — Run this once to set up the agent-profiler project
set -euo pipefail

PROJECT=~/projects/agent-profiler

echo "Creating project at $PROJECT ..."
mkdir -p "$PROJECT"
cd "$PROJECT"

# --- Git ---
git init

# --- Python project ---
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install it: brew install uv"
    exit 1
fi

uv init --name agent-profiler --no-readme
uv venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv add pydantic orjson rich typer
uv add --dev pytest pytest-snapshot ruff

# --- Directory structure ---
mkdir -p src/agent_profiler/{schema,collector,analyzer,verifier,reporter}
mkdir -p tests/fixtures
mkdir -p demos/output
mkdir -p docs
mkdir -p .claude/skills/add-metric

# --- __init__.py files ---
for dir in src/agent_profiler src/agent_profiler/{schema,collector,analyzer,verifier,reporter}; do
    touch "$dir/__init__.py"
done

# --- README ---
cat > README.md << 'EOF'
# Agent Profiler

A correctness-aware offline profiler for on-device agentic workflows.

## Quick start

```bash
source .venv/bin/activate
uv run agent-profiler demo happy_path          # generate a synthetic trace
uv run agent-profiler analyze demos/output/happy_path.jsonl   # analyze it
uv run pytest                                   # run all tests
```

## What it does

Given a completed agent run (e.g., from OpenClaw), the profiler explains:
- Where time was spent (model inference vs tool execution)
- Why time was wasted (wrong tool choice, bad parameters, retries)
- What the primary bottleneck was
EOF

echo ""
echo "=== Project skeleton created at $PROJECT ==="
echo ""
echo "Next steps:"
echo "  1. Copy CLAUDE.md and docs/v0-design.md into the project"
echo "  2. git add -A && git commit -m 'chore: project skeleton'"
echo "  3. Open Claude Code: cd $PROJECT && claude"
echo "  4. Start with Day 2 prompt from the guide"
echo ""
