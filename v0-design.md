# v0 Design — Agent Profiler

## What this document is

This is the foundational design for a correctness-aware offline profiler
targeting on-device agentic workflows. The first integration target is
OpenClaw running locally on macOS (Apple Silicon).

The profiler's job: given a completed agent run, explain both WHERE time
was spent and WHY it was spent there — distinguishing agent mistakes from
environment slowness.

---

## How OpenClaw executes a task (what we're profiling)

OpenClaw uses a ReAct (Reasoning + Acting) loop:

```
User message
  → Context assembly (system prompt + skills + memory + session history)
  → Model inference (stream to Claude/GPT/Ollama)
  → Parse response for tool calls
  → Execute tool(s): bash, file ops, browser (Chromium/CDP), MCP servers
  → Feed tool results back to model
  → Repeat until model produces a final response (no more tool calls)
```

Key architectural facts that affect profiling:

1. **Sessions are JSONL transcripts** stored at
   `~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl`.
   Each line is a JSON-encoded AgentMessage with role + content.
   This is a natural hook point — we can read these post-hoc.

2. **The agentic loop can repeat multiple times** in a single interaction.
   Each loop iteration is: model call → tool call(s) → result → model call.
   A single user request might trigger 3-15 loop iterations.

3. **Tools are concrete**: bash/exec, file read/write/edit, browser
   (Chromium via CDP), MCP servers, canvas updates. Each tool call has
   observable inputs and outputs.

4. **Skills are injected per-turn**, not all at once. The runtime selects
   relevant skills to keep the prompt lean. Skill selection itself can
   be a source of errors (wrong skill loaded, relevant skill missing).

5. **Sandbox mode is optional**. When Docker sandbox is on, tool execution
   has additional overhead. When it's off (typical local use), tools run
   directly on the host.

6. **Model streaming** means tokens arrive incrementally. For profiling,
   we care about the total model call duration (first token to last token
   of a complete response), not individual token timing.

---

## The question the profiler must answer

Given a completed run, the profiler must answer:

> "The agent took X seconds to complete this task. Here's why:
>  Y% was the agent thinking, Z% was tools executing.
>  The agent needed N attempts. Attempts 1 and 2 failed because
>  [specific reason]. The primary bottleneck was [specific cause]."

Concretely, the profiler must distinguish these 6 causes of slowness:

| Cause | Example | Who's at fault |
|-------|---------|---------------|
| **Reasoning overhead** | Agent spent 8s across 5 model calls deciding what to do | Agent (unavoidable) |
| **Wrong tool choice** | Agent used `bash curl` when it should have used the browser tool | Agent (mistake) |
| **Bad parameters** | Agent ran `rm -rf /wrong/path` then had to undo and retry | Agent (mistake) |
| **Tool slowness** | Correct bash command, but it took 30s to compile | Environment |
| **Transient failure** | API call returned 503; agent retried and it worked | Environment |
| **Reasoning loop** | Agent repeated the same failing approach 3 times | Agent (mistake) |

---

## Event hierarchy

```
Run (1 per user request)
 └─ Attempt (1..N — a complete try at fulfilling the request)
     ├─ LoopIteration (1..M — one cycle of the ReAct loop)
     │   ├─ ModelCall (exactly 1 — the LLM inference)
     │   └─ ToolCall (0..K — tools invoked in this iteration)
     └─ Evaluation (1 — correctness judgment for this attempt)
```

### Why "LoopIteration" instead of "Step"

OpenClaw's ReAct loop has a clear boundary: one model call followed by
zero or more tool calls, then back to the model. This maps cleanly to
a "LoopIteration." Calling it a "Step" is ambiguous — does "step" mean
one model call? One tool call? One reasoning thought? "LoopIteration"
is unambiguous: it's one pass through the ReAct cycle.

### Why "Attempt" exists

OpenClaw doesn't have a built-in retry mechanism at the framework level.
But in practice, agents frequently fail on the first approach and
self-correct: "That didn't work, let me try a different approach."

An "Attempt" boundary is placed when the agent explicitly acknowledges
failure and restarts its approach. This is detected by:
- The agent saying something like "that didn't work" / "let me try again"
- A tool call returning an error followed by a different tool or approach
- A configurable heuristic (e.g., same tool called 3x with different params)

For v0, we detect attempts heuristically. In v1, we can inject a custom
skill that asks the agent to explicitly mark attempt boundaries.

---

## Event fields

### Run

```json
{
  "event_type": "run",
  "run_id": "uuid",
  "task_description": "Book a flight from SEA to SFO for next Friday",
  "start_ns": 1234567890000000,
  "end_ns":   1234572340000000,
  "outcome": "success | failure | timeout | interrupted",
  "attempt_count": 2,
  "total_model_calls": 8,
  "total_tool_calls": 5,
  "model_provider": "anthropic",
  "model_name": "claude-sonnet-4-20250514",
  "sandbox_mode": "off | docker | openshell"
}
```

**Timestamps**: Monotonic nanoseconds via `time.monotonic_ns()`.
Not wall-clock time. Monotonic timestamps don't jump on clock sync
and are correct for measuring durations.

### Attempt

```json
{
  "event_type": "attempt",
  "attempt_id": "uuid",
  "run_id": "uuid",
  "attempt_number": 1,
  "start_ns": 1234567890000000,
  "end_ns":   1234569990000000,
  "outcome": "success | failure",
  "failure_reason": "Agent used bash to curl the airline API but got 403; should have used browser tool with auth cookies",
  "failure_category": "wrong_tool | bad_params | tool_error | transient | timeout | reasoning_loop | hallucinated_tool | context_overflow | null"
}
```

**failure_category values explained**:

| Category | Meaning |
|----------|---------|
| `wrong_tool` | Agent chose an inappropriate tool for the task |
| `bad_params` | Right tool, wrong arguments (bad path, bad query, etc.) |
| `tool_error` | Tool itself failed (crash, uncaught exception) |
| `transient` | Temporary failure (network error, rate limit, 503) |
| `timeout` | Tool or model call exceeded time limit |
| `reasoning_loop` | Agent repeated the same failing approach without changing strategy |
| `hallucinated_tool` | Agent tried to call a tool that doesn't exist |
| `context_overflow` | Context window filled up, agent lost track of the task |
| `null` | Attempt succeeded, no failure |

### LoopIteration

```json
{
  "event_type": "loop_iteration",
  "iteration_id": "uuid",
  "attempt_id": "uuid",
  "iteration_number": 1,
  "start_ns": 1234567890000000,
  "end_ns":   1234568290000000,
  "has_tool_calls": true,
  "iteration_type": "reason_and_act | reason_only | act_only"
}
```

- `reason_and_act`: Model called, then tools executed (the normal ReAct cycle)
- `reason_only`: Model responded with text only, no tool calls (e.g., final answer)
- `act_only`: Rare; tool calls without a preceding model call (e.g., cleanup)

### ModelCall

```json
{
  "event_type": "model_call",
  "model_call_id": "uuid",
  "iteration_id": "uuid",
  "model_provider": "anthropic",
  "model_name": "claude-sonnet-4-20250514",
  "start_ns": 1234567890000000,
  "end_ns":   1234568190000000,
  "input_tokens": 4200,
  "output_tokens": 380,
  "time_to_first_token_ms": 450,
  "requested_tools": ["bash", "browser", "file_read"],
  "tools_called_in_response": ["bash"]
}
```

`input_tokens` and `output_tokens` enable cost computation.
`time_to_first_token_ms` is useful for distinguishing model latency
from generation time.

### ToolCall

```json
{
  "event_type": "tool_call",
  "tool_call_id": "uuid",
  "iteration_id": "uuid",
  "tool_name": "bash",
  "tool_category": "shell | filesystem | browser | mcp | canvas | system",
  "tool_params": {
    "command": "curl -s https://api.airline.com/flights?from=SEA&to=SFO"
  },
  "start_ns": 1234568200000000,
  "end_ns":   1234568290000000,
  "outcome": "success | error | timeout",
  "error_message": null,
  "sandbox_used": false,
  "is_program_under_test": false
}
```

**`tool_category`**: Groups tools for aggregate analysis.

| Category | OpenClaw tools |
|----------|---------------|
| `shell` | bash, exec |
| `filesystem` | file_read, file_write, file_edit, apply_patch |
| `browser` | browser (Chromium/CDP automation) |
| `mcp` | Any MCP server tool |
| `canvas` | Canvas update methods |
| `system` | cron, gateway, config tools |

**`is_program_under_test`**: Flag for the primary program the agent is
operating on (e.g., if the agent is running a build, the `make` command
is the "program under test"). This enables the `program_runtime_ms` metric.

### Evaluation

```json
{
  "event_type": "evaluation",
  "eval_id": "uuid",
  "attempt_id": "uuid",
  "evaluator": "human | llm_judge | script | heuristic",
  "passed": true,
  "score": 0.95,
  "reason": "Agent booked correct flight, but selected wrong seat class initially before correcting",
  "criteria": {
    "task_completed": true,
    "correct_result": true,
    "no_side_effects": true,
    "efficient_path": false
  }
}
```

**`evaluator` types**:
- `human`: A person reviewed the result
- `llm_judge`: A separate LLM scored the output (Langfuse-style)
- `script`: An automated check (e.g., "does the file exist?")
- `heuristic`: Rule-based (e.g., "did the agent finish without errors?")

**`criteria`**: Extensible dict. For v0, we track four booleans:
- `task_completed`: Did the agent produce a final response?
- `correct_result`: Was the result actually correct?
- `no_side_effects`: Did the agent avoid unintended changes?
- `efficient_path`: Did the agent take a reasonable path? (false if excessive retries)

---

## Core metrics (v0)

### Timing metrics

| Metric | Computation | Unit |
|--------|------------|------|
| `e2e_wall_ms` | `run.end_ns - run.start_ns` converted to ms | ms |
| `total_model_time_ms` | Sum of all `ModelCall` durations | ms |
| `total_tool_time_ms` | Sum of all `ToolCall` durations | ms |
| `agent_overhead_ms` | `total_model_time_ms` (reasoning IS the overhead) | ms |
| `tool_execution_ms` | `total_tool_time_ms` | ms |
| `program_runtime_ms` | Sum of ToolCall durations where `is_program_under_test=true` | ms |
| `retry_waste_ms` | Sum of wall time in all failed attempts | ms |
| `gap_time_ms` | `e2e_wall_ms - total_model_time_ms - total_tool_time_ms` (framework overhead, network, etc.) | ms |

### Correctness metrics

| Metric | Computation | Type |
|--------|------------|------|
| `first_pass_success` | `attempts[0].outcome == "success"` | bool |
| `attempt_count` | Length of attempts list | int |
| `failure_categories` | List of `failure_category` from failed attempts | list[str] |
| `correctness_score` | `evaluation.score` from the successful attempt (or last attempt) | float |

### Cost metrics

| Metric | Computation | Type |
|--------|------------|------|
| `total_input_tokens` | Sum of all `ModelCall.input_tokens` | int |
| `total_output_tokens` | Sum of all `ModelCall.output_tokens` | int |
| `estimated_cost_usd` | Token counts × model pricing (configurable) | float |
| `wasted_tokens` | Tokens used in failed attempts | int |

### Derived metric: primary_bottleneck

This is the most important metric — it answers "what should you fix first?"

**Computation (evaluated in order, first match wins)**:

1. If `retry_waste_ms > 50% of e2e_wall_ms` → `"retry_overhead"`
   *The agent wasted most of its time on failed attempts.*

2. If any failed attempt has `failure_category == "reasoning_loop"` → `"reasoning_loop"`
   *The agent got stuck repeating the same mistake.*

3. If `agent_overhead_ms > 60% of e2e_wall_ms` → `"agent_reasoning"`
   *Model inference dominated. Agent is thinking too much, acting too little.*

4. If `program_runtime_ms > 50% of e2e_wall_ms` → `"program_runtime"`
   *The program the agent was controlling was slow. Not the agent's fault.*

5. If `tool_execution_ms > 50% of e2e_wall_ms` → `"tool_slowness"`
   *Tools are slow. Might be network, might be sandbox overhead.*

6. If `gap_time_ms > 30% of e2e_wall_ms` → `"framework_overhead"`
   *Time is being lost in context assembly, prompt construction, etc.*

7. Else → `"balanced"`
   *No single factor dominates. Run is healthy.*

**Why this order?** Retry waste is checked first because it's the most
actionable: if half your time is wasted on failed attempts, nothing else
matters until you fix why attempts fail. Reasoning loops are next because
they indicate a fundamental agent logic problem.

---

## Synthetic demo scenarios

These scenarios exercise every metric and every failure category.
Each produces a `.jsonl` file that serves as both a demo and a test fixture.

### Scenario 1: happy_path

- 1 attempt, succeeds
- 3 loop iterations: reason → bash tool → reason → file_read → reason (final answer)
- Moderate timing: model calls ~500ms each, tool calls ~200ms each
- Expected: `first_pass_success=true`, `primary_bottleneck="balanced"`

### Scenario 2: wrong_tool_retry

- Attempt 1: agent uses `bash curl` for an authenticated page → 403 error → fails
  - `failure_category="wrong_tool"`
- Attempt 2: agent uses `browser` tool → succeeds
- Expected: `first_pass_success=false`, `attempt_count=2`, `retry_waste_ms > 0`

### Scenario 3: slow_program

- 1 attempt, succeeds
- Agent runs `make build` (the program under test) which takes 25 seconds
- Total run is ~28 seconds, of which 25s is program runtime
- Expected: `primary_bottleneck="program_runtime"`

### Scenario 4: reasoning_heavy

- 1 attempt, succeeds
- 8 loop iterations, but only 2 have tool calls
- Model calls average 2 seconds each (complex reasoning)
- Expected: `primary_bottleneck="agent_reasoning"`

### Scenario 5: reasoning_loop

- Attempt 1: agent tries `bash npm install`, fails, tries `bash npm install` again
  with same params, fails again, tries a third time → fails
  - `failure_category="reasoning_loop"`
- Attempt 2: agent tries `bash yarn install` → succeeds
- Expected: `failure_categories=["reasoning_loop"]`

### Scenario 6: transient_failure

- 1 attempt (the retry happens within the same attempt)
- Loop iteration 3: API call returns 503
- Loop iteration 4: Same API call succeeds
- Expected: `first_pass_success=true`, but gap visible in tool call outcomes

### Scenario 7: context_overflow

- Attempt 1: 15+ loop iterations, agent accumulates massive context,
  starts losing track of the original task, produces wrong result
  - `failure_category="context_overflow"`
- Attempt 2 (fresh context): succeeds in 3 iterations
- Expected: `failure_categories=["context_overflow"]`, high `wasted_tokens`

### Scenario 8: hallucinated_tool

- Attempt 1: agent tries to call `send_email` tool (doesn't exist in OpenClaw
  unless an MCP server provides it) → error
  - `failure_category="hallucinated_tool"`
- Attempt 2: agent uses `bash` to send email via `sendmail` → succeeds
- Expected: `failure_categories=["hallucinated_tool"]`

---

## Attempt boundary detection (v0 heuristic)

Since OpenClaw doesn't have explicit "attempt" markers, we detect attempt
boundaries heuristically from the JSONL transcript:

**Rule 1 — Explicit restart language**: If the model's text response
contains phrases like:
- "let me try a different approach"
- "that didn't work"
- "I'll try another way"
- "starting over"

Then a new attempt begins at the next loop iteration.

**Rule 2 — Repeated tool failure**: If the same tool is called 3+ times
in a row with errors, and then a DIFFERENT tool is called, that's an
attempt boundary.

**Rule 3 — Manual annotation**: The evaluation can include an
`attempt_boundaries` field that lists iteration numbers where attempts
start. This overrides heuristic detection.

For v0, Rule 1 + Rule 2 are sufficient. Rule 3 exists for ground truth.

---

## What we read from OpenClaw (data sources)

| Data | Source | Available in v0? |
|------|--------|-----------------|
| Session transcript | `~/.openclaw/sessions/<id>.jsonl` | Yes — post-hoc read |
| Tool call inputs/outputs | Embedded in transcript (assistant messages with tool calls, tool result messages) | Yes |
| Token usage | Model provider response metadata (if logged) | Partial — depends on provider |
| Timing | NOT in transcript today — we need to add timestamps | No — requires instrumentation |

**The v0 gap**: OpenClaw's JSONL transcript records WHAT happened but not
WHEN each event started/ended. For v0, we have two options:

- **Option A (preferred)**: Build a thin shim that wraps the agent runtime
  and adds timestamps to each event as it occurs, writing to a parallel
  `.profiler.jsonl` file alongside the session transcript.

- **Option B (fallback)**: Parse the session transcript post-hoc and
  estimate timing from message ordering + model token counts. Less
  accurate but zero instrumentation needed.

v0 starts with synthetic traces (where timing is known) and adds
Option A instrumentation in week 2.

---

## Non-goals for v0

- Real-time monitoring or live dashboards
- Perfetto or OpenTelemetry conversion
- NemoClaw / OpenShell support
- Multi-agent profiling (agent A delegates to agent B)
- Automatic optimization suggestions
- Web UI of any kind
- Database storage
- Integration with Langfuse, Arize, or any external platform
- macOS Instruments / signpost correlation
- Profiling the model inference itself (that's the provider's problem)

---

## Success criteria for v0

v0 is done when:

1. All 8 synthetic scenarios generate valid JSONL traces
2. `load_trace()` validates all 8 traces without error
3. `compute_metrics()` produces correct metrics for all 8 scenarios
4. Each metric has at least one golden test that asserts an exact value
5. The CLI prints a readable table for any valid trace file
6. A new developer can run `uv run agent-profiler analyze demos/output/wrong_tool_retry.jsonl`
   and understand the output without reading the code

---

## Open questions (to resolve during implementation)

1. **Attempt boundary detection accuracy**: How reliable are the heuristics?
   Should we just ask the user to annotate attempt boundaries manually in v0?

2. **Token counting**: Not all model providers include token usage in their
   streaming responses. Do we estimate from text length, or skip cost metrics
   when data is unavailable?

3. **Evaluation source**: Who provides the Evaluation event? For synthetic
   traces it's built-in. For real runs, does the user fill it in manually?
   Or do we auto-evaluate with a heuristic ("did the agent finish without
   errors = passed")?

4. **Nested tool calls**: If a bash tool call itself invokes something that
   takes a long time, we only see the outer duration. Is that sufficient
   for v0, or do we need to parse tool output for timing information?

5. **Concurrent tool calls**: Can OpenClaw execute multiple tools in parallel?
   If so, our sequential timestamp model breaks. Need to verify.
