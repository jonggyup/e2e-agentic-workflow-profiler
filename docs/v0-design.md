# v0 Design — Agent Profiler

## Goal
Given a completed agent run, answer:
1. Did the agent succeed?
2. How long did it take end-to-end?
3. How many attempts did it need?
4. For each failed attempt: WHY did it fail?
5. What was the primary bottleneck?
   - Agent overhead (reasoning, planning, context assembly)?
   - Wrong tool choice (agent error)?
   - Correct tool, slow execution (environment)?
   - Correct tool, bad parameters (agent error)?
   - Transient failure, successful retry (environment)?

## Event hierarchy

Run (1)
 └─ Attempt (1..N)
     ├─ Step (1..M per attempt)
     │   └─ ToolCall (0..1 per step)
     └─ Evaluation (1 per attempt)

## Event fields

### Run
- run_id: str (uuid)
- task_description: str
- start_ns: int (monotonic)
- end_ns: int (monotonic)
- outcome: "success" | "failure" | "timeout"
- attempt_count: int

### Attempt
- attempt_id: str
- run_id: str
- attempt_number: int (1-indexed)
- start_ns: int
- end_ns: int
- outcome: "success" | "failure"
- failure_reason: str | null
- failure_category: "wrong_tool" | "bad_params" | "tool_error" |
                    "transient" | "timeout" | "reasoning_error" | null

### Step
- step_id: str
- attempt_id: str
- step_type: "reason" | "plan" | "act" | "observe"
- start_ns: int
- end_ns: int
- metadata: dict (free-form, step-type specific)

### ToolCall
- tool_call_id: str
- step_id: str
- tool_name: str
- tool_params: dict
- start_ns: int
- end_ns: int
- outcome: "success" | "error"
- error_message: str | null

### Evaluation
- eval_id: str
- attempt_id: str
- evaluator: str (who/what judged correctness)
- passed: bool
- score: float | null (0.0-1.0)
- reason: str

## Core metrics (v0)

1. e2e_wall_ms — total run duration
2. first_pass_success — did attempt 1 succeed? (bool)
3. attempt_count — total attempts
4. retry_waste_ms — sum of wall time in failed attempts
5. agent_overhead_ms — sum of "reason" + "plan" step durations
6. tool_execution_ms — sum of all ToolCall durations
7. program_runtime_ms — sum of tool calls where tool_name is
   the "program under agent control" (configurable)
8. primary_bottleneck — enum:
   "agent_reasoning" | "wrong_tool_choice" | "tool_slowness" |
   "retry_overhead" | "program_runtime"
   Computed by comparing the above metrics.

## Non-goals for v0
- Real-time monitoring
- Dashboard or web UI
- Perfetto trace conversion
- OpenClaw/NemoClaw integration
- Self-optimization feedback loop
- Multi-agent profiling (single agent only)
