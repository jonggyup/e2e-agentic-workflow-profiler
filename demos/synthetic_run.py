#!/usr/bin/env python3
"""Synthetic trace generator for agent-profiler demos.

Usage:
    uv run python demos/synthetic_run.py <scenario_name>
    uv run python demos/synthetic_run.py all

Scenarios:
    happy_path, wrong_tool_retry, slow_program, reasoning_heavy,
    reasoning_loop, transient_failure, context_overflow, hallucinated_tool
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_profiler.schema.events import (
    AttemptEvent,
    EvaluationCriteria,
    EvaluationEvent,
    LoopIterationEvent,
    ModelCallEvent,
    RunEvent,
    ToolCallEvent,
    event_to_jsonl_bytes,
)
from agent_profiler.schema.trace import RunTrace, load_trace

OUTPUT_DIR = Path(__file__).parent / "output"

_MS = 1_000_000      # nanoseconds per millisecond
_BASE_NS = 1_000_000_000_000  # arbitrary monotonic start (~16 min uptime)

ALL_SCENARIOS = [
    "happy_path",
    "wrong_tool_retry",
    "slow_program",
    "reasoning_heavy",
    "reasoning_loop",
    "transient_failure",
    "context_overflow",
    "hallucinated_tool",
]


# ---------------------------------------------------------------------------
# TraceBuilder
# ---------------------------------------------------------------------------


class TraceBuilder:
    """Accumulates profiler events for one synthetic run."""

    def __init__(self, scenario: str, task: str, rng: random.Random) -> None:
        self.scenario = scenario
        self.task = task
        self.rng = rng
        self.run_id: UUID = uuid4()
        self.t: int = _BASE_NS          # monotonic clock (ns)
        self.run_start: int = _BASE_NS

        self._attempts: list[AttemptEvent] = []
        self._iterations: list[LoopIterationEvent] = []
        self._model_calls: list[ModelCallEvent] = []
        self._tool_calls: list[ToolCallEvent] = []
        self._evaluations: list[EvaluationEvent] = []

    # ------------------------------------------------------------------
    # Clock helpers
    # ------------------------------------------------------------------

    def _tick(self, ms: float) -> None:
        self.t += int(ms * _MS)

    def _now(self) -> int:
        return self.t

    # ------------------------------------------------------------------
    # ID helpers — pre-allocate before child events reference them
    # ------------------------------------------------------------------

    def new_attempt_id(self) -> UUID:
        return uuid4()

    def new_iteration_id(self) -> tuple[UUID, int]:
        """Return (iteration_id, start_ns) at the current clock."""
        return uuid4(), self.t

    # ------------------------------------------------------------------
    # Event adders
    # ------------------------------------------------------------------

    def add_model_call(
        self,
        iteration_id: UUID,
        duration_ms: float,
        input_tokens: int = 2_000,
        output_tokens: int = 200,
        requested_tools: list[str] | None = None,
        tools_called: list[str] | None = None,
    ) -> ModelCallEvent:
        start = self.t
        ttft = round(duration_ms * self.rng.uniform(0.25, 0.45), 1)
        self._tick(duration_ms)
        req = requested_tools if requested_tools is not None else ["bash", "file_read", "browser"]
        called = tools_called if tools_called is not None else []
        mc = ModelCallEvent(
            event_type="model_call",
            model_call_id=uuid4(),
            iteration_id=iteration_id,
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            start_ns=start,
            end_ns=self.t,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_to_first_token_ms=ttft,
            requested_tools=req,
            tools_called_in_response=called,
        )
        self._model_calls.append(mc)
        return mc

    def add_tool_call(
        self,
        iteration_id: UUID,
        tool_name: str,
        category: str,
        duration_ms: float,
        params: dict | None = None,
        outcome: str = "success",
        error_message: str | None = None,
        is_put: bool = False,
    ) -> ToolCallEvent:
        start = self.t
        self._tick(duration_ms)
        tc = ToolCallEvent(
            event_type="tool_call",
            tool_call_id=uuid4(),
            iteration_id=iteration_id,
            tool_name=tool_name,
            tool_category=category,
            tool_params=params or {},
            start_ns=start,
            end_ns=self.t,
            outcome=outcome,
            error_message=error_message if outcome != "success" else None,
            sandbox_used=False,
            is_program_under_test=is_put,
        )
        self._tool_calls.append(tc)
        return tc

    def close_iteration(
        self,
        iteration_id: UUID,
        attempt_id: UUID,
        iteration_number: int,
        start_ns: int,
        has_tool_calls: bool,
        iteration_type: str,
        gap_ms: float = 5.0,
    ) -> LoopIterationEvent:
        self._tick(gap_ms)
        li = LoopIterationEvent(
            event_type="loop_iteration",
            iteration_id=iteration_id,
            attempt_id=attempt_id,
            iteration_number=iteration_number,
            start_ns=start_ns,
            end_ns=self.t,
            has_tool_calls=has_tool_calls,
            iteration_type=iteration_type,
        )
        self._iterations.append(li)
        return li

    def close_attempt(
        self,
        attempt_id: UUID,
        attempt_number: int,
        start_ns: int,
        outcome: str,
        failure_category: str | None = None,
        failure_reason: str | None = None,
        gap_ms: float = 10.0,
    ) -> AttemptEvent:
        self._tick(gap_ms)
        a = AttemptEvent(
            event_type="attempt",
            attempt_id=attempt_id,
            run_id=self.run_id,
            attempt_number=attempt_number,
            start_ns=start_ns,
            end_ns=self.t,
            outcome=outcome,
            failure_category=failure_category,
            failure_reason=failure_reason,
        )
        self._attempts.append(a)
        return a

    def add_evaluation(
        self,
        attempt_id: UUID,
        passed: bool,
        score: float | None = None,
        reason: str | None = None,
        task_completed: bool = True,
        correct_result: bool = True,
        no_side_effects: bool = True,
        efficient_path: bool = True,
    ) -> EvaluationEvent:
        ev = EvaluationEvent(
            event_type="evaluation",
            eval_id=uuid4(),
            attempt_id=attempt_id,
            evaluator="heuristic",
            passed=passed,
            score=score,
            reason=reason,
            criteria=EvaluationCriteria(
                task_completed=task_completed,
                correct_result=correct_result,
                no_side_effects=no_side_effects,
                efficient_path=efficient_path,
            ),
        )
        self._evaluations.append(ev)
        return ev

    def build(self) -> RunTrace:
        """Finalize and return a validated RunTrace."""
        self._tick(10.0)
        run_outcome = "success" if self._attempts[-1].outcome == "success" else "failure"
        run = RunEvent(
            event_type="run",
            run_id=self.run_id,
            task_description=self.task,
            start_ns=self.run_start,
            end_ns=self.t,
            outcome=run_outcome,
            attempt_count=len(self._attempts),
            total_model_calls=len(self._model_calls),
            total_tool_calls=len(self._tool_calls),
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            sandbox_mode="off",
        )
        return RunTrace(
            run=run,
            attempts=self._attempts,
            loop_iterations=self._iterations,
            model_calls=self._model_calls,
            tool_calls=self._tool_calls,
            evaluations=self._evaluations,
        )

    def write(self) -> Path:
        """Build, validate, and write to demos/output/<scenario>.jsonl."""
        trace = self.build()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{self.scenario}.jsonl"
        events = (
            [trace.run]
            + list(trace.attempts)
            + list(trace.loop_iterations)
            + list(trace.model_calls)
            + list(trace.tool_calls)
            + list(trace.evaluations)
        )
        with path.open("wb") as fh:
            for ev in events:
                fh.write(event_to_jsonl_bytes(ev))
                fh.write(b"\n")
        return path


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def build_happy_path(rng: random.Random) -> TraceBuilder:
    """1 attempt, 3 loop iterations, succeeds. Balanced timing."""
    b = TraceBuilder(
        "happy_path",
        "Search for recent papers on transformer efficiency and summarize the top 3",
        rng,
    )
    a1 = b.new_attempt_id()
    a1_start = b._now()

    # Iter 1: reason + bash ls
    # Model calls are kept short (270 ms each) so total model time stays below 60% of
    # e2e wall time → primary_bottleneck="balanced" (no single factor dominates).
    i1, i1s = b.new_iteration_id()
    b.add_model_call(i1, 270, input_tokens=2_100, output_tokens=180,
                     requested_tools=["bash", "file_read", "browser"], tools_called=["bash"])
    b.add_tool_call(i1, "bash", "shell", 200,
                    params={"command": "find ~/papers -name '*.pdf' | head -20"})
    b.close_iteration(i1, a1, 1, i1s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 2: reason + file_read
    i2, i2s = b.new_iteration_id()
    b.add_model_call(i2, 270, input_tokens=2_800, output_tokens=220,
                     requested_tools=["bash", "file_read", "browser"], tools_called=["file_read"])
    b.add_tool_call(i2, "file_read", "filesystem", 350,
                    params={"path": "~/papers/flash_attention_v2.pdf"})
    b.close_iteration(i2, a1, 2, i2s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 3: reason only (final answer)
    i3, i3s = b.new_iteration_id()
    b.add_model_call(i3, 270, input_tokens=3_500, output_tokens=450,
                     requested_tools=["bash", "file_read", "browser"], tools_called=[])
    b.close_iteration(i3, a1, 3, i3s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(a1, 1, a1_start, "success")
    b.add_evaluation(a1, passed=True, score=1.0,
                     reason="Correct summary produced in 3 efficient iterations",
                     efficient_path=True)
    return b


def build_wrong_tool_retry(rng: random.Random) -> TraceBuilder:
    """Attempt 1 fails (wrong_tool: bash curl on auth page), attempt 2 succeeds (browser)."""
    b = TraceBuilder(
        "wrong_tool_retry",
        "Retrieve the authenticated dashboard data from internal.corp/dashboard",
        rng,
    )

    # --- Attempt 1: wrong_tool ---
    a1 = b.new_attempt_id()
    a1_start = b._now()

    i1, i1s = b.new_iteration_id()
    b.add_model_call(i1, 540, input_tokens=2_200, output_tokens=190,
                     requested_tools=["bash", "file_read", "browser"], tools_called=["bash"])
    b.add_tool_call(i1, "bash", "shell", 320,
                    params={"command": "curl -s https://internal.corp/dashboard"},
                    outcome="error",
                    error_message="HTTP 403 Forbidden — authentication cookies required")
    b.close_iteration(i1, a1, 1, i1s, has_tool_calls=True, iteration_type="reason_and_act")

    i2, i2s = b.new_iteration_id()
    b.add_model_call(i2, 460, input_tokens=2_700, output_tokens=120,
                     requested_tools=["bash", "file_read", "browser"], tools_called=[])
    b.close_iteration(i2, a1, 2, i2s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(
        a1, 1, a1_start, "failure",
        failure_category="wrong_tool",
        failure_reason="Used bash curl for an authenticated page that requires browser with session cookies",
    )

    # --- Attempt 2: browser succeeds ---
    a2 = b.new_attempt_id()
    a2_start = b._now()

    i3, i3s = b.new_iteration_id()
    b.add_model_call(i3, 580, input_tokens=2_000, output_tokens=200,
                     requested_tools=["bash", "file_read", "browser"], tools_called=["browser"])
    b.add_tool_call(i3, "browser", "browser", 1_200,
                    params={"url": "https://internal.corp/dashboard", "action": "navigate"})
    b.close_iteration(i3, a2, 1, i3s, has_tool_calls=True, iteration_type="reason_and_act")

    i4, i4s = b.new_iteration_id()
    b.add_model_call(i4, 430, input_tokens=3_800, output_tokens=380,
                     requested_tools=["bash", "file_read", "browser"], tools_called=[])
    b.close_iteration(i4, a2, 2, i4s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(a2, 2, a2_start, "success")
    b.add_evaluation(a2, passed=True, score=0.85,
                     reason="Correct data retrieved on second attempt after switching to browser tool",
                     efficient_path=False)
    return b


def build_slow_program(rng: random.Random) -> TraceBuilder:
    """1 attempt succeeds, make build takes 25 seconds. Primary bottleneck: program_runtime."""
    b = TraceBuilder(
        "slow_program",
        "Build the project from source and verify the binary is correct",
        rng,
    )
    a1 = b.new_attempt_id()
    a1_start = b._now()

    # Iter 1: reason + make build (25 s, is_program_under_test=True)
    i1, i1s = b.new_iteration_id()
    b.add_model_call(i1, 510, input_tokens=1_800, output_tokens=160,
                     requested_tools=["bash", "file_read"], tools_called=["bash"])
    b.add_tool_call(i1, "bash", "shell", 25_000,
                    params={"command": "make build"},
                    is_put=True)
    b.close_iteration(i1, a1, 1, i1s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 2: reason + file_read (check build output)
    i2, i2s = b.new_iteration_id()
    b.add_model_call(i2, 490, input_tokens=2_400, output_tokens=180,
                     requested_tools=["bash", "file_read"], tools_called=["file_read"])
    b.add_tool_call(i2, "file_read", "filesystem", 60,
                    params={"path": "build/agent.log"})
    b.close_iteration(i2, a1, 2, i2s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 3: reason only (final answer)
    i3, i3s = b.new_iteration_id()
    b.add_model_call(i3, 530, input_tokens=3_200, output_tokens=280,
                     requested_tools=["bash", "file_read"], tools_called=[])
    b.close_iteration(i3, a1, 3, i3s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(a1, 1, a1_start, "success")
    b.add_evaluation(a1, passed=True, score=1.0,
                     reason="Build succeeded and binary verified correctly")
    return b


def build_reasoning_heavy(rng: random.Random) -> TraceBuilder:
    """1 attempt, 8 loop iterations, only 2 have tool calls. Model calls avg 2 s.
    Primary bottleneck: agent_reasoning (model time >> 60% of wall time)."""
    b = TraceBuilder(
        "reasoning_heavy",
        "Design a database schema for a multi-tenant SaaS billing system",
        rng,
    )
    a1 = b.new_attempt_id()
    a1_start = b._now()

    # Only iters 3 and 6 have tool calls; all others are reason_only
    tool_iters = {3, 6}
    for num in range(1, 9):
        iid, istart = b.new_iteration_id()
        has_tools = num in tool_iters
        if has_tools:
            b.add_model_call(iid, 2_050, input_tokens=4_000 + num * 600, output_tokens=350,
                             requested_tools=["bash", "file_read"], tools_called=["file_read"])
            b.add_tool_call(iid, "file_read", "filesystem", 80,
                            params={"path": f"schemas/reference_{num}.sql"})
            b.close_iteration(iid, a1, num, istart, has_tool_calls=True,
                              iteration_type="reason_and_act")
        else:
            b.add_model_call(iid, 1_980, input_tokens=3_500 + num * 600, output_tokens=400,
                             requested_tools=["bash", "file_read"], tools_called=[])
            b.close_iteration(iid, a1, num, istart, has_tool_calls=False,
                              iteration_type="reason_only")

    b.close_attempt(a1, 1, a1_start, "success")
    b.add_evaluation(a1, passed=True, score=0.9,
                     reason="Schema design is correct but required excessive reasoning iterations",
                     efficient_path=False)
    return b


def build_reasoning_loop(rng: random.Random) -> TraceBuilder:
    """Attempt 1 fails (reasoning_loop: same npm install 3 times), attempt 2 succeeds (yarn)."""
    b = TraceBuilder(
        "reasoning_loop",
        "Install project dependencies and run the test suite",
        rng,
    )

    # --- Attempt 1: reasoning_loop ---
    a1 = b.new_attempt_id()
    a1_start = b._now()

    for num in range(1, 4):
        iid, istart = b.new_iteration_id()
        b.add_model_call(iid, 420 + num * 30, input_tokens=2_000 + num * 500, output_tokens=150,
                         requested_tools=["bash"], tools_called=["bash"])
        b.add_tool_call(iid, "bash", "shell", 3_500,
                        params={"command": "npm install"},
                        outcome="error",
                        error_message=f"ERESOLVE unable to resolve dependency tree (attempt {num})")
        b.close_iteration(iid, a1, num, istart, has_tool_calls=True,
                          iteration_type="reason_and_act")

    # Iter 4: agent gives up on this approach
    i4, i4s = b.new_iteration_id()
    b.add_model_call(i4, 480, input_tokens=4_500, output_tokens=200,
                     requested_tools=["bash"], tools_called=[])
    b.close_iteration(i4, a1, 4, i4s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(
        a1, 1, a1_start, "failure",
        failure_category="reasoning_loop",
        failure_reason="Agent repeated 'npm install' 3 times with identical parameters without changing strategy",
    )

    # --- Attempt 2: yarn works ---
    a2 = b.new_attempt_id()
    a2_start = b._now()

    i5, i5s = b.new_iteration_id()
    b.add_model_call(i5, 530, input_tokens=2_000, output_tokens=180,
                     requested_tools=["bash"], tools_called=["bash"])
    b.add_tool_call(i5, "bash", "shell", 8_200,
                    params={"command": "yarn install --frozen-lockfile"})
    b.close_iteration(i5, a2, 1, i5s, has_tool_calls=True, iteration_type="reason_and_act")

    i6, i6s = b.new_iteration_id()
    b.add_model_call(i6, 450, input_tokens=3_800, output_tokens=320,
                     requested_tools=["bash"], tools_called=["bash"])
    b.add_tool_call(i6, "bash", "shell", 4_100,
                    params={"command": "yarn test"})
    b.close_iteration(i6, a2, 2, i6s, has_tool_calls=True, iteration_type="reason_and_act")

    i7, i7s = b.new_iteration_id()
    b.add_model_call(i7, 390, input_tokens=4_500, output_tokens=280,
                     requested_tools=["bash"], tools_called=[])
    b.close_iteration(i7, a2, 3, i7s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(a2, 2, a2_start, "success")
    b.add_evaluation(a2, passed=True, score=0.8,
                     reason="Tests passed after switching to yarn; wasted time on npm loop",
                     efficient_path=False)
    return b


def build_transient_failure(rng: random.Random) -> TraceBuilder:
    """1 attempt succeeds; one tool call returns 503, next iteration retries and succeeds."""
    b = TraceBuilder(
        "transient_failure",
        "Fetch the current exchange rates from the currency API and convert 100 USD to EUR",
        rng,
    )
    a1 = b.new_attempt_id()
    a1_start = b._now()

    # Iter 1: bash setup
    i1, i1s = b.new_iteration_id()
    b.add_model_call(i1, 510, input_tokens=1_900, output_tokens=160,
                     requested_tools=["bash", "mcp_currency"], tools_called=["bash"])
    b.add_tool_call(i1, "bash", "shell", 120,
                    params={"command": "echo $CURRENCY_API_KEY | wc -c"})
    b.close_iteration(i1, a1, 1, i1s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 2: bash set env
    i2, i2s = b.new_iteration_id()
    b.add_model_call(i2, 480, input_tokens=2_300, output_tokens=140,
                     requested_tools=["bash", "mcp_currency"], tools_called=["bash"])
    b.add_tool_call(i2, "bash", "shell", 95,
                    params={"command": "export CURRENCY_API_KEY=test_key_abc123"})
    b.close_iteration(i2, a1, 2, i2s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 3: MCP currency call → 503 transient error
    i3, i3s = b.new_iteration_id()
    b.add_model_call(i3, 520, input_tokens=2_600, output_tokens=170,
                     requested_tools=["bash", "mcp_currency"], tools_called=["mcp_currency"])
    b.add_tool_call(i3, "mcp_currency", "mcp", 2_800,
                    params={"endpoint": "latest", "base": "USD", "symbols": "EUR"},
                    outcome="error",
                    error_message="HTTP 503 Service Unavailable — upstream rate-limit, retry after 2s")
    b.close_iteration(i3, a1, 3, i3s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 4: retry same MCP call → succeeds
    i4, i4s = b.new_iteration_id()
    b.add_model_call(i4, 430, input_tokens=3_100, output_tokens=140,
                     requested_tools=["bash", "mcp_currency"], tools_called=["mcp_currency"])
    b.add_tool_call(i4, "mcp_currency", "mcp", 1_100,
                    params={"endpoint": "latest", "base": "USD", "symbols": "EUR"})
    b.close_iteration(i4, a1, 4, i4s, has_tool_calls=True, iteration_type="reason_and_act")

    # Iter 5: final answer
    i5, i5s = b.new_iteration_id()
    b.add_model_call(i5, 460, input_tokens=3_600, output_tokens=310,
                     requested_tools=["bash", "mcp_currency"], tools_called=[])
    b.close_iteration(i5, a1, 5, i5s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(a1, 1, a1_start, "success")
    b.add_evaluation(a1, passed=True, score=0.95,
                     reason="Correct conversion returned; single transient 503 handled by retry")
    return b


def build_context_overflow(rng: random.Random) -> TraceBuilder:
    """Attempt 1 fails (context_overflow: 15 iters, agent loses track).
    Attempt 2 succeeds in 3 iterations with a fresh context."""
    b = TraceBuilder(
        "context_overflow",
        "Refactor the authentication module to support OAuth2 and update all 47 call sites",
        rng,
    )

    # --- Attempt 1: 15 iterations, context grows out of control ---
    a1 = b.new_attempt_id()
    a1_start = b._now()

    # Iters 1, 3, 5, 7, 9 have tool calls; rest are reason_only
    tool_iters_a1 = {1, 3, 5, 7, 9}
    for num in range(1, 16):
        base_tokens = 3_000 + num * 2_100  # context swells each iteration
        iid, istart = b.new_iteration_id()
        has_tools = num in tool_iters_a1
        if has_tools:
            b.add_model_call(iid, 900 + num * 40, input_tokens=base_tokens, output_tokens=320,
                             requested_tools=["bash", "file_read", "file_write"],
                             tools_called=["file_read"])
            b.add_tool_call(iid, "file_read", "filesystem", 70,
                            params={"path": f"src/auth/callsite_{num:02d}.py"})
            b.close_iteration(iid, a1, num, istart, has_tool_calls=True,
                              iteration_type="reason_and_act")
        else:
            b.add_model_call(iid, 1_100 + num * 60, input_tokens=base_tokens, output_tokens=280,
                             requested_tools=["bash", "file_read", "file_write"],
                             tools_called=[])
            b.close_iteration(iid, a1, num, istart, has_tool_calls=False,
                              iteration_type="reason_only")

    b.close_attempt(
        a1, 1, a1_start, "failure",
        failure_category="context_overflow",
        failure_reason=(
            "After 15 iterations the agent's context exceeded 34k tokens; "
            "it began conflating different call sites and produced incorrect refactors"
        ),
    )

    # --- Attempt 2: fresh context, 3 iterations ---
    a2 = b.new_attempt_id()
    a2_start = b._now()

    i_a, i_as = b.new_iteration_id()
    b.add_model_call(i_a, 610, input_tokens=2_200, output_tokens=250,
                     requested_tools=["bash", "file_read", "file_write"],
                     tools_called=["bash"])
    b.add_tool_call(i_a, "bash", "shell", 280,
                    params={"command": "grep -r 'authenticate(' src/ | wc -l"})
    b.close_iteration(i_a, a2, 1, i_as, has_tool_calls=True, iteration_type="reason_and_act")

    i_b, i_bs = b.new_iteration_id()
    b.add_model_call(i_b, 580, input_tokens=3_100, output_tokens=420,
                     requested_tools=["bash", "file_read", "file_write"],
                     tools_called=["file_write"])
    b.add_tool_call(i_b, "file_write", "filesystem", 150,
                    params={"path": "src/auth/oauth2_adapter.py",
                            "content": "# OAuth2 adapter — generated"})
    b.close_iteration(i_b, a2, 2, i_bs, has_tool_calls=True, iteration_type="reason_and_act")

    i_c, i_cs = b.new_iteration_id()
    b.add_model_call(i_c, 540, input_tokens=4_000, output_tokens=600,
                     requested_tools=["bash", "file_read", "file_write"],
                     tools_called=[])
    b.close_iteration(i_c, a2, 3, i_cs, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(a2, 2, a2_start, "success")
    b.add_evaluation(a2, passed=True, score=0.88,
                     reason="Refactoring completed correctly on second attempt with focused context",
                     efficient_path=False)
    return b


def build_hallucinated_tool(rng: random.Random) -> TraceBuilder:
    """Attempt 1 fails (hallucinated_tool: send_email doesn't exist).
    Attempt 2 succeeds via bash sendmail."""
    b = TraceBuilder(
        "hallucinated_tool",
        "Send a summary email to the team with today's deployment results",
        rng,
    )

    # --- Attempt 1: hallucinated send_email tool ---
    a1 = b.new_attempt_id()
    a1_start = b._now()

    i1, i1s = b.new_iteration_id()
    # Model thinks send_email is available but it's not in requested_tools (hallucination)
    # tools_called_in_response intentionally outside requested_tools to reflect the hallucination
    b.add_model_call(i1, 550, input_tokens=2_000, output_tokens=210,
                     requested_tools=["bash", "file_read"],
                     tools_called=["send_email"])   # hallucination — not in requested_tools
    b.add_tool_call(i1, "send_email", "mcp", 180,
                    params={"to": "team@corp.com", "subject": "Deployment results"},
                    outcome="error",
                    error_message="ToolNotFoundError: 'send_email' is not a registered tool")
    b.close_iteration(i1, a1, 1, i1s, has_tool_calls=True, iteration_type="reason_and_act")

    i2, i2s = b.new_iteration_id()
    b.add_model_call(i2, 470, input_tokens=2_600, output_tokens=130,
                     requested_tools=["bash", "file_read"], tools_called=[])
    b.close_iteration(i2, a1, 2, i2s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(
        a1, 1, a1_start, "failure",
        failure_category="hallucinated_tool",
        failure_reason="Agent called 'send_email' which is not registered; no MCP email server is running",
    )

    # --- Attempt 2: bash sendmail succeeds ---
    a2 = b.new_attempt_id()
    a2_start = b._now()

    i3, i3s = b.new_iteration_id()
    b.add_model_call(i3, 510, input_tokens=2_000, output_tokens=190,
                     requested_tools=["bash", "file_read"], tools_called=["bash"])
    b.add_tool_call(i3, "bash", "shell", 340,
                    params={"command": (
                        "echo 'Deployment results: all green' | "
                        "sendmail -v team@corp.com"
                    )})
    b.close_iteration(i3, a2, 1, i3s, has_tool_calls=True, iteration_type="reason_and_act")

    i4, i4s = b.new_iteration_id()
    b.add_model_call(i4, 430, input_tokens=2_800, output_tokens=260,
                     requested_tools=["bash", "file_read"], tools_called=[])
    b.close_iteration(i4, a2, 2, i4s, has_tool_calls=False, iteration_type="reason_only")

    b.close_attempt(a2, 2, a2_start, "success")
    b.add_evaluation(a2, passed=True, score=0.9,
                     reason="Email sent via bash sendmail after hallucinated tool failure",
                     efficient_path=False)
    return b


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, callable] = {
    "happy_path": build_happy_path,
    "wrong_tool_retry": build_wrong_tool_retry,
    "slow_program": build_slow_program,
    "reasoning_heavy": build_reasoning_heavy,
    "reasoning_loop": build_reasoning_loop,
    "transient_failure": build_transient_failure,
    "context_overflow": build_context_overflow,
    "hallucinated_tool": build_hallucinated_tool,
}


def generate(scenario: str) -> Path:
    """Generate a single scenario, validate, and write to disk."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}. Choose from: {list(SCENARIOS)}")
    rng = random.Random(42)
    builder = SCENARIOS[scenario](rng)
    path = builder.write()
    # Re-validate by loading from disk
    load_trace(path, strict=True)
    return path


def generate_all() -> list[Path]:
    paths = []
    for name in ALL_SCENARIOS:
        path = generate(name)
        paths.append(path)
        print(f"  wrote {path}")
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <scenario_name|all>")
        print(f"Scenarios: {', '.join(ALL_SCENARIOS)}")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "all":
        print("Generating all scenarios...")
        generate_all()
        print("Done.")
    else:
        path = generate(arg)
        print(f"Wrote {path}")
