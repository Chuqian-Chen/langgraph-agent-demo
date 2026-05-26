# Multi-Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mock-first Supervisor-routed LangGraph multi-agent runtime while preserving the existing `run_demo()` entry point.

**Architecture:** The runtime uses a shared `AgentState`, a `supervisor` router, focused role agents, and adapter seams for models and tools. The first implementation remains deterministic and offline, with future provider/tool/self-improvement extension points represented as explicit interfaces and state fields.

**Tech Stack:** Python, `unittest`, LangGraph `StateGraph`, LangChain message classes, typed dictionaries and protocols from the standard library.

---

## File Structure

Create and modify these files:

```text
langgraph_agent_demo/
  __init__.py
  demo.py
  state.py
  graph.py
  supervisor.py
  agents.py
  models.py
  tools.py
tests/
  test_demo.py
  test_graph.py
  test_supervisor.py
  test_agents.py
  test_models.py
```

Responsibilities:

- `state.py`: all shared state, review, and error type definitions.
- `models.py`: model protocol and deterministic mock implementation.
- `tools.py`: tool registry and deterministic mock research tool.
- `agents.py`: planner, researcher, executor, and reviewer node functions.
- `supervisor.py`: routing and stopping policy.
- `graph.py`: LangGraph assembly and conditional edges.
- `demo.py`: public `build_agent_graph()` and `run_demo()` compatibility layer.
- Tests: focused tests by module plus integration coverage for CLI/demo behavior.

Commit and push policy:

- Make one commit per completed task.
- After a remote is configured, push each commit with `git push origin main`.
- Keep each commit small enough to review by reading its diff.

---

### Task 1: Define Shared State Types

**Files:**
- Create: `langgraph_agent_demo/state.py`
- Test: `tests/test_agents.py`

- [ ] **Step 1: Write failing tests for owned state fields**

Create `tests/test_agents.py` with:

```python
import unittest

from langgraph_agent_demo.state import AgentState, make_initial_state


class AgentStateTest(unittest.TestCase):
    def test_make_initial_state_sets_defaults(self):
        state = make_initial_state("Build a report")

        self.assertEqual(state["task"], "Build a report")
        self.assertEqual(state["plan"], [])
        self.assertEqual(state["findings"], [])
        self.assertEqual(state["result"], "")
        self.assertEqual(state["review"]["approved"], False)
        self.assertEqual(state["next_agent"], "planner")
        self.assertEqual(state["iteration_count"], 0)
        self.assertEqual(state["max_iterations"], 3)
        self.assertEqual(state["trace"], ["START"])
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["errors"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents.AgentStateTest.test_make_initial_state_sets_defaults -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `state.py` does not exist.

- [ ] **Step 3: Implement state definitions**

Create `langgraph_agent_demo/state.py`:

```python
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage


AgentName = Literal["planner", "researcher", "executor", "reviewer", "end"]
RunStatus = Literal["running", "approved", "max_iterations_reached", "error"]


class ReviewResult(TypedDict):
    approved: bool
    score: float
    issues: list[str]


class AgentError(TypedDict):
    node: str
    message: str
    recoverable: bool


class AgentState(TypedDict):
    messages: list[Any]
    task: str
    plan: list[str]
    needs_research: bool
    findings: list[str]
    result: str
    review: ReviewResult
    next_agent: AgentName
    iteration_count: int
    max_iterations: int
    trace: list[str]
    status: RunStatus
    errors: list[AgentError]


def make_initial_state(user_text: str, max_iterations: int = 3) -> AgentState:
    return {
        "messages": [HumanMessage(content=user_text)],
        "task": user_text,
        "plan": [],
        "needs_research": False,
        "findings": [],
        "result": "",
        "review": {"approved": False, "score": 0.0, "issues": []},
        "next_agent": "planner",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "trace": ["START"],
        "status": "running",
        "errors": [],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents.AgentStateTest.test_make_initial_state_sets_defaults -v
```

Expected: PASS.

- [ ] **Step 5: Commit and push**

Run:

```powershell
git add langgraph_agent_demo/state.py tests/test_agents.py
git commit -m "feat: add shared agent state"
git push origin main
```

If no remote exists yet, skip only the push command and record that push is blocked by missing `origin`.

---

### Task 2: Add Mock Model And Tool Adapters

**Files:**
- Create: `langgraph_agent_demo/models.py`
- Create: `langgraph_agent_demo/tools.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for adapters**

Create `tests/test_models.py`:

```python
import unittest

from langgraph_agent_demo.models import MockModelClient
from langgraph_agent_demo.tools import ToolRegistry


class AdapterTest(unittest.TestCase):
    def test_mock_model_returns_role_specific_text(self):
        model = MockModelClient()

        self.assertIn("Plan", model.generate("planner", "Build a report"))
        self.assertIn("Result", model.generate("executor", "Build a report"))
        self.assertIn("approved", model.generate("reviewer", "Build a report"))

    def test_tool_registry_runs_registered_tool(self):
        registry = ToolRegistry()
        registry.register("research", lambda query: f"finding for {query}")

        self.assertEqual(registry.run("research", "LangGraph"), "finding for LangGraph")

    def test_tool_registry_reports_missing_tool(self):
        registry = ToolRegistry()

        result = registry.run("missing", "LangGraph")

        self.assertIn("Tool not found", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_models -v
```

Expected: FAIL because `models.py` and `tools.py` do not exist.

- [ ] **Step 3: Implement model adapter**

Create `langgraph_agent_demo/models.py`:

```python
from typing import Protocol


class ModelClient(Protocol):
    def generate(self, role: str, prompt: str) -> str:
        ...


class MockModelClient:
    def generate(self, role: str, prompt: str) -> str:
        if role == "planner":
            return f"Plan: clarify goal, gather context, produce answer for {prompt}"
        if role == "researcher":
            return f"Findings: useful context for {prompt}"
        if role == "executor":
            return f"Result: completed response for {prompt}"
        if role == "reviewer":
            return "Review: approved with score 1.0"
        return f"Mock response for {role}: {prompt}"
```

- [ ] **Step 4: Implement tool registry**

Create `langgraph_agent_demo/tools.py`:

```python
from collections.abc import Callable


Tool = Callable[[str], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, tool: Tool) -> None:
        self._tools[name] = tool

    def run(self, name: str, query: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Tool not found: {name}"
        try:
            return tool(query)
        except Exception as exc:
            return f"Tool error from {name}: {exc}"


def build_default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("research", lambda query: f"Mock research finding for {query}")
    return registry
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_models -v
```

Expected: PASS.

- [ ] **Step 6: Commit and push**

Run:

```powershell
git add langgraph_agent_demo/models.py langgraph_agent_demo/tools.py tests/test_models.py
git commit -m "feat: add mock model and tool adapters"
git push origin main
```

If no remote exists yet, skip only the push command and record that push is blocked by missing `origin`.

---

### Task 3: Implement Role Agent Nodes

**Files:**
- Modify: `tests/test_agents.py`
- Create: `langgraph_agent_demo/agents.py`

- [ ] **Step 1: Add failing tests for role agents**

Append to `tests/test_agents.py`:

```python
from langgraph_agent_demo.agents import executor, planner, researcher, reviewer
from langgraph_agent_demo.models import MockModelClient
from langgraph_agent_demo.tools import build_default_tools


class RoleAgentTest(unittest.TestCase):
    def test_planner_writes_plan_and_trace(self):
        state = make_initial_state("Build a report")

        update = planner(state, model=MockModelClient())

        self.assertGreater(len(update["plan"]), 0)
        self.assertEqual(update["next_agent"], "executor")
        self.assertIn("planner", update["trace"])

    def test_planner_can_request_research(self):
        state = make_initial_state("Research LangGraph and summarize it")

        update = planner(state, model=MockModelClient())

        self.assertTrue(update["needs_research"])
        self.assertEqual(update["next_agent"], "researcher")

    def test_researcher_writes_findings(self):
        state = make_initial_state("Research LangGraph")
        state["needs_research"] = True

        update = researcher(state, model=MockModelClient(), tools=build_default_tools())

        self.assertGreater(len(update["findings"]), 0)
        self.assertEqual(update["next_agent"], "executor")
        self.assertIn("researcher", update["trace"])

    def test_executor_writes_result(self):
        state = make_initial_state("Build a report")
        state["plan"] = ["Produce a concise answer"]

        update = executor(state, model=MockModelClient())

        self.assertIn("Result", update["result"])
        self.assertEqual(update["next_agent"], "reviewer")
        self.assertIn("executor", update["trace"])

    def test_reviewer_approves_non_empty_result(self):
        state = make_initial_state("Build a report")
        state["result"] = "A completed result"

        update = reviewer(state, model=MockModelClient())

        self.assertTrue(update["review"]["approved"])
        self.assertEqual(update["status"], "approved")
        self.assertEqual(update["next_agent"], "end")
        self.assertIn("reviewer", update["trace"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents -v
```

Expected: FAIL because `agents.py` does not exist.

- [ ] **Step 3: Implement role agents**

Create `langgraph_agent_demo/agents.py`:

```python
from langgraph_agent_demo.models import MockModelClient, ModelClient
from langgraph_agent_demo.state import AgentState, ReviewResult
from langgraph_agent_demo.tools import ToolRegistry, build_default_tools


def _append_trace(state: AgentState, node: str) -> list[str]:
    return [*state["trace"], node]


def planner(state: AgentState, model: ModelClient | None = None) -> dict:
    client = model or MockModelClient()
    plan_text = client.generate("planner", state["task"])
    needs_research = "research" in state["task"].lower()
    return {
        "plan": [plan_text],
        "needs_research": needs_research,
        "next_agent": "researcher" if needs_research else "executor",
        "trace": _append_trace(state, "planner"),
    }


def researcher(
    state: AgentState,
    model: ModelClient | None = None,
    tools: ToolRegistry | None = None,
) -> dict:
    client = model or MockModelClient()
    registry = tools or build_default_tools()
    model_finding = client.generate("researcher", state["task"])
    tool_finding = registry.run("research", state["task"])
    return {
        "findings": [model_finding, tool_finding],
        "next_agent": "executor",
        "trace": _append_trace(state, "researcher"),
    }


def executor(state: AgentState, model: ModelClient | None = None) -> dict:
    client = model or MockModelClient()
    result = client.generate("executor", state["task"])
    return {
        "result": result,
        "next_agent": "reviewer",
        "trace": _append_trace(state, "executor"),
    }


def reviewer(state: AgentState, model: ModelClient | None = None) -> dict:
    client = model or MockModelClient()
    review_text = client.generate("reviewer", state["result"])
    approved = bool(state["result"].strip())
    review: ReviewResult = {
        "approved": approved,
        "score": 1.0 if approved else 0.0,
        "issues": [] if approved else ["Result is empty"],
    }
    return {
        "review": review,
        "status": "approved" if approved else "running",
        "next_agent": "end" if approved else "executor",
        "messages": [*state["messages"]],
        "trace": _append_trace(state, "reviewer"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents -v
```

Expected: PASS.

- [ ] **Step 5: Commit and push**

Run:

```powershell
git add langgraph_agent_demo/agents.py tests/test_agents.py
git commit -m "feat: add role agent nodes"
git push origin main
```

If no remote exists yet, skip only the push command and record that push is blocked by missing `origin`.

---

### Task 4: Implement Supervisor Routing

**Files:**
- Create: `tests/test_supervisor.py`
- Create: `langgraph_agent_demo/supervisor.py`

- [ ] **Step 1: Write failing supervisor tests**

Create `tests/test_supervisor.py`:

```python
import unittest

from langgraph_agent_demo.state import make_initial_state
from langgraph_agent_demo.supervisor import route_next, supervisor


class SupervisorTest(unittest.TestCase):
    def test_supervisor_records_trace_and_keeps_initial_route(self):
        state = make_initial_state("Build a report")

        update = supervisor(state)

        self.assertEqual(update["next_agent"], "planner")
        self.assertEqual(update["iteration_count"], 1)
        self.assertIn("supervisor", update["trace"])

    def test_route_next_returns_selected_agent(self):
        state = make_initial_state("Build a report")
        state["next_agent"] = "executor"

        self.assertEqual(route_next(state), "executor")

    def test_route_next_ends_on_approval(self):
        state = make_initial_state("Build a report")
        state["review"]["approved"] = True
        state["next_agent"] = "end"

        self.assertEqual(route_next(state), "end")

    def test_supervisor_stops_at_max_iterations(self):
        state = make_initial_state("Build a report", max_iterations=1)
        state["iteration_count"] = 1
        state["next_agent"] = "executor"

        update = supervisor(state)

        self.assertEqual(update["status"], "max_iterations_reached")
        self.assertEqual(update["next_agent"], "end")

    def test_supervisor_handles_unknown_route(self):
        state = make_initial_state("Build a report")
        state["next_agent"] = "unknown"

        update = supervisor(state)

        self.assertEqual(update["status"], "error")
        self.assertEqual(update["next_agent"], "end")
        self.assertEqual(update["errors"][-1]["node"], "supervisor")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_supervisor -v
```

Expected: FAIL because `supervisor.py` does not exist.

- [ ] **Step 3: Implement supervisor**

Create `langgraph_agent_demo/supervisor.py`:

```python
from langgraph_agent_demo.state import AgentName, AgentState


VALID_ROUTES: set[str] = {"planner", "researcher", "executor", "reviewer", "end"}


def supervisor(state: AgentState) -> dict:
    trace = [*state["trace"], "supervisor"]
    iteration_count = state["iteration_count"] + 1
    next_agent = state["next_agent"]

    if state["review"]["approved"]:
        return {
            "next_agent": "end",
            "status": "approved",
            "iteration_count": iteration_count,
            "trace": trace,
        }

    if iteration_count > state["max_iterations"]:
        return {
            "next_agent": "end",
            "status": "max_iterations_reached",
            "iteration_count": iteration_count,
            "trace": trace,
        }

    if next_agent not in VALID_ROUTES:
        return {
            "next_agent": "end",
            "status": "error",
            "iteration_count": iteration_count,
            "trace": trace,
            "errors": [
                *state["errors"],
                {
                    "node": "supervisor",
                    "message": f"Unknown route: {next_agent}",
                    "recoverable": False,
                },
            ],
        }

    return {
        "next_agent": next_agent,
        "iteration_count": iteration_count,
        "trace": trace,
    }


def route_next(state: AgentState) -> AgentName:
    next_agent = state["next_agent"]
    if next_agent in VALID_ROUTES:
        return next_agent  # type: ignore[return-value]
    return "end"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_supervisor -v
```

Expected: PASS.

- [ ] **Step 5: Commit and push**

Run:

```powershell
git add langgraph_agent_demo/supervisor.py tests/test_supervisor.py
git commit -m "feat: add supervisor routing"
git push origin main
```

If no remote exists yet, skip only the push command and record that push is blocked by missing `origin`.

---

### Task 5: Wire LangGraph Runtime

**Files:**
- Create: `tests/test_graph.py`
- Create: `langgraph_agent_demo/graph.py`
- Modify: `langgraph_agent_demo/demo.py`
- Modify: `langgraph_agent_demo/__init__.py`

- [ ] **Step 1: Write failing graph tests**

Create `tests/test_graph.py`:

```python
import unittest

from langgraph_agent_demo.graph import build_agent_graph
from langgraph_agent_demo.state import make_initial_state


class GraphTest(unittest.TestCase):
    def test_graph_compiles_and_runs_mock_runtime(self):
        graph = build_agent_graph()

        final_state = graph.invoke(make_initial_state("Build a report"))

        self.assertEqual(final_state["status"], "approved")
        self.assertIn("supervisor", final_state["trace"])
        self.assertIn("planner", final_state["trace"])
        self.assertIn("executor", final_state["trace"])
        self.assertIn("reviewer", final_state["trace"])
        self.assertTrue(final_state["result"])

    def test_graph_routes_to_researcher_when_needed(self):
        graph = build_agent_graph()

        final_state = graph.invoke(make_initial_state("Research LangGraph"))

        self.assertIn("researcher", final_state["trace"])
        self.assertGreater(len(final_state["findings"]), 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_graph -v
```

Expected: FAIL because `graph.py` does not exist.

- [ ] **Step 3: Implement graph wiring**

Create `langgraph_agent_demo/graph.py`:

```python
from langgraph.graph import END, START, StateGraph

from langgraph_agent_demo.agents import executor, planner, researcher, reviewer
from langgraph_agent_demo.state import AgentState
from langgraph_agent_demo.supervisor import route_next, supervisor


def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("executor", executor)
    builder.add_node("reviewer", reviewer)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "planner": "planner",
            "researcher": "researcher",
            "executor": "executor",
            "reviewer": "reviewer",
            "end": END,
        },
    )
    builder.add_edge("planner", "supervisor")
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("executor", "reviewer")
    builder.add_edge("reviewer", "supervisor")
    return builder.compile()
```

- [ ] **Step 4: Update public demo API**

Replace `langgraph_agent_demo/demo.py` with:

```python
from typing import Any, TypedDict

from langgraph_agent_demo.graph import build_agent_graph
from langgraph_agent_demo.state import make_initial_state


class DemoResult(TypedDict):
    reply: str
    trace: list[str]
    state: dict[str, Any]


def run_demo(user_text: str) -> DemoResult:
    graph = build_agent_graph()
    final_state = graph.invoke(make_initial_state(user_text))
    trace = [*final_state["trace"], "END"]
    return {
        "reply": final_state["result"],
        "trace": trace,
        "state": final_state,
    }
```

Update `langgraph_agent_demo/__init__.py`:

```python
from .demo import run_demo
from .graph import build_agent_graph

__all__ = ["build_agent_graph", "run_demo"]
```

- [ ] **Step 5: Run graph tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_graph -v
```

Expected: PASS.

- [ ] **Step 6: Commit and push**

Run:

```powershell
git add langgraph_agent_demo/graph.py langgraph_agent_demo/demo.py langgraph_agent_demo/__init__.py tests/test_graph.py
git commit -m "feat: wire supervisor routed graph"
git push origin main
```

If no remote exists yet, skip only the push command and record that push is blocked by missing `origin`.

---

### Task 6: Update Demo And Compatibility Tests

**Files:**
- Modify: `tests/test_demo.py`
- Modify: `run_demo.py`
- Modify: `README.zh-CN.md`

- [ ] **Step 1: Update tests for new result shape**

Replace `tests/test_demo.py` with:

```python
import os
import subprocess
import sys
import unittest

from langgraph_agent_demo.demo import run_demo
from langgraph_agent_demo.graph import build_agent_graph


class LangGraphAgentDemoTest(unittest.TestCase):
    def test_agent_graph_runs_multi_agent_runtime(self):
        result = run_demo("你好，LangGraph")

        self.assertIn("Result", result["reply"])
        self.assertEqual(result["trace"][0], "START")
        self.assertEqual(result["trace"][-1], "END")
        self.assertIn("supervisor", result["trace"])
        self.assertIn("planner", result["trace"])
        self.assertIn("executor", result["trace"])
        self.assertIn("reviewer", result["trace"])
        self.assertEqual(result["state"]["status"], "approved")

    def test_build_agent_graph_returns_compiled_graph(self):
        graph = build_agent_graph()

        self.assertTrue(hasattr(graph, "invoke"))

    def test_cli_prints_chinese_when_stdout_encoding_is_narrow(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "cp1252"

        result = subprocess.run(
            [sys.executable, "run_demo.py"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertIn("最终回复:", result.stdout.decode("utf-8"))
```

- [ ] **Step 2: Run tests to verify CLI output still needs updating**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_demo -v
```

Expected: FAIL if `run_demo.py` still expects `messages`; PASS only after the CLI is updated.

- [ ] **Step 3: Update CLI output**

Replace `run_demo.py` with:

```python
import sys

from langgraph_agent_demo import run_demo


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    configure_stdout()
    user_text = "你好，LangGraph"
    result = run_demo(user_text)
    state = result["state"]

    print("LangGraph Multi-Agent Demo")
    print("=" * 28)
    print(f"用户输入: {user_text}")
    print("执行路径:")
    for index, step in enumerate(result["trace"], start=1):
        print(f"  {index}. {step}")
    print("共享状态:")
    print(f"  - plan: {state['plan']}")
    print(f"  - findings: {state['findings']}")
    print(f"  - status: {state['status']}")
    print(f"  - review: {state['review']}")
    print(f"最终回复: {result['reply']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update README run description**

Edit `README.zh-CN.md` so the architecture sections describe:

```text
START -> supervisor -> planner / researcher / executor / reviewer -> supervisor -> END
```

Update expected CLI output title to:

```text
LangGraph Multi-Agent Demo
```

Update the explanation to say the project is now a mock-first multi-agent runtime rather than a single `mock_llm` node.

- [ ] **Step 5: Run full verification**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
python run_demo.py
```

Expected: all tests PASS and CLI prints the multi-agent trace plus final reply.

- [ ] **Step 6: Commit and push**

Run:

```powershell
git add tests/test_demo.py run_demo.py README.zh-CN.md
git commit -m "docs: update demo for multi-agent runtime"
git push origin main
```

If no remote exists yet, skip only the push command and record that push is blocked by missing `origin`.

---

### Task 7: Final Verification And Repository Hygiene

**Files:**
- Modify only if verification exposes a concrete issue.

- [ ] **Step 1: Check repository status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on `main`.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run CLI demo**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python run_demo.py
```

Expected: output includes `LangGraph Multi-Agent Demo`, `supervisor`, `planner`, `executor`, `reviewer`, and `最终回复:`.

- [ ] **Step 4: Inspect commit history**

Run:

```powershell
git log --oneline --decorate -8
```

Expected: one baseline commit plus one implementation commit per task.

- [ ] **Step 5: Push final state**

Run:

```powershell
git push origin main
```

Expected: push succeeds after `origin` exists and credentials are available.

