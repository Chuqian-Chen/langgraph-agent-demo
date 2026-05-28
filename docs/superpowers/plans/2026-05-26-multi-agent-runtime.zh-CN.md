# Multi-Agent Runtime 实施计划

> **给 agent worker 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`。所有步骤使用 checkbox（`- [ ]`）格式便于追踪。

**目标：** 构建一个 mock-first、Supervisor 路由式 LangGraph multi-agent runtime，同时保留现有 `run_demo()` 入口。

**架构：** runtime 使用共享 `AgentState`、`supervisor` 路由器、职责聚焦的角色 agent，以及模型和工具适配层。第一版保持离线和确定性；真实 provider、工具和自改进能力通过接口和状态字段预留。

**技术栈：** Python、`unittest`、LangGraph `StateGraph`、LangChain message classes、标准库 typed dictionaries 和 protocols。

---

## 文件结构

创建和修改这些文件：

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

职责：

- `state.py`：共享状态、review、error 类型定义。
- `models.py`：模型协议和确定性的 mock 实现。
- `tools.py`：工具注册表和确定性的 mock research 工具。
- `agents.py`：planner、researcher、executor、reviewer 节点函数。
- `supervisor.py`：路由和停止策略。
- `graph.py`：LangGraph 图组装和条件边。
- `demo.py`：公开 `build_agent_graph()` 和 `run_demo()` 兼容层。
- tests：按模块拆分测试，同时覆盖 CLI/demo 集成行为。

提交和推送策略：

- 每完成一个任务提交一次。
- 远程仓库配置好后，每个提交后执行 `git push origin main`。
- 每个提交都应足够小，便于通过 diff review。

---

### Task 1：定义共享状态类型

**文件：**
- 创建：`langgraph_agent_demo/state.py`
- 测试：`tests/test_agents.py`

- [ ] **Step 1：先写失败测试，锁定状态字段默认值**

创建 `tests/test_agents.py`：

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

- [ ] **Step 2：运行测试，确认失败**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents.AgentStateTest.test_make_initial_state_sets_defaults -v
```

预期：FAIL，原因是 `state.py` 还不存在，报 `ModuleNotFoundError` 或 `ImportError`。

- [ ] **Step 3：实现状态定义**

创建 `langgraph_agent_demo/state.py`：

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

- [ ] **Step 4：运行测试，确认通过**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents.AgentStateTest.test_make_initial_state_sets_defaults -v
```

预期：PASS。

- [ ] **Step 5：提交并推送**

运行：

```powershell
git add langgraph_agent_demo/state.py tests/test_agents.py
git commit -m "feat: add shared agent state"
git push origin main
```

如果还没有远程仓库，只跳过 push，并记录 push 被缺少 `origin` 阻塞。

---

### Task 2：添加 Mock 模型和工具适配器

**文件：**
- 创建：`langgraph_agent_demo/models.py`
- 创建：`langgraph_agent_demo/tools.py`
- 测试：`tests/test_models.py`

- [ ] **Step 1：先写适配器失败测试**

创建 `tests/test_models.py`：

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

- [ ] **Step 2：运行测试，确认失败**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_models -v
```

预期：FAIL，因为 `models.py` 和 `tools.py` 还不存在。

- [ ] **Step 3：实现模型适配器**

创建 `langgraph_agent_demo/models.py`：

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

- [ ] **Step 4：实现工具注册表**

创建 `langgraph_agent_demo/tools.py`：

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

- [ ] **Step 5：运行测试，确认通过**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_models -v
```

预期：PASS。

- [ ] **Step 6：提交并推送**

运行：

```powershell
git add langgraph_agent_demo/models.py langgraph_agent_demo/tools.py tests/test_models.py
git commit -m "feat: add mock model and tool adapters"
git push origin main
```

如果还没有远程仓库，只跳过 push，并记录 push 被缺少 `origin` 阻塞。

---

### Task 3：实现角色 Agent 节点

**文件：**
- 修改：`tests/test_agents.py`
- 创建：`langgraph_agent_demo/agents.py`

- [ ] **Step 1：添加角色 agent 的失败测试**

追加到 `tests/test_agents.py`：

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

- [ ] **Step 2：运行测试，确认失败**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents -v
```

预期：FAIL，因为 `agents.py` 还不存在。

- [ ] **Step 3：实现角色 agent**

创建 `langgraph_agent_demo/agents.py`：

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

- [ ] **Step 4：运行测试，确认通过**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_agents -v
```

预期：PASS。

- [ ] **Step 5：提交并推送**

运行：

```powershell
git add langgraph_agent_demo/agents.py tests/test_agents.py
git commit -m "feat: add role agent nodes"
git push origin main
```

如果还没有远程仓库，只跳过 push，并记录 push 被缺少 `origin` 阻塞。

---

### Task 4：实现 Supervisor 路由

**文件：**
- 创建：`tests/test_supervisor.py`
- 创建：`langgraph_agent_demo/supervisor.py`

- [ ] **Step 1：先写 supervisor 失败测试**

创建 `tests/test_supervisor.py`：

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

- [ ] **Step 2：运行测试，确认失败**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_supervisor -v
```

预期：FAIL，因为 `supervisor.py` 还不存在。

- [ ] **Step 3：实现 supervisor**

创建 `langgraph_agent_demo/supervisor.py`：

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

- [ ] **Step 4：运行测试，确认通过**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_supervisor -v
```

预期：PASS。

- [ ] **Step 5：提交并推送**

运行：

```powershell
git add langgraph_agent_demo/supervisor.py tests/test_supervisor.py
git commit -m "feat: add supervisor routing"
git push origin main
```

如果还没有远程仓库，只跳过 push，并记录 push 被缺少 `origin` 阻塞。

---

### Task 5：接入 LangGraph Runtime

**文件：**
- 创建：`tests/test_graph.py`
- 创建：`langgraph_agent_demo/graph.py`
- 修改：`langgraph_agent_demo/demo.py`
- 修改：`langgraph_agent_demo/__init__.py`

- [ ] **Step 1：先写 graph 失败测试**

创建 `tests/test_graph.py`：

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

- [ ] **Step 2：运行测试，确认失败**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_graph -v
```

预期：FAIL，因为 `graph.py` 还不存在。

- [ ] **Step 3：实现 graph 接线**

创建 `langgraph_agent_demo/graph.py`：

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

- [ ] **Step 4：更新公开 demo API**

替换 `langgraph_agent_demo/demo.py`：

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

更新 `langgraph_agent_demo/__init__.py`：

```python
from .demo import run_demo
from .graph import build_agent_graph

__all__ = ["build_agent_graph", "run_demo"]
```

- [ ] **Step 5：运行 graph 测试**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_graph -v
```

预期：PASS。

- [ ] **Step 6：提交并推送**

运行：

```powershell
git add langgraph_agent_demo/graph.py langgraph_agent_demo/demo.py langgraph_agent_demo/__init__.py tests/test_graph.py
git commit -m "feat: wire supervisor routed graph"
git push origin main
```

如果还没有远程仓库，只跳过 push，并记录 push 被缺少 `origin` 阻塞。

---

### Task 6：更新 Demo 和兼容测试

**文件：**
- 修改：`tests/test_demo.py`
- 修改：`run_demo.py`
- 修改：`README.zh-CN.md`

- [ ] **Step 1：更新新结果结构的测试**

替换 `tests/test_demo.py`：

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

- [ ] **Step 2：运行测试，确认 CLI 仍需更新**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest tests.test_demo -v
```

预期：如果 `run_demo.py` 仍然读取 `messages`，测试 FAIL；只有 CLI 更新后才 PASS。

- [ ] **Step 3：更新 CLI 输出**

替换 `run_demo.py`：

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

- [ ] **Step 4：更新 README 运行说明**

修改 `README.zh-CN.md`，让架构部分描述：

```text
START -> supervisor -> planner / researcher / executor / reviewer -> supervisor -> END
```

把预期 CLI 输出标题改成：

```text
LangGraph Multi-Agent Demo
```

说明项目现在是 mock-first multi-agent runtime，而不是单个 `mock_llm` 节点。

- [ ] **Step 5：运行完整验证**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
python run_demo.py
```

预期：所有测试 PASS，CLI 打印 multi-agent trace 和最终回复。

- [ ] **Step 6：提交并推送**

运行：

```powershell
git add tests/test_demo.py run_demo.py README.zh-CN.md
git commit -m "docs: update demo for multi-agent runtime"
git push origin main
```

如果还没有远程仓库，只跳过 push，并记录 push 被缺少 `origin` 阻塞。

---

### Task 7：最终验证和仓库卫生

**文件：**
- 只有验证发现具体问题时才修改文件。

- [ ] **Step 1：检查仓库状态**

运行：

```powershell
git status --short --branch
```

预期：`main` 上工作区干净。

- [ ] **Step 2：运行完整测试套件**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
```

预期：所有测试 PASS。

- [ ] **Step 3：运行 CLI demo**

运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python run_demo.py
```

预期：输出包含 `LangGraph Multi-Agent Demo`、`supervisor`、`planner`、`executor`、`reviewer` 和 `最终回复:`。

- [ ] **Step 4：检查提交历史**

运行：

```powershell
git log --oneline --decorate -8
```

预期：一条 baseline commit，加上每个任务一条 implementation commit。

- [ ] **Step 5：推送最终状态**

运行：

```powershell
git push origin main
```

预期：`origin` 存在且凭据可用时，push 成功。
