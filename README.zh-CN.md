# LangGraph Multi-Agent Demo

这个项目从 LangGraph 的最小示例演进而来。当前目标不再是单个 `mock_llm` 节点，而是一个可测试、可扩展、mock-first 的 multi-agent runtime。

第一版不需要 API key，不调用真实模型，也不依赖网络。它用确定性的 mock 模型和 mock 工具展示这些核心概念：

- 共享状态如何在多个 agent 节点之间流动。
- Supervisor 如何根据状态选择下一步。
- Planner、Researcher、Executor、Reviewer 如何分工协作。
- 如何为后续真实模型、工具、记忆和受控自进化预留接口。

## 当前图结构

高层执行路径：

```text
START
  -> supervisor
  -> planner | researcher | executor | reviewer
  -> supervisor
  -> ...
  -> END
```

默认简单任务通常走：

```text
START -> supervisor -> planner -> supervisor -> executor -> reviewer -> supervisor -> END
```

如果任务文本包含 `research`，planner 会要求补充上下文，路径会包含 researcher：

```text
START -> supervisor -> planner -> supervisor -> researcher -> supervisor -> executor -> reviewer -> supervisor -> END
```

## 核心模块

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
```

- `state.py`：定义 `AgentState`、review 和错误结构。
- `models.py`：定义 `ModelClient` 协议和 `MockModelClient`。
- `tools.py`：定义 `ToolRegistry` 和默认 mock research 工具。
- `agents.py`：实现 `planner`、`researcher`、`executor`、`reviewer`。
- `supervisor.py`：实现路由、停止条件、错误路由保护。
- `graph.py`：组装 LangGraph `StateGraph`。
- `demo.py`：保留对外 `run_demo()` API。

## AgentState

multi-agent runtime 共享一个状态对象，主要字段包括：

- `messages`：对话消息，保留 LangGraph/LangChain 兼容性。
- `task`：用户任务。
- `plan`：planner 产出的计划。
- `needs_research`：是否需要 researcher。
- `findings`：researcher 产出的上下文。
- `result`：executor 产出的结果。
- `review`：reviewer 的审批结果、分数和问题。
- `next_agent`：supervisor 的下一跳。
- `iteration_count`：路由轮次。
- `max_iterations`：最大轮次，默认 3。
- `trace`：执行路径。
- `status`：运行状态。
- `errors`：结构化错误列表。

## 运行方式

进入项目目录：

```powershell
cd E:\test_for_codex\langgraph-agent-demo
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果你想把依赖放在项目内 `.deps` 目录：

```powershell
python -m pip install -r requirements.txt --target .deps
```

运行测试：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
```

运行 demo：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python run_demo.py
```

预期输出类似：

```text
LangGraph Multi-Agent Demo
============================
用户输入: 你好，LangGraph
执行路径:
  1. START
  2. supervisor
  3. planner
  4. supervisor
  5. executor
  6. reviewer
  7. supervisor
  8. END
共享状态:
  - plan: [...]
  - findings: []
  - status: approved
  - review: {'approved': True, 'score': 1.0, 'issues': []}
最终回复: Result: completed response for 你好，LangGraph
```

## 设计文档

设计和实施计划在这里：

```text
docs/superpowers/specs/2026-05-26-multi-agent-runtime-design.md
docs/superpowers/plans/2026-05-26-multi-agent-runtime.md
```

## 后续方向

第一版只实现受控、可测试的 runtime 骨架。后续可以继续扩展：

- 接入真实 LLM provider。
- 增加真实工具，例如搜索、文件读取、代码执行。
- 接 LangGraph checkpoint，支持恢复和持久运行。
- 增加 Eval、Reflection、ImprovementProposal 形成受控自进化闭环。
- 在关键改进点加入人工审批。
