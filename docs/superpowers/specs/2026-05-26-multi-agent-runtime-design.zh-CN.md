# Multi-Agent Runtime 设计文档

日期：2026-05-26
项目：`langgraph-agent-demo`

## 目标

把当前最小 LangGraph demo 从单一的 `START -> mock_llm -> END` 图，升级为一个 mock-first、可测试、后续可接真实模型和工具的 multi-agent runtime。

第一版需要展示真实的 multi-agent 编排，但不要求 API key、网络调用或生产部署。它也需要为受控自改进留下清晰扩展点，但不能让 agent 自动改写自己的代码或提示词。

## 选定方案

使用 Supervisor 路由式图。

图通过 `supervisor` 节点进行路由。`supervisor` 读取共享状态，决定下一个 agent，并控制终止条件。角色 agent 只负责具体工作，并返回状态更新。

高层流程：

```text
START
  -> supervisor
  -> planner | researcher | executor | reviewer
  -> supervisor
  -> ...
  -> END
```

这个方案比固定流水线更灵活，也比双图自进化系统更轻。它契合 LangGraph 的条件路由模型，同时保持 demo 可理解。

## 架构

### 节点

`supervisor`

- 读取当前 `AgentState`。
- 选择下一个 agent。
- 执行停止规则。
- 把路由决策记录到 `trace`。
- 不做具体的规划、研究、执行或评审工作。

`planner`

- 读取用户任务。
- 写入结构化计划。
- 可以标记是否需要更多上下文。

`researcher`

- 读取任务和计划。
- 在需要研究时使用工具层。
- 写入 findings。
- 第一版可以使用 mock findings。

`executor`

- 读取任务、计划和 findings。
- 产出主要结果。

`reviewer`

- 读取结果。
- 写入结构化 review，包括是否通过、分数和问题。
- 让 supervisor 决定结束或返工。

### 共享状态

使用自定义 `AgentState`，而不是只使用 `MessagesState`。

状态需要包含：

- `messages`：对话消息，用于 LangGraph 兼容。
- `task`：规范化后的用户请求。
- `plan`：planner 输出。
- `findings`：researcher 输出。
- `result`：executor 输出。
- `review`：reviewer 输出。
- `next_agent`：supervisor 的路由目标。
- `iteration_count`：循环计数器。
- `max_iterations`：安全上限，默认 3。
- `trace`：有序节点执行历史。
- `status`：当前运行状态。
- `errors`：结构化的可恢复错误和终止错误。

## 数据流

1. `run_demo(user_text)` 创建初始状态，包含 `task`、`messages`、`max_iterations` 和空 `trace`。
2. `START` 进入 `supervisor`。
3. 新任务开始时，`supervisor` 设置 `next_agent="planner"`。
4. `planner` 写入 `plan`。
5. 如果计划需要更多上下文，`supervisor` 路由到 `researcher`；否则路由到 `executor`。
6. `researcher` 写入 `findings`，然后回到 `supervisor`。
7. `executor` 写入 `result`。
8. `reviewer` 写入 `review`。
9. 如果 review 通过，`supervisor` 结束图执行。
10. 如果 review 未通过且未超过 `max_iterations`，`supervisor` 路由回 `planner` 或 `executor`。
11. 如果超过循环上限或遇到非法路由，图以安全状态结束。

公开返回结果应保留当前 demo 风格，同时暴露更丰富的状态：

```python
{
    "reply": "...",
    "trace": ["START", "supervisor", "planner", "supervisor", "executor", "reviewer", "supervisor", "END"],
    "state": {...},
}
```

## 模块结构

把当前实现拆成职责明确的小模块：

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

`demo.py`

- 保留公开 `run_demo()` API。
- 准备初始状态。
- 调用 graph。
- 为 CLI 和测试格式化最终结果。

`state.py`

- 定义 `AgentState`。
- 定义结构化 review 和 error 类型。
- 让状态字段显式、可读、可维护。

`graph.py`

- 构建 LangGraph `StateGraph`。
- 注册节点。
- 配置条件路由。
- 编译图。

`supervisor.py`

- 包含路由选择和停止规则。
- 处理未知路由目标。
- 把路由策略和 agent 工作分离。

`agents.py`

- 实现 `planner`、`researcher`、`executor` 和 `reviewer`。
- 每个 agent 只更新自己负责的字段。

`models.py`

- 定义 `ModelClient` 协议。
- 默认提供 `MockModelClient`。
- 为后续 OpenAI、Anthropic 或其他 provider 留出位置。

`tools.py`

- 定义 `ToolRegistry`。
- 第一版提供 mock 工具。
- 避免 agent 节点直接依赖具体工具实现。

## 错误处理

错误应尽量进入可观察状态，而不是失控崩溃。

使用结构化错误：

```python
{
    "node": "executor",
    "message": "...",
    "recoverable": True,
}
```

规则：

- agent 节点错误追加到 `errors`，然后返回 `supervisor`。
- 未知路由目标追加错误，并以 `status="error"` 结束。
- 超过 `max_iterations` 时以 `status="max_iterations_reached"` 结束。
- 模型失败通过模型适配器统一处理。
- 工具失败返回结构化工具错误，不把实现异常泄漏到 agent 逻辑里。

## 测试策略

测试重点是 graph 行为、状态更新和安全规则，而不是模型输出质量。

建议测试文件：

```text
tests/
  test_demo.py
  test_graph.py
  test_supervisor.py
  test_agents.py
  test_models.py
```

关键覆盖：

- `run_demo()` 返回 `reply`、`trace` 和 `state`。
- 默认 mock 流程能成功完成。
- graph 可以编译和调用。
- supervisor 把新任务路由到 `planner`。
- 需要上下文时 supervisor 路由到 `researcher`。
- 计划就绪时 supervisor 路由到 `executor`。
- reviewer 通过后进入 `END`。
- reviewer 拒绝后进入有上限的返工流程。
- `max_iterations` 防止无限循环。
- 未知路由目标安全结束。
- 每个 agent 只更新自己负责的状态字段。
- `MockModelClient` 返回确定性响应。

验证命令：

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
python run_demo.py
```

## 受控自改进扩展

第一版不实现自主自修改。它只保留后续扩展需要的数据。

为未来 meta layer 保留：

- `trace`：发生了什么。
- `review`：结果质量如何。
- `errors`：哪里失败了。
- `status`：为什么结束。
- 未来可选 `metrics`：分数、延迟、重试次数、工具结果等。

未来扩展：

```text
Task Graph result + trace
  -> Eval
  -> Reflection
  -> ImprovementProposal
  -> human approval
  -> config or prompt update
```

所有自改进都应受控、可审计、可 review。

## 第一版不包含

- 真实模型 provider 集成。
- 真实 web search 或代码执行工具。
- 持久化长期记忆数据库。
- LangGraph Platform 部署。
- 自动代码修改。
- 未经审批的自动提示词更新。
- CLI demo 之外的 UI。

## 验收标准

- 现有 `run_demo()` 仍然可用。
- CLI 不需要 API key 即可运行。
- 单元测试通过，mock 行为确定。
- trace 能清晰展示 multi-agent 执行。
- state 暴露 plan、result、review、status 和 errors。
- supervisor 能处理通过、拒绝、未知路由和最大轮次终止。
- 设计为后续真实模型和工具适配器保留扩展能力。
