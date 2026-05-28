"""角色 agent 节点实现。

每个函数都是一个 LangGraph 节点，只返回自己负责的状态更新。
这种边界让节点更容易测试，也避免不同角色互相覆盖状态字段。
"""

from langgraph_agent_demo.models import MockModelClient, ModelClient
from langgraph_agent_demo.state import AgentState, ReviewResult
from langgraph_agent_demo.tools import ToolRegistry, build_default_tools


def _append_trace(state: AgentState, node: str) -> list[str]:
    """返回追加当前节点名后的执行轨迹，不原地修改旧状态。"""

    return [*state["trace"], node]


def planner(state: AgentState, model: ModelClient | None = None) -> dict:
    """根据用户任务生成计划，并决定是否需要 researcher 补充上下文。"""

    client = model or MockModelClient()
    plan_text = client.generate("planner", state["task"])
    # 第一版用显式关键词触发 research，保证路由行为确定且可测试。
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
    """收集任务相关上下文，并把模型发现和工具发现写入 findings。"""

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
    """根据任务、计划和上下文生成候选结果。"""

    client = model or MockModelClient()
    result = client.generate("executor", state["task"])
    return {
        "result": result,
        "next_agent": "reviewer",
        "trace": _append_trace(state, "executor"),
    }


def reviewer(state: AgentState, model: ModelClient | None = None) -> dict:
    """评审 executor 输出，并把通过/失败信息写入 review。"""

    client = model or MockModelClient()
    client.generate("reviewer", state["result"])
    # 第一版的审批规则保持简单：非空结果通过，空结果要求返工。
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
