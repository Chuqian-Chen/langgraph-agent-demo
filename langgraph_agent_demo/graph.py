"""LangGraph 图组装模块。

这里只负责注册节点和边，不放具体 agent 行为。
节点实现和路由策略分别位于 `agents.py` 和 `supervisor.py`。
"""

from langgraph.graph import END, START, StateGraph

from langgraph_agent_demo.agents import executor, planner, researcher, reviewer
from langgraph_agent_demo.state import AgentState
from langgraph_agent_demo.supervisor import route_next, supervisor


def build_agent_graph():
    """构建并编译 Supervisor 路由式 multi-agent graph。"""

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("executor", executor)
    builder.add_node("reviewer", reviewer)

    builder.add_edge(START, "supervisor")
    # supervisor 根据 `state["next_agent"]` 选择下一条条件边。
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
    # executor 的输出必须先由 reviewer 审批，再回到 supervisor 决定结束或返工。
    builder.add_edge("executor", "reviewer")
    builder.add_edge("reviewer", "supervisor")
    return builder.compile()
