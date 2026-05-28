"""Supervisor 路由策略。

`supervisor` 是 graph 的控制节点，只负责决定下一跳和终止条件。
它不做计划、研究、执行或评审，避免控制逻辑和业务逻辑耦合。
"""

from langgraph_agent_demo.state import AgentName, AgentState


VALID_ROUTES: set[str] = {"planner", "researcher", "executor", "reviewer", "end"}


def supervisor(state: AgentState) -> dict:
    """读取当前状态并返回下一跳、计数器、状态和错误更新。"""

    trace = [*state["trace"], "supervisor"]
    iteration_count = state["iteration_count"] + 1
    next_agent = state["next_agent"]

    # Reviewer 已批准时立即结束，避免继续无意义路由。
    if state["review"]["approved"]:
        return {
            "next_agent": "end",
            "status": "approved",
            "iteration_count": iteration_count,
            "trace": trace,
        }

    # 硬性轮次上限防止 reviewer 拒绝后无限返工。
    if iteration_count > state["max_iterations"]:
        return {
            "next_agent": "end",
            "status": "max_iterations_reached",
            "iteration_count": iteration_count,
            "trace": trace,
        }

    # 未知 route 说明状态被错误写入；安全结束并留下结构化错误。
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
    """把 `next_agent` 映射为 LangGraph 条件边目标。"""

    next_agent = state["next_agent"]
    if next_agent in VALID_ROUTES:
        return next_agent  # type: ignore[return-value]
    return "end"
