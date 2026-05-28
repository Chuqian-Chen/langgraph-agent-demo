"""公开 demo API。

`run_demo` 是 CLI、测试和外部调用者使用的稳定入口。
它隐藏 LangGraph 初始化细节，只返回 reply、trace 和完整 state。
"""

from typing import Any, TypedDict

from langgraph_agent_demo.evolution import EvolutionResult, run_controlled_evolution
from langgraph_agent_demo.graph import build_agent_graph
from langgraph_agent_demo.state import make_initial_state


class DemoResult(TypedDict):
    """`run_demo` 的公开返回结构。"""

    reply: str
    trace: list[str]
    state: dict[str, Any]
    evolution: EvolutionResult


def run_demo(user_text: str) -> DemoResult:
    """运行一次 multi-agent demo，并返回主任务结果和受控自进化结果。"""

    graph = build_agent_graph()
    final_state = graph.invoke(make_initial_state(user_text))
    trace = [*final_state["trace"], "END"]
    return {
        "reply": final_state["result"],
        "trace": trace,
        "state": final_state,
        "evolution": run_controlled_evolution(final_state),
    }
