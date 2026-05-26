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
