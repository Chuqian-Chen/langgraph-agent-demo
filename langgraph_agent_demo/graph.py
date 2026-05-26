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
