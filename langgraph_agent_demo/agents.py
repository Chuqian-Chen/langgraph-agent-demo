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
    client.generate("reviewer", state["result"])
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
