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
