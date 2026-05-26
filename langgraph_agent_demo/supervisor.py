from langgraph_agent_demo.state import AgentName, AgentState


VALID_ROUTES: set[str] = {"planner", "researcher", "executor", "reviewer", "end"}


def supervisor(state: AgentState) -> dict:
    trace = [*state["trace"], "supervisor"]
    iteration_count = state["iteration_count"] + 1
    next_agent = state["next_agent"]

    if state["review"]["approved"]:
        return {
            "next_agent": "end",
            "status": "approved",
            "iteration_count": iteration_count,
            "trace": trace,
        }

    if iteration_count > state["max_iterations"]:
        return {
            "next_agent": "end",
            "status": "max_iterations_reached",
            "iteration_count": iteration_count,
            "trace": trace,
        }

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
    next_agent = state["next_agent"]
    if next_agent in VALID_ROUTES:
        return next_agent  # type: ignore[return-value]
    return "end"
