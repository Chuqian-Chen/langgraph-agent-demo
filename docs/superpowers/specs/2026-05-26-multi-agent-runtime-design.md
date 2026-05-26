# Multi-Agent Runtime Design

Date: 2026-05-26
Project: `langgraph-agent-demo`

## Goal

Upgrade the current minimal LangGraph demo from a single `START -> mock_llm -> END` graph into a mock-first, testable multi-agent runtime that can later connect to real model providers and tools.

The first version should demonstrate real multi-agent orchestration without requiring API keys, network calls, or production deployment. It should also leave clean extension points for controlled self-improvement, but it should not let the agent rewrite its own code or prompts automatically.

## Chosen Approach

Use a Supervisor-routed graph.

The graph should route through a `supervisor` node that reads shared state, decides the next agent, and controls termination. Role agents perform focused work and return state updates.

High-level flow:

```text
START
  -> supervisor
  -> planner | researcher | executor | reviewer
  -> supervisor
  -> ...
  -> END
```

This approach is more flexible than a fixed pipeline and lighter than a two-graph self-evolution system. It fits LangGraph's conditional routing model and lets the demo remain understandable.

## Architecture

### Nodes

`supervisor`

- Reads the current `AgentState`.
- Chooses the next agent.
- Enforces stopping rules.
- Records routing decisions in `trace`.
- Does not perform task-specific planning, research, execution, or review.

`planner`

- Reads the user task.
- Writes a structured plan.
- May mark that more context is needed.

`researcher`

- Reads the task and plan.
- Uses the tool layer when research is needed.
- Writes findings.
- In the first version, this can use mock findings.

`executor`

- Reads task, plan, and findings.
- Produces the main result.

`reviewer`

- Reads the result.
- Writes a structured review with approval, score, and issues.
- Allows the supervisor to decide whether to finish or route back for revision.

### Shared State

Use a custom `AgentState` rather than only `MessagesState`.

The state should include:

- `messages`: conversation messages for LangGraph compatibility.
- `task`: normalized user request.
- `plan`: planner output.
- `findings`: researcher output.
- `result`: executor output.
- `review`: reviewer output.
- `next_agent`: supervisor routing target.
- `iteration_count`: loop counter.
- `max_iterations`: safety limit, default 3.
- `trace`: ordered node execution history.
- `status`: current run status.
- `errors`: structured recoverable and terminal errors.

## Data Flow

1. `run_demo(user_text)` creates the initial state with `task`, `messages`, `max_iterations`, and an empty `trace`.
2. `START` enters `supervisor`.
3. `supervisor` sets `next_agent="planner"` for a new task.
4. `planner` writes `plan`.
5. `supervisor` routes to `researcher` if the plan needs more context, otherwise to `executor`.
6. `researcher` writes `findings` and returns to `supervisor`.
7. `executor` writes `result`.
8. `reviewer` writes `review`.
9. `supervisor` ends when the review is approved.
10. If the review is not approved and the run is below `max_iterations`, `supervisor` routes back to `planner` or `executor`.
11. If the run exceeds the limit or reaches an invalid route, the graph ends with a safe status.

The public result should keep the current demo style while exposing richer state:

```python
{
    "reply": "...",
    "trace": ["START", "supervisor", "planner", "supervisor", "executor", "reviewer", "supervisor", "END"],
    "state": {...},
}
```

## Module Structure

Split the current implementation into focused modules:

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

- Keeps the public `run_demo()` API.
- Prepares initial state.
- Invokes the graph.
- Formats the final result for CLI and tests.

`state.py`

- Defines `AgentState`.
- Defines structured review and error types.
- Keeps state keys explicit and documented.

`graph.py`

- Builds the LangGraph `StateGraph`.
- Registers nodes.
- Wires conditional routing.
- Compiles the graph.

`supervisor.py`

- Contains route selection and stopping rules.
- Handles unknown route targets.
- Keeps routing policy separate from agent work.

`agents.py`

- Implements `planner`, `researcher`, `executor`, and `reviewer`.
- Each agent updates only its own fields.

`models.py`

- Defines a `ModelClient` protocol.
- Provides `MockModelClient` as the default.
- Leaves room for OpenAI, Anthropic, or other providers later.

`tools.py`

- Defines a `ToolRegistry`.
- Provides mock tools for the first version.
- Prevents agent nodes from depending directly on concrete tool implementations.

## Error Handling

Errors should become observable state rather than uncontrolled crashes where possible.

Use structured errors:

```python
{
    "node": "executor",
    "message": "...",
    "recoverable": True,
}
```

Rules:

- Agent node errors append to `errors` and return to `supervisor`.
- Unknown route targets append an error and end with `status="error"`.
- Exceeding `max_iterations` ends with `status="max_iterations_reached"`.
- Model failures are handled through the model adapter.
- Tool failures return structured tool errors rather than leaking implementation exceptions into agent logic.

## Testing Strategy

Testing should focus on graph behavior, state updates, and safety rules rather than model output quality.

Proposed tests:

```text
tests/
  test_demo.py
  test_graph.py
  test_supervisor.py
  test_agents.py
  test_models.py
```

Key coverage:

- `run_demo()` returns `reply`, `trace`, and `state`.
- The default mock run completes successfully.
- The graph compiles and can be invoked.
- Supervisor routes a new task to `planner`.
- Supervisor routes to `researcher` when more context is needed.
- Supervisor routes to `executor` when the plan is ready.
- Reviewer approval leads to `END`.
- Reviewer rejection causes bounded revision.
- `max_iterations` prevents infinite loops.
- Unknown route targets end safely.
- Each agent updates only its owned state fields.
- `MockModelClient` gives deterministic responses.

Verification commands:

```powershell
$env:PYTHONPATH=(Resolve-Path .deps).Path
python -m unittest discover -s tests -v
python run_demo.py
```

## Controlled Self-Improvement Extension

The first version should not implement autonomous self-modification. It should only preserve the data needed to add that later.

Prepare for a future meta layer by keeping:

- `trace`: what happened.
- `review`: how good the result was.
- `errors`: what failed.
- `status`: why the run ended.
- Optional future `metrics`: scores, latency, retry counts, and tool outcomes.

Future extension:

```text
Task Graph result + trace
  -> Eval
  -> Reflection
  -> ImprovementProposal
  -> human approval
  -> config or prompt update
```

Any self-improvement should be controlled, auditable, and reviewable.

## Out of Scope For First Version

- Real model provider integration.
- Real web search or code execution tools.
- Persistent long-term memory database.
- LangGraph Platform deployment.
- Automatic code modification.
- Automatic prompt updates without approval.
- UI beyond the existing CLI demo.

## Acceptance Criteria

- Existing `run_demo()` remains available.
- CLI still runs without API keys.
- Unit tests pass with deterministic mock behavior.
- Trace clearly shows multi-agent execution.
- State exposes plan, result, review, status, and errors.
- Supervisor handles approval, rejection, unknown routes, and max-iteration termination.
- The design remains ready for real model and tool adapters.
