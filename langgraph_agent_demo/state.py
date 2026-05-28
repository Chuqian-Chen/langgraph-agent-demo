"""定义 multi-agent runtime 的共享状态结构。

LangGraph 节点之间不直接共享局部变量，而是通过 `AgentState`
传递任务、计划、执行结果、评审结果和路由信息。
"""

from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage


AgentName = Literal["planner", "researcher", "executor", "reviewer", "end"]
RunStatus = Literal["running", "approved", "max_iterations_reached", "error"]


class ReviewResult(TypedDict):
    """Reviewer 对 executor 输出的结构化评审结果。"""

    approved: bool
    score: float
    issues: list[str]


class AgentError(TypedDict):
    """节点执行或路由过程中产生的可观察错误。"""

    node: str
    message: str
    recoverable: bool


class AgentState(TypedDict):
    """所有 LangGraph 节点共享和更新的状态对象。

    字段说明：
    - `messages` 保留 LangChain message 兼容性。
    - `task` 是用户原始任务文本。
    - `plan` 是 planner 产出的执行计划。
    - `needs_research` 决定是否需要 researcher 节点补上下文。
    - `findings` 保存 researcher 从模型或工具拿到的信息。
    - `result` 是 executor 产出的最终候选答案。
    - `review` 是 reviewer 对 `result` 的质量判断。
    - `next_agent` 是 supervisor 的下一跳路由目标。
    - `iteration_count` 和 `max_iterations` 防止循环失控。
    - `trace` 记录节点执行顺序，便于调试和后续评估。
    - `status` 说明当前 run 的结束或运行状态。
    - `errors` 保存结构化错误，避免异常信息散落在节点逻辑里。
    """

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
    """把用户输入转换成 graph 可执行的初始状态。"""

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
