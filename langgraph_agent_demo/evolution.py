"""受控自进化 meta layer。

该模块只读取主任务 graph 的最终 `AgentState`，生成评估、反思和改进提案。
它不写文件、不修改 prompt、不修改配置，也不改变传入的 state。
"""

from typing import Literal, TypedDict

from langgraph_agent_demo.state import AgentState


ProposalTarget = Literal["prompt", "config", "tooling", "routing", "tests"]


class EvaluationResult(TypedDict):
    """对一次任务执行结果的确定性评估。"""

    score: float
    passed: bool
    signals: list[str]


class ReflectionResult(TypedDict):
    """根据评估信号生成的结构化复盘。"""

    strengths: list[str]
    weaknesses: list[str]
    root_causes: list[str]


class ImprovementProposal(TypedDict):
    """需要人工审批的改进提案；第一版永远不会自动应用。"""

    title: str
    rationale: str
    target: ProposalTarget
    change: str
    requires_human_approval: bool
    applied: bool


class EvolutionResult(TypedDict):
    """受控自进化流程的完整输出。"""

    evaluation: EvaluationResult
    reflection: ReflectionResult
    proposal: ImprovementProposal


def run_controlled_evolution(state: AgentState) -> EvolutionResult:
    """评估主任务最终状态，并返回只读改进建议。"""

    evaluation = _evaluate_state(state)
    reflection = _reflect_on_evaluation(state, evaluation)
    proposal = _propose_improvement(evaluation)
    return {
        "evaluation": evaluation,
        "reflection": reflection,
        "proposal": proposal,
    }


def _evaluate_state(state: AgentState) -> EvaluationResult:
    """把 `AgentState` 转换为稳定可断言的质量信号。"""

    signals: list[str] = []
    score = 0.0

    if state["status"] == "approved" and state["review"]["approved"]:
        signals.append("review_approved")
        score += 0.6
    else:
        signals.append(f"status_{state['status']}")

    if state["result"].strip():
        signals.append("result_present")
        score += 0.25
    else:
        signals.append("result_empty")

    if state["trace"]:
        signals.append("trace_available")
        score += 0.1

    if state["errors"]:
        signals.append("errors_present")
        score -= 0.35

    if state["status"] == "max_iterations_reached":
        signals.append("max_iterations_reached")
        score -= 0.25

    score = max(0.0, min(1.0, round(score, 2)))
    return {
        "score": score,
        "passed": score >= 0.8 and "errors_present" not in signals,
        "signals": signals,
    }


def _reflect_on_evaluation(state: AgentState, evaluation: EvaluationResult) -> ReflectionResult:
    """根据评估信号解释当前执行的优势、弱点和可能根因。"""

    strengths: list[str] = []
    weaknesses: list[str] = []
    root_causes: list[str] = []
    signals = set(evaluation["signals"])

    if "review_approved" in signals:
        strengths.append("主任务结果已通过 reviewer 审批。")
    if "trace_available" in signals:
        strengths.append("执行轨迹完整，可用于后续审计和复盘。")
    if "result_empty" in signals:
        weaknesses.append("executor 没有产出可用结果。")
        root_causes.append("executor/reviewer 缺少对空结果的更强质量门禁。")
    if "errors_present" in signals:
        weaknesses.append("执行过程中出现结构化错误。")
        root_causes.append("错误路径需要补充更具体的恢复策略和回归测试。")
    if "max_iterations_reached" in signals:
        weaknesses.append("任务达到最大轮次限制后才结束。")
        root_causes.append("路由或 retry 策略需要更早识别无效返工。")

    if not weaknesses:
        weaknesses.append("第一版 mock runtime 仍缺少真实任务质量信号。")
        root_causes.append("当前评估只基于 deterministic state，尚未接入真实 eval 数据。")

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "root_causes": root_causes,
    }


def _propose_improvement(evaluation: EvaluationResult) -> ImprovementProposal:
    """根据失败优先级生成一个需要人工审批的改进提案。"""

    signals = set(evaluation["signals"])

    if "errors_present" in signals:
        title = "补充错误路径测试和恢复策略"
        target: ProposalTarget = "tests"
        rationale = "执行状态包含 errors，需要先让错误路径可复现、可验证。"
        change = "为产生错误的节点增加回归测试，并定义 supervisor 如何选择恢复或终止。"
    elif "max_iterations_reached" in signals:
        title = "收紧 routing/retry 终止策略"
        target = "routing"
        rationale = "任务达到最大轮次后才结束，说明返工路径缺少提前退出判断。"
        change = "在 supervisor 中增加更具体的返工原因检查，减少无效循环。"
    elif "result_empty" in signals:
        title = "补充 executor/reviewer 空结果质量门禁"
        target = "tests"
        rationale = "空结果会导致任务无法形成有效输出，需要更明确的质量保护。"
        change = "增加空结果测试，并让 reviewer 对空结果给出可执行的修复问题。"
    else:
        title = "增加真实任务质量评估信号"
        target = "prompt"
        rationale = "当前结果通过了 deterministic 评估，但仍缺少真实模型和真实任务质量信号。"
        change = "后续接入真实模型后，补充面向准确性、完整性和工具使用质量的 eval prompt。"

    return {
        "title": title,
        "rationale": rationale,
        "target": target,
        "change": change,
        "requires_human_approval": True,
        "applied": False,
    }
