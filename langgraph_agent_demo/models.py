"""模型适配层。

节点依赖 `ModelClient` 协议，而不是直接依赖某个具体 LLM provider。
第一版使用确定性的 `MockModelClient`，保证测试和 demo 不需要 API key。
"""

from typing import Protocol


class ModelClient(Protocol):
    """所有模型客户端必须实现的最小接口。"""

    def generate(self, role: str, prompt: str) -> str:
        """根据 agent 角色和 prompt 返回文本结果。"""

        ...


class MockModelClient:
    """离线、确定性的模型客户端，用于测试和教学 demo。"""

    def generate(self, role: str, prompt: str) -> str:
        """返回可预测的角色化文本，避免测试依赖真实模型波动。"""

        if role == "planner":
            return f"Plan: clarify goal, gather context, produce answer for {prompt}"
        if role == "researcher":
            return f"Findings: useful context for {prompt}"
        if role == "executor":
            return f"Result: completed response for {prompt}"
        if role == "reviewer":
            return "Review: approved with score 1.0"
        return f"Mock response for {role}: {prompt}"
