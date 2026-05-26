from typing import Protocol


class ModelClient(Protocol):
    def generate(self, role: str, prompt: str) -> str:
        ...


class MockModelClient:
    def generate(self, role: str, prompt: str) -> str:
        if role == "planner":
            return f"Plan: clarify goal, gather context, produce answer for {prompt}"
        if role == "researcher":
            return f"Findings: useful context for {prompt}"
        if role == "executor":
            return f"Result: completed response for {prompt}"
        if role == "reviewer":
            return "Review: approved with score 1.0"
        return f"Mock response for {role}: {prompt}"
