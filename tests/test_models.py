import unittest

from langgraph_agent_demo.models import MockModelClient
from langgraph_agent_demo.tools import ToolRegistry


class AdapterTest(unittest.TestCase):
    """验证模型和工具适配层的确定性行为。"""

    def test_mock_model_returns_role_specific_text(self):
        """Mock 模型应根据 agent 角色返回可断言的固定文本。"""

        model = MockModelClient()

        self.assertIn("Plan", model.generate("planner", "Build a report"))
        self.assertIn("Result", model.generate("executor", "Build a report"))
        self.assertIn("approved", model.generate("reviewer", "Build a report"))

    def test_tool_registry_runs_registered_tool(self):
        """工具注册表应能按名称执行已注册工具。"""

        registry = ToolRegistry()
        registry.register("research", lambda query: f"finding for {query}")

        self.assertEqual(registry.run("research", "LangGraph"), "finding for LangGraph")

    def test_tool_registry_reports_missing_tool(self):
        """缺失工具应返回可观察错误文本，而不是抛异常。"""

        registry = ToolRegistry()

        result = registry.run("missing", "LangGraph")

        self.assertIn("Tool not found", result)


if __name__ == "__main__":
    unittest.main()
