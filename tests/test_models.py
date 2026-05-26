import unittest

from langgraph_agent_demo.models import MockModelClient
from langgraph_agent_demo.tools import ToolRegistry


class AdapterTest(unittest.TestCase):
    def test_mock_model_returns_role_specific_text(self):
        model = MockModelClient()

        self.assertIn("Plan", model.generate("planner", "Build a report"))
        self.assertIn("Result", model.generate("executor", "Build a report"))
        self.assertIn("approved", model.generate("reviewer", "Build a report"))

    def test_tool_registry_runs_registered_tool(self):
        registry = ToolRegistry()
        registry.register("research", lambda query: f"finding for {query}")

        self.assertEqual(registry.run("research", "LangGraph"), "finding for LangGraph")

    def test_tool_registry_reports_missing_tool(self):
        registry = ToolRegistry()

        result = registry.run("missing", "LangGraph")

        self.assertIn("Tool not found", result)


if __name__ == "__main__":
    unittest.main()
