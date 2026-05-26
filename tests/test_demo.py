import unittest
import os
import subprocess
import sys

from langgraph_agent_demo.demo import build_agent_graph, run_demo


class LangGraphAgentDemoTest(unittest.TestCase):
    def test_agent_graph_adds_mock_ai_response(self):
        result = run_demo("你好，LangGraph")

        self.assertEqual(result["reply"], "hello world")
        self.assertEqual(result["trace"], ["START", "mock_llm", "END"])
        self.assertEqual(result["messages"][-1].content, "hello world")

    def test_build_agent_graph_returns_compiled_graph(self):
        graph = build_agent_graph()

        self.assertTrue(hasattr(graph, "invoke"))

    def test_cli_prints_chinese_when_stdout_encoding_is_narrow(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "cp1252"

        result = subprocess.run(
            [sys.executable, "run_demo.py"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertIn("最终回复: hello world", result.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
