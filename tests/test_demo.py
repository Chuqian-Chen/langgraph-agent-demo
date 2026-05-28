import os
import subprocess
import sys
import unittest

from langgraph_agent_demo.demo import run_demo
from langgraph_agent_demo.graph import build_agent_graph


class LangGraphAgentDemoTest(unittest.TestCase):
    """验证公开 demo API 和 CLI 仍然可用。"""

    def test_agent_graph_runs_multi_agent_runtime(self):
        """`run_demo` 应返回主任务结果和受控自进化结果。"""

        result = run_demo("你好，LangGraph")

        self.assertIn("Result", result["reply"])
        self.assertEqual(result["trace"][0], "START")
        self.assertEqual(result["trace"][-1], "END")
        self.assertIn("supervisor", result["trace"])
        self.assertIn("planner", result["trace"])
        self.assertIn("executor", result["trace"])
        self.assertIn("reviewer", result["trace"])
        self.assertEqual(result["state"]["status"], "approved")
        self.assertIn("evaluation", result["evolution"])
        self.assertIn("reflection", result["evolution"])
        self.assertIn("proposal", result["evolution"])
        self.assertTrue(result["evolution"]["proposal"]["requires_human_approval"])
        self.assertFalse(result["evolution"]["proposal"]["applied"])

    def test_build_agent_graph_returns_compiled_graph(self):
        """公开 graph 构建函数应返回可 invoke 的 LangGraph runnable。"""

        graph = build_agent_graph()

        self.assertTrue(hasattr(graph, "invoke"))

    def test_cli_prints_chinese_when_stdout_encoding_is_narrow(self):
        """CLI 在窄 stdout 编码下仍应输出中文和受控自进化状态。"""

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "cp1252"

        result = subprocess.run(
            [sys.executable, "run_demo.py"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        stdout = result.stdout.decode("utf-8")
        self.assertIn("最终回复:", stdout)
        self.assertIn("受控自进化", stdout)
        self.assertIn("requires_human_approval: True", stdout)
        self.assertIn("applied: False", stdout)


if __name__ == "__main__":
    unittest.main()
