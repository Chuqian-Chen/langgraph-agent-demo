import copy
import unittest

from langgraph_agent_demo.evolution import run_controlled_evolution
from langgraph_agent_demo.state import make_initial_state


class ControlledEvolutionTest(unittest.TestCase):
    """验证受控自进化 meta layer 只生成评估、反思和提案。"""

    def test_approved_result_gets_passing_evaluation(self):
        """已通过的主任务结果应得到高分评估和只读改进提案。"""

        state = make_initial_state("Build a report")
        state["result"] = "A completed result"
        state["review"] = {"approved": True, "score": 1.0, "issues": []}
        state["status"] = "approved"

        evolution = run_controlled_evolution(state)

        self.assertTrue(evolution["evaluation"]["passed"])
        self.assertGreaterEqual(evolution["evaluation"]["score"], 0.9)
        self.assertIn("review_approved", evolution["evaluation"]["signals"])
        self.assertTrue(evolution["proposal"]["requires_human_approval"])
        self.assertFalse(evolution["proposal"]["applied"])

    def test_error_state_generates_error_handling_proposal(self):
        """错误状态应生成面向错误处理或测试补强的改进提案。"""

        state = make_initial_state("Build a report")
        state["status"] = "error"
        state["errors"] = [
            {"node": "supervisor", "message": "Unknown route: unknown", "recoverable": False}
        ]

        evolution = run_controlled_evolution(state)

        self.assertFalse(evolution["evaluation"]["passed"])
        self.assertIn("errors_present", evolution["evaluation"]["signals"])
        self.assertEqual(evolution["proposal"]["target"], "tests")
        self.assertIn("错误", evolution["proposal"]["title"])

    def test_max_iteration_state_generates_routing_proposal(self):
        """达到最大轮次时应生成 routing/retry 方向的改进提案。"""

        state = make_initial_state("Build a report")
        state["status"] = "max_iterations_reached"
        state["iteration_count"] = state["max_iterations"] + 1

        evolution = run_controlled_evolution(state)

        self.assertFalse(evolution["evaluation"]["passed"])
        self.assertIn("max_iterations_reached", evolution["evaluation"]["signals"])
        self.assertEqual(evolution["proposal"]["target"], "routing")

    def test_empty_result_generates_executor_reviewer_proposal(self):
        """空结果应生成 executor/reviewer 质量门禁相关提案。"""

        state = make_initial_state("Build a report")
        state["status"] = "running"
        state["result"] = ""

        evolution = run_controlled_evolution(state)

        self.assertFalse(evolution["evaluation"]["passed"])
        self.assertIn("result_empty", evolution["evaluation"]["signals"])
        self.assertEqual(evolution["proposal"]["target"], "tests")
        self.assertIn("executor/reviewer", evolution["proposal"]["title"])

    def test_evolution_does_not_mutate_agent_state(self):
        """受控自进化只能读取 state，不能修改主任务 graph 的最终状态。"""

        state = make_initial_state("Build a report")
        state["result"] = "A completed result"
        state["review"] = {"approved": True, "score": 1.0, "issues": []}
        state["status"] = "approved"
        before = copy.deepcopy(state)

        run_controlled_evolution(state)

        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
