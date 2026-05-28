import unittest

from langgraph_agent_demo.state import make_initial_state
from langgraph_agent_demo.supervisor import route_next, supervisor


class SupervisorTest(unittest.TestCase):
    """验证 supervisor 的路由、终止和错误保护规则。"""

    def test_supervisor_records_trace_and_keeps_initial_route(self):
        """Supervisor 应记录自身执行，并保留初始 planner 路由。"""

        state = make_initial_state("Build a report")

        update = supervisor(state)

        self.assertEqual(update["next_agent"], "planner")
        self.assertEqual(update["iteration_count"], 1)
        self.assertIn("supervisor", update["trace"])

    def test_route_next_returns_selected_agent(self):
        """条件边函数应返回状态中已选择的合法下一跳。"""

        state = make_initial_state("Build a report")
        state["next_agent"] = "executor"

        self.assertEqual(route_next(state), "executor")

    def test_route_next_ends_on_approval(self):
        """评审通过后，条件边函数应把 graph 路由到 end。"""

        state = make_initial_state("Build a report")
        state["review"]["approved"] = True
        state["next_agent"] = "end"

        self.assertEqual(route_next(state), "end")

    def test_supervisor_stops_at_max_iterations(self):
        """超过最大轮次时，supervisor 应安全结束而不是继续返工。"""

        state = make_initial_state("Build a report", max_iterations=1)
        state["iteration_count"] = 1
        state["next_agent"] = "executor"

        update = supervisor(state)

        self.assertEqual(update["status"], "max_iterations_reached")
        self.assertEqual(update["next_agent"], "end")

    def test_supervisor_handles_unknown_route(self):
        """未知下一跳应产生结构化错误，并安全结束 graph。"""

        state = make_initial_state("Build a report")
        state["next_agent"] = "unknown"

        update = supervisor(state)

        self.assertEqual(update["status"], "error")
        self.assertEqual(update["next_agent"], "end")
        self.assertEqual(update["errors"][-1]["node"], "supervisor")


if __name__ == "__main__":
    unittest.main()
