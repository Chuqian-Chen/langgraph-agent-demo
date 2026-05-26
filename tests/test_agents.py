import unittest

from langgraph_agent_demo.state import AgentState, make_initial_state


class AgentStateTest(unittest.TestCase):
    def test_make_initial_state_sets_defaults(self):
        state = make_initial_state("Build a report")

        self.assertEqual(state["task"], "Build a report")
        self.assertEqual(state["plan"], [])
        self.assertEqual(state["findings"], [])
        self.assertEqual(state["result"], "")
        self.assertEqual(state["review"]["approved"], False)
        self.assertEqual(state["next_agent"], "planner")
        self.assertEqual(state["iteration_count"], 0)
        self.assertEqual(state["max_iterations"], 3)
        self.assertEqual(state["trace"], ["START"])
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["errors"], [])


if __name__ == "__main__":
    unittest.main()
