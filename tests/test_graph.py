import unittest

from langgraph_agent_demo.graph import build_agent_graph
from langgraph_agent_demo.state import make_initial_state


class GraphTest(unittest.TestCase):
    def test_graph_compiles_and_runs_mock_runtime(self):
        graph = build_agent_graph()

        final_state = graph.invoke(make_initial_state("Build a report"))

        self.assertEqual(final_state["status"], "approved")
        self.assertIn("supervisor", final_state["trace"])
        self.assertIn("planner", final_state["trace"])
        self.assertIn("executor", final_state["trace"])
        self.assertIn("reviewer", final_state["trace"])
        self.assertTrue(final_state["result"])

    def test_graph_routes_to_researcher_when_needed(self):
        graph = build_agent_graph()

        final_state = graph.invoke(make_initial_state("Research LangGraph"))

        self.assertIn("researcher", final_state["trace"])
        self.assertGreater(len(final_state["findings"]), 0)


if __name__ == "__main__":
    unittest.main()
