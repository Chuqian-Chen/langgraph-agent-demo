import unittest

from langgraph_agent_demo.agents import executor, planner, researcher, reviewer
from langgraph_agent_demo.models import MockModelClient
from langgraph_agent_demo.state import AgentState, make_initial_state
from langgraph_agent_demo.tools import build_default_tools


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


class RoleAgentTest(unittest.TestCase):
    def test_planner_writes_plan_and_trace(self):
        state = make_initial_state("Build a report")

        update = planner(state, model=MockModelClient())

        self.assertGreater(len(update["plan"]), 0)
        self.assertEqual(update["next_agent"], "executor")
        self.assertIn("planner", update["trace"])

    def test_planner_can_request_research(self):
        state = make_initial_state("Research LangGraph and summarize it")

        update = planner(state, model=MockModelClient())

        self.assertTrue(update["needs_research"])
        self.assertEqual(update["next_agent"], "researcher")

    def test_researcher_writes_findings(self):
        state = make_initial_state("Research LangGraph")
        state["needs_research"] = True

        update = researcher(state, model=MockModelClient(), tools=build_default_tools())

        self.assertGreater(len(update["findings"]), 0)
        self.assertEqual(update["next_agent"], "executor")
        self.assertIn("researcher", update["trace"])

    def test_executor_writes_result(self):
        state = make_initial_state("Build a report")
        state["plan"] = ["Produce a concise answer"]

        update = executor(state, model=MockModelClient())

        self.assertIn("Result", update["result"])
        self.assertEqual(update["next_agent"], "reviewer")
        self.assertIn("executor", update["trace"])

    def test_reviewer_approves_non_empty_result(self):
        state = make_initial_state("Build a report")
        state["result"] = "A completed result"

        update = reviewer(state, model=MockModelClient())

        self.assertTrue(update["review"]["approved"])
        self.assertEqual(update["status"], "approved")
        self.assertEqual(update["next_agent"], "end")
        self.assertIn("reviewer", update["trace"])


if __name__ == "__main__":
    unittest.main()
