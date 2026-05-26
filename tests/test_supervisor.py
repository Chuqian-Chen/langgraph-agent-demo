import unittest

from langgraph_agent_demo.state import make_initial_state
from langgraph_agent_demo.supervisor import route_next, supervisor


class SupervisorTest(unittest.TestCase):
    def test_supervisor_records_trace_and_keeps_initial_route(self):
        state = make_initial_state("Build a report")

        update = supervisor(state)

        self.assertEqual(update["next_agent"], "planner")
        self.assertEqual(update["iteration_count"], 1)
        self.assertIn("supervisor", update["trace"])

    def test_route_next_returns_selected_agent(self):
        state = make_initial_state("Build a report")
        state["next_agent"] = "executor"

        self.assertEqual(route_next(state), "executor")

    def test_route_next_ends_on_approval(self):
        state = make_initial_state("Build a report")
        state["review"]["approved"] = True
        state["next_agent"] = "end"

        self.assertEqual(route_next(state), "end")

    def test_supervisor_stops_at_max_iterations(self):
        state = make_initial_state("Build a report", max_iterations=1)
        state["iteration_count"] = 1
        state["next_agent"] = "executor"

        update = supervisor(state)

        self.assertEqual(update["status"], "max_iterations_reached")
        self.assertEqual(update["next_agent"], "end")

    def test_supervisor_handles_unknown_route(self):
        state = make_initial_state("Build a report")
        state["next_agent"] = "unknown"

        update = supervisor(state)

        self.assertEqual(update["status"], "error")
        self.assertEqual(update["next_agent"], "end")
        self.assertEqual(update["errors"][-1]["node"], "supervisor")


if __name__ == "__main__":
    unittest.main()
