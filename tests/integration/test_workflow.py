"""Real LangGraph conditional edges and durable checkpoints across three routes."""

from importlib.util import find_spec


def test_nine_nodes_process_subquestions_once_and_persist_checkpoints(tmp_path):
    assert find_spec("annual_report_agent.graph.builder"), "workflow builder missing"
    from annual_report_agent.graph.builder import build_workflow
    from annual_report_agent.graph.state import NODE_NAMES

    calls = []

    def node(name):
        def execute(state):
            calls.append((name, state.get("cursor", 0)))
            if name == "N4_route":
                return {"plan": {"subtasks": [{"route": r} for r in ["semantic", "data", "hybrid"]]}, "cursor": 0}
            if name == "N9_verify":
                return {"cursor": state["cursor"] + 1, "subtask_error": None,
                        "candidates": [], "evidence": [], "facts": [], "calculations": []}
            return {}
        return execute

    with build_workflow({name: node(name) for name in NODE_NAMES}, tmp_path / "checkpoints.db") as workflow:
        output = workflow.invoke({"run_id": "run-1", "kind": "analysis"},
                                 {"configurable": {"thread_id": "run-1"}, "recursion_limit": 100})
        assert output["cursor"] == 3
        assert [name for name, _ in calls].count("N4_route") == 1
        assert [name for name, i in calls if i == 1] == ["N7_calculate", "N8_compose", "N9_verify"]
        assert [name for name, i in calls if i == 2] == ["N5_retrieve", "N6_rerank", "N7_calculate", "N8_compose", "N9_verify"]
    with build_workflow({name: node(name) for name in NODE_NAMES}, tmp_path / "checkpoints.db") as restarted:
        state = restarted.get_state({"configurable": {"thread_id": "run-1"}})
        assert state.values["cursor"] == 3


def test_failed_subquestion_bypasses_generation_but_next_question_runs(tmp_path):
    assert find_spec("annual_report_agent.graph.builder"), "workflow builder missing"
    from annual_report_agent.graph.builder import build_workflow
    from annual_report_agent.graph.state import NODE_NAMES
    calls = []

    def node(name):
        def execute(state):
            cursor = state.get("cursor", 0)
            calls.append((name, cursor))
            if name == "N4_route":
                return {"plan": {"subtasks": [{"route": "semantic"}, {"route": "semantic"}]}, "cursor": 0}
            if name == "N5_retrieve" and cursor == 0:
                return {"subtask_error": {"code": "EMBEDDING_UNAVAILABLE"}}
            if name == "N9_verify":
                return {"cursor": cursor + 1, "subtask_error": None}
            return {}
        return execute
    with build_workflow({name: node(name) for name in NODE_NAMES}, tmp_path / "g.db") as graph:
        graph.invoke({"run_id": "run-2", "kind": "analysis"},
                     {"configurable": {"thread_id": "run-2"}, "recursion_limit": 100})
    assert ("N8_compose", 0) not in calls
    assert ("N8_compose", 1) in calls
