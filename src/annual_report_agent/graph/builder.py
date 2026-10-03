"""Nine business nodes, serial subtask loop, and persistent SQLite checkpoints."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from annual_report_agent.graph.state import NODE_NAMES, GraphState


def _next_subtask(state):
    if state.get("global_error"):
        return "N9_verify"
    plan = state.get("plan", {})
    if plan.get("needs_clarification"):
        return "N9_verify"
    tasks = plan.get("subtasks", [])
    cursor = state.get("cursor", 0)
    if cursor >= len(tasks):
        return END
    current = tasks[cursor]
    if current.get("needs_clarification") or current.get("unsupported_reason"):
        return "N9_verify"
    return "N7_calculate" if current.get("route") == "data" else "N5_retrieve"


@contextmanager
def build_workflow(nodes: dict, checkpoint_path: Path):
    if set(nodes) != set(NODE_NAMES):
        raise ValueError("all nine business handlers are required")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    graph = StateGraph(GraphState)
    for name in NODE_NAMES:
        graph.add_node(name, nodes[name])
    graph.add_edge(START, "N1_acquire")
    graph.add_conditional_edges("N1_acquire", lambda state:
        "N2_parse" if state.get("kind") == "ingest" else "N4_route")
    graph.add_edge("N2_parse", "N3_index")
    graph.add_edge("N3_index", END)
    graph.add_conditional_edges("N4_route", _next_subtask)
    graph.add_conditional_edges("N5_retrieve", lambda state:
        "N9_verify" if state.get("subtask_error") else "N6_rerank")
    graph.add_conditional_edges("N6_rerank", lambda state:
        "N9_verify" if state.get("subtask_error") else "N7_calculate"
        if state["plan"]["subtasks"][state["cursor"]]["route"] == "hybrid" else "N8_compose")
    graph.add_conditional_edges("N7_calculate", lambda state:
        "N9_verify" if state.get("subtask_error") else "N8_compose")
    graph.add_edge("N8_compose", "N9_verify")
    graph.add_conditional_edges("N9_verify", lambda state:
        END if state.get("answer") or state.get("global_error") else _next_subtask(state))
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        yield graph.compile(checkpointer=saver)
    finally:
        connection.close()
