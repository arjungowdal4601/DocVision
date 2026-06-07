from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from vision_indexer.graph.nodes import (
    finalize_run_node,
    initialize_run_node,
    process_page_node,
    render_pdf_node,
    route_next_page_node,
    save_page_result_node,
    select_next_node,
)
from vision_indexer.graph.state import VisionIndexerState


def build_graph():
    graph = StateGraph(VisionIndexerState)

    graph.add_node("initialize_run_node", initialize_run_node)
    graph.add_node("render_pdf_node", render_pdf_node)
    graph.add_node("process_page_node", process_page_node)
    graph.add_node("save_page_result_node", save_page_result_node)
    graph.add_node("route_next_page_node", route_next_page_node)
    graph.add_node("finalize_run_node", finalize_run_node)

    graph.add_edge(START, "initialize_run_node")
    graph.add_edge("initialize_run_node", "render_pdf_node")
    graph.add_edge("render_pdf_node", "process_page_node")
    graph.add_edge("process_page_node", "save_page_result_node")
    graph.add_edge("save_page_result_node", "route_next_page_node")
    graph.add_conditional_edges(
        "route_next_page_node",
        select_next_node,
        {
            "process_page_node": "process_page_node",
            "finalize_run_node": "finalize_run_node",
        },
    )
    graph.add_edge("finalize_run_node", END)

    return graph.compile()
