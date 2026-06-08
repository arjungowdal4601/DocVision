from vision_indexer.graph.build_graph import build_graph
from vision_indexer.graph.export_graph import graph_to_mermaid


def test_graph_build_mermaid_contains_stage_one_nodes() -> None:
    graph = build_graph()

    mermaid = graph_to_mermaid(graph)

    assert "initialize_run_node" in mermaid
    assert "render_pdf_node" in mermaid
    assert "process_page_node" in mermaid
    assert "save_page_result_node" in mermaid
    assert "route_next_page_node" in mermaid
    assert "build_topic_index_node" in mermaid
    assert "finalize_run_node" in mermaid
