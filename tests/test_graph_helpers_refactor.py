from __future__ import annotations

from pathlib import Path

from vision_indexer.graph import helpers, nodes


def test_nodes_exposes_graph_node_functions() -> None:
    for function_name in [
        "initialize_run_node",
        "render_pdf_node",
        "process_page_node",
        "save_page_result_node",
        "route_next_page_node",
        "select_next_node",
        "build_topic_index_node",
        "finalize_run_node",
    ]:
        assert hasattr(nodes, function_name)


def test_private_graph_helpers_live_in_helpers_module_not_nodes_module() -> None:
    nodes_source = Path(nodes.__file__).read_text(encoding="utf-8")
    helpers_source = Path(helpers.__file__).read_text(encoding="utf-8")

    helper_names = [
        "_run_status_from_state",
        "_require_run_status",
        "_ensure_page_statuses",
        "_page_statuses_payload",
        "_should_skip_completed_page",
        "_existing_page_results",
        "_append_unique",
        "_current_page_number",
        "_persist_page_failure",
        "_load_token_usage_seed",
        "_write_manifest",
    ]

    for helper_name in helper_names:
        assert f"def {helper_name}" not in nodes_source
        assert f"def {helper_name}" in helpers_source
