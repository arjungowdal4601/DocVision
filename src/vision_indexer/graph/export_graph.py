from __future__ import annotations

from pathlib import Path
from typing import Any


def graph_to_mermaid(graph: Any) -> str:
    return graph.get_graph().draw_mermaid()


def export_graph_mermaid(graph: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mermaid = graph_to_mermaid(graph)
    output_path = output_dir / "graph.mmd"
    output_path.write_text(mermaid, encoding="utf-8")
    return output_path
