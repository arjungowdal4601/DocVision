from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def graph_to_mermaid(graph: Any) -> str:
    return graph.get_graph().draw_mermaid()


def export_graph_mermaid(graph: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mermaid = graph_to_mermaid(graph)
    output_path = output_dir / "graph.mmd"
    output_path.write_text(mermaid, encoding="utf-8")
    return output_path


def export_graph_png(graph: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "graph.png"
    output_path.write_bytes(graph.get_graph().draw_mermaid_png())
    return output_path


def export_graph(graph: Any, output_dir: Path) -> dict[str, Path | None]:
    mermaid_path = export_graph_mermaid(graph, output_dir)
    png_path: Path | None = None
    try:
        png_path = export_graph_png(graph, output_dir)
    except Exception as exc:
        logger.warning("Could not export graph PNG: %s", exc)
    return {"mermaid": mermaid_path, "png": png_path}
