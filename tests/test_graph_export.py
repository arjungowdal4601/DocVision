from __future__ import annotations

import logging
from pathlib import Path

from vision_indexer.graph.export_graph import export_graph, export_graph_mermaid


class FakeInnerGraph:
    def __init__(self, fail_png: bool = False) -> None:
        self.fail_png = fail_png

    def draw_mermaid(self) -> str:
        return "graph TD;\nA-->B;"

    def draw_mermaid_png(self) -> bytes:
        if self.fail_png:
            raise RuntimeError("renderer unavailable")
        return b"\x89PNG\r\n\x1a\nfake"


class FakeGraph:
    def __init__(self, fail_png: bool = False) -> None:
        self.inner = FakeInnerGraph(fail_png=fail_png)

    def get_graph(self) -> FakeInnerGraph:
        return self.inner


def test_export_graph_mermaid_still_writes_mmd(tmp_path: Path) -> None:
    output_path = export_graph_mermaid(FakeGraph(), tmp_path)

    assert output_path == tmp_path / "graph.mmd"
    assert output_path.read_text(encoding="utf-8") == "graph TD;\nA-->B;"


def test_export_graph_writes_mermaid_and_png(tmp_path: Path) -> None:
    output_paths = export_graph(FakeGraph(), tmp_path)

    assert output_paths["mermaid"] == tmp_path / "graph.mmd"
    assert output_paths["png"] == tmp_path / "graph.png"
    assert (tmp_path / "graph.png").read_bytes().startswith(b"\x89PNG")


def test_export_graph_png_failure_keeps_mermaid_and_logs_warning(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)

    output_paths = export_graph(FakeGraph(fail_png=True), tmp_path)

    assert output_paths["mermaid"] == tmp_path / "graph.mmd"
    assert output_paths["png"] is None
    assert (tmp_path / "graph.mmd").exists()
    assert not (tmp_path / "graph.png").exists()
    assert "Could not export graph PNG" in caplog.text
