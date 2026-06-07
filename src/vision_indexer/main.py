from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Sequence

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

from vision_indexer.config import AppConfig

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change in a future version.*",
    category=LangChainPendingDeprecationWarning,
)

from vision_indexer.graph.build_graph import build_graph
from vision_indexer.graph.export_graph import export_graph_mermaid
from vision_indexer.logging_config import setup_logging
from vision_indexer.storage.run_store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vision-indexer Stage 1 pipeline.")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the input PDF.")
    parser.add_argument("--out", required=True, type=Path, help="Run output directory.")
    parser.add_argument("--debug", action="store_true", default=None, help="Enable memory debug snapshots.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.from_env(debug_memory=args.debug)

    run_dir = args.out
    RunStore(run_dir).prepare_directories()
    setup_logging(run_dir)

    logger = logging.getLogger("vision_indexer.main")
    logger.info("Starting vision-indexer run")

    graph = build_graph()
    export_graph_mermaid(graph, run_dir / "graph")

    graph.invoke(
        {
            "source_pdf_path": str(args.pdf),
            "run_dir": str(run_dir),
            "dpi": config.pdf_dpi,
            "debug_memory": config.debug_memory,
        }
    )

    logger.info("Finished vision-indexer run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
