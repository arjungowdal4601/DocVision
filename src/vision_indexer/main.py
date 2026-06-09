from __future__ import annotations

import argparse
import logging
import warnings
from os import getenv
from pathlib import Path
from typing import Sequence

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

from vision_indexer.config import AppConfig

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change in a future version.*",
    category=LangChainPendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:.*",
    category=UserWarning,
)

from vision_indexer.graph.build_graph import build_graph
from vision_indexer.graph.export_graph import export_graph
from vision_indexer.logging_config import setup_logging
from vision_indexer.retry.retry_policy import RetryPolicy
from vision_indexer.storage.run_status_store import load_run_status
from vision_indexer.storage.run_store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vision-indexer PDF indexing pipeline.")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the input PDF.")
    parser.add_argument("--out", required=True, type=Path, help="Run output directory.")
    parser.add_argument("--debug", action="store_true", default=None, help="Enable memory debug snapshots.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run from scratch.")
    parser.add_argument("--force-render", action="store_true", help="Render page images even if they already exist.")
    parser.add_argument("--force-page", action="append", type=int, default=[], help="Reprocess a completed page.")
    parser.add_argument("--max-pages", type=int, default=None, help="Process only the first N rendered pages.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.from_env(debug_memory=args.debug)
    if not getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    run_dir = args.out
    run_store = RunStore(run_dir)
    if args.force:
        run_store.reset_run_outputs()

    existing_status = load_run_status(run_dir)
    if existing_status is not None and not args.resume and not args.force:
        raise RuntimeError(f"Run already exists at {run_dir}. Use --resume or --force.")
    if args.resume and existing_status is None:
        raise RuntimeError(f"Cannot resume because run_status.json is missing at {run_dir}.")

    run_store.prepare_directories()
    setup_logging(run_dir, append=args.resume and not args.force)

    logger = logging.getLogger("vision_indexer.main")
    logger.info("Starting vision-indexer run")

    graph = build_graph()
    export_graph(graph, run_dir / "graph")

    graph.invoke(
        {
            "source_pdf_path": str(args.pdf),
            "run_dir": str(run_dir),
            "dpi": config.pdf_dpi,
            "debug_memory": config.debug_memory,
            "resume": args.resume,
            "force": args.force,
            "force_render": args.force_render,
            "force_pages": args.force_page,
            "max_pages": args.max_pages,
            "retry_policy": RetryPolicy.from_env().to_dict(),
            "llm_provider": config.llm_provider,
            "model_name": config.model,
            "reasoning_effort": config.reasoning_effort,
        },
        config={"configurable": {"thread_id": run_dir.name}},
    )

    logger.info("Finished vision-indexer run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
