from __future__ import annotations

import argparse
import site
from datetime import datetime
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PDF_PATH = Path("1512.03385v1.pdf")
DEFAULT_OUT_DIR = Path("runs")

if SRC_DIR.exists():
    site.addsitedir(str(SRC_DIR))

from vision_indexer.main import main as pipeline_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the DocVision pipeline with local defaults.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="Path to the input PDF.")
    parser.add_argument("--out", type=Path, default=None, help="Run output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run from scratch.")
    parser.add_argument("--force-render", action="store_true", help="Render page images even if they already exist.")
    parser.add_argument("--force-page", action="append", type=int, default=[], help="Reprocess a completed page.")
    parser.add_argument("--max-pages", type=int, default=None, help="Process only the first N rendered pages.")
    return parser


def build_launcher_args(argv: Sequence[str] | None = None, now: datetime | None = None) -> list[str]:
    args = build_parser().parse_args(argv)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or DEFAULT_OUT_DIR / f"main_real_{timestamp}"

    pipeline_args = [
        "--pdf",
        str(args.pdf),
        "--out",
        str(out_dir),
        "--debug",
    ]

    if args.resume:
        pipeline_args.append("--resume")
    if args.force:
        pipeline_args.append("--force")
    if args.force_render:
        pipeline_args.append("--force-render")
    for page_number in args.force_page:
        pipeline_args.extend(["--force-page", str(page_number)])
    if args.max_pages is not None:
        pipeline_args.extend(["--max-pages", str(args.max_pages)])

    return pipeline_args


def main(argv: Sequence[str] | None = None) -> int:
    return pipeline_main(build_launcher_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
