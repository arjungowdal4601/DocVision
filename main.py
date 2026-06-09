from __future__ import annotations

import site
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PDF_PATH = Path("1512.03385v1.pdf")
DEFAULT_OUT_DIR = Path("runs")

if SRC_DIR.exists():
    site.addsitedir(str(SRC_DIR))

from vision_indexer.main import main as pipeline_main


def build_launcher_args(
    *,
    pdf: Path | str = DEFAULT_PDF_PATH,
    out: Path | str | None = None,
    resume: bool = False,
    force: bool = False,
    force_render: bool = False,
    force_pages: Sequence[int] | None = None,
    max_pages: int | None = None,
    debug: bool = True,
    now: datetime | None = None,
) -> list[str]:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out) if out is not None else DEFAULT_OUT_DIR / f"main_real_{timestamp}"

    pipeline_args = [
        "--pdf",
        str(Path(pdf)),
        "--out",
        str(out_dir),
    ]

    if debug:
        pipeline_args.append("--debug")
    if resume:
        pipeline_args.append("--resume")
    if force:
        pipeline_args.append("--force")
    if force_render:
        pipeline_args.append("--force-render")
    for page_number in force_pages or []:
        pipeline_args.extend(["--force-page", str(page_number)])
    if max_pages is not None:
        pipeline_args.extend(["--max-pages", str(max_pages)])

    return pipeline_args


def main(
    *,
    pdf: Path | str = DEFAULT_PDF_PATH,
    out: Path | str | None = None,
    resume: bool = False,
    force: bool = False,
    force_render: bool = False,
    force_pages: Sequence[int] | None = None,
    max_pages: int | None = None,
    debug: bool = True,
    now: datetime | None = None,
) -> int:
    return pipeline_main(
        build_launcher_args(
            pdf=pdf,
            out=out,
            resume=resume,
            force=force,
            force_render=force_render,
            force_pages=force_pages,
            max_pages=max_pages,
            debug=debug,
            now=now,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
