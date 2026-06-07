from __future__ import annotations

from pathlib import Path

import fitz


def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    document = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(document, start=1):
            image_path = output_dir / f"page_{page_index:04d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(image_path)
            rendered_paths.append(image_path)
    finally:
        document.close()

    return rendered_paths
