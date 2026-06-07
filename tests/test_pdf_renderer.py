from pathlib import Path

import fitz

from vision_indexer.utils.pdf_renderer import render_pdf_to_images


def create_sample_pdf(path: Path, page_count: int = 2) -> None:
    document = fitz.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {page_number}")
    document.save(path)
    document.close()


def test_render_pdf_to_images_writes_numbered_png_files(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    output_dir = tmp_path / "images"
    create_sample_pdf(pdf_path)

    image_paths = render_pdf_to_images(pdf_path, output_dir, dpi=72)

    assert [path.name for path in image_paths] == ["page_0001.png", "page_0002.png"]
    assert all(path.exists() for path in image_paths)
