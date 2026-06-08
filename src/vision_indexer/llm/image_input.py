from __future__ import annotations

import base64
from pathlib import Path


def build_image_input(image_path: Path) -> dict:
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }
