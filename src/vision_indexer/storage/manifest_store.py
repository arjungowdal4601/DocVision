from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestStore:
    def __init__(self, run_dir: Path) -> None:
        self.manifest_path = run_dir / "manifest.json"

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.manifest_path
