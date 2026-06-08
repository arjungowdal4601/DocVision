from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest


def _value_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_launcher_defaults_to_local_pdf_and_timestamped_real_run() -> None:
    import main as root_main

    launcher_args = root_main.build_launcher_args([], now=datetime(2026, 6, 7, 20, 5, 6))

    assert _value_after(launcher_args, "--pdf") == str(Path("1512.03385v1.pdf"))
    assert _value_after(launcher_args, "--out") == str(Path("runs") / "main_real_20260607_200506")
    assert "--debug" in launcher_args
    assert "--mock-llm" not in launcher_args


def test_launcher_passes_through_overrides() -> None:
    import main as root_main

    launcher_args = root_main.build_launcher_args(
        [
            "--pdf",
            "tests/fixtures/sample.pdf",
            "--out",
            "runs/main_real",
            "--resume",
            "--force",
            "--force-render",
            "--force-page",
            "2",
            "--force-page",
            "5",
            "--max-pages",
            "6",
        ],
        now=datetime(2026, 6, 7, 20, 5, 6),
    )

    assert _value_after(launcher_args, "--pdf") == str(Path("tests/fixtures/sample.pdf"))
    assert _value_after(launcher_args, "--out") == str(Path("runs/main_real"))
    assert "--mock-llm" not in launcher_args
    assert "--resume" in launcher_args
    assert "--force" in launcher_args
    assert "--force-render" in launcher_args
    assert launcher_args.count("--force-page") == 2
    assert _value_after(launcher_args, "--max-pages") == "6"


def test_launcher_rejects_public_mock_flag() -> None:
    import main as root_main

    with pytest.raises(SystemExit):
        root_main.build_launcher_args(["--mock-llm"], now=datetime(2026, 6, 7, 20, 5, 6))


def test_launcher_does_not_import_sys() -> None:
    import main as root_main

    source = Path(root_main.__file__).read_text(encoding="utf-8")

    assert "import sys" not in source
    assert "from sys" not in source
