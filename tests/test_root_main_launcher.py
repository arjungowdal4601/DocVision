from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _value_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_launcher_defaults_to_local_pdf_and_timestamped_real_run(monkeypatch) -> None:
    import main as root_main

    captured: dict[str, list[str]] = {}

    def fake_pipeline_main(args: list[str]) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(root_main, "pipeline_main", fake_pipeline_main)

    exit_code = root_main.main(now=datetime(2026, 6, 7, 20, 5, 6))

    launcher_args = captured["args"]

    assert exit_code == 0
    assert _value_after(launcher_args, "--pdf") == str(Path("1512.03385v1.pdf"))
    assert _value_after(launcher_args, "--out") == str(Path("runs") / "main_real_20260607_200506")
    assert "--debug" in launcher_args
    assert "--mock-llm" not in launcher_args


def test_launcher_passes_through_function_argument_overrides(monkeypatch) -> None:
    import main as root_main

    captured: dict[str, list[str]] = {}

    def fake_pipeline_main(args: list[str]) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(root_main, "pipeline_main", fake_pipeline_main)

    exit_code = root_main.main(
        pdf=Path("tests/fixtures/sample.pdf"),
        out=Path("runs/main_real"),
        resume=True,
        force=True,
        force_render=True,
        force_pages=[2, 5],
        max_pages=6,
        debug=False,
        now=datetime(2026, 6, 7, 20, 5, 6),
    )

    launcher_args = captured["args"]

    assert exit_code == 0
    assert _value_after(launcher_args, "--pdf") == str(Path("tests/fixtures/sample.pdf"))
    assert _value_after(launcher_args, "--out") == str(Path("runs/main_real"))
    assert "--debug" not in launcher_args
    assert "--mock-llm" not in launcher_args
    assert "--resume" in launcher_args
    assert "--force" in launcher_args
    assert "--force-render" in launcher_args
    assert launcher_args.count("--force-page") == 2
    assert _value_after(launcher_args, "--max-pages") == "6"


def test_launcher_does_not_import_argparse_or_sys() -> None:
    import main as root_main

    source = Path(root_main.__file__).read_text(encoding="utf-8")

    assert "import argparse" not in source
    assert "import sys" not in source
    assert "ArgumentParser" not in source
    assert "from sys" not in source
