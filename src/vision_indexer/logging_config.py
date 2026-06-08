from __future__ import annotations

import logging
from pathlib import Path


BASE_LOGGER_NAME = "vision_indexer"
TOKENOMICS_LOGGER_NAME = "vision_indexer.tokenomics"


def _reset_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_logging(run_dir: Path, append: bool = False) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_logger = logging.getLogger(BASE_LOGGER_NAME)
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False
    _reset_handlers(base_logger)

    file_mode = "a" if append else "w"

    run_file_handler = logging.FileHandler(logs_dir / "run.log", mode=file_mode, encoding="utf-8")
    run_file_handler.setFormatter(formatter)
    base_logger.addHandler(run_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    base_logger.addHandler(console_handler)

    tokenomics_logger = logging.getLogger(TOKENOMICS_LOGGER_NAME)
    tokenomics_logger.setLevel(logging.INFO)
    tokenomics_logger.propagate = False
    _reset_handlers(tokenomics_logger)

    tokenomics_file_handler = logging.FileHandler(logs_dir / "tokenomics.log", mode=file_mode, encoding="utf-8")
    tokenomics_file_handler.setFormatter(formatter)
    tokenomics_logger.addHandler(tokenomics_file_handler)
