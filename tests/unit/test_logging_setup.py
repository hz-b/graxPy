from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import grax


def test_grax_logger_has_null_handler_on_import() -> None:
    """Without setup_logging, grax records must not fall through to stderr."""

    handlers = logging.getLogger("grax").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


@pytest.fixture
def restore_grax_logger() -> Iterator[logging.Logger]:
    grax_logger = logging.getLogger("grax")
    original_handlers = list(grax_logger.handlers)
    original_propagate = grax_logger.propagate
    original_level = grax_logger.level
    try:
        yield grax_logger
    finally:
        for handler in list(grax_logger.handlers):
            if handler not in original_handlers:
                grax_logger.removeHandler(handler)
                handler.close()
        grax_logger.propagate = original_propagate
        grax_logger.setLevel(original_level)


def test_setup_logging_disables_propagation_and_dedupes_file_handler(
    restore_grax_logger: logging.Logger,
    tmp_path: Path,
) -> None:
    grax_logger = restore_grax_logger
    log_file = tmp_path / "run.log"

    grax.setup_logging(log_file=str(log_file))
    grax.setup_logging(log_file=str(log_file))

    file_handlers = [
        handler
        for handler in grax_logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    assert grax_logger.propagate is False
