"""Lightweight, dependency-free logging setup used across the pipeline.

A single ``get_logger`` gives every module a consistent, timestamped stream that
writes to stderr (so stdout stays clean for any machine-readable output). The
verbosity honours the ``AIGNITION_LOG_LEVEL`` environment variable, defaulting
to INFO.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("AIGNITION_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("aignition")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger("ingest")``."""
    _configure_root()
    return logging.getLogger(f"aignition.{name}")
