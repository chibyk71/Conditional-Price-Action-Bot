"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", fmt: Optional[str] = None) -> None:
    fmt = fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
