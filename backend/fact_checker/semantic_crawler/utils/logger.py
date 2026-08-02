"""
utils/logger.py
---------------
Configures the root logger once; every module retrieves a child logger via
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from config.settings import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Set up root logger — called once at application startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_cfg = settings.logging
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get(
        "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler
    log_file = log_cfg.get("file", "data/crawler.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.get("max_bytes", 10_485_760),
        backupCount=log_cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Silence noisy third-party libraries
    for lib in ("urllib3", "chardet", "filelock", "httpx", "httpcore"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger, configuring root on first call."""
    configure_logging()
    return logging.getLogger(name)
