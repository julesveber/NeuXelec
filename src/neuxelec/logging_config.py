"""Centralised logging configuration for NeuXelec.

NeuXelec ships as a windowed application (no console), so ``print`` output is
lost once the app is packaged. This module configures the standard
:mod:`logging` library to write rotating log files to a per-user location:

    Windows : %LOCALAPPDATA%/NeuXelec/logs/neuxelec.log
    Other   : ~/.neuxelec/logs/neuxelec.log

Usage
-----
Call :func:`configure_logging` once, as early as possible in ``main()``::

    from neuxelec.logging_config import configure_logging
    configure_logging()

Then, in any module::

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Coregistration finished for %s", modality)
    logger.exception("Failed to render brain")   # inside an except block

The log file is the first thing to ask a user for when diagnosing a problem
with the packaged executable.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

#: Logger name prefix for the whole application.
APP_LOGGER_NAME = "neuxelec"

#: Maximum size of a single log file before it is rotated (bytes).
_MAX_BYTES = 2 * 1024 * 1024  # 2 MB

#: Number of rotated log files to keep.
_BACKUP_COUNT = 5

_configured = False


def get_log_directory() -> Path:
    """Return the per-user directory where log files are written.

    The directory is created if it does not already exist.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        log_dir = Path(base) / "NeuXelec" / "logs"
    else:
        log_dir = Path.home() / ".neuxelec" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_logging(level: int = logging.INFO) -> Path:
    """Configure application-wide logging. Safe to call more than once.

    Parameters
    ----------
    level:
        Minimum level recorded to the log file (default :data:`logging.INFO`).

    Returns
    -------
    pathlib.Path
        The path of the active log file.
    """
    global _configured

    log_dir = get_log_directory()
    log_file = log_dir / "neuxelec.log"

    if _configured:
        return log_file

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler - survives long sessions without unbounded growth.
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    # Also echo to stderr when a console is attached (development runs).
    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root.addHandler(stream_handler)

    logging.getLogger(APP_LOGGER_NAME).info("Logging initialised. Log file: %s", log_file)
    _configured = True
    return log_file
