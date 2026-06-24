from __future__ import annotations

import sys
from pathlib import Path


def resource_path(relative: str) -> Path:
    """Resolve a resource path for both dev and PyInstaller bundles.

    Usage:
        ui = resource_path("resources/ui/MainWindow.ui")

    When bundling with PyInstaller, include resources via:
        pyinstaller --add-data "resources/ui/MainWindow.ui:resources/ui" ...
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        return base / relative

    # dev: go from src/neuxelec/utils/resources.py to project root
    base = Path(__file__).resolve().parents[3]
    return base / relative
