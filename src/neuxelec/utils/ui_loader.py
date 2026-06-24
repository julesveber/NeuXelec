from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader


def load_ui(ui_path: str | Path, parent=None):
    """Load a Qt Designer `.ui` file using QUiLoader."""
    path = Path(ui_path)
    qfile = QFile(str(path))

    if not qfile.open(QFile.ReadOnly):
        raise RuntimeError(f"Cannot open UI file: {path}")

    loader = QUiLoader()
    widget = loader.load(qfile, parent)
    qfile.close()

    if widget is None:
        raise RuntimeError(f"Failed to load UI: {path}")

    return widget
