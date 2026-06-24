"""Small shared Qt helpers used across pages and dialogs."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget


def top_level_window() -> QWidget | None:
    """Return the most relevant top-level window, or ``None``.

    Prefers the active window; otherwise falls back to the first visible
    top-level widget. Used to parent dialogs when no explicit parent is known.
    """
    active = QApplication.activeWindow()
    if active is not None and active.isWindow():
        return active

    for widget in QApplication.topLevelWidgets():
        try:
            if widget is not None and widget.isWindow() and widget.isVisible():
                return widget
        except Exception:
            pass

    return None
