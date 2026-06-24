from __future__ import annotations

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QAbstractButton, QStackedWidget, QWidget


def connect_menu_navigation(
    ui_root: QObject,
    stacked_widget_name: str,
    mapping: dict[str, int],
) -> None:
    """Backwards-compatible wiring: button objectName -> page index."""
    sw = ui_root.findChild(QStackedWidget, stacked_widget_name)
    if sw is None:
        return

    def go(idx: int):
        try:
            sw.setCurrentIndex(idx)
        except Exception:
            pass

    for btn_name, idx in mapping.items():
        btn = ui_root.findChild(QAbstractButton, btn_name)
        if btn is None:
            continue
        btn.clicked.connect(lambda _=False, i=idx: go(i))


def connect_menu_navigation_by_page_name(
    ui_root: QObject,
    stacked_widget_name: str,
    mapping: dict[str, str],
) -> None:
    """Recommended wiring: button objectName -> page widget objectName.

    This does NOT depend on page order in Qt Designer.
    Missing buttons/pages are ignored so you can incrementally implement pages.
    """
    sw = ui_root.findChild(QStackedWidget, stacked_widget_name)
    if sw is None:
        return

    # Build a lookup {pageObjectName: index}
    page_name_to_index: dict[str, int] = {}
    for i in range(sw.count()):
        w = sw.widget(i)
        if isinstance(w, QWidget):
            page_name_to_index[w.objectName()] = i

    def go(idx: int):
        try:
            sw.setCurrentIndex(idx)
        except Exception:
            pass

    for btn_name, page_widget_name in mapping.items():
        btn = ui_root.findChild(QAbstractButton, btn_name)
        if btn is None:
            continue
        idx = page_name_to_index.get(page_widget_name, None)
        if idx is None:
            continue
        btn.clicked.connect(lambda _=False, i=idx: go(i))
