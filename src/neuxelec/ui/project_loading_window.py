from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from .page_loading_overlay import PageLoadingOverlay


class ProjectLoadingWindow(QWidget):
    """
    Temporary loading window displayed while a project is created or opened,
    and while the NeuXelec Main Window is being initialized.
    """

    def __init__(
        self,
        title: str,
        message: str,
    ) -> None:
        super().__init__()

        self.setWindowTitle("NeuXelec")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(560, 470)

        self._overlay = PageLoadingOverlay(
            self,
            title,
            message,
        )

        self._overlay.setStyleSheet("""
            QWidget#pageLoadingOverlay {
                background-color: #07080E;
                border: 1px solid #FF487D;
                border-radius: 16px;
            }
            """)

    def show_loading(self) -> None:
        """
        Show the centered loading window and start the logo animation.
        """
        self.show()

        screen = QApplication.primaryScreen()

        if screen is not None:
            available = screen.availableGeometry()
            self.move(
                available.center().x() - self.width() // 2,
                available.center().y() - self.height() // 2,
            )

        self._overlay.setGeometry(self.rect())
        self._overlay.begin()

        QApplication.processEvents()

    def set_progress(
        self,
        value: float,
        message: str,
    ) -> None:
        """
        Update the visual loading progress while the project is initialized.
        """
        self._overlay.set_progress(value, message)
        QApplication.processEvents()

    def complete(self) -> None:
        """
        Complete the logo animation before closing the loading window.
        """
        self._overlay.complete()
        QApplication.processEvents()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())
