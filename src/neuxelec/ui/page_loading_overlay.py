from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QEvent, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from neuxelec.utils.resources import resource_path


def _find_brain_logo() -> Path | None:
    """
    Locate brain_logo.png during development or after packaging.
    """
    path = resource_path("resources/images/brain_logo.png")
    return path if path.exists() else None


def _load_logo_with_transparent_background() -> QPixmap:
    """
    Load brain_logo.png and remove a possible white background.
    """
    path = _find_brain_logo()

    if path is None:
        return QPixmap()

    image = QImage(str(path)).convertToFormat(QImage.Format.Format_ARGB32)

    if image.isNull():
        return QPixmap()

    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)

            if color.red() > 245 and color.green() > 245 and color.blue() > 245:
                color.setAlpha(0)
                image.setPixelColor(x, y, color)

    return QPixmap.fromImage(image)


class PageLoadingOverlay(QWidget):
    """
    Full-page loading overlay used while heavy PyVista / VTK views are built.

    The logo progressively appears using its own brain silhouette as a mask.
    """

    def __init__(
        self,
        parent: QWidget,
        page_title: str,
        initial_message: str,
    ) -> None:
        super().__init__(parent)

        self._page_title = str(page_title)
        self._message = str(initial_message)

        self._target_progress = 0.0
        self._shown_progress = 0.0
        self._finishing = False

        self._logo = _load_logo_with_transparent_background()

        self._clock = QElapsedTimer()

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate)

        self.setObjectName("pageLoadingOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self.setStyleSheet("""
            QWidget#pageLoadingOverlay {
                background-color: #07080E;
                border: none;
            }
            """)

        # Important on Windows: keep the overlay above native VTK/PyVista widgets.
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
            self.winId()
        except Exception:
            pass

        parent.installEventFilter(self)
        self.hide()

    def begin(self, message: str | None = None) -> None:
        """
        Display the overlay and start the reveal animation.
        """
        if message is not None:
            self._message = str(message)

        self._target_progress = 0.10
        self._shown_progress = 0.0
        self._finishing = False

        self._clock.restart()

        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

        self.show()
        self.raise_()
        self._timer.start()
        self.update()

    def set_progress(
        self,
        value: float,
        message: str | None = None,
    ) -> None:
        """
        Update the target progress and the displayed loading message.
        """
        if message is not None:
            self._message = str(message)

        self._target_progress = max(
            self._target_progress,
            min(0.94, float(value)),
        )

        self.raise_()
        self.update()

    def complete(self) -> None:
        """
        Finish the reveal animation before hiding the overlay.
        """
        self._target_progress = 1.0
        self._finishing = True
        self.raise_()

        if not self._timer.isActive():
            self._timer.start()

    def cancel(self) -> None:
        """
        Immediately stop and hide the overlay.
        """
        self._timer.stop()
        self.hide()
        self._finishing = False

    def eventFilter(self, watched, event):
        """
        Keep the overlay aligned with the parent page when it is resized.
        """
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            parent = self.parentWidget()
            if parent is not None:
                self.setGeometry(parent.rect())

        return super().eventFilter(watched, event)

    def _animate(self) -> None:
        """
        Smoothly animate displayed progress toward the requested target.
        """
        delta = self._target_progress - self._shown_progress

        if abs(delta) > 0.001:
            self._shown_progress += delta * 0.18
        else:
            self._shown_progress = self._target_progress

        self.update()
        self.raise_()

        if self._finishing and self._shown_progress >= 0.985 and self._clock.elapsed() >= 350:
            self._timer.stop()
            self.hide()

    def _logo_rect(self) -> QRect:
        """
        Central drawing area of the animated brain logo.
        """
        size = max(120, min(230, int(self.width() * 0.20)))
        x = int((self.width() - size) / 2)
        y = int((self.height() - size) / 2) - 34

        return QRect(x, y, size, size)

    def _build_revealed_logo(self, size: QSize) -> QPixmap:
        """
        Reveal the full logo progressively using a growing logo-shaped mask.
        """
        if self._logo.isNull():
            return QPixmap()

        full_logo = self._logo.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        canvas = QPixmap(size)
        canvas.fill(Qt.GlobalColor.transparent)

        progress = max(0.01, min(1.0, self._shown_progress))

        mask_size = QSize(
            max(1, int(size.width() * progress)),
            max(1, int(size.height() * progress)),
        )

        growing_brain_shape = self._logo.scaled(
            mask_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        mask = QPixmap(size)
        mask.fill(Qt.GlobalColor.transparent)

        mask_painter = QPainter(mask)
        mask_painter.drawPixmap(
            int((size.width() - growing_brain_shape.width()) / 2),
            int((size.height() - growing_brain_shape.height()) / 2),
            growing_brain_shape,
        )
        mask_painter.end()

        canvas_painter = QPainter(canvas)
        canvas_painter.drawPixmap(
            int((size.width() - full_logo.width()) / 2),
            int((size.height() - full_logo.height()) / 2),
            full_logo,
        )

        canvas_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        canvas_painter.drawPixmap(0, 0, mask)
        canvas_painter.end()

        return canvas

    def paintEvent(self, event) -> None:
        """
        Draw the logo animation, message and progress bar.
        """
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        logo_rect = self._logo_rect()

        if not self._logo.isNull():
            ghost_logo = self._logo.scaled(
                logo_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            painter.setOpacity(0.10)
            painter.drawPixmap(
                logo_rect.x() + int((logo_rect.width() - ghost_logo.width()) / 2),
                logo_rect.y() + int((logo_rect.height() - ghost_logo.height()) / 2),
                ghost_logo,
            )

            painter.setOpacity(1.0)
            visible_logo = self._build_revealed_logo(logo_rect.size())
            painter.drawPixmap(logo_rect.topLeft(), visible_logo)

        painter.setOpacity(1.0)

        title_font = QFont("Segoe UI")
        title_font.setPointSize(11)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_font.setLetterSpacing(
            QFont.SpacingType.AbsoluteSpacing,
            1.1,
        )

        painter.setFont(title_font)
        painter.setPen(QColor("#F4D9D0"))
        painter.drawText(
            QRect(0, logo_rect.bottom() + 22, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter,
            self._page_title,
        )

        message_font = QFont("Segoe UI")
        message_font.setPointSize(9)

        painter.setFont(message_font)
        painter.setPen(QColor("#8B8FA0"))
        painter.drawText(
            QRect(0, logo_rect.bottom() + 53, self.width(), 25),
            Qt.AlignmentFlag.AlignCenter,
            self._message,
        )

        bar_width = max(160, min(250, int(self.width() * 0.20)))
        bar_rect = QRect(
            int((self.width() - bar_width) / 2),
            logo_rect.bottom() + 89,
            bar_width,
            4,
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1D202B"))
        painter.drawRoundedRect(bar_rect, 2, 2)

        progress_rect = QRect(
            bar_rect.x(),
            bar_rect.y(),
            int(bar_rect.width() * self._shown_progress),
            bar_rect.height(),
        )

        if progress_rect.width() > 0:
            painter.setBrush(QColor("#FF487D"))
            painter.drawRoundedRect(progress_rect, 2, 2)
