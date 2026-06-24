from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QMimeData, QSize, Qt, QUrl
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QToolTip,
    QWidget,
)

from neuxelec.utils.resources import resource_path


def _toolbar_icon_path(filename: str) -> str:
    """
    Return the absolute path of a NeuXelec toolbar SVG icon.
    Works during development and after PyInstaller packaging.
    """
    return str(resource_path(f"resources/images/{filename}"))


class PyVistaQuickTools(QWidget):
    """
    Floating quick-tools menu for 3D View:
    - screenshot
    - fixed camera presets
    - save/reload custom camera
    - transparent background screenshots
    """

    def __init__(self, parent: QWidget, owner):
        super().__init__(parent)

        self.owner = owner
        self._expanded = False
        self._transparent_background = False

        # Circular carousel used when the toolbar cannot display every tool.
        self._tool_buttons = []
        self._carousel_index = 0
        self._visible_tool_count = 999
        self._carousel_enabled = False

        self.setObjectName("View3DQuickTools")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self._apply_style()

        try:
            parent.installEventFilter(self)
        except Exception:
            pass

        self._update_panel_visibility()
        self.reposition()

    def _make_btn(
        self,
        icon_filename: str,
        tooltip: str,
        callback,
        checkable: bool = False,
    ) -> QPushButton:
        btn = QPushButton()
        btn.setToolTip(tooltip)
        btn.setFixedSize(34, 34)
        btn.setIcon(QIcon(_toolbar_icon_path(icon_filename)))
        btn.setIconSize(QSize(19, 19))
        btn.setCheckable(bool(checkable))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        if checkable:
            btn.toggled.connect(callback)
        else:
            btn.clicked.connect(callback)

        self._tool_buttons.append(btn)
        btn.installEventFilter(self)

        return btn

    def _build_ui(self) -> None:
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)

        self.btn_arrow = QPushButton()
        self.btn_arrow.setToolTip("Open quick tools")
        self.btn_arrow.setFixedSize(34, 34)
        self.btn_arrow.setIcon(QIcon(_toolbar_icon_path("qt_expand.svg")))
        self.btn_arrow.setIconSize(QSize(18, 18))
        self.btn_arrow.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_arrow.clicked.connect(self.toggle_menu)
        self.layout.addWidget(self.btn_arrow)

        self.panel = QWidget(self)
        self.panel.setObjectName("View3DQuickToolsPanel")
        self.panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.panel.installEventFilter(self)

        self.panel_layout = QHBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(5)

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_screenshot.svg",
                "Save a screenshot of the 3D visualization area and copy it to the clipboard.",
                self._take_screenshot,
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_gif.svg",
                "Export a rotating 3D animation as a GIF. You can choose the rotation axis: X, Y, or Z.",
                self._export_gif,
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_front.svg",
                "Front coronal view. Camera faces the brain from anterior.",
                lambda: self._set_view("front"),
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_back.svg",
                "Back coronal view. Camera faces the brain from posterior.",
                lambda: self._set_view("back"),
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_left.svg",
                "Left sagittal view. Camera faces the left hemisphere side.",
                lambda: self._set_view("left"),
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_right.svg",
                "Right sagittal view. Camera faces the right hemisphere side.",
                lambda: self._set_view("right"),
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_beauty_left.svg",
                "Oblique view from front-left, useful for presentation screenshots.",
                lambda: self._set_view("beauty_left"),
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_beauty_right.svg",
                "Oblique view from front-right, useful for presentation screenshots.",
                lambda: self._set_view("beauty_right"),
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_restore.svg",
                "Apply your saved custom camera position.",
                self._apply_saved_view,
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_save.svg",
                "Save the current camera position into the project JSON.",
                self._save_current_view,
            )
        )

        self.btn_transparent = self._make_btn(
            "qt_transparent_background.svg",
            "Toggle transparent background for screenshots. Works best with PNG.",
            self._toggle_transparent_background,
            checkable=True,
        )
        self.btn_transparent.setObjectName("btn_transparent_background")
        self.panel_layout.addWidget(self.btn_transparent)

        self.layout.addWidget(self.panel)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget#View3DQuickTools {
                background-color: transparent;
                border: none;
            }

            QWidget#View3DQuickToolsPanel {
                background-color: transparent;
                border: none;
            }

            QWidget#View3DQuickTools QPushButton {
                color: white;
                background-color: rgba(10, 12, 19, 210);
                border: 1px solid #D6D8E2;
                border-radius: 8px;
                padding: 0px;
            }

            QWidget#View3DQuickTools QPushButton:hover {
                background-color: rgba(18, 20, 29, 235);
                border: 1px solid #FF487D;
            }

            QWidget#View3DQuickTools QPushButton:pressed {
                background-color: rgba(28, 30, 40, 245);
                border: 1px solid #FF487D;
            }

            QWidget#View3DQuickTools QPushButton:checked {
                border: 1px solid #FF487D;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QWidget#View3DQuickTools QPushButton#btn_transparent_background:checked {
                border: 1px solid #FFFFFF;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QWidget#View3DQuickTools QPushButton#btn_transparent_background:checked:hover {
                border: 1px solid #FFFFFF;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QToolTip {
                color: white;
                background-color: #11131B;
                border: 1px solid #FF487D;
                border-radius: 5px;
                padding: 6px;
            }
        """)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.Show):
            self.reposition()
            return False

        # Circular icon navigation only when the cursor is above the toolbar.
        if (
            event.type() == QEvent.Wheel
            and self._expanded
            and self._carousel_enabled
            and (obj is self.panel or obj in self._tool_buttons)
        ):
            delta = int(event.angleDelta().y())

            if delta > 0:
                self._rotate_carousel(-1)
            elif delta < 0:
                self._rotate_carousel(+1)

            try:
                event.accept()
            except Exception:
                pass

            return True

        return False

    def set_carousel_available_width(
        self,
        available_width: int,
        button_width: int = 32,
        button_height: int = 30,
        spacing_px: int = 4,
    ) -> None:
        """
        Adapt the toolbar to the available parent width.

        If every icon fits, all tools remain visible.
        Otherwise, only a circular subset is shown and can be browsed with
        the mouse wheel while hovering the toolbar.
        """
        try:
            available_width = max(1, int(available_width))
            button_width = max(26, int(button_width))
            button_height = max(26, int(button_height))
            spacing_px = max(1, int(spacing_px))

            button_side = max(34, int(button_height))

            self.btn_arrow.setFixedSize(button_side, button_side)
            self.btn_arrow.setIconSize(QSize(17, 17))

            for btn in self._tool_buttons:
                btn.setFixedSize(button_side, button_side)
                btn.setIconSize(QSize(18, 18))

            self.layout.setSpacing(spacing_px)
            self.panel_layout.setSpacing(spacing_px)

            fixed_width = button_side + spacing_px
            remaining_width = max(
                button_side,
                available_width - fixed_width,
            )

            slots = max(
                5,
                int((remaining_width + spacing_px) // (button_side + spacing_px)),
            )

            self._visible_tool_count = min(len(self._tool_buttons), slots)

            # Keep carousel navigation enabled even when all icons are visible.
            # In a large toolbar, the wheel rotates the order of all visible icons.
            # In a narrow toolbar, the wheel reveals the icons outside the visible subset.
            self._carousel_enabled = len(self._tool_buttons) > 1

            self._apply_carousel_visibility()
            self.adjustSize()
            self.reposition()

        except Exception:
            pass

    def _apply_carousel_visibility(self) -> None:
        """
        Rebuild the visible horizontal sequence of toolbar buttons.
        Re-adding widgets is necessary so the circular order is visually correct
        when the carousel wraps from the last button back to the first.
        """
        try:
            if not self._tool_buttons:
                return

            for btn in self._tool_buttons:
                self.panel_layout.removeWidget(btn)
                btn.hide()

            if self._carousel_enabled:
                count = max(1, int(self._visible_tool_count))
                n_buttons = len(self._tool_buttons)

                visible_buttons = [
                    self._tool_buttons[(self._carousel_index + i) % n_buttons] for i in range(count)
                ]

                self.panel.setToolTip("Use the mouse wheel to browse more tools.")
            else:
                visible_buttons = list(self._tool_buttons)
                self.panel.setToolTip("")

            for btn in visible_buttons:
                self.panel_layout.addWidget(btn)
                btn.show()

            self.panel.adjustSize()
            self.adjustSize()

        except Exception:
            pass

    def _rotate_carousel(self, direction: int) -> None:
        """
        Rotate visible toolbar icons circularly by one position.
        """
        try:
            if not self._carousel_enabled or not self._tool_buttons:
                return

            self._carousel_index = (int(self._carousel_index) + int(direction)) % len(
                self._tool_buttons
            )

            self._apply_carousel_visibility()
            self.reposition()
            self.raise_()

        except Exception:
            pass

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return

        try:
            self.adjustSize()
            margin = 12
            x = margin
            y = max(margin, parent.height() - self.height() - margin)
            self.move(x, y)
            self.raise_()
        except Exception:
            pass

    def toggle_menu(self) -> None:
        self._expanded = not self._expanded
        self._update_panel_visibility()

    def _update_panel_visibility(self) -> None:
        self.panel.setVisible(bool(self._expanded))

        arrow_icon = "qt_collapse.svg" if self._expanded else "qt_expand.svg"
        self.btn_arrow.setIcon(QIcon(_toolbar_icon_path(arrow_icon)))

        self.btn_arrow.setToolTip("Close quick tools" if self._expanded else "Open quick tools")

        self._apply_carousel_visibility()
        self.adjustSize()
        self.reposition()
        self.raise_()

    def _toggle_transparent_background(self, checked: bool) -> None:
        self._transparent_background = bool(checked)

    def _set_view(self, view_name: str) -> None:
        try:
            if hasattr(self.owner, "_set_camera_quick_view"):
                self.owner._set_camera_quick_view(view_name)
        except Exception:
            pass

    def _save_current_view(self) -> None:
        try:
            if hasattr(self.owner, "_save_current_3d_camera_to_state"):
                self.owner._save_current_3d_camera_to_state()
        except Exception:
            pass

    def _apply_saved_view(self) -> None:
        try:
            if hasattr(self.owner, "_apply_saved_3d_camera_from_state"):
                self.owner._apply_saved_3d_camera_from_state()
        except Exception:
            pass

    def _export_gif(self) -> None:
        try:
            if hasattr(self.owner, "_export_3d_view_gif"):
                self.owner._export_3d_view_gif()
        except Exception as e:
            print("[3D Quick Tools] GIF export failed:", e)

    def _copy_image_file_to_clipboard(self, filename: str) -> bool:
        """
        Copy the saved screenshot file itself to the system clipboard.

        This behaves like selecting the PNG in Windows Explorer and pressing Ctrl+C.
        It is more reliable for keeping PNG transparency when pasting into
        Word / PowerPoint than copying decoded image pixels.
        """
        try:
            path = Path(str(filename)).resolve()

            if not path.exists() or not path.is_file():
                return False

            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(path))])
            mime.setText(str(path))

            clipboard = QGuiApplication.clipboard()
            if clipboard is None:
                return False

            clipboard.setMimeData(mime)
            return True

        except Exception as e:
            print("[3D Quick Tools] Clipboard file copy failed:", e)
            return False

    def _take_screenshot(self) -> None:
        try:
            if not hasattr(self.owner, "_save_3d_view_screenshot"):
                return

            parent = self.parentWidget()
            filename, selected_filter = QFileDialog.getSaveFileName(
                parent,
                "Save 3D screenshot",
                "neuxelec_3d_view.png",
                "PNG image (*.png);;JPEG image (*.jpg *.jpeg)",
            )

            if not filename:
                return

            final_filename = str(filename)

            # QFileDialog does not always append the extension automatically.
            if not final_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                if "JPEG" in str(selected_filter):
                    final_filename += ".jpg"
                else:
                    final_filename += ".png"

            # If transparent background is requested, force PNG.
            # JPEG cannot store transparency.
            if bool(self._transparent_background) and not final_filename.lower().endswith(".png"):
                final_filename = str(Path(final_filename).with_suffix(".png"))

            self.owner._save_3d_view_screenshot(
                final_filename,
                transparent_background=bool(self._transparent_background),
            )

            copied = self._copy_image_file_to_clipboard(final_filename)

            if copied:
                if bool(self._transparent_background):
                    msg = "Screenshot saved and copied with transparent background"
                else:
                    msg = "Screenshot saved and copied with background"
            else:
                msg = "Screenshot saved, but could not be copied to clipboard"

            QToolTip.showText(
                self.mapToGlobal(self.rect().center()),
                msg,
                self,
            )

        except Exception as e:
            print("[3D Quick Tools] Screenshot failed:", e)

    def transparent_background_enabled(self) -> bool:
        return bool(self._transparent_background)
