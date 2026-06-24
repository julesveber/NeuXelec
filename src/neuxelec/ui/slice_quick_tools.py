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


class SliceQuickTools(QWidget):
    """
    Floating quick-tools menu for 2D oblique slice QLabel/QPixmap views.

    This toolbar is only a controller. The real actions are implemented in
    ObliqueSlicePage, because that page owns the pixmaps, zoom, pan, rotation,
    screenshot and JSON state.
    """

    def __init__(self, parent: QWidget, owner, slot_index: int):
        super().__init__(parent)

        self.owner = owner
        self.slot_index = int(slot_index)
        self._expanded = False
        self._background_removed = False

        # Circular carousel used when the slice view is too small for all tools.
        self._tool_buttons = []
        self._carousel_index = 0
        self._visible_tool_count = 999
        self._carousel_enabled = False

        self.setObjectName("SliceQuickTools")
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
        self.btn_arrow.setToolTip("Open slice quick tools")
        self.btn_arrow.setFixedSize(34, 34)
        self.btn_arrow.setIcon(QIcon(_toolbar_icon_path("qt_expand.svg")))
        self.btn_arrow.setIconSize(QSize(18, 18))
        self.btn_arrow.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_arrow.clicked.connect(self.toggle_menu)
        self.layout.addWidget(self.btn_arrow)

        self.panel = QWidget(self)
        self.panel.setObjectName("SliceQuickToolsPanel")
        self.panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.panel.installEventFilter(self)
        self.panel_layout = QHBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(5)

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_screenshot.svg",
                "Save a screenshot of this oblique slice and copy it to the clipboard.",
                self._take_screenshot,
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_gif.svg",
                "Export a GIF that rotates the oblique plane around the electrode axis.",
                self._export_gif,
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_save.svg",
                "Save this slice position: zoom, pan, rotation and background mode.",
                self._save_current_view,
            )
        )

        self.panel_layout.addWidget(
            self._make_btn(
                "qt_restore.svg",
                "Restore the saved position for this slice.",
                self._apply_saved_view,
            )
        )

        self.btn_background = self._make_btn(
            "qt_transparent_background.svg",
            "Remove black background for display, PNG screenshots and GIF export.",
            self._toggle_background_removed,
            checkable=True,
        )
        self.btn_background.setObjectName("btn_remove_background")
        self.panel_layout.addWidget(self.btn_background)

        rotations = {
            45: "qt_rotate_45.svg",
            90: "qt_rotate_90.svg",
            135: "qt_rotate_135.svg",
            180: "qt_rotate_180.svg",
        }

        for angle, icon_filename in rotations.items():
            self.panel_layout.addWidget(
                self._make_btn(
                    icon_filename,
                    f"Rotate this slice image by {angle}° around its centre.",
                    lambda _checked=False, a=angle: self._rotate_by(a),
                )
            )

        self.layout.addWidget(self.panel)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget#SliceQuickTools {
                background-color: transparent;
                border: none;
            }

            QWidget#SliceQuickToolsPanel {
                background-color: transparent;
                border: none;
            }

            QWidget#SliceQuickTools QPushButton {
                color: white;
                background-color: rgba(10, 12, 19, 210);
                border: 1px solid #D6D8E2;
                border-radius: 8px;
                padding: 0px;
            }

            QWidget#SliceQuickTools QPushButton:hover {
                background-color: rgba(18, 20, 29, 235);
                border: 1px solid #FF487D;
            }

            QWidget#SliceQuickTools QPushButton:pressed {
                background-color: rgba(28, 30, 40, 245);
                border: 1px solid #FF487D;
            }

            QWidget#SliceQuickTools QPushButton:checked {
                border: 1px solid #FF487D;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QWidget#SliceQuickTools QPushButton#btn_remove_background:checked {
                border: 1px solid #FFFFFF;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QWidget#SliceQuickTools QPushButton#btn_remove_background:checked:hover {
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
        button_width: int = 38,
        button_height: int = 30,
        spacing_px: int = 4,
    ) -> None:
        """
        Adapt the toolbar to the width of one oblique slice frame.
        """
        try:
            available_width = max(1, int(available_width))
            button_width = max(34, int(button_width))
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
                1,
                int((remaining_width + spacing_px) // (button_side + spacing_px)),
            )

            self._visible_tool_count = min(len(self._tool_buttons), slots)
            self._carousel_enabled = self._visible_tool_count < len(self._tool_buttons)

            if not self._carousel_enabled:
                self._carousel_index = 0

            self._apply_carousel_visibility()
            self.adjustSize()
            self.reposition()

        except Exception:
            pass

    def _apply_carousel_visibility(self) -> None:
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

        self.btn_arrow.setToolTip(
            "Close slice quick tools" if self._expanded else "Open slice quick tools"
        )

        self._apply_carousel_visibility()
        self.adjustSize()
        self.reposition()
        self.raise_()

    def _copy_image_file_to_clipboard(self, filename: str) -> bool:
        """
        Copy the saved screenshot file itself to the system clipboard.

        This behaves like selecting the PNG in Windows Explorer and pressing Ctrl+C.
        It keeps the saved PNG file intact, including transparency when the black
        background was removed.
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
            print("[Slice Quick Tools] Clipboard file copy failed:", e)
            return False

    def _take_screenshot(self) -> None:
        try:
            parent = self.parentWidget()
            filename, selected_filter = QFileDialog.getSaveFileName(
                parent,
                "Save oblique slice screenshot",
                f"neuxelec_oblique_slice_{self.slot_index}.png",
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

            # If background removal is enabled, force PNG.
            # JPEG cannot store transparency.
            if bool(self._background_removed) and not final_filename.lower().endswith(".png"):
                final_filename = str(Path(final_filename).with_suffix(".png"))

            if hasattr(self.owner, "_save_oblique_slice_screenshot"):
                self.owner._save_oblique_slice_screenshot(
                    self.slot_index,
                    final_filename,
                    remove_background=bool(self._background_removed),
                )

                copied = self._copy_image_file_to_clipboard(final_filename)

                if copied:
                    if bool(self._background_removed):
                        msg = "Screenshot saved and copied without black background"
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
            print("[Slice Quick Tools] Screenshot failed:", e)

    def _export_gif(self) -> None:
        try:
            parent = self.parentWidget()
            filename, _ = QFileDialog.getSaveFileName(
                parent,
                "Export oblique slice GIF",
                f"neuxelec_oblique_slice_{self.slot_index}_rotation.gif",
                "GIF animation (*.gif)",
            )
            if not filename:
                return

            if hasattr(self.owner, "_export_oblique_slice_gif"):
                self.owner._export_oblique_slice_gif(
                    self.slot_index,
                    filename,
                    remove_background=bool(self._background_removed),
                )
        except Exception as e:
            print("[Slice Quick Tools] GIF export failed:", e)

    def _save_current_view(self) -> None:
        try:
            if hasattr(self.owner, "_save_current_oblique_slice_view_to_state"):
                self.owner._save_current_oblique_slice_view_to_state(self.slot_index)
        except Exception as e:
            print("[Slice Quick Tools] Save view failed:", e)

    def _apply_saved_view(self) -> None:
        try:
            if hasattr(self.owner, "_apply_saved_oblique_slice_view_from_state"):
                self.owner._apply_saved_oblique_slice_view_from_state(self.slot_index)
            self._sync_background_button_from_owner()
        except Exception as e:
            print("[Slice Quick Tools] Apply saved view failed:", e)

    def _toggle_background_removed(self, checked: bool) -> None:
        self._background_removed = bool(checked)
        try:
            if hasattr(self.owner, "_set_oblique_slice_background_removed"):
                self.owner._set_oblique_slice_background_removed(self.slot_index, bool(checked))
        except Exception as e:
            print("[Slice Quick Tools] Background toggle failed:", e)

    def _rotate_by(self, angle_deg: float) -> None:
        try:
            if hasattr(self.owner, "_set_oblique_slice_rotation"):
                self.owner._set_oblique_slice_rotation(
                    self.slot_index,
                    float(angle_deg),
                    refresh=True,
                )
        except Exception as e:
            print("[Slice Quick Tools] Rotation failed:", e)

    def set_background_removed_checked(self, checked: bool) -> None:
        self._background_removed = bool(checked)
        try:
            self.btn_background.blockSignals(True)
            self.btn_background.setChecked(bool(checked))
            self.btn_background.blockSignals(False)
        except Exception:
            pass

    def _sync_background_button_from_owner(self) -> None:
        try:
            if hasattr(self.owner, "_get_oblique_slice_background_removed"):
                self.set_background_removed_checked(
                    bool(self.owner._get_oblique_slice_background_removed(self.slot_index))
                )
        except Exception:
            pass

    def background_removed_enabled(self) -> bool:
        return bool(self._background_removed)
