"""Fullscreen toggle for the 3D View page.

Isolates the floating fullscreen button and the enter/exit-fullscreen logic of
:class:`View3DPage` as a mixin. Methods are unchanged (two ``print`` calls were
replaced by logging). ``View3DPage`` inherits this mixin, so every ``self.*``
reference resolves exactly as before.

Host-provided attributes/methods used here include: ``container_3d``,
``interactor``, ``btn_3d_fullscreen``, ``ui``, ``_layout_3d``, ``_quick_tools``,
``lbl_planes_info``, ``plotter``, ``_resource_image_path``,
``_active_3d_parent_widget``, ``_update_quick_tools_geometry``,
``_update_planes_info_label`` and the ``_view3d_*`` fullscreen state fields.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class _Floating3DButtonResizeFilter(QObject):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def eventFilter(self, obj, event):
        try:
            if event.type() in (
                QEvent.Resize,
                QEvent.Show,
                QEvent.LayoutRequest,
                QEvent.Move,
            ):
                self.owner._schedule_fullscreen_button_geometry_update()
        except Exception:
            pass

        return False


class View3DFullscreenMixin:
    """Floating fullscreen button + enter/exit fullscreen for View3DPage."""

    def _create_fullscreen_button(self) -> None:
        """
        Create the floating fullscreen toggle button in the bottom-right
        corner of the 3D viewport.
        """
        if self.container_3d is None:
            return

        if self.btn_3d_fullscreen is not None:
            return

        self.btn_3d_fullscreen = QToolButton(self.container_3d)
        self.btn_3d_fullscreen.setObjectName("btn_3d_fullscreen")
        self.btn_3d_fullscreen.setCursor(Qt.PointingHandCursor)
        self.btn_3d_fullscreen.setToolTip("Fullscreen 3D view")
        self.btn_3d_fullscreen.setIconSize(QSize(28, 28))
        self.btn_3d_fullscreen.setFixedSize(36, 36)
        self.btn_3d_fullscreen.setAutoRaise(False)
        self.btn_3d_fullscreen.clicked.connect(self._toggle_3d_fullscreen)

        self.btn_3d_fullscreen.setStyleSheet("""
            QToolButton#btn_3d_fullscreen {
                background-color: rgba(18, 19, 25, 110);
                border: 1px solid rgba(180, 185, 195, 120);
                border-radius: 8px;
                padding: 4px;
            }

            QToolButton#btn_3d_fullscreen:hover {
                background-color: rgba(32, 34, 43, 170);
                border: 1px solid rgba(255, 255, 255, 210);
            }

            QToolButton#btn_3d_fullscreen:pressed {
                background-color: rgba(6, 7, 13, 210);
                border: 1px solid #FF3B8A;
            }
            """)

        self._update_fullscreen_button_icon()
        self._update_fullscreen_button_geometry()
        self.btn_3d_fullscreen.show()
        self.btn_3d_fullscreen.raise_()
        self._install_fullscreen_button_position_filter(self.container_3d)

        try:
            self._view3d_escape_shortcut = QShortcut(
                QKeySequence(Qt.Key_Escape),
                self.ui.findChild(QWidget, "page3DView") or self.container_3d,
            )
            self._view3d_escape_shortcut.activated.connect(self._exit_3d_fullscreen)
        except Exception:
            self._view3d_escape_shortcut = None

    def _update_fullscreen_button_icon(self) -> None:
        """
        Update the fullscreen button icon depending on the current mode.
        """
        if self.btn_3d_fullscreen is None:
            return

        if bool(getattr(self, "_view3d_is_fullscreen", False)):
            icon_path = self._resource_image_path("fullscreen_exit.svg")
            tooltip = "Exit fullscreen"
        else:
            icon_path = self._resource_image_path("fullscreen_enter.svg")
            tooltip = "Fullscreen 3D view"

        try:
            self.btn_3d_fullscreen.setIcon(QIcon(icon_path))
            self.btn_3d_fullscreen.setToolTip(tooltip)
        except Exception:
            pass

    def _update_fullscreen_button_geometry(self) -> None:
        """
        Keep the fullscreen button in the bottom-right corner
        of the currently active 3D viewport.

        Normal mode:
            parent = frame_17

        Fullscreen mode:
            parent = temporary fullscreen host
        """
        try:
            btn = self.btn_3d_fullscreen
            parent = self._active_3d_parent_widget()

            if btn is None or parent is None:
                return

            if btn.parentWidget() is not parent:
                btn.setParent(parent)
                btn.show()

            margin = 12
            x = max(0, int(parent.width()) - int(btn.width()) - margin)
            y = max(0, int(parent.height()) - int(btn.height()) - margin)

            btn.move(x, y)
            btn.raise_()
            # Reveal the button only now that it sits at the correct corner;
            # this prevents the brief flash at its previous position when
            # entering fullscreen (the button is hidden during the transition).
            if not btn.isVisible():
                btn.show()

        except Exception:
            pass

    def _schedule_fullscreen_button_geometry_update(self) -> None:
        """
        Reposition the fullscreen button after Qt has finished resizing layouts.
        This avoids the icon staying in the middle after window resize.
        """
        try:
            QTimer.singleShot(0, self._update_fullscreen_button_geometry)
            QTimer.singleShot(40, self._update_fullscreen_button_geometry)
            QTimer.singleShot(120, self._update_fullscreen_button_geometry)
        except Exception:
            pass

    def _install_fullscreen_button_position_filter(self, parent: QWidget | None = None) -> None:
        """
        Keep the fullscreen button anchored when the 3D viewport changes size.

        This is more robust than only patching resizeEvent, because the button
        can move between frame_17 and the fullscreen host.
        """
        try:
            if parent is None:
                parent = self._active_3d_parent_widget()

            if parent is None:
                return

            if self._fullscreen_button_resize_filter is None:
                self._fullscreen_button_resize_filter = _Floating3DButtonResizeFilter(self)

            parent.installEventFilter(self._fullscreen_button_resize_filter)

            try:
                if self.interactor is not None:
                    self.interactor.installEventFilter(self._fullscreen_button_resize_filter)
            except Exception:
                pass

            self._schedule_fullscreen_button_geometry_update()

        except Exception:
            pass

    def _toggle_3d_fullscreen(self) -> None:
        if bool(getattr(self, "_view3d_is_fullscreen", False)):
            self._exit_3d_fullscreen()
        else:
            self._enter_3d_fullscreen()

    def _enter_3d_fullscreen(self) -> None:
        """
        Show the 3D viewport fullscreen without moving frame_17.

        Important:
        We do NOT remove frame_17 from the main UI layout.
        Only self.interactor is moved into a temporary fullscreen host.
        This avoids the bug where frame_17 comes back in the wrong place.
        """
        if self.container_3d is None or self.interactor is None:
            return

        if bool(getattr(self, "_view3d_is_fullscreen", False)):
            return

        try:
            # Save where the interactor was inside frame_17.
            self._view3d_normal_layout_index = 0

            if self._layout_3d is not None:
                for i in range(self._layout_3d.count()):
                    item = self._layout_3d.itemAt(i)
                    if item is not None and item.widget() is self.interactor:
                        self._view3d_normal_layout_index = int(i)
                        break

                self._layout_3d.removeWidget(self.interactor)

            # Create a temporary fullscreen host.
            host = QWidget()
            host.setObjectName("view3d_fullscreen_host")
            host.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            host.setStyleSheet("""
                QWidget#view3d_fullscreen_host {
                    background-color: #000000;
                }
                """)

            layout = QVBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._view3d_fullscreen_host = host
            self._view3d_fullscreen_layout = layout

            # Move only the PyVista widget, not frame_17.
            self.interactor.setParent(host)
            layout.addWidget(self.interactor)

            # Move floating widgets to the fullscreen host.
            try:
                if self._quick_tools is not None:
                    self._quick_tools.hide()
                    self._quick_tools.setEnabled(False)
            except Exception:
                pass

            try:
                if self.lbl_planes_info is not None:
                    self.lbl_planes_info.setParent(host)
                    self.lbl_planes_info.raise_()
            except Exception:
                pass

            try:
                if self.btn_3d_fullscreen is not None:
                    self.btn_3d_fullscreen.setParent(host)
                    # Keep the button hidden until it has been repositioned to
                    # the new (fullscreen) bottom-right corner. It is revealed
                    # by _update_fullscreen_button_geometry once the host has
                    # its final size, which avoids a brief flash at the old
                    # position during the transition.
                    self.btn_3d_fullscreen.hide()
                    self._install_fullscreen_button_position_filter(self.container_3d)
                    self._schedule_fullscreen_button_geometry_update()
            except Exception:
                pass

            self._view3d_is_fullscreen = True
            self._update_fullscreen_button_icon()

            host.showFullScreen()

            try:
                self._view3d_host_escape_shortcut = QShortcut(
                    QKeySequence(Qt.Key_Escape),
                    host,
                )
                self._view3d_host_escape_shortcut.setContext(Qt.WindowShortcut)
                self._view3d_host_escape_shortcut.activated.connect(self._exit_3d_fullscreen)
            except Exception:
                self._view3d_host_escape_shortcut = None

            try:
                self.interactor.setFocus()
            except Exception:
                pass

            QTimer.singleShot(0, self._after_3d_fullscreen_geometry_changed)
            QTimer.singleShot(120, self._after_3d_fullscreen_geometry_changed)
            QTimer.singleShot(250, self._after_3d_fullscreen_geometry_changed)

        except Exception as e:
            logger.warning("Failed to enter 3D fullscreen: %s", e, exc_info=True)
            self._view3d_is_fullscreen = False

    def _exit_3d_fullscreen(self) -> None:
        """
        Restore the PyVista viewport inside frame_17.

        frame_17 was never moved, so the main 3D page layout remains intact.
        """
        if self.container_3d is None or self.interactor is None:
            return

        if not bool(getattr(self, "_view3d_is_fullscreen", False)):
            return

        try:
            host = getattr(self, "_view3d_fullscreen_host", None)
            layout = getattr(self, "_view3d_fullscreen_layout", None)
            try:
                self._view3d_host_escape_shortcut = None
            except Exception:
                pass

            if layout is not None:
                layout.removeWidget(self.interactor)

            # Put PyVista back into frame_17's internal layout.
            self.interactor.setParent(self.container_3d)

            if self._layout_3d is not None:
                index = int(getattr(self, "_view3d_normal_layout_index", 0))

                try:
                    self._layout_3d.insertWidget(index, self.interactor)
                except Exception:
                    self._layout_3d.addWidget(self.interactor)

            # Move floating widgets back onto frame_17.
            try:
                if self._quick_tools is not None:
                    self._quick_tools.setParent(self.container_3d)
                    self._quick_tools.setEnabled(True)
                    self._quick_tools.show()
                    self._quick_tools.raise_()
            except Exception:
                pass

            try:
                if self.lbl_planes_info is not None:
                    self.lbl_planes_info.setParent(self.container_3d)
                    self.lbl_planes_info.raise_()
            except Exception:
                pass

            try:
                if self.btn_3d_fullscreen is not None:
                    self.btn_3d_fullscreen.setParent(self.container_3d)
                    self.btn_3d_fullscreen.show()
                    self.btn_3d_fullscreen.raise_()
            except Exception:
                pass

            self._view3d_is_fullscreen = False
            self._update_fullscreen_button_icon()

            # Close and delete the temporary fullscreen host.
            if host is not None:
                try:
                    host.hide()
                    host.deleteLater()
                except Exception:
                    pass

            self._view3d_fullscreen_host = None
            self._view3d_fullscreen_layout = None

            try:
                self.interactor.setFocus()
            except Exception:
                pass

            QTimer.singleShot(0, self._after_3d_fullscreen_geometry_changed)
            QTimer.singleShot(120, self._after_3d_fullscreen_geometry_changed)
            QTimer.singleShot(250, self._after_3d_fullscreen_geometry_changed)

        except Exception as e:
            logger.warning("Failed to exit 3D fullscreen: %s", e, exc_info=True)

    def _after_3d_fullscreen_geometry_changed(self) -> None:
        """
        Refresh floating widgets and PyVista after entering/exiting fullscreen.
        """
        try:
            parent = self._active_3d_parent_widget()
            if parent is not None:
                parent.updateGeometry()
                parent.update()
        except Exception:
            pass

        try:
            if self.interactor is not None:
                self.interactor.updateGeometry()
                self.interactor.update()

                parent = self._active_3d_parent_widget()
                if parent is not None:
                    self.interactor.resize(parent.size())
        except Exception:
            pass

        try:
            self._update_quick_tools_geometry()
        except Exception:
            pass

        try:
            self._update_fullscreen_button_geometry()
        except Exception:
            pass
        try:
            self._schedule_fullscreen_button_geometry_update()
        except Exception:
            pass
        try:
            self._update_planes_info_label()
        except Exception:
            pass

        try:
            if self.plotter is not None:
                self.plotter.reset_camera_clipping_range()
                self.plotter.render()
        except Exception:
            pass
