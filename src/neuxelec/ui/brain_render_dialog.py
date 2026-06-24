from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from neuxelec.utils.resources import resource_path

from .neuxelec_color_dialog import NeuXelecColorDialog


class NeuXelecDialogHeader(QFrame):
    """
    Minimal frameless-window header with a custom close button.
    The empty header area can be dragged to move the dialog.
    """

    def __init__(self, dialog: QDialog, parent=None):
        super().__init__(parent)

        self.dialog = dialog
        self._drag_offset: QPoint | None = None

        self.setObjectName("customDialogHeader")
        self.setFixedHeight(34)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addStretch(1)

        self.btn_close_window = QPushButton("✕")
        self.btn_close_window.setObjectName("closeWindowButton")
        self.btn_close_window.setCursor(Qt.PointingHandCursor)
        self.btn_close_window.setFixedSize(30, 30)
        self.btn_close_window.clicked.connect(self.dialog.close)

        layout.addWidget(
            self.btn_close_window,
            0,
            Qt.AlignRight | Qt.AlignTop,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.dialog.frameGeometry().topLeft()
            )
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and bool(event.buttons() & Qt.LeftButton):
            self.dialog.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = None
            event.accept()
            return

        super().mouseReleaseEvent(event)


class BrainRenderDialog(QDialog):
    """
    Styled dialog used to configure brain / pial surface rendering in 3D View.
    Parameters are applied live to the current View3DPage.
    """

    ORIGINAL_PRESET = {
        "ambient": 0.10,
        "diffuse": 0.80,
        "specular": 0.30,
        "specular_power": 40.0,
        "key_light": 0.90,
        "fill_light": 0.10,
        "back_light": 0.15,
        "shadows": True,
    }

    def __init__(self, view3d_page, parent=None):
        super().__init__(parent)

        self.view3d_page = view3d_page
        self._positioned_once = False

        self.setWindowTitle("Render Brain")
        self.setModal(False)

        # Remove the native operating-system title bar and use the
        # NeuXelec custom rounded shell instead.
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # The native size grip is removed because it produces an unwanted
        # light marker on a dark frameless dialog. A transparent custom grip
        # is added inside the rounded shell instead.
        self.setSizeGripEnabled(False)

        # The dialog remains resizable and automatically fits smaller screens.
        self.setMinimumSize(390, 340)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()
        self._load_from_page()
        self._update_brain_color_button()
        self._connect_controls()

    # ============================================================
    # UI
    # ============================================================

    def _set_adapted_initial_size(self) -> None:
        """
        Open the dialog at its preferred size on a regular screen while
        automatically fitting it inside smaller displays.
        """
        preferred_width = 470
        preferred_height = 670

        try:
            screen = None

            parent = self.parentWidget()
            if parent is not None:
                screen = parent.screen()

            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.resize(preferred_width, preferred_height)
                return

            available = screen.availableGeometry()

            # Leave visible margins around the dialog, including space for the
            # operating-system title bar and task bar.
            max_width = max(390, int(available.width()) - 80)
            max_height = max(340, int(available.height()) - 100)

            initial_width = min(preferred_width, max_width)
            initial_height = min(preferred_height, max_height)

            self.resize(initial_width, initial_height)

        except Exception:
            self.resize(preferred_width, preferred_height)

    def showEvent(self, event) -> None:
        super().showEvent(event)

        if self._positioned_once:
            return

        self._positioned_once = True

        try:
            parent = self.parentWidget()

            if parent is None:
                return

            geometry = self.frameGeometry()
            geometry.moveCenter(parent.frameGeometry().center())
            self.move(geometry.topLeft())

        except Exception:
            pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        try:
            if not hasattr(self, "resize_grip"):
                return

            margin = 4
            self.resize_grip.move(
                self.dialog_shell.width() - self.resize_grip.width() - margin,
                self.dialog_shell.height() - self.resize_grip.height() - margin,
            )
            self.resize_grip.raise_()

        except Exception:
            pass

    def _build_ui(self) -> None:
        # ============================================================
        # Transparent outer dialog and rounded NeuXelec shell
        # ============================================================
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.dialog_shell = QFrame()
        self.dialog_shell.setObjectName("dialogShell")

        main_layout = QVBoxLayout(self.dialog_shell)
        main_layout.setContentsMargins(14, 8, 14, 14)
        main_layout.setSpacing(8)

        outer_layout.addWidget(self.dialog_shell)

        # ---------------------------------------------------------
        # Custom frameless header
        # ---------------------------------------------------------
        self.custom_header = NeuXelecDialogHeader(self)
        main_layout.addWidget(self.custom_header)

        # ============================================================
        # Scrollable settings area
        # ============================================================
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("renderBrainScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("renderBrainScrollContent")

        root = QVBoxLayout(self.scroll_content)
        root.setContentsMargins(12, 4, 12, 10)
        root.setSpacing(14)

        self.lbl_title = QLabel("RENDER BRAIN")
        self.lbl_title.setObjectName("dialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel(
            "Adjust the appearance of the brain or pial surface in the 3D view."
        )
        self.lbl_subtitle.setObjectName("dialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        self.lbl_subtitle.setWordWrap(True)
        root.addWidget(self.lbl_subtitle)

        root.addSpacing(4)

        # ---------------------------------------------------------
        # Material
        # ---------------------------------------------------------
        self.grp_material = QGroupBox("Material")
        self.grp_material.setMinimumHeight(180)

        form_mat = QFormLayout(self.grp_material)
        form_mat.setContentsMargins(16, 18, 16, 14)
        form_mat.setHorizontalSpacing(16)
        form_mat.setVerticalSpacing(9)

        self.spn_ambient = self._make_spin(0.0, 1.0, 0.01)
        self.spn_diffuse = self._make_spin(0.0, 1.0, 0.01)
        self.spn_specular = self._make_spin(0.0, 1.0, 0.01)
        self.spn_specular_power = self._make_spin(1.0, 200.0, 1.0)

        form_mat.addRow("Ambient", self.spn_ambient)
        form_mat.addRow("Diffuse", self.spn_diffuse)
        form_mat.addRow("Specular", self.spn_specular)
        form_mat.addRow("Specular power", self.spn_specular_power)

        root.addWidget(self.grp_material)

        # ---------------------------------------------------------
        # Lights
        # ---------------------------------------------------------
        self.grp_lights = QGroupBox("Lights")
        self.grp_lights.setMinimumHeight(145)

        form_lights = QFormLayout(self.grp_lights)
        form_lights.setContentsMargins(16, 18, 16, 14)
        form_lights.setHorizontalSpacing(16)
        form_lights.setVerticalSpacing(9)

        self.spn_key = self._make_spin(0.0, 5.0, 0.05)
        self.spn_fill = self._make_spin(0.0, 5.0, 0.05)
        self.spn_back = self._make_spin(0.0, 5.0, 0.05)

        form_lights.addRow("Key light", self.spn_key)
        form_lights.addRow("Fill light", self.spn_fill)
        form_lights.addRow("Back light", self.spn_back)

        root.addWidget(self.grp_lights)

        # ---------------------------------------------------------
        # Appearance
        # ---------------------------------------------------------
        self.grp_misc = QGroupBox("Appearance")
        self.grp_misc.setMinimumHeight(112)

        form_misc = QFormLayout(self.grp_misc)
        form_misc.setContentsMargins(16, 18, 16, 14)
        form_misc.setHorizontalSpacing(16)
        form_misc.setVerticalSpacing(10)

        self.chk_shadows = QCheckBox("Enable shadows")

        self.btn_color = QPushButton()
        self.btn_color.setObjectName("brainColorButton")
        self.btn_color.setMinimumHeight(34)
        self.btn_color.setCursor(Qt.PointingHandCursor)

        form_misc.addRow("Shadows", self.chk_shadows)
        form_misc.addRow("Brain color", self.btn_color)

        root.addWidget(self.grp_misc)
        root.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # ============================================================
        # Fixed bottom buttons
        # ============================================================
        btns = QHBoxLayout()
        btns.setSpacing(10)
        btns.setContentsMargins(12, 0, 12, 4)

        self.btn_reset_original = QPushButton("Reset original preset")
        self.btn_reset_original.setObjectName("secondaryButton")
        self.btn_reset_original.setCursor(Qt.PointingHandCursor)
        self.btn_reset_original.setMinimumHeight(42)

        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("primaryButton")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setMinimumHeight(42)
        self.btn_close.setMinimumWidth(108)

        btns.addWidget(self.btn_reset_original)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)

        main_layout.addLayout(btns)

        # ============================================================
        # Invisible but functional resize grip
        # ============================================================
        # This preserves manual resizing without displaying the native white
        # corner marker on the dark frameless dialog.
        self.resize_grip = QSizeGrip(self.dialog_shell)
        self.resize_grip.setObjectName("dialogResizeGrip")
        self.resize_grip.setFixedSize(16, 16)
        self.resize_grip.raise_()

    def _make_spin(
        self,
        vmin: float,
        vmax: float,
        step: float,
    ) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(vmin, vmax)
        sp.setSingleStep(step)
        sp.setDecimals(2 if step < 1 else 1)

        # Fixed height prevents the rows from being visually compressed.
        sp.setFixedHeight(34)
        sp.setMinimumWidth(128)
        sp.setAlignment(Qt.AlignRight)

        return sp

    def _connect_controls(self) -> None:
        # Live render updates
        self.spn_ambient.valueChanged.connect(self._push_to_page)
        self.spn_diffuse.valueChanged.connect(self._push_to_page)
        self.spn_specular.valueChanged.connect(self._push_to_page)
        self.spn_specular_power.valueChanged.connect(self._push_to_page)
        self.spn_key.valueChanged.connect(self._push_to_page)
        self.spn_fill.valueChanged.connect(self._push_to_page)
        self.spn_back.valueChanged.connect(self._push_to_page)
        self.chk_shadows.toggled.connect(self._push_to_page)

        self.btn_color.clicked.connect(self._choose_brain_color)
        self.btn_reset_original.clicked.connect(self._reset_original)
        self.btn_close.clicked.connect(self.close)

    # ============================================================
    # State / rendering
    # ============================================================

    def _load_from_page(self) -> None:
        params = getattr(self.view3d_page, "_brain_render_params", {})
        self._set_controls_from_params(params)

    def _set_controls_from_params(self, params: dict) -> None:
        """
        Fill controls without triggering repeated live rendering.
        """
        controls = (
            self.spn_ambient,
            self.spn_diffuse,
            self.spn_specular,
            self.spn_specular_power,
            self.spn_key,
            self.spn_fill,
            self.spn_back,
            self.chk_shadows,
        )

        for control in controls:
            control.blockSignals(True)

        try:
            self.spn_ambient.setValue(float(params.get("ambient", 0.10)))
            self.spn_diffuse.setValue(float(params.get("diffuse", 0.80)))
            self.spn_specular.setValue(float(params.get("specular", 0.30)))
            self.spn_specular_power.setValue(float(params.get("specular_power", 40.0)))
            self.spn_key.setValue(float(params.get("key_light", 1.2)))
            self.spn_fill.setValue(float(params.get("fill_light", 0.4)))
            self.spn_back.setValue(float(params.get("back_light", 0.25)))
            self.chk_shadows.setChecked(bool(params.get("shadows", True)))
        finally:
            for control in controls:
                control.blockSignals(False)

    def _current_parameters(self) -> dict:
        return {
            "ambient": float(self.spn_ambient.value()),
            "diffuse": float(self.spn_diffuse.value()),
            "specular": float(self.spn_specular.value()),
            "specular_power": float(self.spn_specular_power.value()),
            "key_light": float(self.spn_key.value()),
            "fill_light": float(self.spn_fill.value()),
            "back_light": float(self.spn_back.value()),
            "shadows": bool(self.chk_shadows.isChecked()),
        }

    def _push_to_page(self) -> None:
        self.view3d_page._brain_render_params = self._current_parameters()

        try:
            self.view3d_page._render_brain()
        except Exception:
            pass

    def _choose_brain_color(self) -> None:
        """
        Open the shared NeuXelec color dialog used throughout the application.
        The selected color is applied to the displayed brain or pial surface.
        """
        initial = self.view3d_page._tuple_to_qcolor(self.view3d_page._brain_color)

        color_hex = NeuXelecColorDialog.get_color(
            initial_color=initial,
            parent=self,
            title="Choose Brain color",
        )

        if color_hex is None:
            return

        color = QColor(color_hex)

        if not color.isValid():
            return

        self.view3d_page._brain_color = self.view3d_page._qcolor_to_tuple(color)

        self._update_brain_color_button()

        try:
            self.view3d_page._render_brain()
        except Exception:
            pass

    def _update_brain_color_button(self) -> None:
        """
        Display the current selected brain color directly in the button.
        """
        try:
            qcolor = self.view3d_page._tuple_to_qcolor(self.view3d_page._brain_color)
            color_hex = qcolor.name().upper()
        except Exception:
            color_hex = "#C9C0B5"

        self.btn_color.setText(color_hex)

        self.btn_color.setStyleSheet(f"""
            QPushButton#brainColorButton {{
                color: white;
                font-size: 12px;
                font-weight: 600;
                background-color: {color_hex};
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 8px;
                padding-left: 14px;
                padding-right: 14px;
            }}

            QPushButton#brainColorButton:hover {{
                border: 2px solid #FF487D;
            }}
            """)

    def _reset_original(self) -> None:
        """
        Restore original material and light parameters, then immediately update
        the 3D brain rendering.
        """
        self._set_controls_from_params(self.ORIGINAL_PRESET)

        self.view3d_page._brain_render_params = dict(self.ORIGINAL_PRESET)

        try:
            self.view3d_page._render_brain()
        except Exception:
            pass

    # ============================================================
    # Styling
    # ============================================================

    def _apply_style(self) -> None:
        """
        Apply the NeuXelec rounded frameless theme to the Render Brain dialog.
        """
        icon_dir = resource_path("resources/images")

        spin_up_path = (icon_dir / "spin_up.svg").as_posix()
        spin_down_path = (icon_dir / "spin_down.svg").as_posix()
        checkbox_cross_path = (icon_dir / "neuxelec_checkbox_cross.svg").as_posix()

        style = """
            QDialog {
                background: transparent;
            }

            QFrame#dialogShell {
                background-color: #06070D;
                border: 1.5px solid #FF487D;
                border-radius: 16px;
            }

            QFrame#customDialogHeader {
                background-color: transparent;
                border: none;
            }

            QPushButton#closeWindowButton {
                color: #D8DAE4;
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 700;
                padding: 0px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                text-align: center;
            }

            QPushButton#closeWindowButton:hover {
                color: white;
                border: 1px solid transparent;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#closeWindowButton:pressed {
                color: white;
                border: 1px solid transparent;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }

            QSizeGrip#dialogResizeGrip {
                background-color: transparent;
                border: none;
                image: none;
            }

            QWidget#renderBrainScrollContent {
                background-color: transparent;
            }

            QScrollArea#renderBrainScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#renderBrainScrollArea > QWidget > QWidget {
                background-color: transparent;
            }

            QScrollArea#renderBrainScrollArea QScrollBar:vertical {
                background-color: transparent;
                width: 13px;
                margin: 5px 2px 5px 2px;
                border: none;
                border-radius: 7px;
            }

            QScrollArea#renderBrainScrollArea QScrollBar::handle:vertical {
                background-color: #3F424C;
                border: 1px solid #FF487D;
                border-radius: 6px;
                min-height: 22px;
            }

            QScrollArea#renderBrainScrollArea QScrollBar::handle:vertical:hover {
                background-color: #4B4E59;
                border: 1px solid #FF6F9D;
                border-radius: 6px;
            }

            QScrollArea#renderBrainScrollArea QScrollBar::handle:vertical:pressed {
                background-color: #343743;
                border: 1px solid #FF487D;
                border-radius: 6px;
            }

            QScrollArea#renderBrainScrollArea QScrollBar::add-line:vertical,
            QScrollArea#renderBrainScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }

            QScrollArea#renderBrainScrollArea QScrollBar::add-page:vertical,
            QScrollArea#renderBrainScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }

            QLabel {
                color: #F2F2F5;
                font-size: 12px;
                border: none;
                background-color: transparent;
            }

            QLabel#dialogTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
            }

            QLabel#dialogSubtitle {
                color: #8E8E98;
                font-size: 12px;
                font-weight: 400;
                padding-bottom: 4px;
            }

            QGroupBox {
                color: #F2F2F5;
                font-size: 13px;
                font-weight: 600;
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 7px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0px 6px;
                color: #CFCFD6;
                background-color: #06070D;
            }

            QDoubleSpinBox {
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 7px;
                padding-left: 8px;
                padding-right: 27px;
                selection-background-color: #FF008F;
                font-size: 12px;
            }

            QDoubleSpinBox:hover {
                border: 1px solid #3A3D4A;
            }

            QDoubleSpinBox:focus {
                border: 1px solid #FF487D;
            }

            QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px;
                height: 17px;
                background-color: #24262F;
                border: none;
                border-left: 1px solid #2B2D38;
                border-top-right-radius: 7px;
            }

            QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px;
                height: 17px;
                background-color: #24262F;
                border: none;
                border-left: 1px solid #2B2D38;
                border-bottom-right-radius: 7px;
            }

            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover {
                background-color: #353844;
            }

            QDoubleSpinBox::up-button:pressed,
            QDoubleSpinBox::down-button:pressed {
                background-color: #191B22;
            }

            QDoubleSpinBox::up-arrow {
                image: url(__SPIN_UP__);
                width: 12px;
                height: 8px;
            }

            QDoubleSpinBox::down-arrow {
                image: url(__SPIN_DOWN__);
                width: 12px;
                height: 8px;
            }

            QCheckBox {
                color: #F2F2F5;
                spacing: 9px;
                font-size: 12px;
                font-weight: 500;
            }

            QCheckBox:disabled {
                color: #676A75;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: #171922;
                border: 1px solid #353844;
                border-radius: 4px;
            }

            QCheckBox::indicator:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
                border-radius: 4px;
            }

            QCheckBox::indicator:checked {
                background-color: #171922;
                border: 1px solid #FF487D;
                border-radius: 4px;
                image: url(__CHECKBOX_CROSS__);
            }

            QCheckBox::indicator:checked:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
                border-radius: 4px;
                image: url(__CHECKBOX_CROSS__);
            }

            QCheckBox::indicator:disabled {
                background-color: #121319;
                border: 1px solid #20222A;
                image: none;
            }

            QPushButton#secondaryButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding-left: 16px;
                padding-right: 16px;
            }

            QPushButton#secondaryButton:hover {
                background-color: #20222B;
                border: 1px solid #FF487D;
            }

            QPushButton#secondaryButton:pressed {
                background-color: #14151B;
                border: 1px solid #FF487D;
            }

            QPushButton#primaryButton {
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding-left: 20px;
                padding-right: 20px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
            }

            QPushButton#primaryButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QPushButton#primaryButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }
        """

        style = style.replace("__SPIN_UP__", spin_up_path)
        style = style.replace("__SPIN_DOWN__", spin_down_path)
        style = style.replace("__CHECKBOX_CROSS__", checkbox_cross_path)

        self.setStyleSheet(style)
