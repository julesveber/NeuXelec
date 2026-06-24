from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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
        self.btn_close_window.clicked.connect(self.dialog.reject)

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


class MarkerDialog(QDialog):
    """
    Dialog used to create or edit a 3D anatomical marker.

    Public data:
        - name
        - type
        - color
        - size_mm
        - description
        - ras
        - voxel_xyz

    Visibility is intentionally not edited here:
    it is controlled from the 3D context menu.
    """

    def __init__(
        self,
        marker: dict | None = None,
        ras_to_voxel: Callable[[list[float]], list[float] | None] | None = None,
        coordinate_space: str = "native",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._marker = dict(marker or {})
        self._ras_to_voxel = ras_to_voxel
        self._coordinate_space = "mni" if str(coordinate_space).lower() == "mni" else "native"
        self._color_hex = str(self._marker.get("color", "#FF3B30"))
        self._positioned_once = False

        self.setWindowTitle("Edit marker" if self._marker.get("id") else "Add marker")
        self.setModal(True)

        # Frameless rounded NeuXelec dialog.
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Disable the native white size grip and replace it with a transparent
        # internal grip compatible with the dark rounded design.
        self.setSizeGripEnabled(False)

        self.setMinimumSize(500, 420)
        self._set_adapted_initial_size()

        self._build_ui()
        self._apply_style()
        self._load_values()
        self._update_voxel_label()

    def _set_adapted_initial_size(self) -> None:
        """
        Open the marker editor at a comfortable size while fitting inside
        smaller screens. Internal content becomes scrollable if necessary.
        """
        preferred_width = 610
        preferred_height = 700

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

            max_width = max(500, int(available.width()) - 80)
            max_height = max(420, int(available.height()) - 100)

            self.resize(
                min(preferred_width, max_width),
                min(preferred_height, max_height),
            )

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
        # Transparent dialog and rounded NeuXelec shell
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
        # Frameless custom header
        # ---------------------------------------------------------
        self.custom_header = NeuXelecDialogHeader(self)
        main_layout.addWidget(self.custom_header)

        # ============================================================
        # Scrollable form content
        # ============================================================
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("markerScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("markerScrollContent")
        self.scroll_content.setMinimumHeight(538)

        content = QVBoxLayout(self.scroll_content)
        content.setContentsMargins(12, 4, 12, 10)
        content.setSpacing(14)

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------
        self.lbl_title = QLabel("EDIT MARKER" if self._marker.get("id") else "ADD MARKER")
        self.lbl_title.setObjectName("markerDialogTitle")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_title)

        space_label = "MNI space" if self._coordinate_space == "mni" else "patient native space"

        self.lbl_subtitle = QLabel(f"Define an anatomical point displayed in {space_label}")
        self.lbl_subtitle.setObjectName("markerDialogSubtitle")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        content.addWidget(self.lbl_subtitle)

        # ---------------------------------------------------------
        # Marker information card
        # ---------------------------------------------------------
        self.info_card = QFrame()
        self.info_card.setObjectName("markerInfoCard")

        card_layout = QVBoxLayout(self.info_card)
        card_layout.setContentsMargins(16, 12, 16, 16)
        card_layout.setSpacing(10)

        self.lbl_information = QLabel("Marker information")
        self.lbl_information.setObjectName("sectionLabel")
        card_layout.addWidget(self.lbl_information)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(11)

        self.edit_name = QLineEdit()
        self.edit_name.setObjectName("markerField")
        self.edit_name.setPlaceholderText("e.g. Lesion 1")
        self.edit_name.setMinimumHeight(38)

        self.combo_type = QComboBox()
        self.combo_type.setObjectName("markerField")
        self.combo_type.setMinimumHeight(38)
        self.combo_type.setCursor(Qt.PointingHandCursor)
        self.combo_type.addItems(
            [
                "Lesion",
                "ROI",
                "Resection",
                "Stimulation",
                "Anatomical landmark",
                "Other",
            ]
        )

        self.btn_color = QPushButton()
        self.btn_color.setObjectName("markerColorButton")
        self.btn_color.setMinimumHeight(38)
        self.btn_color.setCursor(Qt.PointingHandCursor)
        self.btn_color.clicked.connect(self._choose_color)

        self.spin_size = QDoubleSpinBox()
        self.spin_size.setObjectName("markerField")
        self.spin_size.setRange(0.5, 30.0)
        self.spin_size.setDecimals(1)
        self.spin_size.setSingleStep(0.5)
        self.spin_size.setSuffix(" mm")
        self.spin_size.setMinimumHeight(38)
        self.spin_size.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.edit_description = QTextEdit()
        self.edit_description.setObjectName("markerDescription")
        self.edit_description.setPlaceholderText("Optional description")
        self.edit_description.setMinimumHeight(76)

        form.addRow("Name", self.edit_name)
        form.addRow("Type", self.combo_type)
        form.addRow("Color", self.btn_color)
        form.addRow("Marker size", self.spin_size)
        form.addRow("Description", self.edit_description)

        card_layout.addLayout(form)
        content.addWidget(self.info_card)

        # ---------------------------------------------------------
        # Coordinates card
        # ---------------------------------------------------------
        self.coords_card = QFrame()
        self.coords_card.setObjectName("markerCoordinatesCard")

        coords_card_layout = QVBoxLayout(self.coords_card)
        coords_card_layout.setContentsMargins(16, 12, 16, 16)
        coords_card_layout.setSpacing(11)

        self.lbl_coordinates = QLabel("Coordinates")
        self.lbl_coordinates.setObjectName("sectionLabel")
        coords_card_layout.addWidget(self.lbl_coordinates)

        coords_form = QFormLayout()
        coords_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        coords_form.setHorizontalSpacing(18)
        coords_form.setVerticalSpacing(11)

        coords_widget = QWidget()
        coords_widget.setObjectName("coordinatesRow")

        coords_layout = QHBoxLayout(coords_widget)
        coords_layout.setContentsMargins(0, 0, 0, 0)
        coords_layout.setSpacing(8)

        self.spin_x = self._make_coordinate_spinbox()
        self.spin_y = self._make_coordinate_spinbox()
        self.spin_z = self._make_coordinate_spinbox()

        self.spin_x.valueChanged.connect(self._update_voxel_label)
        self.spin_y.valueChanged.connect(self._update_voxel_label)
        self.spin_z.valueChanged.connect(self._update_voxel_label)

        for axis, spin in (
            ("X", self.spin_x),
            ("Y", self.spin_y),
            ("Z", self.spin_z),
        ):
            label = QLabel(axis)
            label.setObjectName("axisLabel")
            coords_layout.addWidget(label)
            coords_layout.addWidget(spin, 1)

        self.lbl_voxel = QLabel("-")
        self.lbl_voxel.setObjectName("voxelValueLabel")

        voxel_label = (
            "Voxel coordinates (MNI template)"
            if self._coordinate_space == "mni"
            else "Voxel coordinates (T1)"
        )

        coords_form.addRow("RAS coordinates (mm)", coords_widget)
        coords_form.addRow(voxel_label, self.lbl_voxel)

        coords_card_layout.addLayout(coords_form)
        content.addWidget(self.coords_card)

        content.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, 1)

        # ============================================================
        # Fixed bottom actions
        # ============================================================
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(12, 0, 12, 4)
        bottom_layout.setSpacing(10)

        self.btn_export = QPushButton("Export")
        self.btn_export.setObjectName("secondaryButton")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setMinimumHeight(42)
        self.btn_export.clicked.connect(self._export_current_marker)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_save = QPushButton("Save marker")
        self.btn_save.setObjectName("primaryButton")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setMinimumHeight(42)
        self.btn_save.setMinimumWidth(126)
        self.btn_save.clicked.connect(self.accept)

        bottom_layout.addWidget(self.btn_export)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.btn_cancel)
        bottom_layout.addWidget(self.btn_save)

        main_layout.addLayout(bottom_layout)

        # ---------------------------------------------------------
        # Transparent functional resize grip
        # ---------------------------------------------------------
        self.resize_grip = QSizeGrip(self.dialog_shell)
        self.resize_grip.setObjectName("dialogResizeGrip")
        self.resize_grip.setFixedSize(16, 16)
        self.resize_grip.raise_()

    def _make_coordinate_spinbox(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName("coordinateSpin")
        spin.setRange(-10000.0, 10000.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setMinimumWidth(96)
        spin.setMinimumHeight(38)
        spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        return spin

    def _load_values(self) -> None:
        self.edit_name.setText(str(self._marker.get("name", "")))

        marker_type = str(self._marker.get("type", "Lesion"))
        idx = self.combo_type.findText(marker_type)
        self.combo_type.setCurrentIndex(idx if idx >= 0 else 0)

        self.spin_size.setValue(float(self._marker.get("size_mm", 4.0)))

        self.edit_description.setPlainText(str(self._marker.get("description", "")))

        ras = self._marker.get("ras", [0.0, 0.0, 0.0])
        if not isinstance(ras, (list, tuple)) or len(ras) != 3:
            ras = [0.0, 0.0, 0.0]

        self.spin_x.setValue(float(ras[0]))
        self.spin_y.setValue(float(ras[1]))
        self.spin_z.setValue(float(ras[2]))

        self._refresh_color_button()

    def _choose_color(self) -> None:
        color_hex = NeuXelecColorDialog.get_color(
            initial_color=self._color_hex,
            parent=self,
            title="Choose marker color",
        )

        if color_hex is None:
            return

        self._color_hex = color_hex
        self._refresh_color_button()

    def _refresh_color_button(self) -> None:
        self.btn_color.setText(self._color_hex)
        self.btn_color.setStyleSheet(f"""
            QPushButton#markerColorButton {{
                color: white;
                font-size: 12px;
                font-weight: 600;
                background-color: {self._color_hex};
                border: 1px solid rgba(255, 255, 255, 85);
                border-radius: 8px;
                min-height: 38px;
                padding-left: 14px;
                padding-right: 14px;
            }}

            QPushButton#markerColorButton:hover {{
                border: 2px solid #FF487D;
            }}
            """)

    def current_ras(self) -> list[float]:
        return [
            float(self.spin_x.value()),
            float(self.spin_y.value()),
            float(self.spin_z.value()),
        ]

    def _current_voxel(self) -> list[float] | None:
        if self._ras_to_voxel is None:
            return None

        try:
            voxel = self._ras_to_voxel(self.current_ras())
            if voxel is None or len(voxel) != 3:
                return None
            return [float(v) for v in voxel]
        except Exception:
            return None

    def _update_voxel_label(self) -> None:
        voxel = self._current_voxel()

        if voxel is None:
            self.lbl_voxel.setText("Unavailable")
            return

        self.lbl_voxel.setText(f"i = {voxel[0]:.2f}    j = {voxel[1]:.2f}    k = {voxel[2]:.2f}")

    def marker_data(self) -> dict:
        voxel = self._current_voxel()

        return {
            "name": self.edit_name.text().strip(),
            "type": self.combo_type.currentText().strip(),
            "color": self._color_hex,
            "size_mm": float(self.spin_size.value()),
            "description": self.edit_description.toPlainText().strip(),
            "ras": self.current_ras(),
            "voxel_xyz": voxel,
        }

    def accept(self) -> None:
        if not self.edit_name.text().strip():
            QMessageBox.warning(
                self,
                "Missing marker name",
                "Please enter a name for this marker.",
            )
            self.edit_name.setFocus()
            return

        super().accept()

    def _export_current_marker(self) -> None:
        data = self.marker_data()

        if not data["name"]:
            QMessageBox.warning(
                self,
                "Missing marker name",
                "Please enter a name before exporting the marker.",
            )
            return

        default_name = data["name"].replace(" ", "_").replace("/", "_") + ".txt"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export marker",
            default_name,
            "Text files (*.txt)",
        )

        if not path:
            return

        if not path.lower().endswith(".txt"):
            path += ".txt"

        voxel = data.get("voxel_xyz")

        lines = [
            "NeuXelec marker information",
            "--------------------------",
            f"Name: {data['name']}",
            f"Type: {data['type']}",
            f"Color: {data['color']}",
            f"Marker size: {data['size_mm']:.1f} mm",
            "",
            "Description:",
            data["description"] or "None",
            "",
            "RAS coordinates (mm):",
            f"X: {data['ras'][0]:.2f}",
            f"Y: {data['ras'][1]:.2f}",
            f"Z: {data['ras'][2]:.2f}",
            "",
            "Voxel coordinates in T1:",
        ]

        if voxel is None:
            lines.append("Unavailable")
        else:
            lines.extend(
                [
                    f"i: {voxel[0]:.2f}",
                    f"j: {voxel[1]:.2f}",
                    f"k: {voxel[2]:.2f}",
                ]
            )

        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def _resource_image_path(self, filename: str) -> str:
        """
        Return an absolute resource path usable inside Qt stylesheet url(...).
        Expected location:
            NeuXelec/resources/images/<filename>
        """
        try:
            here = Path(__file__).resolve()

            candidates = [
                here.parents[3] / "resources" / "images" / filename,
                here.parents[2] / "resources" / "images" / filename,
                Path.cwd() / "resources" / "images" / filename,
                Path.cwd() / "NeuXelec" / "resources" / "images" / filename,
                Path.cwd() / "Neuxelec" / "resources" / "images" / filename,
            ]

            for p in candidates:
                if p.exists():
                    return p.as_posix()

        except Exception:
            pass

        return filename

    def _apply_style(self) -> None:
        spin_down_path = self._resource_image_path("spin_down.svg")

        stylesheet = """
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

            QWidget#markerScrollContent {
                background-color: transparent;
            }

            QScrollArea#markerScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea#markerScrollArea > QWidget > QWidget {
                background-color: transparent;
            }

            QLabel {
                color: #F2F2F5;
                font-size: 12px;
                background-color: transparent;
                border: none;
            }

            QLabel#markerDialogTitle {
                color: #F4D9D0;
                font-size: 18px;
                font-weight: 500;
                letter-spacing: 1px;
            }

            QLabel#markerDialogSubtitle {
                color: #8E8E98;
                font-size: 12px;
                font-weight: 400;
                padding-bottom: 3px;
            }

            QLabel#sectionLabel {
                color: #CFCFD6;
                font-size: 12px;
                font-weight: 600;
                padding-bottom: 3px;
            }

            QLabel#axisLabel {
                color: #FF487D;
                font-size: 13px;
                font-weight: 700;
            }

            QFrame#markerInfoCard,
            QFrame#markerCoordinatesCard {
                background-color: #10121A;
                border: 1px solid #2B2D38;
                border-radius: 12px;
            }

            QLineEdit#markerField,
            QComboBox#markerField,
            QDoubleSpinBox#markerField,
            QDoubleSpinBox#coordinateSpin,
            QTextEdit#markerDescription {
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                padding-left: 10px;
                padding-right: 10px;
                selection-background-color: #FF008F;
                font-size: 12px;
            }


            QLineEdit#markerField:hover,
            QComboBox#markerField:hover,
            QDoubleSpinBox#markerField:hover,
            QDoubleSpinBox#coordinateSpin:hover,
            QTextEdit#markerDescription:hover {
                border: 1px solid #3A3D4A;
            }

            QLineEdit#markerField:focus,
            QComboBox#markerField:focus,
            QDoubleSpinBox#markerField:focus,
            QDoubleSpinBox#coordinateSpin:focus,
            QTextEdit#markerDescription:focus {
                border: 1px solid #FF487D;
            }

            QComboBox#markerField {
                padding-right: 30px;
            }

            QComboBox#markerField::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border: none;
                background-color: transparent;
            }

            QComboBox#markerField::down-arrow {
                image: url("__SPIN_DOWN_PATH__");
                width: 9px;
                height: 6px;
            }

            QComboBox#markerField::down-arrow:on {
                top: 1px;
            }

            QComboBox QAbstractItemView {
                background-color: #171922;
                color: #F2F2F5;
                border: 1px solid #2B2D38;
                selection-background-color: #35202F;
                selection-color: white;
            }

            QLabel#voxelValueLabel {
                color: #A6A8B2;
                background-color: #171922;
                border: 1px solid #2B2D38;
                border-radius: 8px;
                padding: 10px;
                min-height: 20px;
            }

            QPushButton#secondaryButton {
                color: white;
                background-color: #17181F;
                border: 1px solid #2B2D38;
                border-radius: 10px;
                min-height: 42px;
                padding-left: 18px;
                padding-right: 18px;
                font-size: 13px;
                font-weight: 600;
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
                min-height: 42px;
                padding-left: 18px;
                padding-right: 18px;
                font-size: 13px;
                font-weight: 600;
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

            QScrollArea#markerScrollArea QScrollBar:vertical {
                background-color: #111218;
                width: 13px;
                margin: 5px 2px 5px 2px;
                border: none;
                border-radius: 6px;
            }

            QScrollArea#markerScrollArea QScrollBar::handle:vertical {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF8000,
                    stop:1 #FF00A0
                );
                border-radius: 6px;
                min-height: 22px;
            }

            QScrollArea#markerScrollArea QScrollBar::handle:vertical:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF922B,
                    stop:1 #FF33B8
                );
            }

            QScrollArea#markerScrollArea QScrollBar::handle:vertical:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E56F00,
                    stop:1 #E0008C
                );
            }

            QScrollArea#markerScrollArea QScrollBar::add-line:vertical,
            QScrollArea#markerScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }

            QScrollArea#markerScrollArea QScrollBar::add-page:vertical,
            QScrollArea#markerScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """

        self.setStyleSheet(stylesheet.replace("__SPIN_DOWN_PATH__", spin_down_path))
