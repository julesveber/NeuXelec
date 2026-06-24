from __future__ import annotations

from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QPoint,
    Qt,
    QTimer,
)
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QLabel,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
)

from ..ui.context_menus import exec_electrode_tree_menu
from ..ui.neuxelec_color_dialog import NeuXelecColorDialog
from ..ui.neuxelec_message_dialog import (
    NeuXelecMessageDialog,
    NeuXelecTextInputDialog,
)
from ..utils.resources import resource_path

ROLE_KIND = Qt.UserRole + 1  # "electrode" | "meta" | "contact"
ROLE_ELEC_ID = Qt.UserRole + 2  # int
ROLE_CONTACT_INDEX = Qt.UserRole + 3  # int
ROLE_ROW_RGB = Qt.UserRole + 4  # tuple[int, int, int] used by the dark row delegate
ROLE_ROW_ALPHA = Qt.UserRole + 5  # int


def _qcolor_from_rgb(rgb: tuple[int, int, int], alpha: int = 255) -> QColor:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return QColor(r, g, b, int(alpha))


def _composited_row_color(
    rgb: tuple[int, int, int],
    alpha: int,
    base: QColor | None = None,
) -> QColor:
    """
    Return the visible row colour after blending it with the dark tree
    background. Contacts therefore keep the electrode hue while remaining
    visually secondary to the parent electrode row.
    """
    base = QColor("#0B0D14") if base is None else QColor(base)
    source = _qcolor_from_rgb(rgb, alpha)
    a = max(0.0, min(1.0, source.alpha() / 255.0))

    return QColor(
        round((source.red() * a) + (base.red() * (1.0 - a))),
        round((source.green() * a) + (base.green() * (1.0 - a))),
        round((source.blue() * a) + (base.blue() * (1.0 - a))),
    )


def _text_color_for_background(background: QColor) -> QColor:
    """Choose readable row text for bright or dark electrode colours."""
    luminance = 0.2126 * background.red() + 0.7152 * background.green() + 0.0722 * background.blue()
    return QColor("#080A10") if luminance >= 148 else QColor("#F2F2F5")


class ElectrodeRowColorDelegate(QStyledItemDelegate):
    """
    Paint full-width coloured electrode/contact rows on the dark interface.

    Using a delegate is intentional: the global dark QSS no longer masks the
    background set on QTreeWidgetItem rows. The colour remains visible during
    hover and selection, with a pink outline added on top.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        kind = index.data(ROLE_KIND)
        rgb = index.data(ROLE_ROW_RGB)
        alpha = index.data(ROLE_ROW_ALPHA)

        if kind == "meta":
            row_background = QColor("#151720")

        elif (
            kind
            in (
                "electrode",
                "contact",
                "mni_set",
                "mni_group",
                "mni_contact",
                "mni_electrode",
            )
            and rgb is not None
        ):
            try:
                tuple_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                row_background = _composited_row_color(
                    tuple_rgb,
                    int(alpha if alpha is not None else 255),
                )
            except Exception:
                row_background = QColor("#0B0D14")

        else:
            row_background = QColor("#0B0D14")

        row_rect = option.rect.adjusted(2, 1, -3, -1)

        tree = self.parent()
        dragging = bool(tree.property("neuxelec_dragging")) if tree is not None else False

        hovered = bool(option.state & QStyle.State_MouseOver) and not dragging
        selected = bool(option.state & QStyle.State_Selected) and not dragging

        # ---------------------------------------------------------
        # Row background
        # ---------------------------------------------------------
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(row_background))
        painter.drawRoundedRect(row_rect, 6, 6)

        # ---------------------------------------------------------
        # Hover / selection feedback
        # ---------------------------------------------------------
        if selected:
            painter.setBrush(QColor(255, 72, 125, 24))
            painter.setPen(QPen(QColor("#FF487D"), 1))
            painter.drawRoundedRect(row_rect, 6, 6)

        elif hovered:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#FF487D"), 1))
            painter.drawRoundedRect(row_rect, 6, 6)

        painter.restore()

        # ---------------------------------------------------------
        # Readable text depending on row colour
        # ---------------------------------------------------------
        text_color = (
            QColor("#9A9EAE") if kind == "meta" else _text_color_for_background(row_background)
        )

        # ---------------------------------------------------------
        # Let Qt draw text, checkbox and tree controls over our background
        # ---------------------------------------------------------
        draw_option = QStyleOptionViewItem(option)
        self.initStyleOption(draw_option, index)

        draw_option.backgroundBrush = QBrush(Qt.NoBrush)

        draw_option.palette.setColor(
            QPalette.ColorRole.Text,
            text_color,
        )
        draw_option.palette.setColor(
            QPalette.ColorRole.WindowText,
            text_color,
        )
        draw_option.palette.setColor(
            QPalette.ColorRole.HighlightedText,
            text_color,
        )

        # Selection, hover and native focus are already handled manually.
        # Removing HasFocus prevents Qt from drawing an unwanted blue focus mark.
        draw_option.state &= ~QStyle.State_Selected
        draw_option.state &= ~QStyle.State_MouseOver
        draw_option.state &= ~QStyle.State_HasFocus

        super().paint(painter, draw_option, index)


def _top_level_window():
    aw = QApplication.activeWindow()
    if aw is not None and aw.isWindow():
        return aw

    for w in QApplication.topLevelWidgets():
        try:
            if w is not None and w.isWindow() and w.isVisible():
                return w
        except Exception:
            pass

    return None


class ElectrodesController(QObject):
    """
    Manages the electrode/contact tree widgets in all pages.

    Uses QTreeWidget (not model/proxy) for robustness and Voxeloc-like behavior:
      - electrode is a parent item (colored)
      - meta rows + contact rows are children (meta grey, contacts same color with 50% alpha)
      - expand/collapse is handled by the native triangle only
      - parent checkbox controls all children; children can be toggled individually
      - selecting a contact can jump the crosshair in ReconstructionPage
    """

    def __init__(self, ui, state, reco_page=None):
        super().__init__()
        self.ui = ui
        self.state = state
        self.reco_page = reco_page
        # Allow View3DPage to force a clean rebuild of the 3D electrode tree
        # when switching between native patient mode and MNI atlas mode.
        try:
            self.state.electrodes_controller = self
        except Exception:
            pass

        self._updating_checks = False
        self._trees: list[QTreeWidget] = []

        # Counters displayed above the electrode lists.
        self._count_labels: list[QLabel] = []

        for label_name in (
            "lbl_ElectrodesCount",
            "lbl_ElectrodesCount_2",
            "lbl_ElectrodesCount_3",
        ):
            label = getattr(ui, label_name, None)

            if label is None:
                try:
                    label = ui.findChild(QLabel, label_name)
                except Exception:
                    label = None

            if isinstance(label, QLabel):
                self._count_labels.append(label)

        self._ignore_next_tree_click = False
        self._tree_drag_tree = None
        self._tree_drag_start_pos = None
        self._tree_drag_source_ids = []
        self._tree_drag_started = False
        self._tree_drop_indicator = None
        self._tree_drop_indicator_tree = None
        self._tree_drop_target_id = None
        self._tree_drop_after = False
        self._tree_drag_count_badge = None
        self._tree_previous_selection_modes = {}
        self._tree_drag_saved_styles = {}

        self._check_drag_active = False
        self._check_drag_tree = None
        self._check_drag_state = None
        self._check_drag_seen = set()
        self._check_drag_changed = False

        # Coalesce repeated refresh requests triggered by checkbox cascades
        self._pending_refresh_reco = False
        self._pending_refresh_view3d = False
        self._pending_refresh_oblique = False

        self._views_refresh_timer = QTimer(self)
        self._views_refresh_timer.setSingleShot(True)
        self._views_refresh_timer.timeout.connect(self._flush_views_refresh)

        for name in ("tv_Electrodes", "tv_Electrodes_2", "tv_Electrodes_3"):
            w = getattr(ui, name, None)
            if isinstance(w, QTreeWidget):
                self._trees.append(w)

        # Configure trees
        for t in self._trees:
            t.setColumnCount(1)
            t.setHeaderHidden(True)
            t.setIndentation(20)
            t.setRootIsDecorated(True)
            t.setExpandsOnDoubleClick(False)
            # Ctrl-click and Shift-click multi-selection
            t.setSelectionMode(QAbstractItemView.ExtendedSelection)
            t.setSelectionBehavior(QAbstractItemView.SelectRows)

            # We implement our own drag-reorder logic to keep control of state.electrodes
            t.setDragEnabled(False)
            t.setAcceptDrops(False)
            t.setDropIndicatorShown(False)
            t.viewport().installEventFilter(self)
            t.itemChanged.connect(self._on_item_changed)
            t.itemSelectionChanged.connect(self._on_selection_changed)
            t.itemExpanded.connect(self._on_item_expanded)
            t.itemCollapsed.connect(self._on_item_collapsed)

            # RIGHT CLICK MENU
            t.setContextMenuPolicy(Qt.CustomContextMenu)
            t.customContextMenuRequested.connect(self._open_context_menu)

            # hover only in the tree UI, with no Python-side hover logic
            t.setMouseTracking(True)
            t.viewport().setMouseTracking(True)

            # Row backgrounds are painted by ElectrodeRowColorDelegate so that
            # they remain visible on the dark Main Window theme.
            t.setItemDelegate(ElectrodeRowColorDelegate(t))
            t.setProperty("neuxelec_dragging", False)

            # SVG arrows used to expand / collapse electrode rows.
            # electrodes.py is in src/neuxelec/controllers/, therefore
            # parents[3] corresponds to the NeuXelec project directory.
            icons_dir = resource_path("resources/images")

            spin_up_path = (icons_dir / "spin_up.svg").as_posix()
            spin_down_path = (icons_dir / "spin_down.svg").as_posix()

            tree_style = """
                QTreeWidget {
                    color: #F2F2F5;
                    background-color: #0B0D14;
                    border: none;
                    outline: none;
                    padding: 4px;
                    selection-background-color: transparent;
                    selection-color: #F2F2F5;
                }

                QTreeWidget::item {
                    background-color: transparent;
                    border: none;
                    min-height: 27px;
                    padding: 2px 5px;
                }

                QTreeWidget::item:hover,
                QTreeWidget::item:selected,
                QTreeWidget::item:selected:hover {
                    background-color: transparent;
                    border: none;
                }

                QTreeWidget::branch {
                    background-color: transparent;
                    border: none;
                    width: 14px;
                    min-width: 14px;
                    selection-background-color: transparent;
                }

                /*
                Closed electrode: display the custom white SVG arrow.
                Explicit selected rules prevent Qt from drawing its native
                blue branch indicator when the electrode row is selected.
                */
                QTreeWidget::branch:closed:has-children,
                QTreeWidget::branch:closed:has-children:selected,
                QTreeWidget::branch:closed:has-children:active:selected,
                QTreeWidget::branch:closed:has-children:!active:selected {
                    image: url(__SPIN_DOWN__);
                    background-color: transparent;
                    border: none;
                    width: 9px;
                    height: 6px;
                }

                /*
                Open electrode: display the custom white SVG arrow even
                when the row is currently selected.
                */
                QTreeWidget::branch:open:has-children,
                QTreeWidget::branch:open:has-children:selected,
                QTreeWidget::branch:open:has-children:active:selected,
                QTreeWidget::branch:open:has-children:!active:selected {
                    image: url(__SPIN_UP__);
                    background-color: transparent;
                    border: none;
                    width: 9px;
                    height: 6px;
                }

                QTreeWidget::branch:selected,
                QTreeWidget::branch:active:selected,
                QTreeWidget::branch:!active:selected {
                    background-color: transparent;
                    border: none;
                }

                QTreeWidget::branch:closed:has-children:hover,
                QTreeWidget::branch:closed:has-children:selected:hover,
                QTreeWidget::branch:open:has-children:hover,
                QTreeWidget::branch:open:has-children:selected:hover {
                    background-color: rgba(255, 72, 125, 35);
                    border: none;
                    border-radius: 5px;
                }
            """

            self._tree_normal_style = tree_style.replace("__SPIN_UP__", spin_up_path).replace(
                "__SPIN_DOWN__", spin_down_path
            )

            self._tree_drag_style = self._tree_normal_style

            t.setStyleSheet(self._tree_normal_style)

        # Refresh on state changes (estimate / visibility / color)
        if hasattr(self.state, "register_electrodes_changed"):
            self.state.register_electrodes_changed(self.refresh_all)

        self.refresh_all()

    def _schedule_views_refresh(self, reco: bool = True, view3d: bool = True, oblique: bool = True):
        if reco:
            self._pending_refresh_reco = True
        if view3d:
            self._pending_refresh_view3d = True
        if oblique:
            self._pending_refresh_oblique = True

        self._views_refresh_timer.start(20)

    def _flush_views_refresh(self):
        do_reco = self._pending_refresh_reco
        do_view3d = self._pending_refresh_view3d
        do_oblique = self._pending_refresh_oblique

        self._pending_refresh_reco = False
        self._pending_refresh_view3d = False
        self._pending_refresh_oblique = False

        # Reconstruction page
        if do_reco:
            try:
                if hasattr(self, "reco_page") and self.reco_page is not None:
                    self.reco_page.render_all()
            except Exception:
                pass

        # 3D view
        if do_view3d:
            try:
                vp = getattr(self.state, "view3d_page", None)
                if vp is not None and hasattr(vp, "update_electrodes"):
                    vp.update_electrodes()
            except Exception:
                pass

        # Oblique slice page
        if do_oblique:
            try:
                op = getattr(self.state, "oblique_page", None)
                if op is not None:
                    if hasattr(op, "_schedule_refresh"):
                        op._schedule_refresh(slices=True, brain=True)
                    elif hasattr(op, "render_all"):
                        op.render_all()
            except Exception:
                pass

    def _refresh_for_current_page_only(self) -> None:
        current_page = self._get_current_page_name()

        if current_page == "pageObliqueSlices":
            try:
                op = getattr(self.state, "oblique_page", None)
                if op is not None and hasattr(op, "_schedule_refresh"):
                    op._schedule_refresh(slices=True, brain=True)
            except Exception:
                pass
            return

        if current_page == "page3DView":
            # Do not call update_electrodes() here.
            # Visibility changes are already handled immediately by
            # vp.set_electrode_visible() / vp.set_contact_visible().
            try:
                vp = getattr(self.state, "view3d_page", None)
                if vp is not None and hasattr(vp, "_render"):
                    vp._render()
            except Exception:
                pass
            return

        if current_page == "pageReconstruction":
            try:
                if self.reco_page is not None and hasattr(self.reco_page, "render_all"):
                    self.reco_page.render_all()
            except Exception:
                pass
            return

    def _set_page_visibility_silent(
        self, page, kind: str, elec_id: int, contact_idx: int = -1, checked: bool = True
    ) -> None:
        """
        Update page-local visibility state without rendering.
        Used during checkbox drag; final render happens once on mouse release.
        """
        if page is None:
            return

        try:
            elec_id = int(elec_id)
            elec = self.state.electrodes[elec_id]
            n = len(elec.get("contacts_lps", []) or [])
        except Exception:
            return

        try:
            if not hasattr(page, "_page_electrode_visible"):
                page._page_electrode_visible = {}
            if not hasattr(page, "_page_contacts_visible"):
                page._page_contacts_visible = {}
        except Exception:
            return

        if kind == "electrode":
            try:
                page._page_electrode_visible[elec_id] = bool(checked)
                page._page_contacts_visible[elec_id] = [bool(checked)] * int(n)
            except Exception:
                pass
            return

        if kind == "contact":
            try:
                contact_idx = int(contact_idx)

                vals = page._page_contacts_visible.get(elec_id)
                if not isinstance(vals, list) or len(vals) != int(n):
                    vals = [True] * int(n)

                if 0 <= contact_idx < len(vals):
                    vals[contact_idx] = bool(checked)

                page._page_contacts_visible[elec_id] = vals
                page._page_electrode_visible[elec_id] = any(vals)
            except Exception:
                pass

    def _dispatch_local_visibility_update(
        self,
        kind: str,
        elec_id: int,
        contact_idx: int = -1,
        checked: bool = True,
        refresh: bool = True,
    ) -> None:
        current_page = self._get_current_page_name()

        if current_page == "pageObliqueSlices":
            op = getattr(self.state, "oblique_page", None)
            if op is None:
                return

            if not refresh:
                self._set_page_visibility_silent(op, kind, elec_id, contact_idx, checked)
                return

            try:
                if kind == "electrode":
                    op.set_electrode_visible(int(elec_id), bool(checked))
                elif kind == "contact":
                    op.set_contact_visible(int(elec_id), int(contact_idx), bool(checked))
            except Exception:
                pass
            return

        if current_page == "page3DView":
            vp = getattr(self.state, "view3d_page", None)
            if vp is None:
                return

            if not refresh:
                self._set_page_visibility_silent(vp, kind, elec_id, contact_idx, checked)
                return

            try:
                if kind == "electrode":
                    vp.set_electrode_visible(int(elec_id), bool(checked))
                elif kind == "contact":
                    vp.set_contact_visible(int(elec_id), int(contact_idx), bool(checked))
            except Exception:
                pass
            return

        if current_page == "pageReconstruction":
            rp = self.reco_page
            if rp is None:
                return

            if not refresh:
                self._set_page_visibility_silent(rp, kind, elec_id, contact_idx, checked)
                return

            try:
                if kind == "electrode":
                    rp.set_electrode_visible(int(elec_id), bool(checked))
                elif kind == "contact":
                    rp.set_contact_visible(int(elec_id), int(contact_idx), bool(checked))
            except Exception:
                pass
            return

    def _dispatch_local_visual_update(
        self, action: str, elec_id: int, contact_idx: int = -1
    ) -> None:
        current_page = self._get_current_page_name()

        if current_page == "pageObliqueSlices":
            op = getattr(self.state, "oblique_page", None)
            if op is None:
                return

            if action == "toggle_labels":
                try:
                    elec = self.state.electrodes[int(elec_id)]
                    n = len(elec.get("contacts_lps", []) or [])
                    vals = op._get_local_contact_labels_visible(int(elec_id), n)
                    new_state = not any(vals)
                    op.set_labels_visible(int(elec_id), new_state)
                except Exception:
                    pass
                return

            if action == "toggle_label":
                try:
                    elec = self.state.electrodes[int(elec_id)]
                    n = len(elec.get("contacts_lps", []) or [])
                    vals = op._get_local_contact_labels_visible(int(elec_id), n)
                    old = (
                        bool(vals[int(contact_idx)]) if 0 <= int(contact_idx) < len(vals) else False
                    )
                    op.set_contact_label_visible(int(elec_id), int(contact_idx), not old)
                except Exception:
                    pass
                return

            # check/uncheck on oblique = immediate local slice update
            if action == "color":
                try:
                    if hasattr(op, "refresh_after_electrode_color_change"):
                        op.refresh_after_electrode_color_change()
                    else:
                        op._last_plane1_key = None
                        op._last_plane2_key = None
                        op._schedule_refresh(slices=True, brain=True)
                except Exception:
                    pass
                return

            if action == "visibility":
                try:
                    op._schedule_refresh(slices=True, brain=True)
                except Exception:
                    pass
                return

        if current_page == "page3DView":
            vp = getattr(self.state, "view3d_page", None)
            if vp is None:
                return

            if action == "toggle_labels":
                try:
                    elec = self.state.electrodes[int(elec_id)]
                    n = len(elec.get("contacts_lps", []) or [])
                    vals = vp._get_local_contact_labels_visible(int(elec_id), n)
                    new_state = not any(vals)
                    vp.set_labels_visible(int(elec_id), new_state)
                except Exception:
                    pass
                return

            if action == "toggle_label":
                try:
                    elec = self.state.electrodes[int(elec_id)]
                    n = len(elec.get("contacts_lps", []) or [])
                    vals = vp._get_local_contact_labels_visible(int(elec_id), n)
                    old = (
                        bool(vals[int(contact_idx)]) if 0 <= int(contact_idx) < len(vals) else False
                    )
                    vp.set_contact_label_visible(int(elec_id), int(contact_idx), not old)
                except Exception:
                    pass
                return

            if action in ("visibility", "color"):
                try:
                    vp.update_electrodes()
                except Exception:
                    pass
                return

        if current_page == "pageReconstruction":
            if self.reco_page is not None:
                try:
                    self.reco_page.render_all()
                except Exception:
                    pass

    def _mni_mode_active(self) -> bool:
        """
        True when the 3D View is currently in MNI atlas mode.

        In this mode, tv_Electrodes_3 must show only MNI electrodes,
        not patient-native electrodes from state.electrodes.
        """
        try:
            vp = getattr(self.state, "view3d_page", None)

            if vp is None:
                return False

            if hasattr(vp, "is_mni_atlas_active"):
                return bool(vp.is_mni_atlas_active())

            cb = getattr(vp, "chk_mni_atlas", None)
            return bool(cb is not None and cb.isChecked())

        except Exception:
            return False

    def _refresh_mni_tree_if_needed(self) -> None:
        """
        Reinsert MNI items in tv_Electrodes_3 after a global tree refresh.

        Without this, changing the color of a native electrode can clear
        tv_Electrodes_3 and make MNI electrodes disappear from the list.
        """
        try:
            if not self._mni_mode_active():
                return

            vp = getattr(self.state, "view3d_page", None)

            if vp is not None and hasattr(vp, "_refresh_mni_tree_items"):
                vp._refresh_mni_tree_items()

        except Exception:
            pass

    def _dispatch_global_structural_update(self) -> None:
        # 1) refresh all electrode trees / UI lists
        self.refresh_all()

        # 2) refresh Reconstruction page immediately
        try:
            if self.reco_page is not None and hasattr(self.reco_page, "render_all"):
                self.reco_page.render_all()
        except Exception:
            pass

        # 3) refresh 3D View immediately
        try:
            vp = getattr(self.state, "view3d_page", None)

            if vp is not None:
                mni_active = False

                try:
                    if hasattr(vp, "is_mni_atlas_active"):
                        mni_active = bool(vp.is_mni_atlas_active())
                    else:
                        cb = getattr(vp, "chk_mni_atlas", None)
                        mni_active = bool(cb is not None and cb.isChecked())
                except Exception:
                    mni_active = False

                if mni_active:
                    # Never redraw patient-native electrodes in MNI mode.
                    if hasattr(vp, "_clear_native_scene_for_mni_mode"):
                        vp._clear_native_scene_for_mni_mode()

                    if hasattr(vp, "_render_mni_scene"):
                        vp._render_mni_scene(reset_camera=False)

                    if hasattr(vp, "_refresh_mni_tree_items"):
                        vp._refresh_mni_tree_items()

                else:
                    if hasattr(vp, "update_electrodes"):
                        vp.update_electrodes()

                    if hasattr(vp, "render_all_surface_projections"):
                        vp.render_all_surface_projections()

                    if hasattr(vp, "_refresh_multiplanar_clipped_scene"):
                        vp._refresh_multiplanar_clipped_scene()

        except Exception:
            pass

        # 4) refresh Oblique Slice immediately
        try:
            op = getattr(self.state, "oblique_page", None)
            if op is not None:
                if hasattr(op, "_schedule_refresh"):
                    op._schedule_refresh(slices=True, brain=True)
                elif hasattr(op, "render_all"):
                    op.render_all()
        except Exception:
            pass

    def _update_electrodes_count_labels(self) -> None:
        """
        Update the counters displayed above the three electrode lists.
        """
        try:
            n_electrodes = len(getattr(self.state, "electrodes", []) or [])

            if n_electrodes == 1:
                text = "1 electrode"
            else:
                text = f"{n_electrodes} electrodes"

            for label in getattr(self, "_count_labels", []):
                try:
                    label.setText(text)
                    label.adjustSize()
                except Exception:
                    pass

        except Exception:
            pass

    # ---------- Public ----------
    def refresh_all(self) -> None:
        """
        Rebuild electrode trees.

        Native patient mode:
            all trees display state.electrodes.

        MNI atlas mode:
            tv_Electrodes_3 displays only MNI electrodes.
            Patient-native electrodes are hidden from the 3D View list.
        """
        self._hide_tree_drop_indicator()
        expanded = self._get_expanded_state()
        mni_active = self._mni_mode_active()

        for t in self._trees:
            t.blockSignals(True)

            try:
                t.clear()

                # In 3D MNI mode, the 3D electrode list must not show
                # patient-native electrodes. MNI items will be reinserted after.
                is_3d_tree = str(t.objectName()) == "tv_Electrodes_3"

                if mni_active and is_3d_tree:
                    continue

                for elec_id, elec in enumerate(self.state.electrodes):
                    self._add_electrode_node(t, elec_id, elec)

            finally:
                t.blockSignals(False)

        # Restore expanded state only for native patient trees.
        for t in self._trees:
            try:
                if mni_active and str(t.objectName()) == "tv_Electrodes_3":
                    continue

                self._apply_expanded_state(t, expanded)

            except Exception:
                pass

        # Re-add MNI items after clearing tv_Electrodes_3.
        self._refresh_mni_tree_if_needed()

        # Update counters above all three electrode lists.
        self._update_electrodes_count_labels()

    def _update_electrode_color_items_in_trees(self, elec_ids: list[int]) -> None:
        """
        Update only the row color data for the given native electrodes in all trees.
        Does not rebuild the tree and does not touch other electrodes.
        """
        valid_ids = sorted(
            {
                int(eid)
                for eid in elec_ids
                if 0 <= int(eid) < len(getattr(self.state, "electrodes", []) or [])
            }
        )

        if not valid_ids:
            return

        for tree in self._trees:
            try:
                tree.blockSignals(True)

                for row in range(tree.topLevelItemCount()):
                    item = tree.topLevelItem(row)

                    try:
                        raw_id = item.data(0, ROLE_ELEC_ID)
                        if raw_id is None:
                            continue

                        elec_id = int(raw_id)
                    except Exception:
                        continue

                    if elec_id not in valid_ids:
                        continue

                    rgb = tuple(self.state.electrodes[elec_id].get("color", (255, 165, 0)))

                    # Parent electrode row.
                    item.setData(0, ROLE_ROW_RGB, tuple(rgb))
                    item.setData(0, ROLE_ROW_ALPHA, 255)

                    # Contact rows only.
                    for child_index in range(item.childCount()):
                        child = item.child(child_index)
                        if child.data(0, ROLE_KIND) != "contact":
                            continue

                        child.setData(0, ROLE_ROW_RGB, tuple(rgb))
                        child.setData(0, ROLE_ROW_ALPHA, 148)

                tree.viewport().update()

            except Exception:
                pass

            finally:
                try:
                    tree.blockSignals(False)
                except Exception:
                    pass

    def _dispatch_color_update_only(self, elec_ids: list[int]) -> None:
        """
        Dispatch a native electrode color update without rebuilding all electrodes.
        """
        valid_ids = sorted(
            {
                int(eid)
                for eid in elec_ids
                if 0 <= int(eid) < len(getattr(self.state, "electrodes", []) or [])
            }
        )

        if not valid_ids:
            return

        current_page = self._get_current_page_name()

        # 3D View: update only the selected electrode actors.
        if current_page == "page3DView":
            try:
                vp = getattr(self.state, "view3d_page", None)

                if vp is not None:
                    for elec_id in valid_ids:
                        if hasattr(vp, "update_single_electrode_color_only"):
                            vp.update_single_electrode_color_only(elec_id, render=False)

                    if hasattr(vp, "_render"):
                        vp._render()

            except Exception:
                pass

            return

        # Oblique Slice: images contain electrode colors, so refresh slices only.
        if current_page == "pageObliqueSlices":
            try:
                op = getattr(self.state, "oblique_page", None)

                if op is not None:
                    if hasattr(op, "refresh_after_electrode_color_change"):
                        op.refresh_after_electrode_color_change()
                    elif hasattr(op, "_schedule_refresh"):
                        op._schedule_refresh(slices=True, brain=True)

            except Exception:
                pass

            return

        # Reconstruction: color affects overlays; redraw only this page.
        if current_page == "pageReconstruction":
            try:
                if self.reco_page is not None and hasattr(self.reco_page, "render_all"):
                    self.reco_page.render_all()
            except Exception:
                pass

    def pick_color_for_selected_electrode(self) -> None:
        elec_id = self._current_selected_electrode_id()
        if elec_id is None:
            return

        self.pick_color_for_electrodes([int(elec_id)])

    def pick_color_for_electrodes(self, elec_ids: list[int]) -> None:
        elec_ids = sorted({int(i) for i in elec_ids if i is not None})

        if not elec_ids:
            return

        elec_ids = [
            i for i in elec_ids if 0 <= int(i) < len(getattr(self.state, "electrodes", []) or [])
        ]

        if not elec_ids:
            return

        first_elec = self.state.electrodes[elec_ids[0]]
        rgb = first_elec.get("color", (0, 255, 0))
        current = _qcolor_from_rgb(rgb, 255)

        if len(elec_ids) == 1:
            dialog_title = "Choose electrode color"
        else:
            dialog_title = f"Choose color for {len(elec_ids)} electrodes"

        color_hex = NeuXelecColorDialog.get_color(
            initial_color=current,
            parent=self.ui.window(),
            title=dialog_title,
        )

        if color_hex is None:
            return

        c = QColor(color_hex)

        if not c.isValid():
            return

        new_rgb = (c.red(), c.green(), c.blue())

        for elec_id in elec_ids:
            try:
                self.state.electrodes[int(elec_id)]["color"] = new_rgb
            except Exception:
                pass

        # Update only the modified rows in the electrode trees.
        self._update_electrode_color_items_in_trees(elec_ids)

        # Update only the modified actors / current page.
        self._dispatch_color_update_only(elec_ids)

        # Keep the selected electrodes selected without rebuilding the tree.
        self._select_electrodes_all_trees(elec_ids)

    def pick_color_for_electrode(self, elec_id: int) -> None:
        self.pick_color_for_electrodes([int(elec_id)])

    def _normalize_electrode_name(self, name: str) -> str:
        """
        Normalize electrode names for duplicate detection.

        Examples considered identical:
            A and a
            " A " and "A"
            "post  P" and "post P"
        """
        return " ".join(str(name).strip().split()).casefold()

    def _update_renamed_electrode_items_in_trees(self, elec_id: int) -> None:
        """
        Update only the renamed electrode and its contact rows in the three
        electrode trees, without rebuilding every other electrode item.
        """
        try:
            elec_id = int(elec_id)
            electrode = self.state.electrodes[elec_id]
        except Exception:
            return

        name = str(electrode.get("name", f"Elec{elec_id + 1}"))
        hemi = str(electrode.get("hemisphere", "?"))
        ref = str(electrode.get("ref") or "").strip()
        contacts_lps = electrode.get("contacts_lps", []) or []

        electrode_label = f"{name} - {hemi} - {ref}" if ref else f"{name} - {hemi}"

        for tree in self._trees:
            try:
                tree.blockSignals(True)

                for row in range(tree.topLevelItemCount()):
                    electrode_item = tree.topLevelItem(row)

                    if int(electrode_item.data(0, ROLE_ELEC_ID)) != elec_id:
                        continue

                    # Update only the electrode parent row.
                    electrode_item.setText(0, electrode_label)

                    # Update only the contact rows belonging to this electrode.
                    for child_index in range(electrode_item.childCount()):
                        child = electrode_item.child(child_index)

                        if child.data(0, ROLE_KIND) != "contact":
                            continue

                        contact_index = int(child.data(0, ROLE_CONTACT_INDEX))

                        if not (0 <= contact_index < len(contacts_lps)):
                            continue

                        lps = contacts_lps[contact_index]

                        try:
                            x, y, z = float(lps[0]), float(lps[1]), float(lps[2])
                            coord_txt = f"({x:.2f}, {y:.2f}, {z:.2f})"
                        except Exception:
                            coord_txt = str(lps)

                        child.setText(
                            0,
                            f"{name}{contact_index + 1} - {hemi} - {coord_txt}",
                        )

                    break

            except Exception:
                pass

            finally:
                try:
                    tree.blockSignals(False)
                except Exception:
                    pass

    def _refresh_visible_labels_after_rename(self, elec_id: int) -> None:
        """
        Refresh only displayed labels affected by the renamed electrode.

        Electrode geometry is unchanged, so no full reconstruction refresh
        is necessary.
        """
        try:
            elec_id = int(elec_id)
            electrode = self.state.electrodes[elec_id]
            n_contacts = len(electrode.get("contacts_lps", []) or [])
        except Exception:
            return

        # ---------------------------------------------------------
        # Reconstruction page:
        # update the name field only if the renamed electrode is selected.
        # No image redraw is needed because coordinates did not change.
        # ---------------------------------------------------------
        try:
            if (
                self.reco_page is not None
                and int(getattr(self.state, "selected_electrode_id", -1)) == elec_id
                and hasattr(self.reco_page, "load_electrode_parameters")
            ):
                self.reco_page.load_electrode_parameters(elec_id)
        except Exception:
            pass

        # ---------------------------------------------------------
        # 3D View:
        # redraw only this electrode if one of its labels is visible.
        # ---------------------------------------------------------
        try:
            view3d_page = getattr(self.state, "view3d_page", None)

            if view3d_page is not None:
                labels_visible = []

                if hasattr(view3d_page, "_get_local_contact_labels_visible"):
                    labels_visible = view3d_page._get_local_contact_labels_visible(
                        elec_id,
                        n_contacts,
                    )

                if any(labels_visible):
                    if hasattr(view3d_page, "_render_single_electrode"):
                        view3d_page._render_single_electrode(elec_id)

                    if hasattr(view3d_page, "_refresh_visible_slice_electrode_overlays"):
                        view3d_page._refresh_visible_slice_electrode_overlays()

                    if hasattr(view3d_page, "_render"):
                        view3d_page._render()

        except Exception:
            pass

        # ---------------------------------------------------------
        # Oblique Slice:
        # labels are drawn inside the displayed images, so only refresh
        # the slices if this electrode currently has visible labels.
        # ---------------------------------------------------------
        try:
            oblique_page = getattr(self.state, "oblique_page", None)

            if oblique_page is not None:
                labels_visible = []

                if hasattr(oblique_page, "_get_local_contact_labels_visible"):
                    labels_visible = oblique_page._get_local_contact_labels_visible(
                        elec_id,
                        n_contacts,
                    )

                if any(labels_visible):
                    if hasattr(oblique_page, "_schedule_refresh"):
                        oblique_page._schedule_refresh(
                            slices=True,
                            brain=False,
                        )

        except Exception:
            pass

    def rename_electrode(self, elec_id: int) -> None:
        """
        Rename one electrode and refresh all associated contact labels and
        displayed views without rebuilding unrelated electrode rows.
        """
        try:
            elec_id = int(elec_id)

            if not (0 <= elec_id < len(getattr(self.state, "electrodes", []) or [])):
                return

            electrode = self.state.electrodes[elec_id]
            current_name = str(electrode.get("name", f"Electrode {elec_id + 1}")).strip()

        except Exception:
            return

        parent = self.ui.window() if self.ui is not None else _top_level_window()
        text_to_display = current_name

        while True:
            new_name = NeuXelecTextInputDialog.get_text(
                parent,
                "Rename electrode",
                "Enter the new electrode name:",
                initial_text=text_to_display,
                accept_text="Rename",
                reject_text="Cancel",
            )

            if new_name is None:
                return

            new_name = str(new_name).strip()

            if not new_name:
                NeuXelecMessageDialog.warning(
                    parent,
                    "Missing electrode name",
                    "Please enter a name for the electrode.",
                )
                text_to_display = ""
                continue

            # No effective modification.
            if new_name == current_name:
                return

            normalized_new_name = self._normalize_electrode_name(new_name)

            duplicate_found = False

            for other_id, other_electrode in enumerate(self.state.electrodes):
                if int(other_id) == elec_id:
                    continue

                other_name = str(other_electrode.get("name", "")).strip()

                if other_name and self._normalize_electrode_name(other_name) == normalized_new_name:
                    duplicate_found = True
                    break

            if duplicate_found:
                keep_duplicate = NeuXelecMessageDialog.question(
                    parent,
                    "Electrode already created",
                    (
                        f'An electrode named "{new_name}" has already been created.\n\n'
                        "Do you want to continue with this name or change it?"
                    ),
                    accept_text="Continue",
                    reject_text="Change",
                )

                if not keep_duplicate:
                    # Reopen the rename window with an empty field.
                    text_to_display = ""
                    continue

            # Save the new electrode name.
            electrode["name"] = new_name

            # Update only this electrode and its contacts in the three lists.
            self._update_renamed_electrode_items_in_trees(elec_id)

            # Update only visible labels linked to this electrode.
            self._refresh_visible_labels_after_rename(elec_id)

            # Immediately write the renamed electrode into the project JSON,
            # when the project already has a saved JSON path.
            try:
                project_path = getattr(self.state, "project_path", None)

                if project_path:
                    from neuxelec.project_io import save_project_json

                    save_project_json(self.state, project_path)

            except Exception as e:
                print("[Rename electrode] Could not save project JSON:", e)

            return

    # ---------- Build helpers ----------
    def _add_electrode_node(self, tree: QTreeWidget, elec_id: int, elec: dict) -> None:
        name = elec.get("name", f"Elec{elec_id+1}")
        hemi = elec.get("hemisphere", "?")
        ref = (elec.get("ref") or "").strip()
        rgb = elec.get("color", (255, 165, 0))
        visible = True

        elec_label = f"{name} - {hemi} - {ref}" if ref else f"{name} - {hemi}"
        elec_item = QTreeWidgetItem([elec_label])
        elec_item.setData(0, ROLE_KIND, "electrode")
        elec_item.setData(0, ROLE_ELEC_ID, elec_id)
        elec_item.setData(0, ROLE_CONTACT_INDEX, -1)
        elec_item.setFlags(
            elec_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        )
        elec_item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
        elec_item.setData(0, ROLE_ROW_RGB, tuple(rgb))
        elec_item.setData(0, ROLE_ROW_ALPHA, 255)
        tree.addTopLevelItem(elec_item)

        n_contacts = elec.get("n_contacts")
        contact_dist = elec.get("contact_dist_mm")

        if contact_dist is None:
            contact_dist = elec.get("d_mm")

        if n_contacts is None:
            n_contacts = len(elec.get("contacts_lps", []) or [])

        if contact_dist is None:
            contact_dist = elec.get("ref_spacing_mm")

        spacing_profile = elec.get("spacing_profile_mm", []) or []

        if spacing_profile:
            contact_dist = elec.get("spacing_label") or "Variable"

        meta1 = QTreeWidgetItem(
            [f"Number of contacts: {n_contacts if n_contacts is not None else '-'}"]
        )
        meta2 = QTreeWidgetItem(
            [f"Inter-contact distance (mm): {contact_dist if contact_dist is not None else '-'}"]
        )
        for m in (meta1, meta2):
            m.setData(0, ROLE_KIND, "meta")
            m.setData(0, ROLE_ELEC_ID, elec_id)
            m.setData(0, ROLE_CONTACT_INDEX, -1)
            m.setFlags(Qt.ItemIsEnabled)
            m.setData(0, ROLE_ROW_RGB, None)
            m.setData(0, ROLE_ROW_ALPHA, 255)
            elec_item.addChild(m)

        contacts_lps = elec.get("contacts_lps", []) or []
        contacts_idx = elec.get("contacts_idx", []) or []
        contacts_visible = [True] * len(contacts_lps)

        for ci, lps in enumerate(contacts_lps):
            vox = contacts_idx[ci] if ci < len(contacts_idx) else None

            try:
                x, y, z = float(lps[0]), float(lps[1]), float(lps[2])
                coord_txt = f"({x:.2f}, {y:.2f}, {z:.2f})"
            except Exception:
                coord_txt = str(lps)

            label = f"{name}{ci+1} - {hemi} - {coord_txt}"
            contact_item = QTreeWidgetItem([label])
            contact_item.setData(0, ROLE_KIND, "contact")
            contact_item.setData(0, ROLE_ELEC_ID, elec_id)
            contact_item.setData(0, ROLE_CONTACT_INDEX, ci)
            contact_item.setFlags(
                contact_item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsSelectable
                | Qt.ItemIsEnabled
            )

            is_vis = bool(visible) and bool(contacts_visible[ci])
            contact_item.setCheckState(0, Qt.Checked if is_vis else Qt.Unchecked)
            contact_item.setData(0, ROLE_ROW_RGB, tuple(rgb))
            contact_item.setData(0, ROLE_ROW_ALPHA, 148)

            try:
                x, y, z = float(lps[0]), float(lps[1]), float(lps[2])
                lps_str = f"LPS(mm): ({x:.4f}, {y:.4f}, {z:.4f})"
            except Exception:
                lps_str = f"LPS(mm): {lps}"
            if vox is not None:
                try:
                    vx, vy, vz = int(vox[0]), int(vox[1]), int(vox[2])
                    vox_str = f"Voxel: ({vx}, {vy}, {vz})"
                except Exception:
                    vox_str = f"Voxel: {vox}"
            else:
                vox_str = "Voxel: (n/a)"
            contact_item.setToolTip(0, f"{lps_str}\n{vox_str}")

            elec_item.addChild(contact_item)

        if bool(elec.get("expanded", False)):
            elec_item.setExpanded(True)
        else:
            elec_item.setExpanded(False)

        self._update_parent_check_state_from_children(elec_item)

    def _get_expanded_state(self) -> dict[int, bool]:
        out = {elec_id: False for elec_id, _ in enumerate(self.state.electrodes)}

        if self._trees:
            t = self._trees[0]
            for i in range(t.topLevelItemCount()):
                item = t.topLevelItem(i)
                elec_id = item.data(0, ROLE_ELEC_ID)
                if isinstance(elec_id, int):
                    out[elec_id] = bool(item.isExpanded())

        for elec_id, is_expanded in out.items():
            if 0 <= elec_id < len(self.state.electrodes):
                self.state.electrodes[elec_id]["expanded"] = bool(is_expanded)

        return out

    def _apply_expanded_state(self, tree: QTreeWidget, expanded: dict[int, bool]) -> None:
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            elec_id = item.data(0, ROLE_ELEC_ID)
            if isinstance(elec_id, int):
                item.setExpanded(bool(expanded.get(elec_id, False)))

    # ---------- Signals ----------
    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_checks:
            return

        kind = item.data(0, ROLE_KIND)
        if kind not in ("electrode", "contact"):
            return

        batching = bool(getattr(self, "_check_drag_active", False))

        self._updating_checks = True

        try:
            raw_elec_id = item.data(0, ROLE_ELEC_ID)

            if raw_elec_id is None:
                return

            elec_id = int(raw_elec_id)

            if not (0 <= elec_id < len(getattr(self.state, "electrodes", []) or [])):
                return

            elec = self.state.electrodes[elec_id]

            if kind == "electrode":
                checked = item.checkState(0) == Qt.Checked

                # update children only in the current clicked tree UI
                for j in range(item.childCount()):
                    ch = item.child(j)
                    if ch.data(0, ROLE_KIND) == "contact":
                        ch.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

                self._dispatch_local_visibility_update(
                    "electrode",
                    elec_id,
                    checked=checked,
                    refresh=not batching,
                )

            elif kind == "contact":
                ci = int(item.data(0, ROLE_CONTACT_INDEX))
                checked = item.checkState(0) == Qt.Checked

                self._dispatch_local_visibility_update(
                    "contact",
                    elec_id,
                    contact_idx=ci,
                    checked=checked,
                    refresh=not batching,
                )

                # update only the parent state in this tree UI
                parent = item.parent()
                if parent is not None:
                    self._update_parent_check_state_from_children(parent)

            if batching:
                self._check_drag_changed = True
            else:
                # In 3D View, the immediate update is already done inside:
                # - vp.set_electrode_visible(...)
                # - vp.set_contact_visible(...)
                # So do not call _refresh_for_current_page_only(), because it can
                # trigger a global electrode refresh.
                if self._get_current_page_name() != "page3DView":
                    self._refresh_for_current_page_only()

        finally:
            self._updating_checks = False

    def _on_selection_changed(self) -> None:
        selected = None
        for t in self._trees:
            items = t.selectedItems()
            if items:
                selected = items[0]
                break
        if selected is None:
            return

        kind = selected.data(0, ROLE_KIND)
        if kind == "electrode":
            elec_id = int(selected.data(0, ROLE_ELEC_ID))
            self.state.selected_electrode_id = elec_id
            self.state.selected_contact_index = None

            if self.reco_page is not None and hasattr(self.reco_page, "load_electrode_parameters"):
                self.reco_page.load_electrode_parameters(elec_id)

        elif kind == "contact":
            elec_id = int(selected.data(0, ROLE_ELEC_ID))
            ci = int(selected.data(0, ROLE_CONTACT_INDEX))
            self.state.selected_electrode_id = elec_id
            self.state.selected_contact_index = ci

            if self.reco_page is not None:
                if hasattr(self.reco_page, "load_electrode_parameters"):
                    self.reco_page.load_electrode_parameters(elec_id)
                if hasattr(self.reco_page, "jump_to_contact"):
                    self.reco_page.jump_to_contact(elec_id, ci)

    def _get_current_page_name(self) -> str:
        try:
            sw = self.ui.findChild(QObject, "stackedWidget")
            if sw is None:
                return ""
            w = sw.currentWidget()
            if w is None:
                return ""
            return str(w.objectName() or "")
        except Exception:
            return ""

    def _confirm_delete_electrodes(self, elec_ids: list[int]) -> bool:
        """
        Show one styled warning listing every electrode that will be deleted.
        """
        try:
            valid_ids = sorted(
                {
                    int(eid)
                    for eid in elec_ids
                    if 0 <= int(eid) < len(getattr(self.state, "electrodes", []) or [])
                }
            )

            if not valid_ids:
                return False

            names = []

            for elec_id in valid_ids:
                elec = self.state.electrodes[elec_id]
                names.append(str(elec.get("name", f"Electrode {elec_id + 1}")))

            n = len(names)
            title = "Delete electrode" if n == 1 else "Delete electrodes"

            if n == 1:
                question = "Are you sure you want to delete this electrode?"
            else:
                question = f"Are you sure you want to delete these {n} electrodes?"

            items_text = "\n".join(f"• {name}" for name in names)

            return NeuXelecMessageDialog.question(
                self.ui.window(),
                title,
                (f"{question}\n\n" f"{items_text}\n\n" "This action cannot be undone."),
                accept_text="Delete",
                reject_text="Cancel",
            )

        except Exception:
            return False

    def _confirm_delete_contacts(self, contacts: list[tuple[int, int]]) -> bool:
        """
        Show one styled warning listing every contact that will be deleted.
        """
        try:
            valid_contacts = []

            for elec_id, contact_idx in contacts:
                elec_id = int(elec_id)
                contact_idx = int(contact_idx)

                if not (0 <= elec_id < len(getattr(self.state, "electrodes", []) or [])):
                    continue

                elec = self.state.electrodes[elec_id]
                contacts_lps = elec.get("contacts_lps", []) or []

                if 0 <= contact_idx < len(contacts_lps):
                    valid_contacts.append((elec_id, contact_idx))

            valid_contacts = sorted(set(valid_contacts))

            if not valid_contacts:
                return False

            names = [
                self._contact_display_name(elec_id, contact_idx)
                for elec_id, contact_idx in valid_contacts
            ]

            n = len(names)
            title = "Delete contact" if n == 1 else "Delete contacts"

            if n == 1:
                question = "Are you sure you want to delete this contact?"
            else:
                question = f"Are you sure you want to delete these {n} contacts?"

            items_text = "\n".join(f"• {name}" for name in names)

            return NeuXelecMessageDialog.question(
                self.ui.window(),
                title,
                (f"{question}\n\n" f"{items_text}\n\n" "This action cannot be undone."),
                accept_text="Delete",
                reject_text="Cancel",
            )

        except Exception:
            return False

    def _tree_from_viewport(self, viewport):
        for t in self._trees:
            try:
                if t.viewport() is viewport:
                    return t
            except Exception:
                pass
        return None

    def _electrode_item_from_any_item(self, item):
        if item is None:
            return None

        try:
            kind = item.data(0, ROLE_KIND)
            if kind == "electrode":
                return item

            parent = item.parent()
            if parent is not None and parent.data(0, ROLE_KIND) == "electrode":
                return parent
        except Exception:
            pass

        return None

    def _selected_electrode_ids_in_tree(self, tree: QTreeWidget) -> list[int]:
        ids = []

        try:
            for item in tree.selectedItems():
                elec_item = self._electrode_item_from_any_item(item)
                if elec_item is None:
                    continue

                elec_id = int(elec_item.data(0, ROLE_ELEC_ID))
                if elec_id not in ids:
                    ids.append(elec_id)
        except Exception:
            pass

        ids.sort()
        return ids

    def _selected_items_of_kind(self, tree: QTreeWidget, kind: str) -> list:
        """
        Return selected items of one exact kind from the active tree.

        Important:
        - selecting contacts must not imply deleting their parent electrodes;
        - selecting electrodes must not imply deleting their contacts individually.
        """
        out = []

        try:
            for item in tree.selectedItems():
                if item.data(0, ROLE_KIND) == str(kind):
                    out.append(item)
        except Exception:
            pass

        return out

    def _prepare_context_selection(self, tree: QTreeWidget, clicked_item, kind: str) -> list:
        """
        Preserve a multi-selection only when the right-clicked row is already part
        of the selection and has the same kind as the requested action.

        Example:
            selected contacts + right-click on one selected contact
            -> keep all selected contacts

            selected contacts + right-click on another unselected contact
            -> act only on the newly clicked contact
        """
        selected_items = self._selected_items_of_kind(tree, kind)

        clicked_is_selected = any(it is clicked_item for it in selected_items)

        if clicked_is_selected and selected_items:
            try:
                tree.setCurrentItem(clicked_item, 0, QItemSelectionModel.NoUpdate)
            except Exception:
                pass

            return selected_items

        try:
            tree.clearSelection()
            tree.setCurrentItem(clicked_item, 0, QItemSelectionModel.ClearAndSelect)
        except Exception:
            try:
                clicked_item.setSelected(True)
            except Exception:
                pass

        return [clicked_item]

    def _contact_display_name(self, elec_id: int, contact_idx: int) -> str:
        """
        Return the displayed name of one contact.
        """
        try:
            elec = self.state.electrodes[int(elec_id)]
            elec_name = str(elec.get("name", f"Electrode {int(elec_id) + 1}"))

            contact_names = elec.get("contact_names", None)
            if isinstance(contact_names, (list, tuple)):
                if 0 <= int(contact_idx) < len(contact_names):
                    custom_name = contact_names[int(contact_idx)]
                    if custom_name:
                        return str(custom_name)

            return f"{elec_name}{int(contact_idx) + 1}"

        except Exception:
            return f"Contact {int(contact_idx) + 1}"

    def _hide_tree_drag_count_badge(self) -> None:
        """
        Hide the small badge showing how many electrodes are being dragged.
        """
        try:
            if self._tree_drag_count_badge is not None:
                self._tree_drag_count_badge.hide()
        except Exception:
            pass

    def _ensure_tree_drag_count_badge(self, tree: QTreeWidget):
        """
        Create one floating badge inside the tree viewport.
        """
        try:
            if tree is None:
                return None

            if (
                self._tree_drag_count_badge is None
                or self._tree_drag_count_badge.parent() is not tree.viewport()
            ):
                self._tree_drag_count_badge = QLabel(tree.viewport())
                self._tree_drag_count_badge.setObjectName("ElectrodeDragCountBadge")
                self._tree_drag_count_badge.setAlignment(Qt.AlignCenter)
                self._tree_drag_count_badge.setStyleSheet("""
                    QLabel#ElectrodeDragCountBadge {
                        color: white;
                        background-color: rgba(0, 0, 0, 210);
                        border: 1px solid rgba(255, 255, 255, 210);
                        border-radius: 9px;
                        padding: 2px 7px;
                        font-weight: bold;
                        font-size: 11px;
                    }
                """)
                self._tree_drag_count_badge.hide()

            return self._tree_drag_count_badge

        except Exception:
            return None

    def _update_tree_drag_count_badge(self, tree: QTreeWidget, pos, count: int) -> None:
        """
        Update badge position next to the mouse while dragging.
        """
        try:
            count = int(count)

            if tree is None or count <= 0:
                self._hide_tree_drag_count_badge()
                return

            badge = self._ensure_tree_drag_count_badge(tree)
            if badge is None:
                return

            # Just the number is visually cleaner.
            badge.setText(str(count))
            badge.adjustSize()

            margin = 8
            x = int(pos.x()) + 14
            y = int(pos.y()) + 12

            max_x = max(0, int(tree.viewport().width()) - int(badge.width()) - margin)
            max_y = max(0, int(tree.viewport().height()) - int(badge.height()) - margin)

            x = max(margin, min(x, max_x))
            y = max(margin, min(y, max_y))

            badge.move(x, y)
            badge.show()
            badge.raise_()

        except Exception:
            self._hide_tree_drag_count_badge()

    def _hide_tree_drop_indicator(self) -> None:
        try:
            if self._tree_drop_indicator is not None:
                self._tree_drop_indicator.hide()
        except Exception:
            pass

        self._tree_drop_indicator_tree = None
        self._tree_drop_target_id = None
        self._tree_drop_after = False

        self._hide_tree_drag_count_badge()

    def _ensure_tree_drop_indicator(self, tree: QTreeWidget):
        """
        Create one floating black line inside the tree viewport.
        """
        try:
            if tree is None:
                return None

            if (
                self._tree_drop_indicator is None
                or self._tree_drop_indicator.parent() is not tree.viewport()
            ):
                self._tree_drop_indicator = QFrame(tree.viewport())
                self._tree_drop_indicator.setObjectName("ElectrodeDropIndicator")
                self._tree_drop_indicator.setFixedHeight(3)
                self._tree_drop_indicator.setStyleSheet("""
                    QFrame#ElectrodeDropIndicator {
                        background-color: black;
                        border: none;
                        border-radius: 1px;
                    }
                """)
                self._tree_drop_indicator.hide()

            return self._tree_drop_indicator

        except Exception:
            return None

    def _drop_target_from_pos(self, tree: QTreeWidget, pos, source_ids: list[int] | None = None):
        """
        Return (target_item, target_id, drop_after).

        The returned target is always an electrode parent item.
        The visual insertion position is:
        - before target_item if drop_after is False
        - after target_item if drop_after is True
        """
        if tree is None:
            return None, None, False

        source_ids = set(int(i) for i in (source_ids or []))

        try:
            n = int(tree.topLevelItemCount())
        except Exception:
            n = 0

        if n <= 0:
            return None, None, False

        try:
            item = tree.itemAt(pos)
            elec_item = self._electrode_item_from_any_item(item)

            # If mouse is not exactly on an item, use first/last fallback.
            if elec_item is None:
                first = tree.topLevelItem(0)
                last = tree.topLevelItem(n - 1)

                first_rect = tree.visualItemRect(first)
                last_rect = tree.visualItemRect(last)

                if pos.y() <= first_rect.center().y():
                    elec_item = first
                    drop_after = False
                else:
                    elec_item = last
                    drop_after = True

            else:
                rect = tree.visualItemRect(elec_item)
                drop_after = bool(pos.y() > rect.center().y())

            target_id = int(elec_item.data(0, ROLE_ELEC_ID))

            # Do not show a drop target inside the selected dragged block.
            if target_id in source_ids:
                return None, None, False

            return elec_item, target_id, bool(drop_after)

        except Exception:
            return None, None, False

    def _update_tree_drop_indicator(self, tree: QTreeWidget, pos) -> None:
        """
        Update the black insertion line while dragging electrodes.
        """
        try:
            source_ids = list(self._tree_drag_source_ids or [])

            target_item, target_id, drop_after = self._drop_target_from_pos(
                tree,
                pos,
                source_ids=source_ids,
            )

            if target_item is None or target_id is None:
                self._hide_tree_drop_indicator()
                return

            rect = tree.visualItemRect(target_item)
            if not rect.isValid():
                self._hide_tree_drop_indicator()
                return

            line = self._ensure_tree_drop_indicator(tree)
            if line is None:
                return

            viewport_w = max(20, int(tree.viewport().width()))

            # Position the line exactly between rows.
            if drop_after:
                y = int(rect.bottom()) + 1
            else:
                y = max(0, int(rect.top()) - 1)

            line.setGeometry(4, y, viewport_w - 8, 3)
            line.show()
            line.raise_()

            self._tree_drop_indicator_tree = tree
            self._tree_drop_target_id = int(target_id)
            self._tree_drop_after = bool(drop_after)

        except Exception:
            self._hide_tree_drop_indicator()

    def _select_electrodes_all_trees(self, elec_ids: list[int]) -> None:
        elec_ids = sorted({int(i) for i in elec_ids if i is not None})

        if elec_ids:
            self.state.selected_electrode_id = int(elec_ids[0])
            self.state.selected_contact_index = None

        for t in self._trees:
            try:
                t.blockSignals(True)
                t.clearSelection()

                first_item = None

                for i in range(t.topLevelItemCount()):
                    item = t.topLevelItem(i)
                    raw_eid = item.data(0, ROLE_ELEC_ID)

                    if raw_eid is None:
                        continue

                    eid = int(raw_eid)

                    if eid in elec_ids:
                        item.setSelected(True)
                        if first_item is None:
                            first_item = item

                if first_item is not None:
                    t.setCurrentItem(first_item)
                    try:
                        t.scrollToItem(first_item)
                    except Exception:
                        pass

            finally:
                try:
                    t.blockSignals(False)
                except Exception:
                    pass

    def _is_pos_on_expand_arrow(
        self,
        tree: QTreeWidget,
        item: QTreeWidgetItem,
        pos,
    ) -> bool:
        """
        Return True when the user clicks in the expand/collapse arrow area
        of an electrode row.

        Important:
        this click must be left entirely to QTreeWidget, otherwise the
        custom electrode drag logic intercepts it when the row is selected.
        """
        if tree is None or item is None:
            return False

        try:
            if item.data(0, ROLE_KIND) != "electrode":
                return False

            if item.childCount() <= 0:
                return False

            rect = tree.visualItemRect(item)

            if not rect.isValid():
                return False

            if pos.y() < rect.top() or pos.y() > rect.bottom():
                return False

            # The expand/collapse SVG arrow is drawn in the indentation area,
            # before the coloured electrode row and before the checkbox.
            branch_zone_right = max(
                int(tree.indentation()) + 10,
                int(rect.left()),
            )

            return 0 <= int(pos.x()) < branch_zone_right

        except Exception:
            return False

    def _is_pos_on_checkbox(self, tree: QTreeWidget, item: QTreeWidgetItem, pos) -> bool:
        """
        Approximate hit-test for the checkbox area of an item.

        This is intentionally simple and robust across the 3 electrode trees:
        if the click is in the left part of the item row, where the checkbox is drawn,
        we treat it as a checkbox drag.
        """
        if tree is None or item is None:
            return False

        try:
            rect = tree.visualItemRect(item)
            if not rect.isValid():
                return False

            # Only accept clicks inside the row vertically
            if pos.y() < rect.top() or pos.y() > rect.bottom():
                return False

            # Checkbox area is near the left side of the item's visual rect.
            # 34 px is enough for the checkbox + small padding, without catching text clicks.
            x0 = rect.left()
            x1 = rect.left() + 34

            return x0 <= pos.x() <= x1

        except Exception:
            return False

    def _set_item_check_state_from_drag(self, tree: QTreeWidget, item: QTreeWidgetItem) -> None:
        """
        Apply the current drag check-state to an electrode/contact item.
        Reuses _on_item_changed through item.setCheckState().
        """
        if tree is None or item is None:
            return

        if self._check_drag_state is None:
            return

        try:
            kind = item.data(0, ROLE_KIND)
            if kind not in ("electrode", "contact"):
                return

            elec_id = int(item.data(0, ROLE_ELEC_ID))
            contact_idx = int(item.data(0, ROLE_CONTACT_INDEX))

            key = (kind, elec_id, contact_idx)
            if key in self._check_drag_seen:
                return

            self._check_drag_seen.add(key)

            desired_state = Qt.Checked if bool(self._check_drag_state) else Qt.Unchecked

            if item.checkState(0) != desired_state:
                item.setCheckState(0, desired_state)

        except Exception:
            pass

    def _reset_check_drag_state(self) -> None:
        self._check_drag_active = False
        self._check_drag_tree = None
        self._check_drag_state = None
        self._check_drag_seen = set()
        self._check_drag_changed = False

    def _finish_check_drag(self) -> None:
        changed = bool(getattr(self, "_check_drag_changed", False))
        current_page = self._get_current_page_name()

        # Important: keep the touched items before reset,
        # because _reset_check_drag_state() clears _check_drag_seen.
        touched = list(getattr(self, "_check_drag_seen", set()) or [])

        self._reset_check_drag_state()

        if not changed:
            if current_page == "pageObliqueSlices":
                op = getattr(self.state, "oblique_page", None)
                if op is not None and hasattr(op, "end_electrode_visibility_freeze"):
                    try:
                        op.end_electrode_visibility_freeze(refresh=False)
                    except Exception:
                        pass
            return

        if current_page == "pageObliqueSlices":
            op = getattr(self.state, "oblique_page", None)
            if op is not None and hasattr(op, "end_electrode_visibility_freeze"):
                try:
                    op.end_electrode_visibility_freeze(refresh=True)
                except Exception:
                    pass
            return

        if current_page == "page3DView":
            vp = getattr(self.state, "view3d_page", None)
            if vp is None:
                return

            # During checkbox drag, visibility was updated silently in:
            # vp._page_electrode_visible / vp._page_contacts_visible.
            # Now apply the change only to the touched electrodes.
            touched_elec_ids = set()

            for key in touched:
                try:
                    _kind, elec_id, _contact_idx = key
                    touched_elec_ids.add(int(elec_id))
                except Exception:
                    pass

            for elec_id in touched_elec_ids:
                try:
                    elec = self.state.electrodes[int(elec_id)]
                    n = len(elec.get("contacts_lps", []) or [])

                    elec_visible = bool(
                        getattr(vp, "_page_electrode_visible", {}).get(int(elec_id), True)
                    )

                    contacts_visible = None
                    try:
                        contacts_visible = vp._get_local_contacts_visible(int(elec_id), n)
                    except Exception:
                        contacts_visible = [True] * n

                    # If the electrode is completely hidden, hide its existing actors.
                    if not elec_visible or not any(contacts_visible):
                        if hasattr(vp, "_set_single_electrode_actor_visibility"):
                            vp._set_single_electrode_actor_visibility(int(elec_id), False)
                        else:
                            if hasattr(vp, "_render_single_electrode"):
                                vp._render_single_electrode(int(elec_id))

                    # If visible, rebuild only this electrode to account for contact-level visibility.
                    else:
                        if hasattr(vp, "_render_single_electrode"):
                            vp._render_single_electrode(int(elec_id))

                except Exception:
                    pass

            try:
                if hasattr(vp, "_refresh_visible_slice_electrode_overlays"):
                    vp._refresh_visible_slice_electrode_overlays()
            except Exception:
                pass

            try:
                if hasattr(vp, "_render"):
                    vp._render()
            except Exception:
                pass

            return

        self._refresh_for_current_page_only()

    def eventFilter(self, obj, event):
        tree = self._tree_from_viewport(obj)
        if tree is None:
            return False

        try:
            # ---------------------------------------------------------
            # 1) Checkbox drag mode:
            #    click on checkbox + hold left button + move over other checkboxes
            #    => apply the same checked/unchecked state.
            # ---------------------------------------------------------
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                item = tree.itemAt(pos)

                # Never intercept a click on the expand/collapse arrow.
                # Qt must receive this event to open or close the contacts,
                # including when the electrode row is already selected.
                if item is not None and self._is_pos_on_expand_arrow(tree, item, pos):
                    return False

                if item is not None and self._is_pos_on_checkbox(tree, item, pos):
                    kind = item.data(0, ROLE_KIND)

                    if kind in ("electrode", "contact"):
                        if self._get_current_page_name() == "pageObliqueSlices":
                            op = getattr(self.state, "oblique_page", None)
                            if op is not None and hasattr(op, "begin_electrode_visibility_freeze"):
                                try:
                                    op.begin_electrode_visibility_freeze()
                                except Exception:
                                    pass
                        # If the first checkbox is checked, dragging will uncheck.
                        # If it is unchecked, dragging will check.
                        self._check_drag_active = True
                        self._check_drag_tree = tree
                        self._check_drag_state = item.checkState(0) != Qt.Checked
                        self._check_drag_seen = set()

                        # Disable reorder drag while checkbox-dragging
                        self._tree_drag_tree = None
                        self._tree_drag_start_pos = None
                        self._tree_drag_source_ids = []
                        self._tree_drag_started = False

                        self._set_item_check_state_from_drag(tree, item)
                        return True

            if event.type() == QEvent.MouseMove:
                if self._check_drag_active and self._check_drag_tree is tree:
                    try:
                        buttons = event.buttons()
                    except Exception:
                        buttons = Qt.NoButton

                    if not (buttons & Qt.LeftButton):
                        self._hide_tree_drop_indicator()
                        self._set_tree_drag_visual_mode(tree, False)
                        return False

                    pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                    item = tree.itemAt(pos)

                    if item is not None and self._is_pos_on_checkbox(tree, item, pos):
                        self._set_item_check_state_from_drag(tree, item)

                    return True

            if event.type() == QEvent.MouseButtonRelease:
                if self._check_drag_active and self._check_drag_tree is tree:
                    self._finish_check_drag()
                    return True

            # ---------------------------------------------------------
            # 2) Normal electrode drag-reorder mode.
            # ---------------------------------------------------------
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                item = tree.itemAt(pos)
                elec_item = self._electrode_item_from_any_item(item)

                if elec_item is None:
                    self._tree_drag_tree = None
                    self._tree_drag_start_pos = None
                    self._tree_drag_source_ids = []
                    self._tree_drag_started = False
                    return False

                clicked_elec_id = int(elec_item.data(0, ROLE_ELEC_ID))
                selected_ids = self._selected_electrode_ids_in_tree(tree)

                try:
                    modifiers = event.modifiers()
                except Exception:
                    modifiers = Qt.NoModifier

                self._tree_drag_tree = tree
                self._tree_drag_start_pos = pos
                self._tree_drag_started = False

                # If user clicks on an already selected electrode without Ctrl/Shift,
                # preserve the full selection for drag.
                if (
                    clicked_elec_id in selected_ids
                    and not (modifiers & Qt.ControlModifier)
                    and not (modifiers & Qt.ShiftModifier)
                ):
                    self._tree_drag_source_ids = list(selected_ids)

                    try:
                        tree.setCurrentItem(elec_item, 0, QItemSelectionModel.NoUpdate)
                    except Exception:
                        pass

                    return True

                # Otherwise let Qt handle normal selection / Ctrl / Shift.
                self._tree_drag_source_ids = []
                return False

            if event.type() == QEvent.MouseMove:
                if self._tree_drag_tree is not tree or self._tree_drag_start_pos is None:
                    return False

                try:
                    buttons = event.buttons()
                except Exception:
                    buttons = Qt.NoButton

                if not (buttons & Qt.LeftButton):
                    self._hide_tree_drop_indicator()
                    self._set_tree_drag_visual_mode(tree, False)
                    return False

                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                dist = (pos - self._tree_drag_start_pos).manhattanLength()

                if dist >= QApplication.startDragDistance():
                    self._tree_drag_started = True

                    if not self._tree_drag_source_ids:
                        ids = self._selected_electrode_ids_in_tree(tree)

                        if not ids:
                            item = tree.itemAt(self._tree_drag_start_pos)
                            elec_item = self._electrode_item_from_any_item(item)
                            if elec_item is not None:
                                ids = [int(elec_item.data(0, ROLE_ELEC_ID))]

                        self._tree_drag_source_ids = list(ids)

                    # New behavior:
                    # show a black insertion line between electrodes instead of relying
                    # on the hovered/selected row background.
                    self._set_tree_drag_visual_mode(tree, True)
                    self._update_tree_drop_indicator(tree, pos)
                    self._update_tree_drag_count_badge(
                        tree,
                        pos,
                        count=len(self._tree_drag_source_ids or []),
                    )
                    return True

            if event.type() == QEvent.MouseButtonRelease:
                if self._tree_drag_tree is not tree:
                    return False

                was_dragging = bool(self._tree_drag_started)
                source_ids = list(self._tree_drag_source_ids or [])

                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()

                # Prefer the live drop indicator target, because it is what the user saw.
                target_id = self._tree_drop_target_id
                drop_after = bool(self._tree_drop_after)

                # Fallback if the indicator was not available.
                if target_id is None:
                    _target_item, target_id, drop_after = self._drop_target_from_pos(
                        tree,
                        pos,
                        source_ids=source_ids,
                    )

                self._hide_tree_drop_indicator()
                self._set_tree_drag_visual_mode(tree, False)

                self._tree_drag_tree = None
                self._tree_drag_start_pos = None
                self._tree_drag_source_ids = []
                self._tree_drag_started = False

                if not was_dragging or not source_ids:
                    return False

                if target_id is None:
                    try:
                        self._select_electrodes_all_trees(source_ids)
                    except Exception:
                        pass
                    return True

                self._move_electrodes_to_target(
                    source_ids,
                    int(target_id),
                    drop_after=bool(drop_after),
                )
                return True

        except Exception:
            self._reset_check_drag_state()
            self._hide_tree_drop_indicator()
            try:
                if self._tree_drag_tree is not None:
                    self._set_tree_drag_visual_mode(self._tree_drag_tree, False)
            except Exception:
                pass
            self._tree_drag_tree = None
            self._tree_drag_start_pos = None
            self._tree_drag_source_ids = []
            self._tree_drag_started = False
            return False

        return False

    def _move_electrodes_to_target(
        self, elec_ids: list[int], target_id: int, drop_after: bool = False
    ) -> None:
        try:
            old_electrodes = list(getattr(self.state, "electrodes", []) or [])
            n = len(old_electrodes)
            if n <= 1:
                return

            selected_ids = sorted({int(i) for i in elec_ids if 0 <= int(i) < n})
            target_id = int(target_id)

            if not selected_ids or target_id < 0 or target_id >= n:
                return

            # Dropping onto the selected block itself: nothing to do.
            if target_id in selected_ids:
                return

            # Original insertion boundary.
            target_boundary = target_id + (1 if drop_after else 0)

            moving = [old_electrodes[i] for i in selected_ids]
            remaining = [e for i, e in enumerate(old_electrodes) if i not in selected_ids]

            # Convert original target boundary to index in remaining list.
            insert_at = 0
            for old_idx, _elec in enumerate(old_electrodes):
                if old_idx >= target_boundary:
                    break
                if old_idx not in selected_ids:
                    insert_at += 1

            insert_at = max(0, min(insert_at, len(remaining)))

            new_electrodes = remaining[:insert_at] + moving + remaining[insert_at:]

            if new_electrodes == old_electrodes:
                return

            self.state.electrodes[:] = new_electrodes

            # Remap page-local states keyed by old elec_id.
            self._remap_page_local_state_after_reorder(old_electrodes, new_electrodes)

            # Restore selected ids based on object identity after reorder.
            selected_obj_ids = {id(e) for e in moving}
            new_selected_ids = [
                i for i, e in enumerate(new_electrodes) if id(e) in selected_obj_ids
            ]

            if new_selected_ids:
                self.state.selected_electrode_id = int(new_selected_ids[0])
                self.state.selected_contact_index = None

            self._dispatch_global_structural_update()
            self._select_electrodes_all_trees(new_selected_ids)

        except Exception:
            pass

    def _is_edit_mode(self) -> bool:
        """
        Return True only when the project is opened in Edit mode.
        """
        mode = str(getattr(self.state, "app_mode", "edit") or "edit").lower().strip()

        return mode == "edit"

    def _open_context_menu(self, pos: QPoint) -> None:
        tree = self.sender()
        if tree is None or not isinstance(tree, QTreeWidget):
            return

        item = tree.itemAt(pos)
        if item is None:
            return

        kind = item.data(0, ROLE_KIND)
        if kind not in ("electrode", "contact"):
            return

        # Keep the multi-selection only when the clicked row belongs to that same
        # selection and is of the same type: electrode or contact.
        selected_action_items = self._prepare_context_selection(tree, item, kind)

        raw_elec_id = item.data(0, ROLE_ELEC_ID)
        raw_contact_idx = item.data(0, ROLE_CONTACT_INDEX)

        if raw_elec_id is None:
            return

        try:
            elec_id = int(raw_elec_id)
        except Exception:
            return

        if not (0 <= elec_id < len(getattr(self.state, "electrodes", []) or [])):
            return

        try:
            contact_idx = int(raw_contact_idx) if raw_contact_idx is not None else -1
        except Exception:
            contact_idx = -1

        selected_elec_ids = []
        selected_contacts = []

        if kind == "electrode":
            try:
                selected_elec_ids = sorted(
                    {
                        int(it.data(0, ROLE_ELEC_ID))
                        for it in selected_action_items
                        if it.data(0, ROLE_KIND) == "electrode"
                    }
                )
            except Exception:
                selected_elec_ids = [elec_id]

            if not selected_elec_ids:
                selected_elec_ids = [elec_id]

        elif kind == "contact":
            try:
                selected_contacts = sorted(
                    {
                        (
                            int(it.data(0, ROLE_ELEC_ID)),
                            int(it.data(0, ROLE_CONTACT_INDEX)),
                        )
                        for it in selected_action_items
                        if it.data(0, ROLE_KIND) == "contact"
                    }
                )
            except Exception:
                selected_contacts = [(elec_id, contact_idx)]

            if not selected_contacts:
                selected_contacts = [(elec_id, contact_idx)]

        global_pos = tree.viewport().mapToGlobal(pos)
        current_page = self._get_current_page_name()

        # Edit actions must be hidden and blocked in View Only mode.
        editable = self._is_edit_mode()

        if kind == "electrode":
            elec = self.state.electrodes[elec_id]

            contacts_lps = elec.get("contacts_lps", []) or []

            labels_on = False
            if current_page == "pageObliqueSlices":
                op = getattr(self.state, "oblique_page", None)
                if op is not None and hasattr(op, "_get_local_contact_labels_visible"):
                    vals = op._get_local_contact_labels_visible(elec_id, len(contacts_lps))
                    labels_on = any(vals)

            elif current_page == "page3DView":
                vp = getattr(self.state, "view3d_page", None)
                if vp is not None and hasattr(vp, "_get_local_contact_labels_visible"):
                    vals = vp._get_local_contact_labels_visible(elec_id, len(contacts_lps))
                    labels_on = any(vals)

            vp = getattr(self.state, "view3d_page", None)
            projection_on = False
            if vp is not None and hasattr(vp, "has_surface_projection"):
                try:
                    projection_on = bool(vp.has_surface_projection(elec_id))
                except Exception:
                    projection_on = False

            choice = exec_electrode_tree_menu(
                global_pos,
                kind="electrode",
                current_page=current_page,
                labels_on=labels_on,
                projection_on=projection_on,
                selection_count=len(selected_elec_ids),
                editable=editable,
            )
            self._ignore_next_tree_click = True

            if choice == "toggle_labels":
                self._dispatch_local_visual_update("toggle_labels", elec_id)
                return

            if choice == "toggle_projection":
                vp = getattr(self.state, "view3d_page", None)
                if vp is not None:
                    try:
                        if hasattr(vp, "has_surface_projection") and vp.has_surface_projection(
                            elec_id
                        ):
                            if hasattr(vp, "remove_surface_projection"):
                                vp.remove_surface_projection(elec_id)
                        else:
                            if hasattr(vp, "project_electrode_on_surface"):
                                vp.project_electrode_on_surface(elec_id)
                    except Exception:
                        pass
                return

            if choice == "rename_electrode":
                if not editable:
                    return

                self.rename_electrode(elec_id)
                return

            if choice == "color_electrode":
                self.pick_color_for_electrodes(selected_elec_ids or [elec_id])
                return
            if choice == "delete_electrode":
                if not editable:
                    return

                if self.reco_page is None:
                    print("Context menu error: reco_page is None")
                    return

                if not hasattr(self.reco_page, "delete_electrodes"):
                    print("Context menu error: " "ReconstructionPage has no delete_electrodes")
                    return

                elec_ids_to_delete = list(selected_elec_ids or [elec_id])

                if not self._confirm_delete_electrodes(elec_ids_to_delete):
                    return

                self.reco_page.delete_electrodes(elec_ids_to_delete)
                self._dispatch_global_structural_update()
                return

        if kind == "contact":
            elec = self.state.electrodes[elec_id]

            contacts_lps = elec.get("contacts_lps", []) or []
            label_on = False

            if current_page == "pageObliqueSlices":
                op = getattr(self.state, "oblique_page", None)
                if op is not None and hasattr(op, "_get_local_contact_labels_visible"):
                    vals = op._get_local_contact_labels_visible(elec_id, len(contacts_lps))
                    if 0 <= contact_idx < len(vals):
                        label_on = bool(vals[contact_idx])

            elif current_page == "page3DView":
                vp = getattr(self.state, "view3d_page", None)
                if vp is not None and hasattr(vp, "_get_local_contact_labels_visible"):
                    vals = vp._get_local_contact_labels_visible(elec_id, len(contacts_lps))
                    if 0 <= contact_idx < len(vals):
                        label_on = bool(vals[contact_idx])

            choice = exec_electrode_tree_menu(
                global_pos,
                kind="contact",
                current_page=current_page,
                label_on=label_on,
                selection_count=len(selected_contacts),
                editable=editable,
            )
            self._ignore_next_tree_click = True

            if choice == "toggle_label":
                self._dispatch_local_visual_update("toggle_label", elec_id, contact_idx)
                return

            if choice in ("show_coronal_slice", "show_axial_slice", "show_sagittal_slice"):
                vp = getattr(self.state, "view3d_page", None)
                if vp is not None and hasattr(vp, "show_contact_in_slice"):
                    plane_name = {
                        "show_coronal_slice": "coronal",
                        "show_axial_slice": "axial",
                        "show_sagittal_slice": "sagittal",
                    }[choice]
                    try:
                        vp.show_contact_in_slice(elec_id, contact_idx, plane_name)
                    except Exception:
                        pass
                return

            if self.reco_page is None:
                print("Context menu error: reco_page is None")
                return

            if choice == "edit_contact":
                if not editable:
                    return

                if not hasattr(
                    self.reco_page,
                    "open_edit_contact_dialog",
                ):
                    print(
                        "Context menu error: "
                        "ReconstructionPage has no "
                        "open_edit_contact_dialog"
                    )
                    return

                self.reco_page.open_edit_contact_dialog(
                    elec_id,
                    contact_idx,
                )
                self.refresh_all()
                return

            if choice == "delete_contact":
                if not editable:
                    return

                if not hasattr(self.reco_page, "delete_contacts"):
                    print("Context menu error: " "ReconstructionPage has no delete_contacts")
                    return

                contacts_to_delete = list(selected_contacts or [(elec_id, contact_idx)])

                if not self._confirm_delete_contacts(contacts_to_delete):
                    return

                self.reco_page.delete_contacts(contacts_to_delete)
                self._dispatch_global_structural_update()
                return

    def _select_electrode_all_trees(self, elec_id: int) -> None:
        try:
            self.state.selected_electrode_id = int(elec_id)
        except Exception:
            pass

        for t in self._trees:
            try:
                t.clearSelection()
                for i in range(t.topLevelItemCount()):
                    item = t.topLevelItem(i)
                    if int(item.data(0, ROLE_ELEC_ID)) == int(elec_id):
                        t.setCurrentItem(item)
                        item.setSelected(True)
                        try:
                            t.scrollToItem(item)
                        except Exception:
                            pass
                        break
            except Exception:
                pass

    def _remap_page_local_state_after_reorder(
        self, old_electrodes: list, new_electrodes: list
    ) -> None:
        old_index_by_obj = {id(e): i for i, e in enumerate(old_electrodes)}

        def _remap_page(page, attr_names):
            if page is None:
                return

            for attr in attr_names:
                try:
                    old_dict = getattr(page, attr, None)
                    if not isinstance(old_dict, dict):
                        continue

                    new_dict = {}
                    for new_idx, elec in enumerate(new_electrodes):
                        old_idx = old_index_by_obj.get(id(elec))
                        if old_idx is not None and old_idx in old_dict:
                            new_dict[new_idx] = old_dict[old_idx]

                    setattr(page, attr, new_dict)
                except Exception:
                    pass

        # Reconstruction page
        _remap_page(
            self.reco_page,
            [
                "_page_electrode_visible",
                "_page_contacts_visible",
            ],
        )

        # Oblique page
        _remap_page(
            getattr(self.state, "oblique_page", None),
            [
                "_page_electrode_visible",
                "_page_contacts_visible",
                "_page_contact_labels_visible",
            ],
        )

        # 3D page
        _remap_page(
            getattr(self.state, "view3d_page", None),
            [
                "_page_electrode_visible",
                "_page_contacts_visible",
                "_page_contact_labels_visible",
                "_surface_projection_defs",
                "_surface_projection_actors",
            ],
        )

    # ---------- Internal helpers ----------
    def _current_selected_electrode_id(self) -> int | None:
        sid = getattr(self.state, "selected_electrode_id", None)
        if isinstance(sid, int):
            return sid

        for t in self._trees:
            items = t.selectedItems()
            if items:
                it = items[0]
                kind = it.data(0, ROLE_KIND)
                if kind == "electrode":
                    return int(it.data(0, ROLE_ELEC_ID))
                if kind == "contact":
                    return int(it.data(0, ROLE_ELEC_ID))
        return None

    def _set_children_checkstates(self, elec_id: int, checked: bool) -> None:
        for t in self._trees:
            for i in range(t.topLevelItemCount()):
                parent = t.topLevelItem(i)
                if int(parent.data(0, ROLE_ELEC_ID)) != elec_id:
                    continue
                for c in range(parent.childCount()):
                    child = parent.child(c)
                    if child.data(0, ROLE_KIND) == "contact":
                        child.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                parent.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

    def _update_parent_states_all_trees(self, elec_id: int) -> None:
        for t in self._trees:
            for i in range(t.topLevelItemCount()):
                parent = t.topLevelItem(i)
                if int(parent.data(0, ROLE_ELEC_ID)) != elec_id:
                    continue
                self._update_parent_check_state_from_children(parent)

    def _update_parent_check_state_from_children(self, parent: QTreeWidgetItem) -> None:
        states = []
        for c in range(parent.childCount()):
            child = parent.child(c)
            if child.data(0, ROLE_KIND) == "contact":
                states.append(child.checkState(0))
        if not states:
            return
        if all(s == Qt.Checked for s in states):
            parent.setCheckState(0, Qt.Checked)
        elif all(s == Qt.Unchecked for s in states):
            parent.setCheckState(0, Qt.Unchecked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        try:
            if item.data(0, ROLE_KIND) != "electrode":
                return
            elec_id = int(item.data(0, ROLE_ELEC_ID))
            if 0 <= elec_id < len(self.state.electrodes):
                self.state.electrodes[elec_id]["expanded"] = True
        except Exception:
            pass

    def _on_item_collapsed(self, item: QTreeWidgetItem) -> None:
        try:
            if item.data(0, ROLE_KIND) != "electrode":
                return
            elec_id = int(item.data(0, ROLE_ELEC_ID))
            if 0 <= elec_id < len(self.state.electrodes):
                self.state.electrodes[elec_id]["expanded"] = False
        except Exception:
            pass

    def _set_tree_drag_visual_mode(self, tree: QTreeWidget, active: bool) -> None:
        """
        During electrode drag-reorder:
        - keep electrode item colors
        - remove white hover/selection feedback
        - show only the black insertion line + count badge
        """
        try:
            if tree is None:
                return

            key = id(tree)

            if bool(active):
                # Save current style once.
                if key not in self._tree_drag_saved_styles:
                    try:
                        self._tree_drag_saved_styles[key] = tree.styleSheet()
                    except Exception:
                        self._tree_drag_saved_styles[key] = self._tree_normal_style

                # Keep row colours visible while hiding hover/selection feedback
                # in the custom delegate during drag.
                tree.setProperty("neuxelec_dragging", True)
                try:
                    tree.setStyleSheet(self._tree_drag_style)
                except Exception:
                    pass

                # Remove visible selection/current row while dragging.
                # The moved electrodes are still stored in self._tree_drag_source_ids.
                try:
                    tree.clearSelection()
                except Exception:
                    pass

                try:
                    sm = tree.selectionModel()
                    if sm is not None:
                        sm.clearSelection()
                        sm.clearCurrentIndex()
                except Exception:
                    pass

                try:
                    tree.setCurrentIndex(QModelIndex())
                except Exception:
                    pass

                # Disable hover painting while dragging.
                try:
                    tree.setMouseTracking(False)
                    tree.viewport().setMouseTracking(False)
                    tree.viewport().setAttribute(Qt.WA_Hover, False)
                except Exception:
                    pass

            else:
                # Restore normal style and hover behavior.
                tree.setProperty("neuxelec_dragging", False)
                try:
                    saved = self._tree_drag_saved_styles.pop(key, self._tree_normal_style)
                    tree.setStyleSheet(saved if saved else self._tree_normal_style)
                except Exception:
                    try:
                        tree.setStyleSheet(self._tree_normal_style)
                    except Exception:
                        pass

                try:
                    tree.setMouseTracking(True)
                    tree.viewport().setMouseTracking(True)
                    tree.viewport().setAttribute(Qt.WA_Hover, True)
                except Exception:
                    pass

            try:
                tree.viewport().update()
            except Exception:
                pass

        except Exception:
            pass
