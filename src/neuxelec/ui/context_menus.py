from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMenu


def top_level_window():
    for w in QApplication.topLevelWidgets():
        try:
            if w.isVisible():
                return w
        except Exception:
            pass
    return None


def make_base_menu() -> QMenu:
    """
    Create a NeuXelec-styled context menu shared by all pages.

    Hovered actions receive the same pink outline used in the startup
    interface input fields.
    """
    menu = QMenu(top_level_window())

    menu.setMinimumWidth(220)
    menu.setSeparatorsCollapsible(False)

    menu.setStyleSheet("""
        QMenu {
            background-color: #0B0B0F;
            color: #F2F2F5;
            border: 1px solid #2B2D38;
            border-radius: 10px;
            padding: 6px;
            font-size: 13px;
            font-weight: 500;
        }

        QMenu::item {
            background-color: transparent;
            color: #F2F2F5;

            /* Transparent border prevents the item from moving
               when the pink hover border appears. */
            border: 1px solid transparent;
            border-radius: 7px;

            padding: 8px 28px 8px 12px;
            margin: 2px 1px;
        }

        QMenu::item:selected {
            background-color: #17181F;
            color: white;
            border: 1px solid #FF487D;
            border-radius: 7px;
        }

        QMenu::item:pressed {
            background-color: #20222B;
            color: white;
            border: 1px solid #FF487D;
            border-radius: 7px;
        }

        QMenu::item:disabled {
            background-color: transparent;
            color: #62646E;
            border: 1px solid transparent;
        }

        QMenu::separator {
            height: 1px;
            background-color: #2B2D38;
            margin: 6px 8px;
        }

        QMenu::indicator {
            width: 16px;
            height: 16px;
            margin-left: 5px;
            margin-right: 6px;
        }

        QMenu::indicator:unchecked {
            background-color: #151720;
            border: 1px solid #353844;
            border-radius: 4px;
        }

        QMenu::indicator:unchecked:selected {
            background-color: #181A24;
            border: 1px solid #FF487D;
            border-radius: 4px;
        }

        QMenu::indicator:checked {
            background-color: #151720;
            border: 1px solid #FF487D;
            border-radius: 4px;
            image: url(resources/images/neuxelec_checkbox_cross.svg);
        }

        QMenu::indicator:checked:selected {
            background-color: #181A24;
            border: 1px solid #FF487D;
            border-radius: 4px;
            image: url(resources/images/neuxelec_checkbox_cross.svg);
        }

        QMenu::indicator:disabled {
            background-color: #10121A;
            border: 1px solid #252834;
            border-radius: 4px;
        }

        QMenu::indicator:checked:disabled {
            background-color: #10121A;
            border: 1px solid #555967;
            border-radius: 4px;
            image: url(resources/images/neuxelec_checkbox_cross.svg);
        }
    """)

    return menu


def exec_3d_view_menu(
    global_pos,
    *,
    has_lh: bool,
    has_rh: bool,
    show_lh: bool,
    show_rh: bool,
    color_scale_visible: bool = True,
    show_pial_options: bool = True,
    show_color_scale_option: bool = True,
    show_mni_load_option: bool = False,
    show_mni_t1_option: bool = False,
    mni_t1_visible: bool = False,
    native_actions_enabled: bool = True,
    show_mni_parcellation_table_option: bool = False,
    show_keep_electrodes_through_slices_option: bool = False,
    keep_electrodes_through_slices: bool = False,
    show_slice_plane_frames_option: bool = False,
    slice_plane_frames_visible: bool = True,
    can_add_marker: bool = False,
    marker_under_cursor: bool = False,
    has_hidden_markers: bool = False,
) -> str | None:

    menu = make_base_menu()

    act_marker_list = None
    act_add_marker = None
    act_edit_marker = None
    act_hide_marker = None
    act_export_marker = None
    act_delete_marker = None
    act_show_hidden_markers = None

    act_marker_list = menu.addAction("Marker list")
    menu.addSeparator()

    if marker_under_cursor:
        act_edit_marker = menu.addAction("Edit marker…")
        act_hide_marker = menu.addAction("Hide marker")
        act_export_marker = menu.addAction("Export marker…")
        act_delete_marker = menu.addAction("Delete marker")
        menu.addSeparator()

    elif can_add_marker:
        act_add_marker = menu.addAction("Add marker…")
        menu.addSeparator()

    if has_hidden_markers:
        act_show_hidden_markers = menu.addAction("Show hidden markers")
        menu.addSeparator()

    act_pet = None
    act_siscom = None
    act_ct = None

    if native_actions_enabled:
        act_pet = menu.addAction("Color PET")
        act_siscom = menu.addAction("Color SISCOM")
        act_ct = menu.addAction("Color CT")

    act_toggle_color_scale = None
    if show_color_scale_option:
        act_toggle_color_scale = menu.addAction(
            "Remove color scale" if color_scale_visible else "Add color scale"
        )
    act_toggle_slice_plane_frames = None
    if show_slice_plane_frames_option:
        act_toggle_slice_plane_frames = menu.addAction(
            "Remove frame" if slice_plane_frames_visible else "Add frame"
        )
    act_render_brain = None

    if native_actions_enabled:
        menu.addSeparator()
        act_render_brain = menu.addAction("Render Brain…")

    act_load_mni_electrodes = None
    if show_mni_load_option:
        act_load_mni_electrodes = menu.addAction("Load MNI electrodes.tsv…")

    act_toggle_mni_t1_slices = None
    if show_mni_t1_option:
        act_toggle_mni_t1_slices = menu.addAction(
            "Remove MNI T1 slices" if mni_t1_visible else "Add MNI T1 slices"
        )
    act_mni_parcellation_table = None
    if show_mni_parcellation_table_option:
        act_mni_parcellation_table = menu.addAction("Parcellation table…")

    act_keep_electrodes_through_slices = None
    if show_keep_electrodes_through_slices_option:
        menu.addSeparator()
        act_keep_electrodes_through_slices = menu.addAction(
            "Keep electrodes visible through slices"
        )
        act_keep_electrodes_through_slices.setCheckable(True)
        act_keep_electrodes_through_slices.setChecked(bool(keep_electrodes_through_slices))

    act_toggle_lh = None
    act_toggle_rh = None

    if show_pial_options and (has_lh or has_rh):
        menu.addSeparator()

        if has_lh:
            act_toggle_lh = menu.addAction("Remove LH" if show_lh else "Add LH")

        if has_rh:
            act_toggle_rh = menu.addAction("Remove RH" if show_rh else "Add RH")

    action = menu.exec(global_pos)

    if action is None:
        return None

    if act_marker_list is not None and action == act_marker_list:
        return "marker_list"

    if act_add_marker is not None and action == act_add_marker:
        return "add_marker"

    if act_edit_marker is not None and action == act_edit_marker:
        return "edit_marker"

    if act_hide_marker is not None and action == act_hide_marker:
        return "hide_marker"

    if act_export_marker is not None and action == act_export_marker:
        return "export_marker"

    if act_delete_marker is not None and action == act_delete_marker:
        return "delete_marker"

    if act_show_hidden_markers is not None and action == act_show_hidden_markers:
        return "show_hidden_markers"

    if act_pet is not None and action == act_pet:
        return "pet"

    if act_siscom is not None and action == act_siscom:
        return "siscom"

    if act_ct is not None and action == act_ct:
        return "ct"

    if act_render_brain is not None and action == act_render_brain:
        return "render_brain"

    if act_load_mni_electrodes is not None and action == act_load_mni_electrodes:
        return "load_mni_electrodes"

    if act_toggle_mni_t1_slices is not None and action == act_toggle_mni_t1_slices:
        return "toggle_mni_t1_slices"

    if act_toggle_slice_plane_frames is not None and action == act_toggle_slice_plane_frames:
        return "toggle_slice_plane_frames"

    if act_mni_parcellation_table is not None and action == act_mni_parcellation_table:
        return "mni_parcellation_table"

    if (
        act_keep_electrodes_through_slices is not None
        and action == act_keep_electrodes_through_slices
    ):
        return "toggle_keep_electrodes_through_slices"

    if act_toggle_lh is not None and action == act_toggle_lh:
        return "toggle_lh"

    if act_toggle_rh is not None and action == act_toggle_rh:
        return "toggle_rh"

    if act_toggle_color_scale is not None and action == act_toggle_color_scale:
        return "toggle_color_scale"

    return None


def exec_oblique_slice_menu(
    global_pos,
    *,
    color_scale_visible: bool = True,
    show_color_scale_option: bool = True,
) -> str | None:
    menu = make_base_menu()

    act_pet = menu.addAction("Color PET")
    act_siscom = menu.addAction("Color SISCOM")

    act_toggle_color_scale = None
    if show_color_scale_option:
        act_toggle_color_scale = menu.addAction(
            "Remove color scale" if color_scale_visible else "Add color scale"
        )

    action = menu.exec(global_pos)

    if action is None:
        return None

    if action == act_pet:
        return "pet"
    if action == act_siscom:
        return "siscom"

    if act_toggle_color_scale is not None and action == act_toggle_color_scale:
        return "toggle_color_scale"

    return None


def exec_electrode_tree_menu(
    global_pos,
    *,
    kind: str,
    current_page: str,
    labels_on: bool = False,
    label_on: bool = False,
    projection_on: bool = False,
    selection_count: int = 1,
    editable: bool = True,
):
    menu = make_base_menu()

    if kind == "electrode":
        act_toggle_labels = None
        act_toggle_projection = None
        act_rename_elec = None
        act_color_elec = None
        act_delete_elec = None

        # Labels on Oblique Slice and 3D View
        if current_page in ("pageObliqueSlices", "page3DView"):
            act_toggle_labels = menu.addAction("Remove labels" if labels_on else "Add labels")

        # Projection ONLY on 3D View
        if current_page == "page3DView":
            act_toggle_projection = menu.addAction(
                "Remove surface projection" if projection_on else "Add surface projection"
            )

        # Rename is available only when a single electrode is selected.
        # Renaming several electrodes at once would be ambiguous.
        n_selected = max(1, int(selection_count))

        # Structural editing actions are only available in Edit mode.
        if editable:
            if n_selected == 1:
                act_rename_elec = menu.addAction("Rename electrode…")

        # Color remains available in both Edit and View Only modes.
        act_color_elec = menu.addAction("Color")

        # Electrode deletion is only available in Edit mode.
        if editable:
            if n_selected > 1:
                act_delete_elec = menu.addAction(f"Delete {n_selected} electrodes")
            else:
                act_delete_elec = menu.addAction("Delete electrode")

        action = menu.exec(global_pos)

        if action is None:
            return None

        if act_toggle_labels is not None and action == act_toggle_labels:
            return "toggle_labels"
        if act_toggle_projection is not None and action == act_toggle_projection:
            return "toggle_projection"
        if act_rename_elec is not None and action == act_rename_elec:
            return "rename_electrode"
        if act_color_elec is not None and action == act_color_elec:
            return "color_electrode"
        if act_delete_elec is not None and action == act_delete_elec:
            return "delete_electrode"

        return None

    if kind == "contact":
        act_toggle_label = None
        act_show_coronal = None
        act_show_axial = None
        act_show_sagittal = None
        act_edit = None
        act_delete = None

        # Add/remove label only on Oblique Slice and 3D View
        if current_page in ("pageObliqueSlices", "page3DView"):
            act_toggle_label = menu.addAction("Remove label" if label_on else "Add label")

        # Show slice actions only on 3D View
        if current_page == "page3DView":
            menu.addSeparator()
            act_show_coronal = menu.addAction("Show coronal slice")
            act_show_axial = menu.addAction("Show axial slice")
            act_show_sagittal = menu.addAction("Show sagittal slice")

        # Edit coordinates only on Reconstruction
        if current_page == "pageReconstruction":
            act_edit = menu.addAction("Edit coordinates")

        n_selected = max(1, int(selection_count))

        # Contact deletion is only available in Edit mode.
        if editable:
            if n_selected > 1:
                act_delete = menu.addAction(f"Delete {n_selected} contacts")
            else:
                act_delete = menu.addAction("Delete contact")

        action = menu.exec(global_pos)

        if action is None:
            return None

        if act_toggle_label is not None and action == act_toggle_label:
            return "toggle_label"
        if act_show_coronal is not None and action == act_show_coronal:
            return "show_coronal_slice"
        if act_show_axial is not None and action == act_show_axial:
            return "show_axial_slice"
        if act_show_sagittal is not None and action == act_show_sagittal:
            return "show_sagittal_slice"
        if act_edit is not None and action == act_edit:
            return "edit_contact"
        if act_delete is not None and action == act_delete:
            return "delete_contact"

        return None


def exec_mni_electrode_tree_menu(
    global_pos,
    *,
    kind: str,
    labels_on: bool = False,
    electrode_names_on: bool = False,
    patient_name_on: bool = False,
) -> str | None:
    """
    Context menu for imported MNI electrodes in tv_Electrodes_3.

    kind:
        - mni_set     : whole patient TSV
        - mni_group   : one electrode
        - mni_contact : one contact
    """
    menu = make_base_menu()

    act_color = None
    act_toggle_labels = None
    act_toggle_electrode_names = None
    act_toggle_patient_name = None
    act_show_coronal = None
    act_show_axial = None
    act_show_sagittal = None

    if kind == "mni_set":
        act_color = menu.addAction("Color")

        act_toggle_electrode_names = menu.addAction(
            "Remove electrode names" if electrode_names_on else "Add electrode names"
        )

        act_toggle_patient_name = menu.addAction(
            "Remove patient name" if patient_name_on else "Add patient name"
        )

    elif kind == "mni_group":
        act_color = menu.addAction("Color")
        act_toggle_labels = menu.addAction("Remove labels" if labels_on else "Add labels")

    elif kind == "mni_contact":
        act_toggle_labels = menu.addAction("Remove label" if labels_on else "Add label")

        menu.addSeparator()

        act_show_coronal = menu.addAction("Show coronal slice")
        act_show_axial = menu.addAction("Show axial slice")
        act_show_sagittal = menu.addAction("Show sagittal slice")

    else:
        return None

    action = menu.exec(global_pos)

    if action is None:
        return None

    if act_color is not None and action == act_color:
        return "color"

    if act_toggle_labels is not None and action == act_toggle_labels:
        return "toggle_labels"

    if act_toggle_electrode_names is not None and action == act_toggle_electrode_names:
        return "toggle_electrode_names"

    if act_toggle_patient_name is not None and action == act_toggle_patient_name:
        return "toggle_patient_name"

    if act_show_coronal is not None and action == act_show_coronal:
        return "show_coronal_slice"

    if act_show_axial is not None and action == act_show_axial:
        return "show_axial_slice"

    if act_show_sagittal is not None and action == act_show_sagittal:
        return "show_sagittal_slice"

    return None
