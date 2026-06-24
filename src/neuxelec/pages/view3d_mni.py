"""MNI atlas mode for the 3D View page.

Isolates the MNI-space rendering and scene management of :class:`View3DPage`
(atlas brain, MNI electrode sets, native<->MNI mode switching, per-patient and
per-group colours/labels) as a mixin. Methods are unchanged. ``View3DPage``
inherits this mixin, so every ``self.*`` reference resolves exactly as before.
"""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTreeWidget

from neuxelec.ui.neuxelec_color_dialog import NeuXelecColorDialog
from neuxelec.ui.neuxelec_message_dialog import NeuXelecMessageDialog

try:
    import pyvista as pv

    _PV_OK = True
except Exception:
    pv = None
    _PV_OK = False

logger = logging.getLogger(__name__)


class View3DMniMixin:
    """MNI atlas mode rendering and scene management for View3DPage."""

    def _get_mni_brain_mask_image(self) -> sitk.Image | None:
        """
        Return the MNI/template brain mask used as atlas brain.
        """
        return self._get_mni_template_mask_image()

    def _render_mni_atlas_brain(self) -> None:
        if not _PV_OK or self.plotter is None:
            return

        if self._mni_atlas_actor is not None:
            try:
                self.plotter.remove_actor(self._mni_atlas_actor, reset_camera=False)
            except Exception:
                pass
            self._mni_atlas_actor = None

        try:
            img = self._get_mni_brain_mask_image()

            if img is None:
                return

            # Your existing _binarymask_to_polydata() already converts LPS image
            # coordinates to RAS PyVista coordinates.
            mesh = self._binarymask_to_polydata(img)

            if mesh is None or getattr(mesh, "n_points", 0) == 0:
                return

            try:
                mesh = mesh.triangulate().clean(tolerance=1e-6)
            except Exception:
                pass
            opacity = 0.22
            try:
                if self.sld_3d_brainMaskOpacity is not None:
                    opacity = float(self.sld_3d_brainMaskOpacity.value()) / 100.0
            except Exception:
                opacity = 0.22
            self._mni_atlas_actor = self.plotter.add_mesh(
                mesh,
                color=(0.78, 0.78, 0.78),
                opacity=float(opacity),
                smooth_shading=True,
                ambient=0.35,
                diffuse=0.60,
                specular=0.08,
                name="mni_atlas_brain",
            )

            try:
                self._mni_atlas_actor.PickableOff()
            except Exception:
                pass

            try:
                self._apply_actor_clipping()
            except Exception:
                pass

        except Exception as e:
            print("[MNI atlas] Failed to render atlas brain:", e)

    def _clear_mni_electrode_actors(self) -> None:
        if self.plotter is None:
            return

        for d in (
            getattr(self, "_mni_electrode_actors", {}),
            getattr(self, "_mni_label_actors", {}),
        ):
            try:
                for _key, actor in list(d.items()):
                    try:
                        self.plotter.remove_actor(actor, reset_camera=False)
                    except Exception:
                        pass

                d.clear()

            except Exception:
                pass

    def _render_mni_electrode_sets(self) -> None:
        if not _PV_OK or self.plotter is None:
            return

        self._clear_mni_electrode_actors()

        sets = getattr(self.state, "mni_electrode_sets", []) or []

        point_size = 8.0
        try:
            if self.spin_contacts_size is not None:
                point_size = float(self.spin_contacts_size.value())
        except Exception:
            pass

        show_shaft = True
        try:
            if self.btn_elec_shaft is not None:
                show_shaft = bool(self.btn_elec_shaft.isChecked())
        except Exception:
            pass

        for si, mni_set in enumerate(sets):
            if not isinstance(mni_set, dict):
                continue

            self._ensure_mni_visibility_fields(mni_set)

            if not bool(mni_set.get("visible", True)):
                continue

            contacts = mni_set.get("contacts", []) or []
            if not contacts:
                continue

            color_rgb = mni_set.get("color", (100, 180, 255))
            try:
                color = (
                    float(color_rgb[0]) / 255.0,
                    float(color_rgb[1]) / 255.0,
                    float(color_rgb[2]) / 255.0,
                )
            except Exception:
                color = (0.4, 0.7, 1.0)

            groups = {}
            for ci, c in enumerate(contacts):
                if not self._mni_contact_is_visible(mni_set, ci, c):
                    continue

                group = self._mni_group_name_from_contact(c)
                try:
                    x, y, z = c.get("mni_ras", [None, None, None])
                    p = [float(x), float(y), float(z)]
                except Exception:
                    continue

                groups.setdefault(group, []).append((ci, c, p))

            for group_name, group_contacts in groups.items():
                group_rgb = self._mni_group_color_rgb(mni_set, group_name)
                try:
                    color = (
                        float(group_rgb[0]) / 255.0,
                        float(group_rgb[1]) / 255.0,
                        float(group_rgb[2]) / 255.0,
                    )
                except Exception:
                    color = (0.4, 0.7, 1.0)
                if not group_contacts:
                    continue

                pts_arr = np.asarray([p for _ci, _c, p in group_contacts], dtype=np.float32)

                # Points
                try:
                    poly = pv.PolyData(pts_arr)
                    actor = self.plotter.add_points(
                        poly,
                        color=color,
                        point_size=float(point_size),
                        render_points_as_spheres=True,
                        name=f"mni_contacts_{si}_{group_name}",
                    )

                    try:
                        actor.PickableOff()
                    except Exception:
                        pass

                    self._mni_electrode_actors[(si, group_name, "points")] = actor

                except Exception as e:
                    print("[MNI electrodes] Failed to render points:", e)

                # Shaft/line per MNI electrode
                if show_shaft and pts_arr.shape[0] >= 2:
                    try:
                        line = pv.lines_from_points(pts_arr, close=False)
                        line_actor = self.plotter.add_mesh(
                            line,
                            color=color,
                            line_width=3,
                            name=f"mni_line_{si}_{group_name}",
                        )

                        try:
                            line_actor.PickableOff()
                        except Exception:
                            pass

                        self._mni_electrode_actors[(si, group_name, "line")] = line_actor

                    except Exception:
                        pass

                # Optional electrode name label near first contact.
                # Controlled from right-click on the patient:
                # Add/Remove electrode names.
                try:
                    if bool(mni_set.get("electrode_names_visible", False)):
                        label_pt = pts_arr[0].copy()
                        label_pt[0] += 3.0
                        label_pt[2] += 3.0

                        label_actor = self.plotter.add_point_labels(
                            np.asarray([label_pt], dtype=np.float32),
                            [str(group_name)],
                            font_size=12,
                            text_color=color,
                            shape_opacity=0.0,
                            show_points=False,
                            always_visible=True,
                            name=f"mni_label_{si}_{group_name}",
                        )

                        try:
                            label_actor.PickableOff()
                        except Exception:
                            pass

                        self._mni_label_actors[(si, group_name, "label")] = label_actor

                except Exception:
                    pass

                # Optional contact labels if later you activate them.
                try:
                    label_pts = []
                    label_txt = []

                    for ci, c, p in group_contacts:
                        if not self._mni_contact_label_is_visible(mni_set, ci):
                            continue

                        lp = np.asarray(p, dtype=np.float32)
                        lp[0] += 2.0
                        lp[2] += 2.0

                        label_pts.append(lp)
                        label_txt.append(str(c.get("name", f"{group_name}{ci + 1}")))

                    if label_pts:
                        contact_label_actor = self.plotter.add_point_labels(
                            np.asarray(label_pts, dtype=np.float32),
                            label_txt,
                            font_size=11,
                            text_color=color,
                            shape_opacity=0.0,
                            show_points=False,
                            always_visible=True,
                            name=f"mni_contact_labels_{si}_{group_name}",
                        )

                        try:
                            contact_label_actor.PickableOff()
                        except Exception:
                            pass

                        self._mni_label_actors[(si, group_name, "contact_labels")] = (
                            contact_label_actor
                        )

                except Exception:
                    pass

            # IMPORTANT:
            # Same indentation as "for group_name, group_contacts in groups.items():"
            # This is outside the group loop and is called once per MNI patient.
            try:
                self._render_mni_patient_name_label_only(si)
            except Exception:
                pass

    def _clear_native_scene_for_mni_mode(self) -> None:
        """
        Remove native patient actors to avoid mixing native T1 space and MNI space.
        """
        if self.plotter is None:
            return
        # Remove native anatomical markers when entering MNI mode.
        try:
            self._remove_all_anatomical_marker_actors()
        except Exception:
            pass

        self._hovered_anatomical_marker_id = None

        for attr in (
            "_brain_actor",
            "_ct_actor",
            "_pet_actor",
            "_siscom_actor",
            "_coronal_plane_actor",
            "_axial_plane_actor",
            "_sagittal_plane_actor",
            "_coronal_pet_actor",
            "_axial_pet_actor",
            "_sagittal_pet_actor",
            "_coronal_siscom_actor",
            "_axial_siscom_actor",
            "_sagittal_siscom_actor",
            "_coronal_elec_actor",
            "_axial_elec_actor",
            "_sagittal_elec_actor",
            "_coronal_outline_actor",
            "_axial_outline_actor",
            "_sagittal_outline_actor",
        ):
            actor = getattr(self, attr, None)

            if actor is None:
                continue

            try:
                if isinstance(actor, (list, tuple)):
                    for a in actor:
                        try:
                            self.plotter.remove_actor(a, reset_camera=False)
                        except Exception:
                            pass
                else:
                    self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass

            try:
                setattr(self, attr, None)
            except Exception:
                pass

        try:
            self._remove_actor("electrodes")
        except Exception:
            pass

        try:
            for _, a in list(getattr(self, "_elec_label_actors", {}).items()):
                try:
                    self.plotter.remove_actor(a, reset_camera=False)
                except Exception:
                    pass
            self._elec_label_actors.clear()
        except Exception:
            pass

        # Surface projections are native-space objects.
        # They must never remain visible on the MNI atlas.
        try:
            self._remove_all_surface_projection_actors()
        except Exception:
            pass

        try:
            self._remove_pet_scalar_bar()
        except Exception:
            pass

        try:
            self._remove_siscom_scalar_bar()
        except Exception:
            pass

    def _store_native_checkbox_state_for_mni(self) -> None:
        """
        Remember current native checkboxes before entering MNI mode.
        """
        self._mni_native_checkboxes_state = {}

        for name in (
            "chk_brainmask",
            "chk_iso",
            "chk_pial",
            "chk_ct",
            "chk_pet",
            "chk_siscom",
            "chk_coronal_plane",
            "chk_axial_plane",
            "chk_sagittal_plane",
            "chk_parcel1",
            "chk_parcel2",
        ):
            cb = getattr(self, name, None)

            if cb is None:
                continue

            try:
                self._mni_native_checkboxes_state[name] = bool(cb.isChecked())
            except Exception:
                pass

    def _set_native_checkboxes_for_mni(self, checked: bool) -> None:
        """
        When MNI is active:
        - Brain mask and Pial surface remain clickable.
        If the user clicks one of them, MNI will be disabled automatically.
        - Other native-space overlays are disabled because they are not in MNI space.
        """
        native_items = [
            ("brainmask", getattr(self, "chk_brainmask", None)),
            ("pial", getattr(self, "chk_pial", None)),
            ("iso", getattr(self, "chk_iso", None)),
            ("ct", getattr(self, "chk_ct", None)),
            ("pet", getattr(self, "chk_pet", None)),
            ("siscom", getattr(self, "chk_siscom", None)),
            ("coronal", getattr(self, "chk_coronal_plane", None)),
            ("axial", getattr(self, "chk_axial_plane", None)),
            ("sagittal", getattr(self, "chk_sagittal_plane", None)),
            ("parcel1", getattr(self, "chk_parcel1", None)),
            ("parcel2", getattr(self, "chk_parcel2", None)),
        ]

        for name, cb in native_items:
            if cb is None:
                continue

            try:
                cb.blockSignals(True)

                if checked:
                    # MNI mode ON.
                    # Keep Brain mask and Pial surface clickable, but unchecked.
                    if name in ("brainmask", "pial"):
                        cb.setChecked(False)
                        cb.setEnabled(True)
                    else:
                        cb.setChecked(False)
                        cb.setEnabled(False)

                else:
                    # MNI mode OFF.
                    cb.setEnabled(True)

                    # If we are switching because the user clicked Brain mask/Pial,
                    # do not restore the old checkbox state, otherwise it may undo
                    # the user's click.
                    if not bool(getattr(self, "_switching_mni_to_native_brain", False)):
                        if name in getattr(self, "_mni_native_checkboxes_state", {}):
                            cb.setChecked(bool(self._mni_native_checkboxes_state[name]))

                cb.blockSignals(False)

            except Exception:
                try:
                    cb.blockSignals(False)
                except Exception:
                    pass

    def _leave_mni_mode_and_restore_native(
        self,
        restore_previous_checkboxes: bool = True,
        keep_clicked_native_source: str | None = None,
    ) -> None:
        """
        Leave MNI atlas mode and restore native patient display.

        This function must be used both when:
        - user unchecks MNI atlas directly;
        - user clicks Brain mask or Pial surface while MNI atlas is active.

        keep_clicked_native_source:
            "brainmask" or "pial" when the user clicked one of these checkboxes.
            In this case we do not restore the previous checkbox state because it
            could undo the user's click.
        """
        try:
            self._mni_t1_slices_visible = False

            # 1) Uncheck MNI atlas without recursively triggering this function.
            if self.chk_mni_atlas is not None:
                self.chk_mni_atlas.blockSignals(True)
                self.chk_mni_atlas.setChecked(False)
                self.chk_mni_atlas.blockSignals(False)

            # 2) Remove all MNI actors.
            self._clear_mni_scene()

            # 3) Explicitly remove every MNI slice actor.
            # Checkbox signals are blocked below, so the usual toggled handlers
            # would not remove these actors automatically.
            self._clear_all_slice_plane_actors()

            # 4) Disable MNI-only slice state.
            self._set_checked(self.chk_coronal_plane, False)
            self._set_checked(self.chk_axial_plane, False)
            self._set_checked(self.chk_sagittal_plane, False)

            # 5) Rebuild all slice caches for the native patient geometry.
            self._invalidate_slice_volume_cache(
                base=True,
                pet=True,
                siscom=True,
            )

            self._slice_crop_bounds_xyz = None

            # 4) Re-enable native controls.
            self._switching_mni_to_native_brain = bool(
                keep_clicked_native_source in ("brainmask", "pial")
                or not restore_previous_checkboxes
            )

            self._set_native_checkboxes_for_mni(False)

            self._switching_mni_to_native_brain = False
            # MNI and native images can have different dimensions and crop bounds.
            # Recalculate the slider ranges using the patient T1 geometry.
            try:
                self._update_all_plane_slider_ranges()
            except Exception:
                pass
            # 5) If the user clicked Brainmask/Pial, force that source to stay checked.
            if keep_clicked_native_source == "brainmask":
                self._set_checked(self.chk_brainmask, True)
                self._set_checked(self.chk_pial, False)
                self._set_checked(self.chk_iso, False)

            elif keep_clicked_native_source == "pial":
                self._set_checked(self.chk_pial, True)
                self._set_checked(self.chk_brainmask, False)
                self._set_checked(self.chk_iso, False)

            # 6) Rebuild the electrode tree: remove MNI rows, restore patient electrodes.
            self._refresh_electrode_tree_for_current_3d_mode()

            # 7) Redraw native brain and native electrodes.
            try:
                self._render_brain(reset_camera=False)
            except TypeError:
                self._render_brain()
            except Exception:
                pass

            try:
                self.update_electrodes()
            except Exception:
                pass
            try:
                self._render_anatomical_markers()
            except Exception:
                pass
            try:
                self._refresh_marker_list_dialog()
            except Exception:
                pass
            try:
                self.render_all_surface_projections()
            except Exception:
                pass

            try:
                # Rebuild native slice actors immediately.
                # This avoids waiting for the user to move a slider.
                self._refresh_all_visible_slice_planes_full()
            except Exception:
                try:
                    self._refresh_multiplanar_clipped_scene()
                except Exception:
                    pass

            try:
                self._update_brain_opacity_slider_states()
                self._update_plane_slider_enabled_states()
                self._update_modality_controls_enabled_states()
                self._update_planes_info_label()
            except Exception:
                pass

            self._render()

        except Exception as e:
            print("[MNI atlas] Failed to restore native mode:", e)

    def _on_mni_atlas_toggled(self, checked: bool) -> None:
        checked = bool(checked)
        self._invalidate_slice_volume_cache()

        try:
            if checked:
                self._store_native_checkbox_state_for_mni()

                # This unchecks and disables native-space checkboxes.
                self._set_native_checkboxes_for_mni(True)

                # Remove native T1-space actors.
                self._clear_native_scene_for_mni_mode()

                # Rebuild the 3D electrode tree: show only MNI electrodes.
                self._refresh_electrode_tree_for_current_3d_mode()

                # Make the MNI template T1 immediately available to the slice controls.
                self._set_mni_t1_slices_visible(True)

                # Enter MNI mode without displaying a slice automatically.
                self._set_checked(self.chk_coronal_plane, False)
                self._set_checked(self.chk_axial_plane, False)
                self._set_checked(self.chk_sagittal_plane, False)

                self._remove_actor("coronal_plane")
                self._remove_actor("axial_plane")
                self._remove_actor("sagittal_plane")

                self._remove_actor("coronal_outline")
                self._remove_actor("axial_outline")
                self._remove_actor("sagittal_outline")

                self._render_mni_scene(reset_camera=True)
                # Marker list and actors now belong to the MNI coordinate space.
                self._refresh_marker_list_dialog()

                # In MNI mode:
                # - BrainMask opacity slider controls MNI atlas opacity.
                # - Pial opacity slider is disabled.
                self._update_brain_opacity_slider_states()
                self._update_plane_slider_enabled_states()
                self._update_modality_controls_enabled_states()
                self._update_planes_info_label()

                self._render()
                return

            # Direct uncheck of MNI atlas checkbox.
            self._leave_mni_mode_and_restore_native(
                restore_previous_checkboxes=True,
                keep_clicked_native_source=None,
            )
            self._update_brain_opacity_slider_states()
        except Exception as e:
            print("[MNI atlas] Toggle failed:", e)

            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "MNI atlas",
                f"MNI atlas mode failed:\n{e}",
            )

    def _clear_mni_scene(self) -> None:
        if self.plotter is None:
            return

        if self._mni_atlas_actor is not None:
            try:
                self.plotter.remove_actor(self._mni_atlas_actor, reset_camera=False)
            except Exception:
                pass

            self._mni_atlas_actor = None

        self._clear_mni_electrode_actors()

    def _render_mni_scene(self, reset_camera: bool = False) -> None:
        """
        Render MNI atlas brain + all visible imported MNI electrode sets.
        """
        if not _PV_OK or self.plotter is None:
            return

        try:
            if self.chk_mni_atlas is not None and not self.chk_mni_atlas.isChecked():
                return
        except Exception:
            pass

        # Remove actors from the previous coordinate space.
        try:
            self._remove_all_anatomical_marker_actors()
        except Exception:
            pass

        self._render_mni_atlas_brain()
        self._render_mni_electrode_sets()

        # _markers() is filtered by the current space, therefore this displays
        # only MNI markers while MNI mode is active.
        try:
            self._render_anatomical_markers()
        except Exception:
            pass

        if bool(getattr(self, "_mni_t1_slices_visible", False)):
            try:
                self._update_all_plane_slider_ranges()
                self._refresh_multiplanar_clipped_scene()
            except Exception:
                pass

        if reset_camera:
            try:
                self.plotter.reset_camera()
                self.plotter.camera.zoom(1.15)
            except Exception:
                pass

        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

        self._render()

    def _mni_group_labels_are_visible(self, mni_set: dict, group_name: str) -> bool:
        """
        True if at least one contact label is visible in this MNI electrode/group.
        Used to decide Add labels vs Remove labels.
        """
        try:
            contacts = mni_set.get("contacts", []) or []

            for ci, c in enumerate(contacts):
                if self._mni_group_name_from_contact(c) == str(group_name):
                    if bool(mni_set.get("contact_label_visible", {}).get(str(ci), False)):
                        return True

            return False

        except Exception:
            return False

    def _mni_set_labels_are_visible(self, mni_set: dict) -> bool:
        """
        True if at least one contact label is visible in this MNI patient set.
        """
        try:
            return any(bool(v) for v in (mni_set.get("contact_label_visible", {}) or {}).values())
        except Exception:
            return False

    def _mni_electrode_names_are_visible(self, mni_set: dict) -> bool:
        """
        True when electrode names are visible for the whole MNI patient set.
        This controls labels like AD, AG, Hipp, etc.
        It does NOT control contact labels like AD1, AD2.
        """
        try:
            return bool(mni_set.get("electrode_names_visible", False))
        except Exception:
            return False

    def _mni_group_color_rgb(self, mni_set: dict, group_name: str):
        """
        Return the RGB color of one MNI electrode/group.
        If the group has no specific color, fallback to patient color.
        """
        try:
            group_color = mni_set.get("group_color", {}) or {}
            if str(group_name) in group_color:
                c = group_color[str(group_name)]
                return [int(c[0]), int(c[1]), int(c[2])]
        except Exception:
            pass

        try:
            c = mni_set.get("color", (100, 180, 255))
            return [int(c[0]), int(c[1]), int(c[2])]
        except Exception:
            return [100, 180, 255]

    def _set_mni_patient_color(self, si: int) -> None:
        """
        Change color for the whole MNI patient.
        This also removes group-specific colors, so every electrode uses the same color.
        """
        try:
            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= int(si) < len(sets)):
                return

            mni_set = sets[int(si)]
            color = mni_set.get("color", (100, 180, 255))

            try:
                current = QColor(int(color[0]), int(color[1]), int(color[2]))
            except Exception:
                current = QColor(100, 180, 255)

            color_hex = NeuXelecColorDialog.get_color(
                initial_color=current,
                parent=self._dialog_parent(),
                title="Choose MNI patient color",
            )

            if color_hex is None:
                return

            qcolor = QColor(color_hex)

            if not qcolor.isValid():
                return

            mni_set["color"] = [
                int(qcolor.red()),
                int(qcolor.green()),
                int(qcolor.blue()),
            ]

            # Patient-level color should apply to all electrodes.
            # Therefore remove electrode-specific overrides.
            mni_set["group_color"] = {}

            self._update_mni_tree_patient_color(si)
            self._update_mni_patient_color_only(si)

        except Exception as e:
            print("[MNI patient color] failed:", e)

    def _update_mni_tree_patient_color(self, si: int) -> None:
        """
        Update only tree colors for one MNI patient.

        Signals must be blocked because this is a color-only update,
        not a checkbox/visibility update.
        """
        tree = None

        try:
            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
            if tree is None:
                return

            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= int(si) < len(sets)):
                return

            mni_set = sets[int(si)]
            rgb = mni_set.get("color", (255, 255, 255))
            rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            old_updating = bool(getattr(self, "_mni_tree_updating", False))
            self._mni_tree_updating = True
            tree.blockSignals(True)

            for i in range(tree.topLevelItemCount()):
                root = tree.topLevelItem(i)

                if root.data(0, Qt.UserRole + 50) != "mni_set":
                    continue

                if int(root.data(0, Qt.UserRole + 51)) != int(si):
                    continue

                self._apply_mni_tree_row_style(
                    root,
                    rgb_tuple,
                    alpha=255,
                    kind="set",
                )

                for gi in range(root.childCount()):
                    group_item = root.child(gi)

                    self._apply_mni_tree_row_style(
                        group_item,
                        rgb_tuple,
                        alpha=255,
                        kind="group",
                    )

                    for ci in range(group_item.childCount()):
                        self._apply_mni_tree_row_style(
                            group_item.child(ci),
                            rgb_tuple,
                            alpha=148,
                            kind="contact",
                        )

                try:
                    tree.viewport().update()
                except Exception:
                    pass

                return

        except Exception:
            pass

        finally:
            try:
                if tree is not None:
                    tree.blockSignals(False)
            except Exception:
                pass

            try:
                self._mni_tree_updating = old_updating
            except Exception:
                self._mni_tree_updating = False

    def _set_mni_group_color(self, si: int, group_name: str) -> None:
        """
        Change color for one MNI electrode/group only.
        """
        try:
            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= int(si) < len(sets)):
                return

            mni_set = sets[int(si)]
            self._ensure_mni_visibility_fields(mni_set)

            current_rgb = self._mni_group_color_rgb(mni_set, group_name)
            current = QColor(int(current_rgb[0]), int(current_rgb[1]), int(current_rgb[2]))

            color_hex = NeuXelecColorDialog.get_color(
                initial_color=current,
                parent=self._dialog_parent(),
                title=f"Choose color for {group_name}",
            )

            if color_hex is None:
                return

            qcolor = QColor(color_hex)

            if not qcolor.isValid():
                return

            mni_set["group_color"][str(group_name)] = [
                int(qcolor.red()),
                int(qcolor.green()),
                int(qcolor.blue()),
            ]

            self._update_mni_tree_group_color(si, group_name)

            self._update_mni_group_color_only(
                si,
                group_name,
                render=False,
            )

            try:
                self._render()
            except Exception:
                pass

        except Exception as e:
            print("[MNI electrode color] failed:", e)

    def _update_mni_tree_group_color(self, si: int, group_name: str) -> None:
        """
        Update only the QTreeWidget colors for one MNI electrode/group.

        Important:
        This is only a visual color update. It must not trigger
        _on_mni_tree_item_changed(), because that function handles visibility
        checkboxes and can rebuild MNI actors.
        """
        tree = None

        try:
            tree = self.ui.findChild(QTreeWidget, "tv_Electrodes_3")
            if tree is None:
                return

            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= int(si) < len(sets)):
                return

            mni_set = sets[int(si)]
            rgb = self._mni_group_color_rgb(mni_set, group_name)
            rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            old_updating = bool(getattr(self, "_mni_tree_updating", False))
            self._mni_tree_updating = True
            tree.blockSignals(True)

            for i in range(tree.topLevelItemCount()):
                root = tree.topLevelItem(i)

                if root.data(0, Qt.UserRole + 50) != "mni_set":
                    continue

                if int(root.data(0, Qt.UserRole + 51)) != int(si):
                    continue

                for gi in range(root.childCount()):
                    group_item = root.child(gi)

                    if str(group_item.data(0, Qt.UserRole + 52)) != str(group_name):
                        continue

                    self._apply_mni_tree_row_style(
                        group_item,
                        rgb_tuple,
                        alpha=255,
                        kind="group",
                    )

                    for ci in range(group_item.childCount()):
                        self._apply_mni_tree_row_style(
                            group_item.child(ci),
                            rgb_tuple,
                            alpha=148,
                            kind="contact",
                        )

                    try:
                        tree.viewport().update()
                    except Exception:
                        pass

                    return

        except Exception:
            pass

        finally:
            try:
                if tree is not None:
                    tree.blockSignals(False)
            except Exception:
                pass

            try:
                self._mni_tree_updating = old_updating
            except Exception:
                self._mni_tree_updating = False

    def _render_mni_patient_name_label_only(self, si: int) -> None:
        """
        Add/remove only the patient name label for one MNI patient.
        Does not rebuild electrodes or atlas.
        """
        if not _PV_OK or self.plotter is None:
            return

        try:
            si = int(si)

            old = self._mni_label_actors.pop((si, "patient", "label"), None)
            if old is not None:
                try:
                    self.plotter.remove_actor(old, reset_camera=False)
                except Exception:
                    pass

            sets = getattr(self.state, "mni_electrode_sets", []) or []
            if not (0 <= si < len(sets)):
                return

            mni_set = sets[si]
            self._ensure_mni_visibility_fields(mni_set)

            if not bool(mni_set.get("patient_name_visible", False)):
                self._render()
                return

            contacts = mni_set.get("contacts", []) or []
            pts = []

            for ci, c in enumerate(contacts):
                if not self._mni_contact_is_visible(mni_set, ci, c):
                    continue

                try:
                    x, y, z = c.get("mni_ras", [None, None, None])
                    pts.append([float(x), float(y), float(z)])
                except Exception:
                    continue

            if not pts:
                self._render()
                return

            pts = np.asarray(pts, dtype=np.float32)
            center = np.nanmean(pts, axis=0)
            center[2] += 14.0

            rgb = mni_set.get("color", (100, 180, 255))
            color = self._mni_rgb_to_float_color(rgb)

            subject = str(mni_set.get("subject", f"MNI set {si + 1}"))

            actor = self.plotter.add_point_labels(
                np.asarray([center], dtype=np.float32),
                [subject],
                font_size=14,
                text_color=color,
                shape_opacity=0.0,
                show_points=False,
                always_visible=True,
                name=f"mni_patient_label_{si}",
            )

            try:
                actor.PickableOff()
            except Exception:
                pass

            self._mni_label_actors[(si, "patient", "label")] = actor
            self._render()

        except Exception as e:
            print("[MNI patient name] failed:", e)
