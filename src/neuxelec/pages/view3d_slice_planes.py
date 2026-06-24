"""Anatomical slice planes for the 3D View page.

Isolates the coronal/axial/sagittal textured slice planes shown inside the 3D
scene: geometry construction, VTK clipping between planes, slider ranges, and
the per-plane electrode overlays. Methods are unchanged. ``View3DPage``
inherits this mixin, so every ``self.*`` reference resolves exactly as before.
"""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk
import vtk

try:
    import pyvista as pv

    _PV_OK = True
except Exception:
    pv = None
    _PV_OK = False

logger = logging.getLogger(__name__)


class View3DSlicePlanesMixin:
    """Coronal/axial/sagittal slice-plane rendering for View3DPage."""

    def _build_coronal_plane_geometry(self):
        img = self._get_3d_plane_reference_img()
        if img is None:
            return None

        y_idx = self._get_coronal_slice_index()
        if y_idx is None:
            return None

        size = img.GetSize()
        x_dim, y_dim, z_dim = int(size[0]), int(size[1]), int(size[2])

        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)
        direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

        def idx_to_lps(idx_xyz: np.ndarray) -> np.ndarray:
            return origin + ((idx_xyz * spacing) @ direction.T)

        def lps_to_ras(p):
            q = np.array(p, dtype=np.float64).copy()
            q[0] *= -1.0
            q[1] *= -1.0
            return q

        x0, x1, y0, y1, z0, z1 = self._get_slice_crop_bounds_xyz(
            img,
            padding_vox=4,
        )

        p00_idx = np.array([x0, float(y_idx), z0], dtype=np.float64)
        p10_idx = np.array([x1, float(y_idx), z0], dtype=np.float64)
        p01_idx = np.array([x0, float(y_idx), z1], dtype=np.float64)
        p11_idx = np.array([x1, float(y_idx), z1], dtype=np.float64)

        p00 = lps_to_ras(idx_to_lps(p00_idx))
        p10 = lps_to_ras(idx_to_lps(p10_idx))
        p01 = lps_to_ras(idx_to_lps(p01_idx))
        p11 = lps_to_ras(idx_to_lps(p11_idx))

        i_vec = p10 - p00
        j_vec = p01 - p00

        normal = np.cross(i_vec, j_vec)
        nrm = np.linalg.norm(normal)
        if nrm > 0:
            normal = normal / nrm

        center = (p00 + p10 + p01 + p11) / 4.0

        return {
            "p00": p00,
            "p10": p10,
            "p01": p01,
            "p11": p11,
            "center": center,
            "normal": normal,
            "x_dim": x_dim,
            "y_idx": y_idx,
            "z_dim": z_dim,
            "crop_bounds_xyz": (x0, x1, y0, y1, z0, z1),
        }

    def _point_is_inside_active_plane_clips(self, pt_ras: np.ndarray, skip_plane: str) -> bool:
        """
        Return True if a RAS point is inside the visible side of the other active planes.
        skip_plane is one of: 'coronal', 'axial', 'sagittal'
        """
        p = np.asarray(pt_ras, dtype=np.float64)

        # Coronal
        if (
            skip_plane != "coronal"
            and self.chk_coronal_plane is not None
            and self.chk_coronal_plane.isChecked()
        ):
            geom = self._build_coronal_plane_geometry()
            if geom is not None:
                n = self._get_effective_plane_normal("coronal", geom)
                if np.dot(p - np.asarray(geom["center"], dtype=np.float64), n) > 0:
                    return False

        # Axial
        if (
            skip_plane != "axial"
            and self.chk_axial_plane is not None
            and self.chk_axial_plane.isChecked()
        ):
            geom = self._build_axial_plane_geometry()
            if geom is not None:
                n = self._get_effective_plane_normal("axial", geom)
                if np.dot(p - np.asarray(geom["center"], dtype=np.float64), n) > 0:
                    return False

        # Sagittal
        if (
            skip_plane != "sagittal"
            and self.chk_sagittal_plane is not None
            and self.chk_sagittal_plane.isChecked()
        ):
            geom = self._build_sagittal_plane_geometry()
            if geom is not None:
                n = self._get_effective_plane_normal("sagittal", geom)
                if np.dot(p - np.asarray(geom["center"], dtype=np.float64), n) > 0:
                    return False

        return True

    def _render_coronal_electrodes_overlay(self) -> None:
        self._remove_actor("coronal_elec")

        if bool(getattr(self, "_keep_electrodes_visible_through_slices", False)):
            return

        if self._mni_t1_slices_are_visible():
            return

        if not _PV_OK or self.plotter is None:
            return
        if self.chk_coronal_plane is None or not self.chk_coronal_plane.isChecked():
            return
        if self._t1_img is None:
            return

        geom = self._build_coronal_plane_geometry()
        if geom is None:
            return

        try:
            y_idx = float(geom["y_idx"])
            tol = 0.49
            radius_mm = self._get_overlay_contact_radius_mm()
            plane_normal = self._get_effective_plane_normal("coronal", geom)

            actors = []

            for elec_id, elec in enumerate(getattr(self.state, "electrodes", []) or []):
                if not self._get_local_electrode_visible(elec_id):
                    continue

                rgb = tuple(elec.get("color", (0, 255, 0)))
                color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

                contacts_idx = elec.get("contacts_idx", []) or []
                contacts_visible = self._get_local_contacts_visible(elec_id, len(contacts_idx))

                contacts_lps = elec.get("contacts_lps", []) or []
                meshes = []

                for ci, idx in enumerate(contacts_idx):
                    if not bool(contacts_visible[ci]):
                        continue
                    try:
                        ix, iy, iz = idx
                        if abs(float(iy) - y_idx) <= tol:
                            ras = np.array(contacts_lps[ci], dtype=np.float64)
                            ras[0] *= -1.0
                            ras[1] *= -1.0

                            if not self._point_is_inside_active_plane_clips(
                                ras, skip_plane="coronal"
                            ):
                                continue

                            ras_on_plane = self._project_point_to_plane(
                                ras,
                                np.asarray(geom["center"], dtype=np.float64),
                                plane_normal,
                            )

                            disc = self._build_contact_disc_mesh(
                                ras_on_plane,
                                -plane_normal,
                                radius_mm=radius_mm,
                                offset_mm=0.1,
                            )
                            if disc is not None and disc.n_points > 0:
                                meshes.append(disc)
                    except Exception:
                        continue

                if not meshes:
                    continue

                try:
                    merged = meshes[0]
                    for m in meshes[1:]:
                        merged = merged.merge(m)
                except Exception:
                    continue

                actor = self.plotter.add_mesh(
                    merged,
                    color=color,
                    opacity=1.0,
                    smooth_shading=True,
                )
                actors.append(actor)

            if actors:
                self._coronal_elec_actor = actors

        except Exception:
            self._remove_actor("coronal_elec")

    def _update_coronal_slider_range(self) -> None:
        img = self._get_3d_plane_reference_img()
        if img is None or self.sld_coronal_plane is None:
            return
        try:
            size = img.GetSize()
            full_y_dim = int(size[1])

            y_min = 0
            y_max = full_y_dim - 1

            active_mask_t1 = self._get_3d_plane_mask_for_slices(img)
            if active_mask_t1 is not None:
                mask_np = sitk.GetArrayFromImage(active_mask_t1).astype(np.uint8)  # z, y, x

                # On cherche les indices y où il existe au moins un voxel du cerveau
                ys = np.where(np.any(mask_np > 0, axis=(0, 2)))[0]

                if ys.size > 0:
                    y_min = int(ys.min())
                    y_max = int(ys.max())

            self._coronal_y_min = y_min
            self._coronal_y_max = y_max

            slider_max = max(0, y_max - y_min)

            current_val = self.sld_coronal_plane.value()
            self.sld_coronal_plane.blockSignals(True)
            self.sld_coronal_plane.setMinimum(0)
            self.sld_coronal_plane.setMaximum(slider_max)

            if current_val < 0 or current_val > slider_max:
                self.sld_coronal_plane.setValue(slider_max // 2)
            else:
                self.sld_coronal_plane.setValue(current_val)

            self.sld_coronal_plane.blockSignals(False)

        except Exception:
            # fallback : volume complet
            try:
                size = img.GetSize()
                full_y_dim = int(size[1])

                self._coronal_y_min = 0
                self._coronal_y_max = full_y_dim - 1

                self.sld_coronal_plane.blockSignals(True)
                self.sld_coronal_plane.setMinimum(0)
                self.sld_coronal_plane.setMaximum(max(0, full_y_dim - 1))
                self.sld_coronal_plane.setValue(full_y_dim // 2)
                self.sld_coronal_plane.blockSignals(False)
            except Exception:
                pass

    def _refresh_coronal_clipped_scene(self) -> None:
        self._refresh_multiplanar_clipped_scene()

    def _refresh_single_plane_full(self, which: str) -> None:
        """
        Refresh the moved slice content only.

        Colored outlines are rebuilt together afterwards by
        _refresh_all_visible_plane_outlines(), once the full clipping state
        is up to date.
        """
        which = str(which).lower().strip()

        try:
            if which == "coronal":
                self._render_coronal_plane()
                self._render_coronal_pet_overlay()
                self._render_coronal_siscom_overlay()
                self._render_coronal_electrodes_overlay()

            elif which == "axial":
                self._render_axial_plane()
                self._render_axial_pet_overlay()
                self._render_axial_siscom_overlay()
                self._render_axial_electrodes_overlay()

            elif which == "sagittal":
                self._render_sagittal_plane()
                self._render_sagittal_pet_overlay()
                self._render_sagittal_siscom_overlay()
                self._render_sagittal_electrodes_overlay()

        except Exception:
            pass

    def _reclip_other_planes(self, changed_plane: str) -> None:
        changed_plane = str(changed_plane).lower().strip()

        try:
            if changed_plane != "coronal":
                if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
                    self._render_coronal_plane()
                    self._render_coronal_pet_overlay()
                    self._render_coronal_siscom_overlay()
                    self._render_coronal_outline()
                    self._render_coronal_electrodes_overlay()
        except Exception:
            pass

        try:
            if changed_plane != "axial":
                if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
                    self._render_axial_plane()
                    self._render_axial_pet_overlay()
                    self._render_axial_siscom_overlay()
                    self._render_axial_outline()
                    self._render_axial_electrodes_overlay()
        except Exception:
            pass

        try:
            if changed_plane != "sagittal":
                if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
                    self._render_sagittal_plane()
                    self._render_sagittal_pet_overlay()
                    self._render_sagittal_siscom_overlay()
                    self._render_sagittal_outline()
                    self._render_sagittal_electrodes_overlay()
        except Exception:
            pass

    def _on_single_plane_changed(self, which: str) -> None:
        try:
            if (
                self.chk_mni_atlas is not None
                and self.chk_mni_atlas.isChecked()
                and not bool(getattr(self, "_mni_t1_slices_visible", False))
            ):
                self._mni_t1_slices_visible = True
                self._invalidate_slice_volume_cache(
                    base=True,
                    pet=True,
                    siscom=True,
                )
                self._update_all_plane_slider_ranges()
        except Exception:
            pass
        if bool(getattr(self, "_view3d_is_fullscreen", False)):
            self._exit_3d_fullscreen()
        """
        Refresh one moved slice and update every dependent clipped frame.

        When one plane moves, its position modifies the visible boundary of the
        two orthogonal planes. Therefore all visible outlines must be rebuilt.
        """
        which = str(which).lower().strip()

        try:
            self._apply_actor_clipping()
        except Exception:
            pass

        # Rebuild the moved plane itself: texture, overlays, outline and contacts.
        try:
            self._refresh_single_plane_full(which)
        except Exception:
            pass

        # Re-clip the two other anatomical/PET/SISCOM planes against the moved plane.
        try:
            self._reclip_existing_plane_actors_only(which)
        except Exception:
            pass

        # Important: redraw ALL visible outlines after all current plane
        # positions have been taken into account.
        try:
            self._refresh_all_visible_plane_outlines()
        except Exception:
            pass

        try:
            self._update_planes_info_label()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def _refresh_all_visible_slice_planes_full(self) -> None:
        """
        Rebuild the full texture of every currently visible anatomical slice.

        This is needed when parcellation visibility/opacity changes:
        the slice position does not move, but its texture must be rebuilt.
        """
        try:
            if self.chk_coronal_plane is not None and self.chk_coronal_plane.isChecked():
                self._refresh_single_plane_full("coronal")
        except Exception:
            pass

        try:
            if self.chk_axial_plane is not None and self.chk_axial_plane.isChecked():
                self._refresh_single_plane_full("axial")
        except Exception:
            pass

        try:
            if self.chk_sagittal_plane is not None and self.chk_sagittal_plane.isChecked():
                self._refresh_single_plane_full("sagittal")
        except Exception:
            pass

        try:
            self._refresh_all_visible_plane_outlines()
        except Exception:
            pass

        try:
            self._update_planes_info_label()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass

    def _refresh_multiplanar_clipped_scene(self) -> None:
        try:
            self._apply_actor_clipping()
        except Exception:
            pass

        try:
            self._refresh_all_visible_plane_outlines()
        except Exception:
            pass

        try:
            self._render_coronal_pet_overlay()
        except Exception:
            pass
        try:
            self._render_axial_pet_overlay()
        except Exception:
            pass
        try:
            self._render_sagittal_pet_overlay()
        except Exception:
            pass

        try:
            self._render_coronal_siscom_overlay()
        except Exception:
            pass
        try:
            self._render_axial_siscom_overlay()
        except Exception:
            pass
        try:
            self._render_sagittal_siscom_overlay()
        except Exception:
            pass
        try:
            self._render_coronal_outline()
        except Exception:
            pass

        try:
            self._render_axial_outline()
        except Exception:
            pass

        try:
            self._render_sagittal_outline()
        except Exception:
            pass

        try:
            self._render_coronal_electrodes_overlay()
        except Exception:
            pass

        try:
            self._render_axial_electrodes_overlay()
        except Exception:
            pass

        try:
            self._render_sagittal_electrodes_overlay()
        except Exception:
            pass

        try:
            self._render()
        except Exception:
            pass
        try:
            self._update_planes_info_label()
        except Exception:
            pass

    def _update_all_plane_slider_ranges(self) -> None:
        self._update_coronal_slider_range()
        self._update_axial_slider_range()
        self._update_sagittal_slider_range()

    def _build_vtk_axial_clip_plane(self):
        geom = self._build_axial_plane_geometry()
        if geom is None:
            return None

        plane = vtk.vtkPlane()
        n = self._get_effective_plane_normal("axial", geom)

        plane.SetOrigin(*geom["center"])
        plane.SetNormal(*n)
        return plane

    def _build_vtk_sagittal_clip_plane(self):
        geom = self._build_sagittal_plane_geometry()
        if geom is None:
            return None

        plane = vtk.vtkPlane()
        n = self._get_effective_plane_normal("sagittal", geom)

        plane.SetOrigin(*geom["center"])
        plane.SetNormal(*n)
        return plane

    def _get_axial_slice_index(self) -> int | None:
        img = self._get_3d_plane_reference_img()
        if img is None or self.sld_axial_plane is None:
            return None

        try:
            size = img.GetSize()
            full_z_dim = int(size[2])

            z_min = int(self._axial_z_min) if self._axial_z_min is not None else 0
            z_max = int(self._axial_z_max) if self._axial_z_max is not None else (full_z_dim - 1)

            if z_max < z_min:
                z_min, z_max = 0, full_z_dim - 1

            slider_max = max(0, z_max - z_min)
            slider_val = int(np.clip(self.sld_axial_plane.value(), 0, slider_max))

            if self._axial_from_inferior:
                z_idx = z_min + slider_val
            else:
                z_idx = z_max - slider_val

            return int(np.clip(z_idx, 0, full_z_dim - 1))
        except Exception:
            return None

    def _get_sagittal_slice_index(self) -> int | None:
        img = self._get_3d_plane_reference_img()
        if img is None or self.sld_sagittal_plane is None:
            return None

        try:
            size = img.GetSize()
            full_x_dim = int(size[0])

            x_min = int(self._sagittal_x_min) if self._sagittal_x_min is not None else 0
            x_max = (
                int(self._sagittal_x_max) if self._sagittal_x_max is not None else (full_x_dim - 1)
            )

            if x_max < x_min:
                x_min, x_max = 0, full_x_dim - 1

            slider_max = max(0, x_max - x_min)
            slider_val = int(np.clip(self.sld_sagittal_plane.value(), 0, slider_max))

            if self._sagittal_from_left:
                x_idx = x_min + slider_val
            else:
                x_idx = x_max - slider_val

            return int(np.clip(x_idx, 0, full_x_dim - 1))
        except Exception:
            return None

    def _build_axial_plane_geometry(self):
        img = self._get_3d_plane_reference_img()
        if img is None:
            return None
        z_idx = self._get_axial_slice_index()
        if z_idx is None:
            return None

        size = img.GetSize()
        x_dim, y_dim, z_dim = int(size[0]), int(size[1]), int(size[2])
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)
        direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

        def idx_to_lps(idx_xyz: np.ndarray) -> np.ndarray:
            return origin + ((idx_xyz * spacing) @ direction.T)

        def lps_to_ras(p):
            q = np.array(p, dtype=np.float64).copy()
            q[0] *= -1.0
            q[1] *= -1.0
            return q

        x0, x1, y0, y1, z0, z1 = self._get_slice_crop_bounds_xyz(
            img,
            padding_vox=4,
        )

        p00_idx = np.array([float(x0), float(y0), float(z_idx)], dtype=np.float64)
        p10_idx = np.array([float(x1), float(y0), float(z_idx)], dtype=np.float64)
        p01_idx = np.array([float(x0), float(y1), float(z_idx)], dtype=np.float64)
        p11_idx = np.array([float(x1), float(y1), float(z_idx)], dtype=np.float64)

        p00 = lps_to_ras(idx_to_lps(p00_idx))
        p10 = lps_to_ras(idx_to_lps(p10_idx))
        p01 = lps_to_ras(idx_to_lps(p01_idx))
        p11 = lps_to_ras(idx_to_lps(p11_idx))

        i_vec = p10 - p00
        j_vec = p01 - p00
        normal = np.cross(i_vec, j_vec)
        nrm = np.linalg.norm(normal)
        if nrm > 0:
            normal = normal / nrm
        center = (p00 + p10 + p01 + p11) / 4.0

        return {
            "p00": p00,
            "p10": p10,
            "p01": p01,
            "p11": p11,
            "center": center,
            "normal": normal,
            "z_idx": z_idx,
            "crop_bounds_xyz": (x0, x1, y0, y1, z0, z1),
        }

    def _build_sagittal_plane_geometry(self):
        img = self._get_3d_plane_reference_img()
        if img is None:
            return None
        x_idx = self._get_sagittal_slice_index()
        if x_idx is None:
            return None

        size = img.GetSize()
        x_dim, y_dim, z_dim = int(size[0]), int(size[1]), int(size[2])
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)
        direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

        def idx_to_lps(idx_xyz: np.ndarray) -> np.ndarray:
            return origin + ((idx_xyz * spacing) @ direction.T)

        def lps_to_ras(p):
            q = np.array(p, dtype=np.float64).copy()
            q[0] *= -1.0
            q[1] *= -1.0
            return q

        x0, x1, y0, y1, z0, z1 = self._get_slice_crop_bounds_xyz(
            img,
            padding_vox=4,
        )

        p00_idx = np.array([float(x_idx), float(y0), float(z0)], dtype=np.float64)
        p10_idx = np.array([float(x_idx), float(y1), float(z0)], dtype=np.float64)
        p01_idx = np.array([float(x_idx), float(y0), float(z1)], dtype=np.float64)
        p11_idx = np.array([float(x_idx), float(y1), float(z1)], dtype=np.float64)

        p00 = lps_to_ras(idx_to_lps(p00_idx))
        p10 = lps_to_ras(idx_to_lps(p10_idx))
        p01 = lps_to_ras(idx_to_lps(p01_idx))
        p11 = lps_to_ras(idx_to_lps(p11_idx))

        i_vec = p10 - p00
        j_vec = p01 - p00
        normal = np.cross(i_vec, j_vec)
        nrm = np.linalg.norm(normal)
        if nrm > 0:
            normal = normal / nrm
        center = (p00 + p10 + p01 + p11) / 4.0

        return {
            "p00": p00,
            "p10": p10,
            "p01": p01,
            "p11": p11,
            "center": center,
            "normal": normal,
            "x_idx": x_idx,
            "crop_bounds_xyz": (x0, x1, y0, y1, z0, z1),
        }

    def _update_all_planes(self):
        try:
            self._refresh_multiplanar_clipped_scene()
        except Exception:
            pass

        try:
            self._update_planes_info_label()
        except Exception:
            pass

    def _update_axial_slider_range(self) -> None:
        img = self._get_3d_plane_reference_img()
        if img is None or self.sld_axial_plane is None:
            return
        try:
            size = img.GetSize()
            full_z_dim = int(size[2])
            z_min, z_max = 0, full_z_dim - 1
            active_mask_t1 = self._get_3d_plane_mask_for_slices(img)
            if active_mask_t1 is not None:
                mask_np = sitk.GetArrayFromImage(active_mask_t1).astype(np.uint8)
                zs = np.where(np.any(mask_np > 0, axis=(1, 2)))[0]
                if zs.size > 0:
                    z_min = int(zs.min())
                    z_max = int(zs.max())
            self._axial_z_min = z_min
            self._axial_z_max = z_max
            slider_max = max(0, z_max - z_min)
            current_val = self.sld_axial_plane.value()
            self.sld_axial_plane.blockSignals(True)
            self.sld_axial_plane.setMinimum(0)
            self.sld_axial_plane.setMaximum(slider_max)
            self.sld_axial_plane.setValue(current_val if 0 <= current_val <= slider_max else 0)
            self.sld_axial_plane.blockSignals(False)
        except Exception:
            pass

    def _update_sagittal_slider_range(self) -> None:
        img = self._get_3d_plane_reference_img()
        if img is None or self.sld_sagittal_plane is None:
            return
        try:
            size = img.GetSize()
            full_x_dim = int(size[0])
            x_min, x_max = 0, full_x_dim - 1
            active_mask_t1 = self._get_3d_plane_mask_for_slices(img)
            if active_mask_t1 is not None:
                mask_np = sitk.GetArrayFromImage(active_mask_t1).astype(np.uint8)
                xs = np.where(np.any(mask_np > 0, axis=(0, 1)))[0]
                if xs.size > 0:
                    x_min = int(xs.min())
                    x_max = int(xs.max())
            self._sagittal_x_min = x_min
            self._sagittal_x_max = x_max
            slider_max = max(0, x_max - x_min)
            current_val = self.sld_sagittal_plane.value()
            self.sld_sagittal_plane.blockSignals(True)
            self.sld_sagittal_plane.setMinimum(0)
            self.sld_sagittal_plane.setMaximum(slider_max)
            self.sld_sagittal_plane.setValue(
                current_val if 0 <= current_val <= slider_max else slider_max // 2
            )
            self.sld_sagittal_plane.blockSignals(False)
        except Exception:
            pass

    def _render_axial_plane(self) -> None:
        self._remove_actor("axial_plane")

        if not _PV_OK or self.plotter is None:
            return

        show = bool(self.chk_axial_plane.isChecked()) if self.chk_axial_plane is not None else False
        if not show or self._get_3d_plane_reference_img() is None or self.sld_axial_plane is None:
            self._render()
            return

        try:
            import pyvista as pv

            geom = self._build_axial_plane_geometry()
            if geom is None:
                self._render()
                return

            rgba = self._build_plane_rgba_highres(geom)
            if rgba is None:
                self._render()
                return

            texture = pv.numpy_to_texture(rgba)

            source_quad = self._build_textured_plane_mesh(geom, "axial_plane")
            if source_quad is None or source_quad.n_points == 0:
                self._render()
                return

            self._axial_plane_source_mesh = source_quad.copy()

            quad = self._clip_plane_mesh_with_other_planes("axial", source_quad.copy())
            if quad is None or quad.n_points == 0:
                self._render()
                return

            self._axial_plane_actor = self.plotter.add_mesh(
                quad,
                texture=texture,
                opacity=1.0,
                lighting=False,
                show_scalar_bar=False,
                name="axial_plane",
            )
            try:
                prop = self._axial_plane_actor.GetProperty()
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
            except Exception:
                pass

            self._render()
        except Exception:
            self._remove_actor("axial_plane")
            self._render()

    def _render_sagittal_plane(self) -> None:
        self._remove_actor("sagittal_plane")

        if not _PV_OK or self.plotter is None:
            return

        show = (
            bool(self.chk_sagittal_plane.isChecked())
            if self.chk_sagittal_plane is not None
            else False
        )
        if (
            not show
            or self._get_3d_plane_reference_img() is None
            or self.sld_sagittal_plane is None
        ):
            self._render()
            return

        try:
            import pyvista as pv

            geom = self._build_sagittal_plane_geometry()
            if geom is None:
                self._render()
                return

            rgba = self._build_plane_rgba_highres(geom)
            if rgba is None:
                self._render()
                return

            texture = pv.numpy_to_texture(rgba)

            source_quad = self._build_textured_plane_mesh(geom, "sagittal_plane")
            if source_quad is None or source_quad.n_points == 0:
                self._render()
                return

            self._sagittal_plane_source_mesh = source_quad.copy()

            quad = self._clip_plane_mesh_with_other_planes("sagittal", source_quad.copy())
            if quad is None or quad.n_points == 0:
                self._render()
                return

            self._sagittal_plane_actor = self.plotter.add_mesh(
                quad,
                texture=texture,
                opacity=1.0,
                lighting=False,
                show_scalar_bar=False,
                name="sagittal_plane",
            )
            try:
                prop = self._sagittal_plane_actor.GetProperty()
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetSpecular(0.0)
            except Exception:
                pass

            self._render()
        except Exception:
            self._remove_actor("sagittal_plane")
            self._render()

    def _render_axial_electrodes_overlay(self) -> None:
        self._remove_actor("axial_elec")

        if bool(getattr(self, "_keep_electrodes_visible_through_slices", False)):
            return

        if self._mni_t1_slices_are_visible():
            return
        if not _PV_OK or self.plotter is None:
            return
        if self.chk_axial_plane is None or not self.chk_axial_plane.isChecked():
            return
        if self._t1_img is None:
            return

        geom = self._build_axial_plane_geometry()
        if geom is None:
            return

        try:
            z_idx = float(geom["z_idx"])
            tol = 0.49
            radius_mm = self._get_overlay_contact_radius_mm()
            plane_normal = self._get_effective_plane_normal("axial", geom)

            actors = []

            for elec_id, elec in enumerate(getattr(self.state, "electrodes", []) or []):
                if not self._get_local_electrode_visible(elec_id):
                    continue

                rgb = tuple(elec.get("color", (0, 255, 0)))
                color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

                contacts_idx = elec.get("contacts_idx", []) or []
                contacts_visible = self._get_local_contacts_visible(elec_id, len(contacts_idx))

                contacts_lps = elec.get("contacts_lps", []) or []
                meshes = []

                for ci, idx in enumerate(contacts_idx):
                    if not bool(contacts_visible[ci]):
                        continue
                    try:
                        ix, iy, iz = idx
                        if abs(float(iz) - z_idx) <= tol:
                            ras = np.array(contacts_lps[ci], dtype=np.float64)
                            ras[0] *= -1.0
                            ras[1] *= -1.0

                            if not self._point_is_inside_active_plane_clips(
                                ras, skip_plane="axial"
                            ):
                                continue

                            ras_on_plane = self._project_point_to_plane(
                                ras,
                                np.asarray(geom["center"], dtype=np.float64),
                                plane_normal,
                            )

                            disc = self._build_contact_disc_mesh(
                                ras_on_plane,
                                -plane_normal,
                                radius_mm=radius_mm,
                                offset_mm=0.1,
                            )
                            if disc is not None and disc.n_points > 0:
                                meshes.append(disc)
                    except Exception:
                        continue

                if not meshes:
                    continue

                try:
                    merged = meshes[0]
                    for m in meshes[1:]:
                        merged = merged.merge(m)
                except Exception:
                    continue

                actor = self.plotter.add_mesh(
                    merged,
                    color=color,
                    opacity=1.0,
                    smooth_shading=True,
                )
                actors.append(actor)

            if actors:
                self._axial_elec_actor = actors

        except Exception:
            self._remove_actor("axial_elec")

    def _render_sagittal_electrodes_overlay(self) -> None:
        self._remove_actor("sagittal_elec")

        if bool(getattr(self, "_keep_electrodes_visible_through_slices", False)):
            return

        if self._mni_t1_slices_are_visible():
            return
        if not _PV_OK or self.plotter is None:
            return
        if self.chk_sagittal_plane is None or not self.chk_sagittal_plane.isChecked():
            return
        if self._t1_img is None:
            return

        geom = self._build_sagittal_plane_geometry()
        if geom is None:
            return

        try:
            x_idx = float(geom["x_idx"])
            tol = 0.49
            radius_mm = self._get_overlay_contact_radius_mm()
            plane_normal = self._get_effective_plane_normal("sagittal", geom)

            actors = []

            for elec_id, elec in enumerate(getattr(self.state, "electrodes", []) or []):
                if not self._get_local_electrode_visible(elec_id):
                    continue

                rgb = tuple(elec.get("color", (0, 255, 0)))
                color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

                contacts_idx = elec.get("contacts_idx", []) or []
                contacts_visible = self._get_local_contacts_visible(elec_id, len(contacts_idx))

                contacts_lps = elec.get("contacts_lps", []) or []
                meshes = []

                for ci, idx in enumerate(contacts_idx):
                    if not bool(contacts_visible[ci]):
                        continue
                    try:
                        ix, iy, iz = idx
                        if abs(float(ix) - x_idx) <= tol:
                            ras = np.array(contacts_lps[ci], dtype=np.float64)
                            ras[0] *= -1.0
                            ras[1] *= -1.0

                            if not self._point_is_inside_active_plane_clips(
                                ras, skip_plane="sagittal"
                            ):
                                continue

                            ras_on_plane = self._project_point_to_plane(
                                ras,
                                np.asarray(geom["center"], dtype=np.float64),
                                plane_normal,
                            )

                            disc = self._build_contact_disc_mesh(
                                ras_on_plane,
                                -plane_normal,
                                radius_mm=radius_mm,
                                offset_mm=0.1,
                            )
                            if disc is not None and disc.n_points > 0:
                                meshes.append(disc)
                    except Exception:
                        continue

                if not meshes:
                    continue

                try:
                    merged = meshes[0]
                    for m in meshes[1:]:
                        merged = merged.merge(m)
                except Exception:
                    continue

                actor = self.plotter.add_mesh(
                    merged,
                    color=color,
                    opacity=1.0,
                    smooth_shading=True,
                )
                actors.append(actor)

            if actors:
                self._sagittal_elec_actor = actors

        except Exception:
            self._remove_actor("sagittal_elec")
