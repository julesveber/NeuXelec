"""Camera control and standard views for the 3D View page.

Isolates camera helpers (scene framing, quick views, save/restore camera to
the project, reset-to-plane / front / sagittal / axial views) of
:class:`View3DPage` as a mixin. Methods are unchanged. ``View3DPage`` inherits
this mixin, so every ``self.*`` reference resolves exactly as before.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class View3DCameraMixin:
    """Camera framing, quick views and saved-camera handling for View3DPage."""

    def _get_camera_scene_center_and_distance(self):
        """
        Compute a stable camera center and distance from the visible brain actor if possible.
        Fallback to world origin.
        """
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        dist = 400.0

        try:
            actor = getattr(self, "_brain_actor", None)
            if actor is not None:
                bounds = actor.GetBounds()  # xmin, xmax, ymin, ymax, zmin, zmax
                if bounds is not None and len(bounds) == 6:
                    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
                    center = np.array(
                        [
                            0.5 * (xmin + xmax),
                            0.5 * (ymin + ymax),
                            0.5 * (zmin + zmax),
                        ],
                        dtype=np.float64,
                    )

                    dx = xmax - xmin
                    dy = ymax - ymin
                    dz = zmax - zmin
                    dist = max(dx, dy, dz) * 2.4

                    if not np.isfinite(dist) or dist <= 1:
                        dist = 400.0
        except Exception:
            pass

        return center, float(dist)

    def _set_camera_quick_view(self, view_name: str) -> None:
        """
        Fixed camera presets for reproducible screenshots.
        RAS display convention:
        +X = right, +Y = anterior, +Z = superior.
        """
        if self.plotter is None:
            return

        try:
            view_name = str(view_name).lower().strip()
            center, dist = self._get_camera_scene_center_and_distance()

            if view_name == "front":
                pos = center + np.array([0.0, dist, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "back":
                pos = center + np.array([0.0, -dist, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "left":
                pos = center + np.array([-dist, 0.0, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "right":
                pos = center + np.array([dist, 0.0, 0.0])
                up = (0.0, 0.0, 1.0)

            elif view_name == "top":
                pos = center + np.array([0.0, 0.0, dist])
                up = (0.0, 1.0, 0.0)

            elif view_name == "beauty_left":
                # Front-left-superior oblique view.
                pos = center + np.array([-0.65 * dist, 0.95 * dist, 0.35 * dist])
                up = (0.0, 0.0, 1.0)

            elif view_name == "beauty_right":
                # Front-right-superior oblique view.
                pos = center + np.array([0.65 * dist, 0.95 * dist, 0.35 * dist])
                up = (0.0, 0.0, 1.0)

            else:
                return

            cam = self.plotter.camera
            cam.position = tuple(float(v) for v in pos)
            cam.focal_point = tuple(float(v) for v in center)
            cam.up = up

            try:
                self.plotter.reset_camera_clipping_range()
            except Exception:
                pass

            self._render()

        except Exception:
            pass

    def _current_camera_dict(self):
        if self.plotter is None:
            return None

        try:
            cam = self.plotter.camera
            return {
                "position": [float(v) for v in cam.position],
                "focal_point": [float(v) for v in cam.focal_point],
                "up": [float(v) for v in cam.up],
                "clipping_range": [float(v) for v in cam.clipping_range],
            }
        except Exception:
            return None

    def _apply_camera_dict(self, camera_dict) -> bool:
        if self.plotter is None or not isinstance(camera_dict, dict):
            return False

        try:
            pos = camera_dict.get("position")
            focal = camera_dict.get("focal_point")
            up = camera_dict.get("up")
            clip = camera_dict.get("clipping_range")

            if not (isinstance(pos, list) and isinstance(focal, list) and isinstance(up, list)):
                return False
            if len(pos) != 3 or len(focal) != 3 or len(up) != 3:
                return False

            cam = self.plotter.camera
            cam.position = tuple(float(v) for v in pos)
            cam.focal_point = tuple(float(v) for v in focal)
            cam.up = tuple(float(v) for v in up)

            if isinstance(clip, list) and len(clip) == 2:
                try:
                    cam.clipping_range = tuple(float(v) for v in clip)
                except Exception:
                    pass
            else:
                try:
                    self.plotter.reset_camera_clipping_range()
                except Exception:
                    pass

            self._render()
            return True

        except Exception:
            return False

    def _save_current_3d_camera_to_state(self) -> None:
        cam = self._current_camera_dict()
        if cam is None:
            return

        try:
            self.state.view3d_saved_camera = cam
        except Exception:
            pass

        # Save immediately in the project JSON if available.
        try:
            project_path = getattr(self.state, "project_path", None)
            if project_path:
                from neuxelec.project_io import save_project_json

                save_project_json(self.state, project_path)
        except Exception as e:
            print("[3D View] Could not save camera to project JSON:", e)

    def _apply_saved_3d_camera_from_state(self) -> None:
        try:
            cam = getattr(self.state, "view3d_saved_camera", None)
            if cam is not None:
                self._apply_camera_dict(cam)
        except Exception:
            pass

    def _reset_camera_to_active_plane_view(self) -> None:
        plane = self._active_single_plane_name()

        if plane == "sagittal":
            self._reset_camera_sagittal_view()
        elif plane == "axial":
            self._reset_camera_axial_view()
        else:
            self._reset_camera_front_view()

    def _reset_camera_front_view(self) -> None:
        if self.plotter is None:
            return

        try:
            self.plotter.reset_camera()
            cam = self.plotter.camera

            # RAS convention:
            # +Y = Anterior, -Y = Posterior
            # If _coronal_from_caudal is True, view from the opposite side.
            if bool(getattr(self, "_coronal_from_caudal", False)):
                cam.position = (0.0, -400.0, 0.0)
            else:
                cam.position = (0.0, 400.0, 0.0)

            cam.focal_point = (0.0, 0.0, 0.0)
            cam.up = (0.0, 0.0, 1.0)

            self.plotter.reset_camera()
            self.plotter.camera.zoom(1.2)
            self._render()

        except Exception:
            pass

    def _reset_camera_sagittal_view(self) -> None:
        if self.plotter is None:
            return

        try:
            self.plotter.reset_camera()
            cam = self.plotter.camera

            # RAS convention:
            # +X = Right, -X = Left
            # If _sagittal_from_left is True, view from left side.
            if bool(getattr(self, "_sagittal_from_left", False)):
                cam.position = (-400.0, 0.0, 0.0)
            else:
                cam.position = (400.0, 0.0, 0.0)

            cam.focal_point = (0.0, 0.0, 0.0)
            cam.up = (0.0, 0.0, 1.0)

            self.plotter.reset_camera()
            self.plotter.camera.zoom(1.2)
            self._render()

        except Exception:
            pass

    def _reset_camera_axial_view(self) -> None:
        if self.plotter is None:
            return

        try:
            self.plotter.reset_camera()
            cam = self.plotter.camera

            # RAS convention:
            # +Z = Superior, -Z = Inferior
            # If _axial_from_inferior is True, view from bottom.
            if bool(getattr(self, "_axial_from_inferior", False)):
                cam.position = (0.0, 0.0, -400.0)
                cam.up = (0.0, 1.0, 0.0)
            else:
                cam.position = (0.0, 0.0, 400.0)
                cam.up = (0.0, 1.0, 0.0)

            cam.focal_point = (0.0, 0.0, 0.0)

            self.plotter.reset_camera()
            self.plotter.camera.zoom(1.2)
            self._render()

        except Exception:
            pass
