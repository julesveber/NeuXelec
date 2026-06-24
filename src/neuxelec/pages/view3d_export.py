"""Screenshot and rotating-GIF export for the 3D View page.

This module isolates the export subsystem of :class:`View3DPage` as a mixin.
The methods are unchanged; they are grouped here to keep ``view3d.py`` focused
on scene construction and interaction. ``View3DPage`` inherits this mixin, so
every ``self.*`` reference resolves exactly as before.

The mixin relies on the following attributes/methods provided by the host
page: ``plotter``, ``container_3d``, ``_loading_overlay``,
``_gif_export_active``, ``_gif_export_state``, ``_gif_export_timer``,
``_block_export_if_fullscreen``, ``_dialog_parent``, ``_render``,
``_apply_camera_dict``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog

from neuxelec.ui.neuxelec_message_dialog import (
    NeuXelecMessageDialog,
    NeuXelecSelectionDialog,
)
from neuxelec.utils.qt_helpers import top_level_window as _top_level_window


class View3DExportMixin:
    """Screenshot and rotating-GIF export methods for :class:`View3DPage`."""

    def _save_3d_view_screenshot(self, filename: str, transparent_background: bool = False) -> None:
        """
        Save only the 3D visualization area, not the full application window.
        """
        if self.plotter is None:
            return

        if self._block_export_if_fullscreen("Screenshot"):
            return

        try:
            filename = str(filename)

            if transparent_background and not filename.lower().endswith(".png"):
                # Transparency only makes sense for PNG.
                filename = filename.rsplit(".", 1)[0] + ".png"

            old_bg = None
            try:
                old_bg = self.plotter.background_color
            except Exception:
                old_bg = None

            try:
                self.plotter.screenshot(
                    filename,
                    transparent_background=bool(transparent_background),
                    return_img=False,
                )
            except TypeError:
                # Compatibility fallback for older PyVista versions
                self.plotter.screenshot(filename, return_img=False)

            # Restore render explicitly
            try:
                if old_bg is not None:
                    self.plotter.set_background(old_bg)
            except Exception:
                pass

            try:
                self._render()
            except Exception:
                pass

        except Exception as e:
            NeuXelecMessageDialog.warning(
                self._dialog_parent(),
                "Screenshot",
                f"Failed to save 3D view screenshot:\n{e}",
            )

    def _rotation_axis_from_user(self) -> str:
        """
        Ask around which anatomical/display axis the rotating GIF is generated.
        Returns 'X', 'Y', or 'Z'. Empty string means cancelled.
        """
        try:
            selected_axis = NeuXelecSelectionDialog.select_item(
                self._dialog_parent(),
                "GIF rotation axis",
                "Choose the anatomical axis used to rotate the brain:",
                options=[
                    "Z axis - axial / vertical rotation",
                    "Y axis - coronal rotation",
                    "X axis - sagittal rotation",
                ],
                current_index=0,
                accept_text="Continue",
                reject_text="Cancel",
            )

            if not selected_axis:
                return ""

            normalized_axis = str(selected_axis).strip().upper()

            if normalized_axis.startswith("X"):
                return "X"

            if normalized_axis.startswith("Y"):
                return "Y"

            return "Z"

        except Exception:
            return ""

    def _camera_state_tuple(self):
        """
        Store the current camera position in a PyVista-compatible form.
        """
        try:
            cam = self.plotter.camera
            return (
                tuple(float(v) for v in cam.position),
                tuple(float(v) for v in cam.focal_point),
                tuple(float(v) for v in cam.up),
                tuple(float(v) for v in cam.clipping_range),
            )
        except Exception:
            return None

    def _restore_camera_state_tuple(self, state_tuple) -> None:
        """
        Restore a camera state created by _camera_state_tuple().
        """
        if self.plotter is None or state_tuple is None:
            return

        try:
            pos, focal, up, clip = state_tuple
            cam = self.plotter.camera
            cam.position = pos
            cam.focal_point = focal
            cam.up = up

            try:
                cam.clipping_range = clip
            except Exception:
                try:
                    self.plotter.reset_camera_clipping_range()
                except Exception:
                    pass

            self._render()
        except Exception:
            pass

    def _rotate_camera_for_gif_frame(self, axis: str, step_deg: float) -> None:
        """
        Rotate the camera around the current focal point.
        X/Y/Z are RAS/display axes.
        """
        if self.plotter is None:
            return

        try:
            cam = self.plotter.camera

            pos = np.asarray(cam.position, dtype=np.float64)
            focal = np.asarray(cam.focal_point, dtype=np.float64)
            up = np.asarray(cam.up, dtype=np.float64)

            vec = pos - focal

            axis = str(axis).upper().strip()
            if axis == "X":
                rot_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            elif axis == "Y":
                rot_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            theta = np.deg2rad(float(step_deg))
            k = rot_axis / max(np.linalg.norm(rot_axis), 1e-9)

            def _rotate(v):
                v = np.asarray(v, dtype=np.float64)
                return (
                    v * np.cos(theta)
                    + np.cross(k, v) * np.sin(theta)
                    + k * np.dot(k, v) * (1.0 - np.cos(theta))
                )

            new_vec = _rotate(vec)
            new_up = _rotate(up)

            cam.position = tuple(float(v) for v in (focal + new_vec))
            cam.up = tuple(float(v) for v in new_up)

            try:
                self.plotter.reset_camera_clipping_range()
            except Exception:
                pass

        except Exception:
            pass

    def _export_3d_view_gif(self) -> None:
        """
        Start a non-blocking rotating GIF export.

        PyVista/VTK rendering remains in the Qt main thread, but only one frame
        is generated per timer iteration. This lets the loading overlay animate
        and prevents the application from appearing frozen.
        """
        if self.plotter is None:
            return

        if self._block_export_if_fullscreen("GIF export"):
            return

        # Prevent a second export while the first one is still running.
        if bool(getattr(self, "_gif_export_active", False)):
            NeuXelecMessageDialog.information(
                self._dialog_parent(),
                "GIF export",
                "A GIF export is already in progress.",
            )
            return

        parent = self.container_3d if self.container_3d is not None else _top_level_window()

        axis = self._rotation_axis_from_user()
        if not axis:
            return

        filename, _ = QFileDialog.getSaveFileName(
            parent,
            "Export rotating 3D GIF",
            f"neuxelec_3d_rotation_{axis}.gif",
            "GIF animation (*.gif)",
        )

        if not filename:
            return

        filename = str(filename)

        if not filename.lower().endswith(".gif"):
            filename += ".gif"

        n_frames = 72

        self._gif_export_active = True
        self._gif_export_state = {
            "filename": filename,
            "axis": str(axis),
            "n_frames": int(n_frames),
            "frame_index": 0,
            "step_deg": 360.0 / float(n_frames),
            "original_camera": self._camera_state_tuple(),
            "writer_open": False,
        }

        if self._loading_overlay is not None:
            self._loading_overlay.begin("Preparing rotating 3D animation")
            self._loading_overlay.set_progress(
                0.06,
                "Opening GIF encoder",
            )

        # Let Qt paint the overlay before the first expensive render.
        QTimer.singleShot(
            0,
            self._start_3d_gif_export,
        )

    def _start_3d_gif_export(self) -> None:
        """
        Open the GIF writer and schedule the first frame.
        """
        state = getattr(self, "_gif_export_state", None)

        if not self._gif_export_active or not isinstance(state, dict):
            return

        try:
            self.plotter.open_gif(state["filename"])
            state["writer_open"] = True

            if self._loading_overlay is not None:
                self._loading_overlay.set_progress(
                    0.10,
                    "Rendering frame 1 of 72",
                )

            self._gif_export_timer.start(0)

        except Exception as e:
            self._fail_3d_gif_export(str(e))

    def _export_next_3d_gif_frame(self) -> None:
        """
        Generate one GIF frame, update the overlay, then return control to Qt.
        """
        state = getattr(self, "_gif_export_state", None)

        if not self._gif_export_active or not isinstance(state, dict):
            return

        try:
            frame_index = int(state["frame_index"])
            n_frames = int(state["n_frames"])

            # First frame uses the original camera.
            if frame_index > 0:
                self._rotate_camera_for_gif_frame(
                    state["axis"],
                    state["step_deg"],
                )

            self.plotter.render()
            self.plotter.write_frame()

            frame_index += 1
            state["frame_index"] = frame_index

            progress = 0.10 + (0.80 * float(frame_index) / float(n_frames))

            if self._loading_overlay is not None:
                self._loading_overlay.set_progress(
                    min(0.90, progress),
                    (f"Rendering frame " f"{frame_index} of {n_frames}"),
                )

            if frame_index >= n_frames:
                QTimer.singleShot(
                    0,
                    self._finish_3d_gif_export,
                )
                return

            # Schedule the following frame on the next event-loop iteration.
            self._gif_export_timer.start(0)

        except Exception as e:
            self._fail_3d_gif_export(str(e))

    def _close_3d_gif_writer(self) -> None:
        """
        Close only the GIF writer, never the PyVista plotter.
        """
        try:
            writer = getattr(self.plotter, "mwriter", None)

            if writer is not None:
                writer.close()

        except Exception:
            pass

        try:
            self.plotter.mwriter = None
        except Exception:
            pass

    def _finish_3d_gif_export(self) -> None:
        """
        Finalize a successful 3D GIF export.
        """
        state = getattr(self, "_gif_export_state", None)

        if not isinstance(state, dict):
            return

        filename = str(state.get("filename", ""))

        try:
            if self._loading_overlay is not None:
                self._loading_overlay.set_progress(
                    0.94,
                    "Finalizing GIF file",
                )

            self._close_3d_gif_writer()

            try:
                self._restore_camera_state_tuple(state.get("original_camera"))
            except Exception:
                pass

            try:
                self._render()
            except Exception:
                pass

            if self._loading_overlay is not None:
                self._loading_overlay.complete()

            NeuXelecMessageDialog.information(
                self._dialog_parent(),
                "GIF export completed",
                ("The rotating 3D GIF was exported successfully.\n\n" f"{filename}"),
            )

        except Exception as e:
            self._fail_3d_gif_export(str(e))
            return

        finally:
            self._gif_export_active = False
            self._gif_export_state = None

    def _fail_3d_gif_export(self, error_message: str) -> None:
        """
        Restore the camera and UI after a failed export.
        """
        state = getattr(self, "_gif_export_state", None)

        try:
            self._gif_export_timer.stop()
        except Exception:
            pass

        self._close_3d_gif_writer()

        if isinstance(state, dict):
            try:
                self._restore_camera_state_tuple(state.get("original_camera"))
            except Exception:
                pass

        try:
            self._render()
        except Exception:
            pass

        if self._loading_overlay is not None:
            self._loading_overlay.cancel()

        self._gif_export_active = False
        self._gif_export_state = None

        NeuXelecMessageDialog.warning(
            self._dialog_parent(),
            "GIF export failed",
            ("The rotating 3D GIF could not be exported.\n\n" f"Details:\n{error_message}"),
        )
