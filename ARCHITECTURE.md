# NeuXelec - Architecture

This document describes how NeuXelec is organized, for reviewers and
contributors who need to read, verify or extend the code.

## High-level design

NeuXelec is a single-window PySide6 (Qt) desktop application. The UI is built
from a Qt Designer file (`resources/ui/MainWindow.ui`) and organized as a
`QStackedWidget` of four **pages**, one per workflow step. All pages share a
single application state object.

```
                         ┌────────────────────────┐
                         │        AppState        │  single source of truth
                         │  (state.py)            │  paths, images, electrodes,
                         └───────────┬────────────┘  validation flags, markers
                                     │ read / write
        ┌───────────────┬───────────┼───────────────┬───────────────┐
        │               │           │               │               │
   FilesPage      ReconstructionPage  ObliqueSlicePage   View3DPage   controllers/
 (coreg, SISCOM)   (electrode recon)   (reslice, parcel)  (3D, MNI)   (electrodes…)
        │               │           │               │
        └───────────────┴───────────┴───────────────┘
                         │ long-running tasks
                         ▼
                   workers/  (QThread: coregistration, brain mask, SISCOM)
                         │ calls
                         ▼
              coregistration.py / siscom.py / utils/  →  ANTs, SimpleITK, VTK
```

## Key components

### `state.py` - `AppState`
The single source of truth for a loaded patient: image paths, SimpleITK image
objects, coregistration results, electrodes, markers, MNI transforms and
validation flags. Pages read from and write to it; they never own patient data
directly. A lightweight observer (`register_electrodes_changed` /
`notify_electrodes_changed`) lets the UI refresh when electrodes change.

### `app.py` - application entry point
Sets up logging, the Windows taskbar identity/icon, then runs a
**create/open-project loop**: show the startup dialog, build a
`NeuxelecWindow`, run the Qt event loop, and - when the user returns to the
menu - fully destroy the window before building the next one. This explicit
teardown is what keeps the VTK/OpenGL contexts from leaking between projects.

### `main_window.py` - `NeuxelecWindow`
Owns the four pages and the page lifecycle. Its `closeEvent` performs an
ordered, safe teardown of the VTK render windows (see *3D teardown* below).

### `pages/` - workflow pages
- `files_page.py` - import, coregistration, brain mask, SISCOM.
- `reconstruction_page.py` - two-point electrode reconstruction.
- `oblique_slice_page.py` - oblique reslicing, parcellation overlay, 3D preview.
- `view3d.py` - main interactive 3D scene, MNI mode, markers, export.

### `workers/` - background threads
`QThread` subclasses (coregistration, brain mask, SISCOM) communicate with the
UI through Qt signals (`progress`, `finished_ok`, `failed`). Heavy computation
never runs on the UI thread.

### `project_io.py` - persistence
A project is a single portable `.json` file with a `schema_version` field for
forward migration. `build_project_dict_from_state` / `apply_project_dict_to_state`
are the serialization boundary; everything that must survive a save lives there.

### `logging_config.py` - diagnostics
Rotating file logging to `%LOCALAPPDATA%/NeuXelec/logs`. Because the packaged
app is windowed (no console), this log is the primary diagnostic artifact.

## 3D teardown (VTK / OpenGL)

VTK render windows hold native OpenGL contexts tied to a window handle.
Destroying the Qt widget before VTK releases its context causes
`wglMakeCurrent` failures and corrupts the next project's 3D views. NeuXelec
therefore, in `NeuxelecWindow.closeEvent`, **closes the QtInteractors first**
(`QtInteractor.close()` finalizes the render window and marks it closed so no
later repaint re-initializes it) and only then proceeds with the rest of the
teardown. `app.py` then deletes the window between projects.

## Coordinate conventions

- Volumes are handled in physical space via SimpleITK.
- Electrode contacts are stored in **LPS millimetres**; voxel indices are
  derived from the reference image.
- MNI coordinates (MNI152NLin2009cAsym) are produced by ANTs normalization in
  `utils/mni_coordinates.py`.

## Threading model

The Qt main thread owns all UI and all VTK rendering. Only pure computation
(registration, masking, SISCOM) is moved to worker threads. Results are passed
back via signals and applied to `AppState` on the main thread.
