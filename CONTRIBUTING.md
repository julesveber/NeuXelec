# Contributing to NeuXelec

Thank you for your interest in NeuXelec. This guide describes how to set up a
development environment and the conventions used in the codebase.

> **License note.** NeuXelec is distributed under the GNU General Public
> License v3.0. Contributions are accepted under the same license; any
> distributed derivative work must also be released under GPL-3.0 with its
> complete source code.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_neuxelec.py
```

You also need `tools/ants/` (ANTs executables) and `templates/` (MNI templates
and atlases) present locally - these are not tracked in git. See the README.

## Before committing

Run, from the project root:

```bash
pytest                 # tests must pass
ruff check src tests   # lint must be clean
black src tests        # auto-format
mypy                   # type checks (lenient, do not regress)
```

## Coding conventions

- **Python 3.10+**, `from __future__ import annotations` at the top of modules.
- **Type hints** on public functions and methods.
- **Docstrings** on modules, classes and non-trivial functions (what it does,
  units and conventions for any coordinates).
- **Logging, not `print`.** Use a module-level
  `logger = logging.getLogger(__name__)` and `logger.info/.warning/.exception`.
- **Error handling.** Catch the *specific* exception you expect and log it with
  `logger.exception(...)`. Avoid bare `except Exception: pass`; only swallow
  errors at UI boundaries (refresh callbacks) and log them even there.
- **No GUI work off the main thread.** Long computations go through a
  `workers/` `QThread`; results return via Qt signals.
- **Coordinates.** Document the space (voxel / LPS mm / MNI mm) of any value
  that crosses a function boundary.

## Tests

- Tests live in `tests/` and must be **GUI-free and data-free** (no display,
  no patient imaging) so they run deterministically in CI.
- Prioritise the scientific core: coordinate transforms, reconstruction
  geometry, SISCOM, and project save/load round-trips.
- Use `tmp_path` for any file output.

## Commits

- One logical change per commit, with a clear message (what and why).
- Keep refactors behaviour-preserving; verify the app still launches and the
  affected page still works before committing.

## Releasing

1. Bump the version in `pyproject.toml` and `NeuXelec_setup.iss`.
2. `pytest` green.
3. `pyinstaller NeuXelec_windows.spec --clean --noconfirm`.
4. Compile the installer with Inno Setup (`NeuXelec_setup.iss`).
5. Smoke-test: open a project, visit each page, return to menu, reopen.
