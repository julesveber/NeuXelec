"""In-app update checker for NeuXelec.

On launch, NeuXelec fetches a small static manifest hosted on the website
(``https://neuxelec.com/latest.json``) and, if it advertises a newer version
than the running build, offers the user to download and install it.

No backend is required: the manifest and the installer are plain static files
served by the web hosting. The flow is entirely opt-in - the user is asked
before anything is downloaded or installed.

Manifest format (latest.json)::

    {
      "version": "1.1.0",
      "url": "https://neuxelec.com/downloads/NeuXelec_Setup_1.1.0.exe",
      "notes": "What changed in this release (optional).",
      "sha256": "<hex sha-256 of the installer, optional but recommended>",
      "mandatory": false
    }

The pure helpers (:func:`is_newer`, :func:`parse_manifest`) are unit-tested;
the network and Qt parts fail silently (no popup) when offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

#: Where the version manifest is published.
DEFAULT_MANIFEST_URL = "https://neuxelec.com/latest.json"

_HTTP_TIMEOUT = 6.0
_USER_AGENT = "NeuXelec-Updater"


@dataclass
class UpdateInfo:
    """A parsed update manifest entry."""

    version: str
    url: str
    notes: str = ""
    sha256: Optional[str] = None
    mandatory: bool = False


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, no network / no Qt)
# ---------------------------------------------------------------------------
def _version_tuple(version: str) -> tuple[int, ...]:
    """Convert a dotted version string to a comparable tuple of ints.

    Tolerant of pre-release suffixes: ``"1.2.0-beta"`` -> ``(1, 2, 0)``.
    Non-numeric or empty parts are treated as 0.
    """
    parts: list[int] = []
    for chunk in str(version).strip().split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(remote_version: str, current_version: str) -> bool:
    """Return True if ``remote_version`` is strictly newer than current."""
    return _version_tuple(remote_version) > _version_tuple(current_version)


def parse_manifest(data: dict) -> UpdateInfo:
    """Build an :class:`UpdateInfo` from a manifest dict. Raises on bad data."""
    version = str(data["version"]).strip()
    url = str(data["url"]).strip()
    if not version or not url:
        raise ValueError("Manifest must contain non-empty 'version' and 'url'.")
    sha256 = data.get("sha256")
    return UpdateInfo(
        version=version,
        url=url,
        notes=str(data.get("notes", "")).strip(),
        sha256=str(sha256).strip().lower() if sha256 else None,
        mandatory=bool(data.get("mandatory", False)),
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def fetch_manifest(url: str = DEFAULT_MANIFEST_URL, timeout: float = _HTTP_TIMEOUT) -> UpdateInfo:
    """Download and parse the manifest. Raises on network/parse errors."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return parse_manifest(payload)


def check_for_update(
    current_version: str,
    url: str = DEFAULT_MANIFEST_URL,
    timeout: float = _HTTP_TIMEOUT,
) -> Optional[UpdateInfo]:
    """Return an :class:`UpdateInfo` if a newer version exists, else None.

    Never raises: any failure (offline, bad manifest) returns None and is
    logged at INFO level.
    """
    try:
        info = fetch_manifest(url, timeout)
    except Exception:
        logger.info("Update check skipped (offline or no manifest).", exc_info=True)
        return None

    if is_newer(info.version, current_version):
        logger.info("Update available: %s (current %s)", info.version, current_version)
        return info

    logger.info("NeuXelec is up to date (%s).", current_version)
    return None


def download_installer(
    info: UpdateInfo,
    dest_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    timeout: float = 30.0,
) -> str:
    """Download the installer to ``dest_dir`` and verify its SHA-256.

    Parameters
    ----------
    progress_cb:
        Optional callback ``(downloaded_bytes, total_bytes)``.

    Returns
    -------
    str
        Path to the downloaded installer.

    Raises
    ------
    ValueError
        If the SHA-256 of the downloaded file does not match the manifest.
    """
    dest_dir = dest_dir or tempfile.gettempdir()
    filename = os.path.basename(info.url.split("?")[0]) or "NeuXelec_Setup.exe"
    dest_path = os.path.join(dest_dir, filename)

    request = urllib.request.Request(info.url, headers={"User-Agent": _USER_AGENT})
    hasher = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        with open(dest_path, "wb") as out:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)
                if progress_cb is not None:
                    progress_cb(downloaded, total)

    if info.sha256:
        actual = hasher.hexdigest().lower()
        if actual != info.sha256:
            try:
                os.remove(dest_path)
            except OSError:
                pass
            raise ValueError(
                "Downloaded installer failed integrity check "
                f"(expected {info.sha256}, got {actual})."
            )

    return dest_path


# ---------------------------------------------------------------------------
# Qt integration
# ---------------------------------------------------------------------------
class UpdateChecker(QThread):
    """Background thread that checks for an update without blocking the UI.

    Emits :data:`update_available` with the :class:`UpdateInfo` only when a
    newer version is found. Stays silent otherwise.
    """

    update_available = Signal(object)  # UpdateInfo

    def __init__(
        self,
        current_version: str,
        url: str = DEFAULT_MANIFEST_URL,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._current_version = current_version
        self._url = url

    def run(self) -> None:  # noqa: D102 - QThread entry point
        info = check_for_update(self._current_version, self._url)
        if info is not None:
            self.update_available.emit(info)


def prompt_and_install(info: UpdateInfo, parent=None) -> None:
    """Ask the user to install the update; if accepted, download and launch it.

    On acceptance the installer is downloaded (with a progress dialog and
    integrity check), launched, and the application quits so the installer can
    update the existing installation in place.
    """
    from PySide6.QtWidgets import (
        QApplication,
        QMessageBox,
        QProgressDialog,
    )

    box = QMessageBox(parent)
    box.setWindowTitle("Mise à jour disponible")
    box.setIcon(QMessageBox.Icon.Information)
    box.setText(
        f"Une nouvelle version de NeuXelec ({info.version}) est disponible.\n\n"
        "Voulez-vous la télécharger et l'installer maintenant ?"
    )
    if info.notes:
        box.setInformativeText(info.notes)
    install_btn = box.addButton("Installer", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Plus tard", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is not install_btn:
        return

    progress = QProgressDialog("Téléchargement de la mise à jour…", "Annuler", 0, 100, parent)
    progress.setWindowTitle("Mise à jour")
    progress.setMinimumDuration(0)
    progress.setValue(0)

    cancelled = {"value": False}

    def _on_progress(done: int, total: int) -> None:
        if progress.wasCanceled():
            cancelled["value"] = True
            raise RuntimeError("Download cancelled by user.")
        if total > 0:
            progress.setMaximum(total)
            progress.setValue(done)
        QApplication.processEvents()

    try:
        path = download_installer(info, progress_cb=_on_progress)
    except Exception as exc:
        progress.close()
        if not cancelled["value"]:
            logger.exception("Update download failed")
            QMessageBox.warning(
                parent,
                "Mise à jour",
                f"Le téléchargement de la mise à jour a échoué :\n{exc}",
            )
        return

    progress.close()

    # Launch the installer and quit so it can update the running app.
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
        else:
            import subprocess

            subprocess.Popen([path])  # noqa: S603
    except Exception as exc:
        logger.exception("Failed to launch installer")
        QMessageBox.warning(
            parent,
            "Mise à jour",
            f"Impossible de lancer l'installeur :\n{exc}\n\nFichier téléchargé : {path}",
        )
        return

    QApplication.quit()
