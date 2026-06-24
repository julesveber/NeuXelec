# Releasing a new version of NeuXelec

NeuXelec has an in-app updater (`neuxelec/updater.py`). On launch it reads a
static manifest published on the website
(`https://neuxelec.com/latest.json`). If that manifest advertises a version
newer than the running build, the user is offered to download and install it.

Publishing an update therefore means: **build a new installer, upload it, and
update `latest.json`.** No backend or server code is involved.

## 1. Bump the version number (3 places, keep them identical)

- `src/neuxelec/__init__.py` → `__version__ = "X.Y.Z"`
- `pyproject.toml` → `version = "X.Y.Z"`
- `NeuXelec_setup.iss` → `#define MyAppVersion "X.Y.Z"`

> The installed app compares its embedded `__version__` against the manifest,
> so this number must reflect the build you ship.

## 2. Run the test suite

```bash
pytest
```

## 3. Build the executable and the installer

```bash
pyinstaller NeuXelec_windows.spec --clean --noconfirm
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" NeuXelec_setup.iss
```

Result: `installer/NeuXelec_Setup_X.Y.Z.exe`.

> The installer keeps the same `AppId`, so it upgrades an existing
> installation in place (no uninstall needed).

## 4. Compute the installer SHA-256

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" installer/NeuXelec_Setup_X.Y.Z.exe
```

## 5. Upload the installer to the website

Upload to `neuxelec.com/downloads/` (FTP). Example:

```bash
curl --ftp-ssl --ftp-create-dirs -u "<ftp_user>:<ftp_pass>" \
  -T "installer/NeuXelec_Setup_X.Y.Z.exe" \
  "ftp://146wkx.ftp.infomaniak.com/sites/neuxelec.com/downloads/NeuXelec_Setup_X.Y.Z.exe"
```

## 6. Update and upload `latest.json`

Edit `Neuxelec_site/latest.json`:

```json
{
  "version": "X.Y.Z",
  "url": "https://neuxelec.com/downloads/NeuXelec_Setup_X.Y.Z.exe",
  "notes": "Short description of what changed in this release.",
  "sha256": "<sha-256 from step 4>",
  "mandatory": false
}
```

Upload it to the site root:

```bash
curl --ftp-ssl -u "<ftp_user>:<ftp_pass>" \
  -T "Neuxelec_site/latest.json" \
  "ftp://146wkx.ftp.infomaniak.com/sites/neuxelec.com/latest.json"
```

## 7. Verify

```bash
curl -s https://neuxelec.com/latest.json
```

That's it. Any running NeuXelec older than `X.Y.Z` will, on its next launch,
detect the update and offer to install it. `mandatory` is reserved for a
future "forced update" policy; keep it `false` for normal releases.

## Notes

- The updater never blocks startup and stays silent when the machine is
  offline or the manifest is unreachable.
- The SHA-256 is verified after download; a mismatch aborts the install.
- If you host the installer somewhere else, just point `url` at it (must be
  HTTPS and publicly reachable).
