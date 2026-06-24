# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

block_cipher = None

project_root = Path(SPEC).resolve().parent
src_dir = project_root / "src"
entry_script = project_root / "scripts" / "run_neuxelec.py"
icon_file = project_root / "resources" / "images" / "brain_logo.ico"

# Application data required at runtime.
datas = [
    (str(project_root / "resources"), "resources"),
    (str(project_root / "templates"), "templates"),
    (str(src_dir / "neuxelec" / "utils" / "FreeSurferColorLUT.txt"), "neuxelec/utils"),
    (str(src_dir / "neuxelec" / "utils" / "electrodes_ref.json"), "neuxelec/utils"),
    (str(src_dir / "neuxelec" / "utils" / "electrodes_ref.txt"), "neuxelec/utils"),
]

# ANTs executables and MNI/template files. Keep the complete tools directory.
tools_dir = project_root / "tools"
if not tools_dir.exists():
    raise FileNotFoundError(
        "Missing tools directory. Restore tools/ before building NeuXelec."
    )
datas.append((str(tools_dir), "tools"))

# Explicitly collect scientific/3D packages with dynamic modules and data files.
hiddenimports = []
binaries = []
for package_name in (
    "pyvista",
    "pyvistaqt",
    "vtk",
    "skimage",
    "scipy",
    "matplotlib",
    "nibabel",
    "SimpleITK",
    "openpyxl",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [str(entry_script)],
    pathex=[str(src_dir), str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "pytest",
        "IPython",
        "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuXelec",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(icon_file),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="NeuXelec",
)
