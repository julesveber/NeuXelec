from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any


def _safe_float(v):
    try:
        if v is None:
            return None

        s = str(v).strip()

        if not s or s.lower() in ("n/a", "na", "nan", "none"):
            return None

        return float(s)

    except Exception:
        return None


def _make_color_from_string(text: str):
    seed = abs(hash(str(text))) % (2**32)
    rng = random.Random(seed)

    return (
        int(rng.randint(60, 255)),
        int(rng.randint(60, 255)),
        int(rng.randint(60, 255)),
    )


def _infer_subject_from_path(path: Path) -> str:
    name = path.name

    if name.startswith("sub-"):
        parts = name.split("_")
        if parts:
            return parts[0]

    return path.stem.replace("_electrodes", "")


def load_bids_mni_electrodes_tsv(path: str) -> dict[str, Any]:
    """
    Load a BIDS MNI electrodes.tsv file.

    Expected columns:
        name, x, y, z

    Coordinates are assumed to be MNI RAS-like x/y/z in mm.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found:\n{p}")

    if p.suffix.lower() != ".tsv":
        raise ValueError("Please drop a BIDS electrodes.tsv file.")

    subject = _infer_subject_from_path(p)
    color = _make_color_from_string(subject)

    contacts: list[dict[str, Any]] = []

    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError("The TSV file has no header.")

        fields = {str(k).strip().lower(): k for k in reader.fieldnames}

        required = ("name", "x", "y", "z")
        missing = [k for k in required if k not in fields]

        if missing:
            raise ValueError(
                "This file does not look like a BIDS electrodes.tsv.\n"
                f"Missing columns: {', '.join(missing)}"
            )

        for row in reader:
            name = str(row.get(fields["name"], "")).strip()

            x = _safe_float(row.get(fields["x"]))
            y = _safe_float(row.get(fields["y"]))
            z = _safe_float(row.get(fields["z"]))

            if x is None or y is None or z is None:
                continue

            group = ""
            if "group" in fields:
                group = str(row.get(fields["group"], "")).strip()

            hemi = ""
            if "hemisphere" in fields:
                hemi = str(row.get(fields["hemisphere"], "")).strip()

            contact_type = ""
            if "type" in fields:
                contact_type = str(row.get(fields["type"], "")).strip()

            parcel1_region = ""
            if "parcel1_region" in fields:
                parcel1_region = str(row.get(fields["parcel1_region"], "")).strip()

            parcel2_region = ""
            if "parcel2_region" in fields:
                parcel2_region = str(row.get(fields["parcel2_region"], "")).strip()

            contacts.append(
                {
                    "name": name,
                    "group": group,
                    "hemisphere": hemi,
                    "type": contact_type,
                    "parcel1_region": parcel1_region,
                    "parcel2_region": parcel2_region,
                    "mni_ras": [float(x), float(y), float(z)],
                }
            )

    if not contacts:
        raise ValueError(
            "No valid contacts were found in this electrodes.tsv.\n"
            "Check that x, y, z columns contain numeric MNI coordinates."
        )

    return {
        "subject": subject,
        "path": str(p),
        "space": "MNI152NLin2009cAsym",
        "coordinate_convention": "MNI_RAS",
        "color": color,
        "visible": True,
        "contacts": contacts,
    }
