import sys
from pathlib import Path

# Allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuxelec.app import main

if __name__ == "__main__":
    raise SystemExit(main())

