import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("data", "eval", "scripts"):
    sys.path.insert(0, str(ROOT / sub))
