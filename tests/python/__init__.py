from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package_root in (ROOT / "ml", ROOT / "services" / "inference"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
