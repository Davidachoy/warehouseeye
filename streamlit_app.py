"""Streamlit Cloud / local entry for pre-rendered mode."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from frontend.app_space import main

if __name__ == "__main__":
    main()
