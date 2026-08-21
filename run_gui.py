#!/usr/bin/env python
"""Run the ANY3dView interactive demo from this checkout."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
for _source in (_ROOT / "src", _ROOT.parent / "ANYtk3D" / "src"):
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))


def main() -> None:
    """Launch the maintained backend-neutral demo."""

    from any3dview.demo import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
