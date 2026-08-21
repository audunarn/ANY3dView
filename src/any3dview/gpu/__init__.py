"""Optional Tk-embedded ModernGL backend.

Importing :mod:`any3dview` never imports this package.  Install
``ANY3dView[gpu]`` before importing it directly.
"""

from .host import GLHostProtocol, TkinterGLHost
from .renderer import ModernGLRenderer
from .widget import Any3DView

__all__ = ["Any3DView", "GLHostProtocol", "ModernGLRenderer", "TkinterGLHost"]
