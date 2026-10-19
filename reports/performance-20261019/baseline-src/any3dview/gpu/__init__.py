"""Optional Tk-embedded ModernGL backend.

Importing :mod:`any3dview` never imports this package.  Install
``ANY3dView[gpu]`` before importing it directly.
"""

def __getattr__(name):
    # Importing the shared renderer from a Qt host must not initialize Tk.
    if name in {"GLHostProtocol", "TkinterGLHost"}:
        from . import host
        return getattr(host, name)
    if name == "ModernGLRenderer":
        from .renderer import ModernGLRenderer
        return ModernGLRenderer
    if name == "Any3DView":
        from .widget import Any3DView
        return Any3DView
    raise AttributeError(name)

__all__ = ["Any3DView", "GLHostProtocol", "ModernGLRenderer", "TkinterGLHost"]
