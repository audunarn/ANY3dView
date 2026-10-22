"""Public viewer initialization errors."""


class GPUUnavailableError(RuntimeError):
    """The optional GPU backend could not be imported or initialized."""

    def __init__(self, message: str, *, diagnostics: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.diagnostics = tuple(diagnostics) or (str(message),)


class FaceColorUpdateError(ValueError):
    """An in-place per-face colour update was refused.

    Raised for a colour count that matches neither the triangle nor the
    element count, for a handle registered without face colours, for a
    handle the renderer does not hold, and for out-of-range numeric colours.
    """
