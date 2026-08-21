"""Public viewer initialization errors."""


class GPUUnavailableError(RuntimeError):
    """The optional GPU backend could not be imported or initialized."""

    def __init__(self, message: str, *, diagnostics: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.diagnostics = tuple(diagnostics) or (str(message),)
