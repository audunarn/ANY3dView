"""Thread-safe handoff of immutable update payloads to a viewer thread."""

from __future__ import annotations

from queue import Empty, SimpleQueue
from threading import get_ident
from typing import Any, Callable


class ViewerScheduler:
    """Queue callbacks for execution by the thread that owns a viewer."""

    __slots__ = ("_owner_thread", "_queue", "_closed")

    def __init__(self) -> None:
        self._owner_thread = get_ident()
        self._queue: SimpleQueue[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = (
            SimpleQueue()
        )
        self._closed = False

    def submit(
        self,
        callback: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self._closed:
            raise RuntimeError("viewer scheduler has been closed")
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._queue.put((callback, args, kwargs))

    def drain(self, limit: int = 1024) -> int:
        if get_ident() != self._owner_thread:
            raise RuntimeError("viewer updates must be drained on the owning thread")
        completed = 0
        while completed < max(1, int(limit)):
            try:
                callback, args, kwargs = self._queue.get_nowait()
            except Empty:
                break
            callback(*args, **kwargs)
            completed += 1
        return completed

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return


__all__ = ["ViewerScheduler"]
