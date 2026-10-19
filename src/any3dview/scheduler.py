"""Thread-safe handoff of immutable update payloads to a viewer thread."""

from __future__ import annotations

from queue import Empty, SimpleQueue
from threading import Lock, get_ident
from typing import Any, Callable, Optional


class ViewerScheduler:
    """Queue callbacks for execution by the thread that owns a viewer.

    An optional notifier hook lets a host queue a wakeup on the owning
    thread as soon as work arrives, replacing idle polling.  The hook may
    be invoked from any thread and must be thread-safe; a failing hook
    never removes the queued item, so a later drain still executes it.
    The first hook failure is recorded and re-raised by the next drain so
    a broken wakeup cannot disappear silently.
    """

    __slots__ = ("_owner_thread", "_queue", "_closed", "_notifier",
                 "_lock", "_notifier_error")

    def __init__(self) -> None:
        self._owner_thread = get_ident()
        self._queue: SimpleQueue[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = (
            SimpleQueue()
        )
        self._closed = False
        self._notifier: Optional[Callable[[], None]] = None
        # Guards the closed flag together with the enqueue it covers, so a
        # close racing a submit can neither strand accepted work nor drop
        # a wakeup for work that was already queued.
        self._lock = Lock()
        self._notifier_error: Optional[BaseException] = None

    def set_notifier(self, notifier: Optional[Callable[[], None]]) -> None:
        """Attach or detach the wakeup hook (owner thread setup only)."""

        self._notifier = notifier

    def _notify(self) -> None:
        notifier = self._notifier
        if notifier is None:
            return
        try:
            notifier()
        except Exception as error:
            # The payload is already queued; a broken wakeup must not
            # lose it or crash the producing thread.  Record the first
            # failure so the owner thread observes it on the next drain.
            if self._notifier_error is None:
                self._notifier_error = error

    def submit(
        self,
        callback: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("viewer scheduler has been closed")
            if not callable(callback):
                raise TypeError("callback must be callable")
            self._queue.put((callback, args, kwargs))
        self._notify()

    def pending(self) -> bool:
        """Report whether queued work remains (approximate across threads)."""

        return not self._queue.empty()

    def drain(self, limit: int = 1024) -> int:
        if get_ident() != self._owner_thread:
            raise RuntimeError("viewer updates must be drained on the owning thread")
        completed = 0
        try:
            while completed < max(1, int(limit)):
                try:
                    callback, args, kwargs = self._queue.get_nowait()
                except Empty:
                    break
                # Fail fast: the first callback error reaches the caller
                # before any later callback runs, exactly as an unbounded
                # drain would propagate it.
                callback(*args, **kwargs)
                completed += 1
        finally:
            if not self._queue.empty():
                # Work queued behind a failure (or beyond the bound) must
                # not wait for a new submission; re-queue the wakeup so the
                # next owner event drains the remainder.
                self._notify()
        if self._notifier_error is not None:
            # A broken wakeup must not disappear silently; surface the
            # first recorded notifier failure once the batch has drained.
            error, self._notifier_error = self._notifier_error, None
            raise error
        return completed

    def close(self) -> None:
        with self._lock:
            # Detach first so no wakeup can observe the half-closed state.
            self._notifier = None
            self._closed = True
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
        self._notifier_error = None


__all__ = ["ViewerScheduler"]
