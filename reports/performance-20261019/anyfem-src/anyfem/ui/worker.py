"""Running a solve without freezing the window.

The solve runs on a worker thread and reports through a queue that the Tk main
loop drains on a timer.  Nothing touches a widget from the worker.

Cancelling abandons the *result*, not the computation: the solver has no
interruption point, so a cancelled run finishes in the background and its
answer is discarded.  The status text says so rather than implying the machine
went idle.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = ["JobWorkerFacade", "SolveWorker"]


from ..application.worker import JobWorkerFacade

@dataclass
class _Message:
    kind: str
    payload: Any = None


class SolveWorker:
    """Runs one analysis at a time on a background thread."""

    POLL_MS = 60

    def __init__(
        self,
        widget,
        *,
        on_status: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self._widget = widget
        self._queue: "queue.Queue[_Message]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._generation = 0
        self._abandoned: set[int] = set()
        self._poll_job: Optional[str] = None

        self.on_status = on_status
        self.on_done = on_done
        self.on_error = on_error
        self.on_state_change = on_state_change

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        function: Callable[..., Any],
        *,
        progress_key: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Run ``function(**kwargs)`` off the main thread.  False if busy.

        ``progress_key`` names a keyword the analysis takes for live progress.
        The callable passed in only enqueues, so the analysis can report from
        the worker thread without ever touching a widget.
        """

        if self.running:
            return False

        self._generation += 1
        generation = self._generation

        def report(text: str) -> None:
            self._queue.put(_Message("status", (generation, text)))

        if progress_key is not None:
            kwargs[progress_key] = report

        def run() -> None:
            try:
                report("solving")
                result = function(**kwargs)
            except BaseException as error:  # noqa: BLE001 - reported verbatim
                self._queue.put(
                    _Message(
                        "error",
                        (generation, f"{type(error).__name__}: {error}", traceback.format_exc()),
                    )
                )
            else:
                self._queue.put(_Message("done", (generation, result)))

        self._thread = threading.Thread(
            target=run, name="anyfem-solve", daemon=True
        )
        self._thread.start()
        self._notify_state()
        self._schedule_poll()
        return True

    def cancel(self) -> None:
        """Discard the running solve's result when it arrives."""

        if self.running:
            self._abandoned.add(self._generation)
            self._emit_status("cancelled - the run finishes in the background")
            self._notify_state()

    # ------------------------------------------------------------------
    def _schedule_poll(self) -> None:
        self._poll_job = self._widget.after(self.POLL_MS, self._poll)

    def _poll(self) -> None:
        self._poll_job = None
        try:
            while True:
                message = self._queue.get_nowait()
                self._handle(message)
        except queue.Empty:
            pass

        if self.running or not self._queue.empty():
            self._schedule_poll()
        else:
            self._notify_state()

    def _handle(self, message: _Message) -> None:
        if message.kind == "status":
            generation, text = message.payload
            if generation not in self._abandoned:
                self._emit_status(text)
        elif message.kind == "done":
            generation, result = message.payload
            if generation in self._abandoned:
                self._abandoned.discard(generation)
                return
            if self.on_done is not None:
                self.on_done(result)
        elif message.kind == "error":
            generation, text, detail = message.payload
            if generation in self._abandoned:
                self._abandoned.discard(generation)
                return
            if self.on_error is not None:
                self.on_error(text)

    def _emit_status(self, text: str) -> None:
        if self.on_status is not None:
            self.on_status(text)

    def _notify_state(self) -> None:
        if self.on_state_change is not None:
            self.on_state_change()

    def stop(self) -> None:
        """Cancel any pending timer, for teardown."""

        if self._poll_job is not None:
            try:
                self._widget.after_cancel(self._poll_job)
            except Exception:  # pragma: no cover - widget already destroyed
                pass
            self._poll_job = None
