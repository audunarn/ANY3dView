"""Run one numerical job in a separate worker process.

Why a process and not a thread
------------------------------
The solvers are largely Python code.  A worker *thread* shares the GUI's
interpreter lock, so while it computes the GUI thread has to win the lock back
from it on every call that releases it -- painting the viewport, handling a
click, a timer.  Each hand-off can cost milliseconds, a paint needs hundreds of
them. A separate process has its own
lock and its own memory, so the GUI and the solver stop competing.

What crosses the boundary
-------------------------
* in: the job function (by reference), its keyword arguments, and the immutable
  project snapshot (or the detached project clone), pickled once;
* out, one pipe, pickled messages: ``progress`` (message and structured
  payload), then exactly one of ``result`` or ``error``;
* cancellation: the parent forwards its cancellation token to the worker, which
  trips its own token, so the solver's cooperative safe points work unchanged.

The pickler used for transport knows how to carry read-only mapping views
(``mappingproxy``), which the solution objects hold.  That is registered on a
private pickler only; ANYfem's rule that such views are not silently copyable
elsewhere (``deepcopy``, global ``copyreg``) is untouched.

The worker runs at below-normal priority so the GUI wins any contention, and it
exits when the GUI process disappears.  A crash in native solver code ends the
worker, not the application, and is reported as a failed job.
"""

from __future__ import annotations

import copyreg
import dataclasses
import inspect
import io
import multiprocessing
import os
import pickle
import select
import signal
import sys
import threading
import time
import traceback
import types
from copy import deepcopy
from typing import Any, Callable, Mapping

__all__ = [
    "WorkerProcessError",
    "WorkerTree",
    "dumps",
    "loads",
    "prepare_call",
    "run_in_worker_process",
]

_POLL_SECONDS = 0.05
_CONTAINED_POLL_SECONDS = 0.1
_HANDSHAKE_SECONDS = 10.0
_HANDSHAKE_POLL_SECONDS = 0.1
_CLEANUP_JOIN_SECONDS = 5.0
_RETIRED_POLL_SECONDS = 0.05
_GUARDIAN_POLL_SECONDS = 0.1
_GUARDIAN_ACK_SECONDS = 5.0


class WorkerProcessError(RuntimeError):
    """A job failed inside the worker process (or the worker itself died).

    ``remote_type`` and ``remote_traceback`` carry the original exception's name
    and formatted traceback so diagnostics read the same as for a thread job.
    """

    def __init__(
        self,
        message: str,
        *,
        remote_type: str = "WorkerProcessError",
        remote_traceback: str = "",
        remote_diagnostic: Any = None,
    ) -> None:
        super().__init__(message)
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback
        # Structured evidence computed in the worker by the job's own ``error_diagnostic``
        # hook (see _worker_main); None when the job has none or the worker died.
        self.remote_diagnostic = remote_diagnostic


# ----------------------------------------------------------------------
# transport serialisation
# ----------------------------------------------------------------------
def _rebuild_mappingproxy(contents: dict) -> types.MappingProxyType:
    return types.MappingProxyType(contents)


def _new_thread_local() -> threading.local:
    return threading.local()


class _TransportPickler(pickle.Pickler):
    # Private dispatch table; nothing else about pickling changes.
    #  * Read-only mapping views travel as their contents and are rebuilt as read-only views.
    #  * Thread-local storage (solver work buffers inside cached contact geometry) travels
    #    as a fresh, empty one.  That is exact, not a loss: by definition no other thread
    #    ever sees what a thread stored in it, so the receiving process starts where any
    #    other thread of the worker would have.
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[types.MappingProxyType] = lambda view: (_rebuild_mappingproxy, (dict(view),))
    dispatch_table[type(threading.local())] = lambda local: (_new_thread_local, ())


def dumps(value: Any) -> bytes:
    buffer = io.BytesIO()
    _TransportPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def loads(data: bytes) -> Any:
    return pickle.loads(data)


# ----------------------------------------------------------------------
# mesh-only opt-in containment for an outer worker and its child pool
# ----------------------------------------------------------------------
# A parallel mesh job's worker process creates its own children (the ANYmesh
# pool).  Those grandchildren are invisible to ``Process.terminate`` on the
# worker alone, so a contained mesh worker is placed in an owned process tree
# before it is allowed to create any child:
#
# * Windows: the parent creates a Job Object with kill-on-close, assigns the
#   worker to it immediately after start, and holds the only other handle.
#   A startup handshake gates the worker: it waits for the parent's "go"
#   after containment is established, so no pool child can predate the job.
#   If the parent dies its job handle closes and the whole tree dies.
# * Linux: the worker ``setsid``s into its own process group (it is the
#   leader, so the group is owned), arms a strictly checked parent-death
#   signal and forks a small cleanup guardian INSIDE that group, all
#   acknowledged before the parent's "go".  The guardian does not die with
#   the worker: when the worker (or the GUI) disappears it kills the owned
#   group -- the pool children, itself and any remaining descendant -- and
#   while it lives it anchors the group against numeric reuse.  Cleanup
#   signals that group only; a shared resource tracker is never touched.
#   Unsupported POSIX systems decline containment before any dispatch.
#
# Solve workers keep the historical daemon behaviour; nothing below runs
# unless a mesh job explicitly opts in.


def _windows_kernel32():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD
    )
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    return kernel32


def _create_kill_on_close_job() -> Any:
    """A new Job Object that dies with its last handle (fail closed)."""

    import ctypes
    from ctypes import wintypes

    kernel32 = _windows_kernel32()
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    class _IOCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(wintypes.ULONG)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IOCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9
    info = _ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        kernel32.CloseHandle(job)
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    return job


def _assign_process_to_job(job: Any, pid: int) -> None:
    import ctypes

    kernel32 = _windows_kernel32()
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001
    handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess failed")
    try:
        if not kernel32.AssignProcessToJobObject(job, handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    finally:
        kernel32.CloseHandle(handle)


def _windows_job_active_processes(job: Any) -> int:
    """The job object's active process count (checked query, exact SDK struct).

    ``JOBOBJECT_BASIC_ACCOUNTING_INFORMATION`` (winnt.h) carries four
    ``LARGE_INTEGER`` time fields before the accounting ``DWORD``s;
    ``ActiveProcesses`` is the number of processes currently assigned to the
    job.  It reaches 0 when every owned process has terminated -- even while
    the parent still holds open handles to them -- so retirement can be
    verified before the outer worker is joined.  A failed query raises: an
    inability to inspect never reads as retirement.
    """

    import ctypes
    from ctypes import wintypes

    class _BasicAccounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", wintypes.LARGE_INTEGER),
            ("TotalKernelTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    kernel32 = _windows_kernel32()
    info = _BasicAccounting()
    JobObjectBasicAccountingInformation = 1
    if not kernel32.QueryInformationJobObject(
        job,
        JobObjectBasicAccountingInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
    ):
        error = ctypes.get_last_error()
        raise OSError(error, f"QueryInformationJobObject failed ({error})")
    return int(info.ActiveProcesses)


def _linux_group_survivor(group: int, deadline: float | None = None) -> int | None:
    """The first live (non-zombie) member of the owned process group, or ``None``.

    Zombies are excluded: they are already dead, merely unreaped (the outer
    worker and the guardian are exactly that until the caller joins).  A
    process that disappears mid-scan is benign (already dead); every other
    inspection failure raises so the caller fails closed.
    """

    try:
        entries = os.listdir("/proc")
    except OSError as error:
        raise OSError(f"listing /proc failed: {error}") from error
    for entry in entries:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("owned group inspection exceeded the cleanup deadline")
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="surrogateescape") as stat:
                content = stat.read()
        except (FileNotFoundError, ProcessLookupError):
            continue  # gone mid-scan: already dead
        except OSError as error:
            raise OSError(f"reading /proc/{pid}/stat failed: {error}") from error
        try:
            fields = content.rsplit(") ", 1)[1].split()
            state = fields[0]
            pgrp = int(fields[2])
        except (IndexError, ValueError) as error:
            raise OSError(f"/proc/{pid}/stat could not be parsed: {error}") from error
        if pgrp == int(group) and state != "Z":
            return pid
    return None


def _set_parent_death_signal() -> None:
    """Linux only: die with the GUI process.  Strictly checked, never best effort.

    For a contained mesh worker this is mandatory: when it cannot be armed the
    worker reports a containment failure and the caller stays serial.  The
    kernel clears the parent-death signal in fork children, so the guardian
    below never inherits it.
    """

    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


class WorkerTree:
    """The parent's handle to one owned worker process tree.

    ``terminate`` hard-kills exactly the tree this run created (the Windows
    job object, or the worker's own process group on POSIX) and never an
    unknown or shared PID/group.  ``close`` releases the parent's handles;
    on Windows the kill-on-close job then also covers a crashed parent.
    Both are idempotent, never raise, and never close a handle while a
    native operation on it is in flight (the lock covers the native calls).

    Native outcomes are checked and recorded in :attr:`cleanup_failures`;
    success is marked only once the stop is confirmed, so a caller can
    refuse to publish a result whose tree was not retired.
    """

    def __init__(self, process: Any, *, job: Any = None, group: int | None = None) -> None:
        self._process = process
        self._job = job
        self._group = group
        self._proven_group: int | None = None
        self._assigned = False
        self._lock = threading.Lock()
        self._terminated = False
        self._closed = False
        self._failures: list[str] = []

    @property
    def group(self) -> int | None:
        return self._group

    @property
    def owns_worker(self) -> bool:
        """Whether the created worker is proven inside this owned tree.

        Windows: the worker was assigned to the job object.  POSIX: the
        worker reported (or was proven to lead) the owned process group.
        ``False`` means the tree's stop does not cover the worker itself --
        which can only happen before dispatch, when no descendant can exist
        yet -- so the caller must stop the created worker directly instead
        of trusting a vacuous tree stop as its retirement.
        """

        with self._lock:
            return self._assigned or self._group is not None or self._proven_group is not None

    def mark_assigned(self) -> None:
        """Record that the worker was assigned to the job object (Windows)."""

        with self._lock:
            self._assigned = True

    @property
    def group(self) -> int | None:
        return self._group

    @property
    def terminated(self) -> bool:
        """Whether the owned tree's stop was confirmed by the native API."""
        return self._terminated

    @property
    def cleanup_failures(self) -> list[str]:
        """Every recorded native cleanup failure, in occurrence order."""
        with self._lock:
            return list(self._failures)

    def set_group(self, group: int) -> None:
        """Record the owned process group the worker reported (Linux)."""
        with self._lock:
            if self._group is None:
                self._group = int(group)

    def _owned_group(self, group: int | None) -> int | None:
        """The owned process group to signal, or ``None`` when unproven.

        Known group: only the one the worker itself created by setsid and
        reported back.  Unknown group (the worker has not reported yet):
        while the created process is alive, live group leadership
        (``getpgid(pid) == pid``) proves it leads its own new session --
        something only its own setsid produces; the group it was spawned
        into is led by another pid and is never signalled.
        """

        pid = getattr(self._process, "pid", None)
        if group is not None:
            return group if group > 0 and group == pid else None
        if pid is None:
            return None
        try:
            alive = bool(self._process.is_alive())
        except Exception:  # pragma: no cover - a torn-down fake in tests
            alive = False
        if not alive:
            return None
        try:
            pgid = os.getpgid(pid)
        except OSError:
            return None
        return pgid if pgid == pid else None

    def terminate(self) -> None:
        with self._lock:
            if self._terminated or self._closed:
                return
            job = self._job
            try:
                if job is not None:
                    import ctypes

                    if not _windows_kernel32().TerminateJobObject(job, 1):
                        error = ctypes.get_last_error()
                        raise OSError(error, f"TerminateJobObject failed ({error})")
                elif os.name == "posix":
                    target = self._owned_group(self._group)
                    if target is None:
                        # No proven owned group (yet): nothing was stopped,
                        # so success is NOT marked -- the caller's fallback
                        # stop of the created worker itself must engage.
                        return
                    try:
                        os.killpg(target, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # the owned group is already gone: retired
                    # A PermissionError is deliberately NOT caught: a refusal
                    # stopped nothing, so it propagates to the recorder
                    # below as a cleanup failure and success stays unmarked.
                    self._proven_group = target
                self._terminated = True
            except Exception as error:  # noqa: BLE001 - recorded, never raised
                self._failures.append(f"terminating the owned worker tree failed: {error}")

    def confirm_retirement(self, deadline: float) -> str | None:
        """Verify every member of the owned tree retired; ``None`` when proven.

        Called only after :meth:`terminate` confirmed the native stop and
        BEFORE the caller joins (reaps) the outer worker: while it is
        unreaped the outer worker (with the guardian) anchors the owned
        group against numeric reuse, so the inspection is exact.  Never
        signals anything -- a stop request is not retirement.  Returns a
        structured failure string when the tree cannot be proven retired by
        ``deadline``: a survivor, a timeout, or an inability to inspect
        (which fails closed, never reads as retirement).
        """

        with self._lock:
            job = self._job
            group = self._proven_group
            terminated = self._terminated
        if not terminated:
            return "the owned worker tree stop was not confirmed; retirement is unverified"
        try:
            if job is not None:
                # Windows: the job object's own accounting must reach zero
                # active processes (the outer worker AND every pool child).
                while True:
                    if time.monotonic() >= deadline:
                        return "owned job retirement was not verified within the cleanup deadline"
                    active = _windows_job_active_processes(job)
                    if time.monotonic() >= deadline:
                        return "owned job retirement was not verified within the cleanup deadline"
                    if active == 0:
                        return None
                    if time.monotonic() >= deadline:
                        return (
                            f"{active} owned worker tree process(es) were still "
                            "active within the cleanup deadline"
                        )
                    time.sleep(min(_RETIRED_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
            if group is not None:
                # Linux: no live (non-zombie) member of the owned process
                # group may remain; zombies are already dead, not survivors.
                while True:
                    if time.monotonic() >= deadline:
                        return "owned group retirement was not verified within the cleanup deadline"
                    survivor = _linux_group_survivor(group, deadline)
                    if time.monotonic() >= deadline:
                        return "owned group retirement was not verified within the cleanup deadline"
                    if survivor is None:
                        return None
                    if time.monotonic() >= deadline:
                        return (
                            f"owned worker tree process (pid {survivor}) was still "
                            "alive within the cleanup deadline"
                        )
                    time.sleep(min(_RETIRED_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
        except OSError as error:
            return f"verifying the owned worker tree retirement failed: {error}"
        return "the owned worker tree's containment mechanism is unknown; retirement is unverified"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            job = self._job
            self._job = None
            if job is None:
                return
            try:
                import ctypes

                if not _windows_kernel32().CloseHandle(job):
                    error = ctypes.get_last_error()
                    raise OSError(error, f"CloseHandle failed ({error})")
            except Exception as error:  # noqa: BLE001 - recorded, never raised
                self._failures.append(f"closing the owned worker tree failed: {error}")


# ----------------------------------------------------------------------
# Linux containment guardian
# ----------------------------------------------------------------------
# The outer worker's own parent-death signal kills only the outer worker; a
# hard kill bypasses its cleanup, so the ANYmesh pool children would leak.
# The guardian closes that gap.  It is a small independent process forked
# INSIDE the outer worker's owned process group before the parent's "go":
#
# * it never sets a parent-death signal, so it does not die automatically
#   with the outer worker; when the outer worker dies (any cause) it kills
#   the owned group -- the pool children, itself and any remaining
#   descendant;
# * it watches the outer worker's liveness pipe and, independently of the
#   outer worker's computation, the multiprocessing parent sentinel of the
#   GUI process, so a lost parent-death signal cannot leak the tree either;
# * while it lives it anchors the process group: the numeric group id
#   (the outer worker's pid) cannot be reused as a new session while
#   cleanup is still pending;
# * it only ever signals its OWN process group (``os.getpgrp()``), never a
#   shared resource tracker or an unrelated group.
#
# The guardian is never dismissed.  Retirement of the owned group is the
# PARENT's terminal cleanup: the outer worker stays alive and anchored
# after delivering its terminal result or error, the parent kills the
# owned group while that anchor is alive and confirms the retirement
# before the outcome is published, and the guardian (together with the
# whole group) dies in that same kill.  If the parent's cleanup never
# arrives, the guardian still kills the group when the outer worker or
# the GUI dies, so no path leaks the pool children.


def _guardian_kill_group() -> None:
    """Kill the guardian's own process group (the contained tree), including the guardian."""

    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except OSError:  # pragma: no cover - the group is already gone
        pass


def _wait_readable(fd: int, timeout: float) -> bool:
    """Wait until ``fd`` has data or EOF, or the timeout passes.

    POSIX ``select`` only: the guardian is a Linux-only production mechanism
    (containment is refused elsewhere before any dispatch), so no
    non-production platform branch exists here.
    """

    ready, _, _ = select.select([fd], [], [], timeout)
    return bool(ready)


def _guardian_loop(dismiss_fd: int, parent: Any) -> str:
    """Watch the outer worker's liveness pipe and the GUI sentinel; return why.

    Returns ``"outer"`` when the outer worker died (its liveness pipe reached
    EOF or broke) and ``"parent"`` when the GUI process disappeared.  There is
    no clean-dismiss outcome: the guardian is retired together with the whole
    owned group by the parent's terminal cleanup, or by its own group kill
    when the outer worker or the GUI dies first.
    """

    while True:
        try:
            ready = _wait_readable(dismiss_fd, _GUARDIAN_POLL_SECONDS)
        except OSError:
            return "outer"  # the liveness pipe is gone: the outer worker died
        if ready:
            try:
                data = os.read(dismiss_fd, 64)
            except OSError:
                data = b""
            if data == b"":
                return "outer"  # EOF: the outer worker is gone
        if parent is not None and not parent.is_alive():
            return "parent"  # the GUI process is gone (sentinel EOF)


def _guardian_main(dismiss_fd: int, ack_fd: int, close_handles: Any) -> None:
    """The guardian process entry point; never returns."""

    try:
        os.write(ack_fd, b"ready")
        os.close(ack_fd)
    except OSError:
        os._exit(0)
    for handle in close_handles:
        # Do not keep the outer worker's transport pipes open: the parent must
        # see EOF when the outer worker dies, not when the guardian exits.
        try:
            handle.close()
        except Exception:  # noqa: BLE001 - a courtesy close, never fatal
            pass
    try:
        _guardian_loop(dismiss_fd, multiprocessing.parent_process())
    except BaseException:  # noqa: BLE001 - a broken watch must fail to "kill the tree"
        pass
    _guardian_kill_group()
    os._exit(0)


def _start_guardian(connection: Any, control: Any) -> tuple[int, int]:
    """Fork and acknowledge the containment guardian (Linux only).

    Returns ``(liveness_write_fd, guardian_pid)``.  The outer worker keeps the
    write end open for its whole life: its death closes the pipe, which is
    the guardian's independent signal that the owned group must die.  Raises
    ``OSError`` when the guardian cannot be forked or does not acknowledge
    startup, so containment is refused before anything is dispatched.
    """

    dismiss_read, dismiss_write = os.pipe()
    ack_read, ack_write = os.pipe()
    pid = os.fork()
    if pid == 0:  # the guardian: same owned process group, no parent-death signal
        os.close(dismiss_write)
        os.close(ack_read)
        _guardian_main(dismiss_read, ack_write, (connection, control))
    # outer worker
    os.close(dismiss_read)
    os.close(ack_write)
    deadline = time.monotonic() + _GUARDIAN_ACK_SECONDS
    try:
        while True:
            ready, _, _ = select.select([ack_read], [], [], 0.05)
            if ready:
                acknowledged = os.read(ack_read, 16)
                break
            if time.monotonic() >= deadline:
                raise OSError("the containment guardian did not acknowledge startup")
    finally:
        os.close(ack_read)
    if acknowledged != b"ready":
        raise OSError("the containment guardian refused to start")
    return dismiss_write, pid


# ----------------------------------------------------------------------
# preparing the call (shared by the thread and the process path)
# ----------------------------------------------------------------------
def prepare_call(
    function: Callable[..., Any],
    kwargs: Mapping[str, Any],
    snapshot: Any,
    project_override: Any,
    progress: Callable[[Any], None],
    token: Any,
) -> dict[str, Any]:
    """Keyword arguments for the job function: a fresh project, progress and token."""

    call = dict(kwargs)
    project = deepcopy(project_override) if project_override is not None else snapshot.thaw()
    call.setdefault("project", project)
    call.setdefault("progress", progress)
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if accepts_kwargs or "cancellation_token" in parameters:
        call.setdefault("cancellation_token", token)
    return call


# ----------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------
def _lower_priority() -> None:
    """Best effort: the GUI should win any CPU contention with the solver."""

    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # Declared types matter: GetCurrentProcess() returns the pseudo-handle -1, which
            # ctypes would otherwise pass as a 32-bit int, an invalid 64-bit handle.
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel32.SetPriorityClass.restype = wintypes.BOOL
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(5)
    except Exception:  # pragma: no cover - priority is a courtesy, never a requirement
        pass


def _exit_with_parent() -> None:
    """Do not outlive the GUI: a solver left running after a crash would burn CPU unseen."""

    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        while parent.is_alive():
            threading.Event().wait(2.0)
        os._exit(3)

    threading.Thread(target=watch, name="anyfem-parent-watch", daemon=True).start()


def _worker_main(
    connection: Any,
    cancel_event: Any,
    payload: bytes,
    control: Any = None,
) -> None:
    from .jobs import _cancellation_token

    _lower_priority()

    def send(message: tuple) -> None:
        connection.send_bytes(dumps(message))

    spec = None
    guardian: tuple[int, int] | None = None
    try:
        spec = loads(payload)
        if not spec.get("containment"):
            # Solve workers keep the historical daemon behaviour: a watch
            # thread exits when the GUI disappears.  A contained worker must
            # not use it -- os._exit would bypass the owned-group cleanup the
            # guardian is responsible for; its death is guaranteed instead by
            # the strictly checked parent-death signal plus the guardian.
            _exit_with_parent()
        if spec.get("containment"):
            # Containment must exist before this process may create any
            # child (the ANYmesh pool).  Linux: become our own process group
            # leader, arm the mandatory parent-death signal and fork the
            # cleanup guardian inside that group, all acknowledged before the
            # parent's "go" confirms the tree is owned.  Windows: the parent
            # assigns this process to its kill-on-close job object before
            # sending "go".  Any other POSIX system safely declines.
            if os.name == "nt":
                control.send(("ready", None))
            elif sys.platform == "linux":
                try:
                    os.setsid()
                    _set_parent_death_signal()
                    guardian = _start_guardian(connection, control)
                except OSError as error:
                    control.send(("failed", f"linux containment bootstrap failed: {error}"))
                    os._exit(4)
                control.send(("ready", os.getpid()))
            else:
                control.send(
                    (
                        "failed",
                        "contained parallel meshing requires Linux process-group containment",
                    )
                )
                os._exit(4)
            try:
                if control.recv() != "go":
                    os._exit(4)
            except (EOFError, OSError):
                os._exit(4)
        token = _cancellation_token()

        def forward_cancellation() -> None:
            cancel_event.wait()
            token.cancel("cancelled by caller")

        threading.Thread(target=forward_cancellation, name="anyfem-cancel", daemon=True).start()

        def progress(value: Any) -> None:
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                # A structured progress record (e.g. a mesh stage) travels as itself.
                try:
                    send(("object", value))
                    return
                except Exception:
                    pass  # fall through to the readable text
            message = getattr(value, "message", None) or str(value)
            raw = getattr(value, "payload", None)
            structured = dict(raw) if isinstance(raw, Mapping) else None
            try:
                send(("progress", str(message), structured))
            except Exception:
                # Telemetry must never end a solve: fall back to the text alone.
                send(("progress", str(message), None))

        call = prepare_call(
            spec["function"], spec["kwargs"], spec["snapshot"], spec["project_override"], progress, token
        )
        result = spec["function"](**call)
        try:
            data = dumps(("result", result))
        except Exception:
            # A completed result can hold live solver objects that refuse
            # pickle by contract (the qualified S3 V2D shells block
            # python_pickle_restart pending a successor gate).  The solve is
            # done; only the transfer needs a supported route.  The result is
            # encoded through its supported plain-data APIs instead of
            # pickling the live objects; anything without such a route
            # fails below with the same structured transfer error as before.
            from .result_transport import encode_result

            progress("preparing the completed result for transfer")
            try:
                data = dumps(("encoded_result", encode_result(result)))
            except Exception as error:
                raise WorkerProcessError(
                    f"the result could not be transferred to the application: "
                    f"{type(error).__name__}: {error}",
                    remote_type="ResultTransferError",
                ) from error
        connection.send_bytes(data)
    except BaseException as error:  # noqa: BLE001 - reported to the parent, which decides
        kind = getattr(error, "remote_type", None) or type(error).__name__
        diagnostic = None
        try:
            hook = getattr(spec["function"], "error_diagnostic", None) if spec is not None else None
            if callable(hook):
                diagnostic = hook(error)
        except BaseException:  # noqa: BLE001 - a broken hook must not hide the job's own error
            diagnostic = None
        try:
            send(("error", kind, str(error), traceback.format_exc(), diagnostic))
        except Exception:  # pragma: no cover - the pipe itself is gone
            pass
    finally:
        try:
            connection.close()
        except Exception:  # pragma: no cover
            pass
        if guardian is not None:
            # Terminal state of a Linux contained worker whose guardian is
            # acknowledged.  The guardian stays armed: it anchors the owned
            # group against numeric reuse and kills it when this worker or
            # the GUI dies.  This worker does NOT dismiss the guardian, does
            # not signal the group itself and does not exit normally --
            # retirement is the PARENT's terminal cleanup, which kills the
            # owned group while the anchor is alive and confirms it before
            # the outcome is published.  If that cleanup never arrives, the
            # control pipe closing (or the GUI's death, watched by the
            # guardian independently) still ends this anchor, and the
            # guardian then kills the group: no path leaks the pool children,
            # and no numeric group operation happens after the anchor is
            # gone.
            try:
                control.recv()  # blocks until the parent's terminal cleanup
            except Exception:  # noqa: BLE001 - the control pipe is gone
                pass
            os._exit(0)  # the guardian sees the liveness pipe close and kills the group


# ----------------------------------------------------------------------
# application side
# ----------------------------------------------------------------------
def _progress_value(message: str, payload: Mapping[str, Any] | None) -> Any:
    """The same value a thread job reports: structured text when there is a payload."""

    if payload is None:
        return message
    try:
        from .solve.run import _StructuredProgressText

        return _StructuredProgressText(message, payload)
    except Exception:  # pragma: no cover - older layouts without the structured type
        return message


def _close_quietly(handle: Any) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except Exception:  # pragma: no cover - teardown must never mask a result
        pass


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _await_startup(control: Any, token: Any) -> tuple:
    """Poll the startup handshake in 100 ms slices, watching cancellation.

    A worker that dies before "ready" and a worker that never confirms within
    the bounded handshake are containment failures, never an uncontained
    parallel job.  A cancellation (or a shutdown, which cancels the same
    token) aborts startup before anything is dispatched.
    """

    deadline = time.monotonic() + _HANDSHAKE_SECONDS
    while True:
        if _is_cancelled(token):
            raise WorkerProcessError(
                "the job was cancelled before the contained worker was dispatched",
                remote_type="CancelledBeforeDispatch",
            )
        try:
            if control.poll(_HANDSHAKE_POLL_SECONDS):
                return control.recv()
        except (EOFError, OSError) as error:
            # A broken control pipe is the worker dying during startup (on
            # Windows the poll itself reports the break).
            raise WorkerProcessError(
                "the contained mesh worker died during startup",
                remote_type="ContainmentUnavailable",
            ) from error
        if time.monotonic() >= deadline:
            raise WorkerProcessError(
                "the contained mesh worker did not confirm startup",
                remote_type="ContainmentUnavailable",
            )


def _cleanup_worker(
    process: Any,
    receiver: Any,
    sender: Any,
    tree: WorkerTree | None,
    control_parent: Any,
    control_child: Any,
    started: bool,
) -> list[str]:
    """Bounded teardown on every path: allocation, start, handshake, run.

    One shared 5 s deadline covers termination, the retirement verification,
    the join and every handle close (there is no second 5 s join).  The owned
    tree is stopped BEFORE any waiting, so stuck work and any pool children
    die immediately; when that stop is confirmed, the whole tree's retirement
    is verified (Windows job accounting / a Linux ``/proc`` group scan) BEFORE
    the join reaps the outer ownership anchor, and an unverified retirement
    is itself a recorded failure.  When the stop is not confirmed the created
    worker itself is stopped at once instead of joining a process that has
    nothing to exit for.  Every native outcome is checked; the recorded
    failures are returned so the caller can refuse to publish an unretired
    tree (the process still being alive after the deadline is itself a
    failure).
    """

    deadline = time.monotonic() + _CLEANUP_JOIN_SECONDS
    failures: list[str] = []
    _close_quietly(sender)
    _close_quietly(receiver)
    _close_quietly(control_child)
    if tree is not None:
        # The owned tree dies first: a stuck worker and any pool children it
        # created are killed now, not after a full join timeout.  Bounded and
        # idempotent; never an unowned PID or group.
        tree.terminate()
    if started:
        if tree is None:
            process.join(_remaining(deadline))
        elif not tree.terminated or not tree.owns_worker:
            # The owned group could not be proven (the worker never became
            # its leader), the native stop failed, or the worker was never
            # placed inside the owned tree (a predispatch refusal, e.g. a
            # failed job-object assignment): no descendant can have been
            # dispatched, so stop the created worker itself immediately,
            # before any waiting.  A failed stop is already recorded as a
            # cleanup failure, and a vacuous tree stop must never read as
            # the worker's retirement.
            if process.is_alive():
                process.terminate()
        else:
            # The native stop succeeding is not yet retirement: verify the
            # WHOLE owned tree (the pool children included) retired BEFORE
            # the join below reaps the outer worker, which anchors the
            # ownership while it is unreaped.  Bounded by the same shared
            # deadline; an unverified tree is a structured cleanup failure
            # and the caller discards the result.
            unretired = tree.confirm_retirement(deadline)
            if unretired is not None:
                failures.append(unretired)
        process.join(_remaining(deadline))
        if process.is_alive():
            process.terminate()
            process.join(_remaining(deadline))
        if process.is_alive():
            failures.append(
                f"the worker process (pid {process.pid}) did not exit within "
                f"{_CLEANUP_JOIN_SECONDS:.0f} s"
            )
    if tree is not None:
        tree.close()
        failures.extend(tree.cleanup_failures)
    _close_quietly(control_parent)
    return failures


def _cleanup_note(error: BaseException, failures: list[str]) -> None:
    """Attach cleanup details to an in-flight failure without replacing it.

    The failures are also exposed structurally as the exception's
    ``cleanup_failures`` attribute, so a caller can refuse a fallback
    without inferring anything from the note text.
    """

    details = "worker tree cleanup did not confirm retirement: " + "; ".join(failures)
    try:
        error.cleanup_failures = list(failures)
    except Exception:  # pragma: no cover - an exception type that forbids attributes
        pass
    try:
        error.add_note(details)
    except AttributeError:  # pragma: no cover - pre-3.11 interpreters
        error.args = (*error.args, f"[{details}]")


def run_in_worker_process(
    *,
    function: Callable[..., Any],
    kwargs: Mapping[str, Any],
    snapshot: Any,
    project_override: Any,
    progress: Callable[[Any], None],
    token: Any,
    on_process: Callable[[Any], None] | None = None,
    containment: bool = False,
    on_tree: Callable[[Any, Any], None] | None = None,
) -> Any:
    """Run ``function`` in a fresh worker process and return its result.

    Blocks the calling (job) thread, which waits on a pipe and so does not hold
    the interpreter lock.  Raises :class:`WorkerProcessError` for a failure in
    the worker or the death of the worker.

    ``containment`` is the mesh-only opt-in for jobs whose worker creates
    child processes (the ANYmesh pool).  The worker is then non-daemonic and
    runs inside an owned, kill-on-close process tree that this call always
    tears down; containment is established and confirmed through a startup
    handshake before the worker may create any child.  When containment
    cannot be established the job never starts and
    ``WorkerProcessError(remote_type="ContainmentUnavailable")`` explains why,
    so the caller can fall back to serial execution.
    """

    try:
        payload = dumps(
            {
                "function": function,
                "kwargs": dict(kwargs),
                "snapshot": snapshot,
                "project_override": project_override,
                "containment": bool(containment),
            }
        )
    except Exception as error:
        raise WorkerProcessError(
            "this job cannot run in a worker process (its function and arguments must be picklable, "
            f"module-level objects): {type(error).__name__}: {error}",
            remote_type="JobNotTransferable",
        ) from error

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    cancel_event = context.Event()
    control_parent = control_child = None
    job = None
    if containment:
        # A contained worker must be able to create its own children, so it
        # cannot be daemonic; the owned tree replaces the daemon guarantee.
        control_parent, control_child = context.Pipe(duplex=True)
        if os.name == "nt":
            # Fail before anything is spawned when the platform cannot
            # contain the tree at all: the caller then stays serial.
            try:
                job = _create_kill_on_close_job()
            except OSError as error:
                _close_quietly(control_parent)
                _close_quietly(control_child)
                _close_quietly(receiver)
                _close_quietly(sender)
                raise WorkerProcessError(
                    "parallel meshing needs a contained worker process tree, but "
                    f"the process-tree containment is unavailable: {error}",
                    remote_type="ContainmentUnavailable",
                ) from error
        elif sys.platform != "linux":
            # Unsupported POSIX containment safely declines: only Linux
            # process groups (with their cleanup guardian) and the Windows
            # job object are claimed.
            _close_quietly(control_parent)
            _close_quietly(control_child)
            _close_quietly(receiver)
            _close_quietly(sender)
            raise WorkerProcessError(
                "parallel meshing needs a contained worker process tree, but "
                "process-tree containment is only supported on Linux and Windows",
                remote_type="ContainmentUnavailable",
            )
    process = context.Process(
        target=_worker_main,
        args=(sender, cancel_event, payload, control_child),
        name="anyfem-mesh" if containment else "anyfem-solve",
        daemon=not containment,
    )
    # Ownership exists before the start attempt: a start failure must not
    # leak the allocated job object or either control pipe end, so the tree
    # that owns them is constructed first and torn down on every path.
    tree: WorkerTree | None = WorkerTree(process, job=job) if containment else None
    started = False
    try:
        try:
            process.start()
        except OSError as error:
            if not containment:
                raise
            raise WorkerProcessError(
                "parallel meshing needs a contained worker process tree, but "
                f"the worker process could not be started: {error}",
                remote_type="ContainmentUnavailable",
            ) from error
        started = True
        sender.close()  # the worker owns its end; EOF here then means the worker is gone
        if control_child is not None:
            control_child.close()
        if containment:
            if job is not None:
                # Strictly checked: a worker that is not inside the job object
                # before "go" would be uncontained.
                try:
                    _assign_process_to_job(job, process.pid)
                except OSError as error:
                    raise WorkerProcessError(
                        "parallel meshing needs a contained worker process tree, but "
                        f"the worker could not be assigned to it: {error}",
                        remote_type="ContainmentUnavailable",
                    ) from error
                # The tree now covers the worker itself; before this point a
                # predispatch refusal must stop the worker directly, never
                # trust the (still empty) job object as its retirement.
                tree.mark_assigned()
            if on_process is not None:
                # Ownership is registered before the handshake: a shutdown or
                # cancellation that lands during startup can already reach the
                # worker, which has no children before "go".
                on_process(process)
            ready = _await_startup(control_parent, token)
            if ready[0] != "ready":
                raise WorkerProcessError(
                    f"the contained mesh worker refused to start: {ready[1]}",
                    remote_type="ContainmentUnavailable",
                )
            if os.name != "nt" and ready[1] is not None:
                # The worker reports the group it created by setsid; the tree
                # identity registered below is the one constructed above.
                tree.set_group(int(ready[1]))
            if on_tree is not None:
                # The full owned tree is registered before "go": nothing may
                # be dispatched unless the caller can already terminate it.
                on_tree(process, tree)
            if _is_cancelled(token):
                raise WorkerProcessError(
                    "the job was cancelled before the contained worker was dispatched",
                    remote_type="CancelledBeforeDispatch",
                )
            try:
                control_parent.send("go")
            except OSError as error:
                raise WorkerProcessError(
                    "the contained mesh worker could not be dispatched",
                    remote_type="ContainmentUnavailable",
                ) from error
        elif on_process is not None:
            on_process(process)

        outcome: tuple | None = None
        while outcome is None:
            if _is_cancelled(token) and not cancel_event.is_set():
                cancel_event.set()
            if receiver.poll(_CONTAINED_POLL_SECONDS if containment else _POLL_SECONDS):
                try:
                    message = loads(receiver.recv_bytes())
                except (EOFError, OSError):
                    break
                if message[0] == "progress":
                    progress(_progress_value(message[1], message[2]))
                elif message[0] == "object":
                    progress(message[1])
                else:
                    outcome = message
            elif not process.is_alive() and not receiver.poll(0):
                break
    finally:
        cleanup_failures = _cleanup_worker(
            process, receiver, sender, tree, control_parent, control_child, started
        )
        if cleanup_failures:
            # An exception is already propagating: keep its identity and add
            # the cleanup details to it (never mask the original failure).
            in_flight = sys.exc_info()[1]
            if in_flight is not None:
                _cleanup_note(in_flight, cleanup_failures)

    if outcome is None:
        error = WorkerProcessError(
            f"the worker process ended unexpectedly (exit code {process.exitcode}); "
            "the application and the model are unaffected",
            remote_type="WorkerProcessExited",
        )
        if cleanup_failures:
            _cleanup_note(error, cleanup_failures)
        raise error
    if outcome[0] == "result":
        if cleanup_failures:
            # Mandatory cleanup failed: the result exists but is not
            # published while its worker tree was not confirmed retired.
            raise WorkerProcessError(
                "the worker process tree could not be retired, so the result is "
                f"discarded: {'; '.join(cleanup_failures)}",
                remote_type="WorkerCleanupFailed",
            )
        return outcome[1]
    if outcome[0] == "encoded_result":
        # A result that refuses pickle travelled as plain data.  The worker
        # tree is already confirmed retired (the cleanup above), so the
        # decode happens here in the waiting job thread, before the result
        # is returned and attached.  A decode failure is a structured
        # transfer failure; the completed solve is never retried.
        if cleanup_failures:
            raise WorkerProcessError(
                "the worker process tree could not be retired, so the result is "
                f"discarded: {'; '.join(cleanup_failures)}",
                remote_type="WorkerCleanupFailed",
            )
        from .result_transport import decode_result

        if _is_cancelled(token):
            # Cancellation stays honoured while the completed result is being
            # prepared for attachment: a job cancelled at this boundary ends
            # cancelled, and its result is never attached.
            raise WorkerProcessError(
                "the job was cancelled while its completed result was being "
                "transferred, so the result is not attached",
                remote_type="CancelledDuringTransfer",
            )
        progress("transferring the completed result")
        try:
            decoded = decode_result(outcome[1])
        except Exception as error:
            raise WorkerProcessError(
                f"the result could not be transferred to the application: "
                f"{type(error).__name__}: {error}",
                remote_type="ResultTransferError",
            ) from error
        if _is_cancelled(token):
            raise WorkerProcessError(
                "the job was cancelled while its completed result was being "
                "transferred, so the result is not attached",
                remote_type="CancelledDuringTransfer",
            )
        return decoded
    _, remote_type, message, remote_traceback, *rest = outcome
    error = WorkerProcessError(
        message,
        remote_type=remote_type,
        remote_traceback=remote_traceback,
        remote_diagnostic=rest[0] if rest else None,
    )
    if cleanup_failures:
        _cleanup_note(error, cleanup_failures)
    raise error


def _is_cancelled(token: Any) -> bool:
    value = getattr(token, "is_cancelled", None)
    if value is None:
        value = getattr(token, "cancelled", False)
    return bool(value() if callable(value) else value)
