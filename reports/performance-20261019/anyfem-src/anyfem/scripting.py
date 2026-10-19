"""Trusted, transactional Python scripting for an ANYfem document.

The scripting console is deliberately *not* an operating-system sandbox.  It
is an engineer automation surface for trusted local code.  What it does
isolate is the live document: code executes on a serialized working copy in a
worker thread and produces a validated proposal.  Only :meth:`ScriptRunner.commit`
can swap that proposal into the live :class:`~anyfem.document.DocumentSession`,
as one command and therefore one undo item.

Typical headless use::

    runner = ScriptRunner(session)
    result = runner.run("commands.run(commands.AddPoint(1, 2, 3))")
    assert result.committed
    runner.shutdown()

For a non-blocking UI, call :meth:`submit`, poll the returned
:class:`ScriptTask`, and call :meth:`commit` from the UI thread when it is
finished.
"""

from __future__ import annotations

import builtins
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import io
import json
import pickle
import sys
from threading import Event
import time
import traceback
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping
from uuid import UUID

import numpy as np

from . import commands as command_types
from .commands import Command
from .document import DocumentRevision, DocumentSession, canonical_hash
from .io.project_file import project_from_dict, project_to_dict
from .model.project import Project
from .selection import Selection, SelectionDomain, SelectionFilter

__all__ = [
    "ScriptCancelled",
    "ScriptCancellationToken",
    "ScriptCommit",
    "ScriptConflictError",
    "ScriptError",
    "ScriptExecutionError",
    "ScriptResult",
    "ScriptRunner",
    "ScriptSelection",
    "ScriptTask",
    "ScriptValidationError",
]


class ScriptError(RuntimeError):
    """Base error carrying console output from a rejected script."""

    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
        traceback_text: str = "",
    ) -> None:
        super().__init__(str(message))
        self.stdout = str(stdout)
        self.stderr = str(stderr)
        self.traceback_text = str(traceback_text)


class ScriptExecutionError(ScriptError):
    """The trusted Python source raised before it made a proposal."""


class ScriptValidationError(ScriptError):
    """The working document was not structurally valid or serializable."""


class ScriptConflictError(ScriptError):
    """The live document changed after the script working copy was made."""


class ScriptCancelled(ScriptError):
    """A queued or executing script was cancelled without committing."""


class ScriptCancellationToken:
    """Cooperative token also checked at Python line boundaries.

    Scripts may call ``cancellation_token.checkpoint()`` around long native
    library operations.  Ordinary Python loops are interrupted by the runner's
    per-thread trace hook.  A native function that does not return to Python
    remains visibly cancelling until that call returns, just like a numerical
    factorization.
    """

    def __init__(self) -> None:
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise ScriptCancelled("script cancelled")

    # Familiar aliases for scripts shared with other cooperative workers.
    check = checkpoint
    raise_if_cancelled = checkpoint


@dataclass(frozen=True)
class ScriptSelection:
    """Immutable, ordered selection/filter state copied with a script."""

    mode: str
    domain: str
    kinds: tuple[str, ...]
    items: tuple[Any, ...] = ()

    @classmethod
    def capture(cls, selection: Selection | None) -> "ScriptSelection | None":
        if selection is None:
            return None
        return cls(
            mode=selection.mode,
            domain=selection.domain.value,
            kinds=tuple(sorted(selection.allowed_kinds)),
            items=tuple(deepcopy(selection.ordered_items)),
        )

    def make(self) -> Selection:
        selection = Selection(
            mode=self.mode,
            domain=SelectionDomain(self.domain),
            kinds=self.kinds,
        )
        selection.restore(deepcopy(self.items))
        return selection

    def apply(self, selection: Selection | None) -> None:
        if selection is None:
            if self.items:
                raise ValueError("the live document has no selection service")
            return
        selection.set_filter(
            SelectionFilter(SelectionDomain(self.domain), frozenset(self.kinds))
        )
        # ``mode`` is normally the first/only filtered kind.  Preserve it when
        # a future multi-kind scripting client chooses a different active kind.
        if selection.mode != self.mode:
            selection._mode = self.mode  # one final notification comes below
        selection.restore(deepcopy(self.items))


@dataclass(frozen=True)
class ScriptResult:
    """Validated, uncommitted output of a worker-side script execution."""

    source: str
    filename: str
    base_revision: DocumentRevision
    base_content_hash: str
    base_selection: ScriptSelection | None
    selection: ScriptSelection | None
    document: Mapping[str, Any]
    document_hash: str
    model_hash: str
    base_model_hash: str
    meshes: Mapping[str, Any] = field(repr=False)
    base_mesh_hash: str = ""
    mesh_hash: str = ""
    stdout: str = ""
    stderr: str = ""
    return_value: Any = field(default=None, repr=False)
    elapsed_seconds: float = 0.0

    @property
    def project_changed(self) -> bool:
        return self.document_hash != self.base_revision.document_hash

    @property
    def selection_changed(self) -> bool:
        return self.selection != self.base_selection

    @property
    def meshes_changed(self) -> bool:
        return self.mesh_hash != self.base_mesh_hash


@dataclass(frozen=True)
class ScriptCommit:
    """Outcome returned after a proposal is applied on the owning thread."""

    result: ScriptResult
    committed: bool
    project_changed: bool
    selection_changed: bool
    meshes_changed: bool
    revision_before: DocumentRevision
    revision_after: DocumentRevision

    @property
    def stdout(self) -> str:
        return self.result.stdout

    @property
    def stderr(self) -> str:
        return self.result.stderr

    @property
    def return_value(self) -> Any:
        return self.result.return_value


class _ScriptCommands:
    """One ergonomic surface combining command constructors and the stack."""

    def __init__(self, session: DocumentSession) -> None:
        self.session = session

    @property
    def stack(self):
        return self.session.commands

    def run(self, command: Command, *, solver_affecting: bool = True) -> Any:
        if isinstance(command, command_types.AddSupport):
            support = command.support
            region = support.region
            if region is not None and region.id not in self.session.project.regions:
                try:
                    generated_singleton = UUID(str(region.id)).version == 5
                except ValueError:
                    generated_singleton = False
                if generated_singleton:
                    # ANYfem 0.3.0 GUI transcripts captured the source
                    # document's deterministic hidden singleton UUID.  A
                    # replay document has another document UUID, so recreate
                    # the singleton from the authoritative EntityRef instead.
                    command.support = replace(support, region=None)
        return self.session.execute(command, solver_affecting=solver_affecting)

    execute = run

    def run_many(
        self,
        commands: Iterable[Command],
        *,
        label: str = "script batch",
        solver_affecting: bool = True,
    ) -> list[Any]:
        pending = list(commands)
        if pending and all(not isinstance(item, Command) for item in pending):
            # Compatibility for transcripts emitted by ANYfem 0.3.0 before
            # the batch-recorder fix.  Those scripts used
            # ``run_many([run(A), run(B)])``: A and B have already executed on
            # this isolated working copy and the list contains only their
            # results.  The eventual ScriptRunner commit is still one atomic
            # live-document command, so returning the results is safe and
            # preserves the intended behavior without executing anything
            # twice.
            return pending
        if any(not isinstance(item, Command) for item in pending):
            raise TypeError(
                "commands.run_many expects Command objects; a batch cannot mix "
                "commands with already-executed return values"
            )
        return self.session.execute_many(
            pending, label=label, solver_affecting=solver_affecting
        )

    execute_many = run_many

    def undo(self) -> bool:
        return self.session.undo()

    def redo(self) -> bool:
        return self.session.redo()

    def __getattr__(self, name: str) -> Any:
        # ``commands.run(commands.AddPoint(...))`` is convenient in an editor,
        # while ``commands.stack.history()`` keeps the underlying API visible.
        try:
            return getattr(command_types, name)
        except AttributeError:
            return getattr(self.session.commands, name)


@dataclass
class _ScriptContext:
    project: Project
    selection: Selection | None
    commands: _ScriptCommands
    meshes: dict[str, Any]
    analyses: dict[str, Any]
    jobs: dict[str, Any]
    cancellation_token: ScriptCancellationToken

    def checkpoint(self) -> None:
        self.cancellation_token.checkpoint()


class ScriptTask:
    """A cancellable future returned by :meth:`ScriptRunner.submit`."""

    def __init__(
        self,
        future: Future[ScriptResult],
        token: ScriptCancellationToken,
    ) -> None:
        self._future = future
        self.cancellation_token = token

    def cancel(self) -> bool:
        self.cancellation_token.cancel()
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled() or self.cancellation_token.cancelled

    def running(self) -> bool:
        return self._future.running()

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> ScriptResult:
        try:
            return self._future.result(timeout=timeout)
        except CancelledError as error:
            raise ScriptCancelled("script cancelled before execution") from error

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)

    def add_done_callback(self, callback: Callable[["ScriptTask"], Any]) -> None:
        self._future.add_done_callback(lambda _future: callback(self))


class ScriptRunner:
    """Run trusted Python on isolated document clones and commit atomically."""

    def __init__(
        self,
        session: DocumentSession,
        *,
        validator: Callable[[Project], None] | None = None,
        max_workers: int = 1,
    ) -> None:
        if not isinstance(session, DocumentSession):
            raise TypeError("ScriptRunner needs a DocumentSession")
        if int(max_workers) != 1:
            raise ValueError("one ScriptRunner intentionally serializes scripts")
        self.session = session
        self.validator = validator
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="anyfem-script"
        )
        self._closed = False
        self._tasks: set[ScriptTask] = set()

    def submit(
        self,
        source: str,
        *,
        filename: str = "<ANYfem script>",
    ) -> ScriptTask:
        """Start one isolated execution and return immediately.

        Capturing the canonical document occurs before dispatch so the worker
        has an exact revision even if the engineer keeps selecting or editing.
        The potentially more expensive reconstruction and user code run on the
        worker thread.
        """

        if self._closed:
            raise RuntimeError("this ScriptRunner has been shut down")
        if not isinstance(source, str):
            raise TypeError("script source must be text")
        if not isinstance(filename, str) or not filename:
            raise ValueError("script filename cannot be empty")

        snapshot = self.session.snapshot()
        document = deepcopy(dict(snapshot.document))
        selection = ScriptSelection.capture(self.session.selection)
        meshes = deepcopy(dict(self.session.mesh_cache))
        token = ScriptCancellationToken()
        future = self._executor.submit(
            self._execute,
            source,
            filename,
            snapshot.revision,
            document,
            selection,
            meshes,
            token,
        )
        task = ScriptTask(future, token)
        self._tasks.add(task)
        future.add_done_callback(lambda _future: self._tasks.discard(task))
        return task

    def run(
        self,
        source: str,
        *,
        filename: str = "<ANYfem script>",
        timeout: float | None = None,
        commit: bool = True,
        label: str = "run script",
    ) -> ScriptCommit | ScriptResult:
        """Blocking headless convenience; execution still occurs off-thread."""

        result = self.submit(source, filename=filename).result(timeout=timeout)
        return self.commit(result, label=label) if commit else result

    def commit(
        self,
        result: ScriptResult,
        *,
        label: str = "run script",
    ) -> ScriptCommit:
        """Apply a validated proposal as exactly one undoable transaction.

        This method mutates the session and must therefore be called from the
        session's owning/UI thread.  Revision conflicts fail closed rather than
        replacing intervening engineer edits.
        """

        if not isinstance(result, ScriptResult):
            raise TypeError("commit expects a ScriptResult")
        if self._closed:
            raise RuntimeError("this ScriptRunner has been shut down")
        if self.session.read_only:
            raise PermissionError("this project is open read-only")

        current_document = project_to_dict(self.session.project)
        current_content_hash = canonical_hash(current_document)
        if current_content_hash != result.base_content_hash:
            raise ScriptConflictError(
                "the model changed while the script was running; its working "
                "copy was not committed",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        current_selection = ScriptSelection.capture(self.session.selection)
        if result.selection_changed and current_selection != result.base_selection:
            raise ScriptConflictError(
                "the selection changed while the script was running; its "
                "selection edit was not committed",
                stdout=result.stdout,
                stderr=result.stderr,
            )

        if result.meshes_changed:
            current_mesh_hash = _object_hash(dict(self.session.mesh_cache))
            if current_mesh_hash != result.base_mesh_hash:
                raise ScriptConflictError(
                    "the active mesh cache changed while the script was running; "
                    "its working copy was not committed",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

        # Validate again at the trust boundary.  ScriptResult contains plain
        # Python mappings and a caller may have retained and modified them.
        after_project = _validated_project(result.document, self.validator)
        canonical_document = project_to_dict(after_project)
        after_session = DocumentSession(after_project)
        project_changed = (
            after_session.revision.document_hash
            != result.base_revision.document_hash
        )
        selection_changed = result.selection_changed
        meshes_changed = result.meshes_changed
        before_revision = self.session.revision

        if not (project_changed or selection_changed or meshes_changed):
            return ScriptCommit(
                result=result,
                committed=False,
                project_changed=False,
                selection_changed=False,
                meshes_changed=False,
                revision_before=before_revision,
                revision_after=before_revision,
            )

        command = _ApplyScriptState(
            session=self.session,
            document=canonical_document,
            selection=result.selection if selection_changed else None,
            meshes=(dict(result.meshes) if meshes_changed else None),
            label=str(label) or "run script",
        )
        solver_affecting = after_session.revision.model_hash != result.base_model_hash
        self.session.execute(command, solver_affecting=solver_affecting)
        return ScriptCommit(
            result=result,
            committed=True,
            project_changed=project_changed,
            selection_changed=selection_changed,
            meshes_changed=meshes_changed,
            revision_before=before_revision,
            revision_after=self.session.revision,
        )

    def cancel_all(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if cancel_futures:
            self.cancel_all()
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> "ScriptRunner":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    def _execute(
        self,
        source: str,
        filename: str,
        base_revision: DocumentRevision,
        base_document: Mapping[str, Any],
        base_selection: ScriptSelection | None,
        base_meshes: Mapping[str, Any],
        token: ScriptCancellationToken,
    ) -> ScriptResult:
        started = time.perf_counter()
        token.checkpoint()
        try:
            project = project_from_dict(deepcopy(dict(base_document)))
            selection = (
                None if base_selection is None else base_selection.make()
            )
            working_session = DocumentSession(project, selection=selection)
            meshes = deepcopy(dict(base_meshes))
            working_session.mesh_cache.update(meshes)
            commands = _ScriptCommands(working_session)
            context = _ScriptContext(
                project=project,
                selection=selection,
                commands=commands,
                meshes=working_session.mesh_cache,
                analyses=project.analyses,
                jobs=project.jobs,
                cancellation_token=token,
            )
        except BaseException as error:
            raise ScriptValidationError(
                f"cannot create script working copy: {error}"
            ) from error

        namespace: dict[str, Any] = {
            "__name__": "__anyfem_script__",
            "__file__": filename,
            "__builtins__": builtins.__dict__,
            "project": project,
            "selection": selection,
            "commands": commands,
            "cmd": command_types,
            "meshes": context.meshes,
            "analyses": context.analyses,
            "jobs": context.jobs,
            "context": context,
            "cancellation_token": token,
            "checkpoint": token.checkpoint,
            "np": np,
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        trace = _cancellation_trace(token, filename)
        previous_trace = sys.gettrace()
        caught: BaseException | None = None
        traceback_text = ""
        try:
            compiled = compile(source, filename, "exec")
            with redirect_stdout(stdout), redirect_stderr(stderr):
                sys.settrace(trace)
                try:
                    exec(compiled, namespace, namespace)
                    token.checkpoint()
                except BaseException as error:
                    caught = error
                    traceback_text = traceback.format_exc()
                finally:
                    sys.settrace(previous_trace)
        except BaseException as error:
            # Compile-time failures occur before redirection/trace setup.
            caught = error
            traceback_text = traceback.format_exc()
            sys.settrace(previous_trace)

        output = stdout.getvalue()
        errors = stderr.getvalue()
        if caught is not None:
            if isinstance(caught, ScriptCancelled) or token.cancelled:
                raise ScriptCancelled(
                    "script cancelled",
                    stdout=output,
                    stderr=errors,
                    traceback_text=traceback_text,
                ) from caught
            raise ScriptExecutionError(
                f"{type(caught).__name__}: {caught}",
                stdout=output,
                stderr=errors,
                traceback_text=traceback_text,
            ) from caught

        try:
            token.checkpoint()
            validated = _validated_project(project_to_dict(project), self.validator)
            document = project_to_dict(validated)
            validated_session = DocumentSession(validated)
            resulting_selection = ScriptSelection.capture(selection)
            _validate_selection(resulting_selection, validated)
            final_meshes = deepcopy(dict(context.meshes))
            base_mesh_hash = _object_hash(dict(base_meshes))
            mesh_hash = _object_hash(final_meshes)
            return_value = _safe_copy(namespace.get("result"))
            token.checkpoint()
        except ScriptCancelled as error:
            raise ScriptCancelled(
                "script cancelled",
                stdout=output,
                stderr=errors,
            ) from error
        except BaseException as error:
            raise ScriptValidationError(
                f"script result is not a valid ANYfem document: {error}",
                stdout=output,
                stderr=errors,
                traceback_text=traceback.format_exc(),
            ) from error

        return ScriptResult(
            source=source,
            filename=filename,
            base_revision=base_revision,
            base_content_hash=canonical_hash(base_document),
            base_selection=base_selection,
            selection=resulting_selection,
            document=MappingProxyType(deepcopy(document)),
            document_hash=validated_session.revision.document_hash,
            model_hash=validated_session.revision.model_hash,
            base_model_hash=base_revision.model_hash,
            meshes=MappingProxyType(final_meshes),
            base_mesh_hash=base_mesh_hash,
            mesh_hash=mesh_hash,
            stdout=output,
            stderr=errors,
            return_value=return_value,
            elapsed_seconds=time.perf_counter() - started,
        )


class _ApplyScriptState(Command):
    """Aggregate project/selection/cache swap retained as one undo item."""

    def __init__(
        self,
        *,
        session: DocumentSession,
        document: Mapping[str, Any],
        selection: ScriptSelection | None,
        meshes: Mapping[str, Any] | None,
        label: str,
    ) -> None:
        self.session = session
        self.document = deepcopy(dict(document))
        self.after_selection = selection
        self.after_meshes = None if meshes is None else deepcopy(dict(meshes))
        self.label = label
        self._before_document: dict[str, Any] | None = None
        self._before_selection: ScriptSelection | None = None
        self._before_meshes: dict[str, Any] | None = None

    def do(self, project: Project) -> Project:
        if self._before_document is None:
            # ANYgeometry exposes immutable MappingProxyType topology views.
            # Project JSON is the supported deep working-copy boundary and is
            # also what was validated before this command was constructed.
            self._before_document = deepcopy(project_to_dict(project))
            self._before_selection = ScriptSelection.capture(self.session.selection)
            self._before_meshes = deepcopy(dict(self.session.mesh_cache))
        self._apply(project, after=True)
        return project

    def undo(self, project: Project) -> None:
        if self._before_document is None:
            raise RuntimeError("script command has not been applied")
        self._apply(project, after=False)

    def redo(self, project: Project) -> Project:
        if self._before_document is None:
            return self.do(project)
        self._apply(project, after=True)
        return project

    def _apply(self, project: Project, *, after: bool) -> None:
        rollback_document = deepcopy(project_to_dict(project))
        rollback_selection = ScriptSelection.capture(self.session.selection)
        rollback_meshes = deepcopy(dict(self.session.mesh_cache))
        try:
            if after:
                replacement = project_from_dict(deepcopy(self.document))
                wanted_selection = self.after_selection
                wanted_meshes = self.after_meshes
            else:
                assert self._before_document is not None
                replacement = project_from_dict(deepcopy(self._before_document))
                wanted_selection = self._before_selection
                wanted_meshes = self._before_meshes

            project.__dict__.clear()
            project.__dict__.update(replacement.__dict__)
            project.__post_init__()

            if wanted_selection is not None:
                wanted_selection.apply(self.session.selection)
            if wanted_meshes is not None:
                self.session.mesh_cache.clear()
                self.session.mesh_cache.update(deepcopy(wanted_meshes))
        except BaseException:
            rollback = project_from_dict(rollback_document)
            project.__dict__.clear()
            project.__dict__.update(rollback.__dict__)
            project.__post_init__()
            if rollback_selection is not None:
                rollback_selection.apply(self.session.selection)
            self.session.mesh_cache.clear()
            self.session.mesh_cache.update(rollback_meshes)
            raise


def _validated_project(
    document: Mapping[str, Any],
    validator: Callable[[Project], None] | None,
) -> Project:
    """Round-trip and structurally validate an editable, possibly partial model."""

    payload = deepcopy(dict(document))
    # Canonical encoding rejects NaN/Infinity and non-serializable values.
    json.dumps(payload, sort_keys=True, allow_nan=False)
    project = project_from_dict(payload)
    topology_errors = project.geometry.validate_topology()
    if topology_errors:
        raise ValueError("invalid geometry topology: " + "; ".join(topology_errors))
    project.geometry.features.validate()
    round_trip = project_to_dict(project)
    if canonical_hash(round_trip) != canonical_hash(payload):
        raise ValueError("the project is not stable across its v4 serialization")
    if validator is not None:
        validator(project)
    return project


def _cancellation_trace(token: ScriptCancellationToken, filename: str):
    def trace(frame, event, arg):
        del event, arg
        if frame.f_code.co_filename == filename:
            token.checkpoint()
        return trace

    return trace


def _validate_selection(
    selection: ScriptSelection | None, project: Project
) -> None:
    """Refuse a successful proposal that would leave dangling geometry picks."""

    if selection is None:
        return
    for reference in selection.items:
        if getattr(reference, "kind", None) not in ("vertex", "edge", "face"):
            # Mesh scopes are revision-bound and may live in a caller-provided
            # cache rather than the editable Project.  Their owner validates
            # them when that mesh is activated.
            continue
        if not project.geometry.resolve_ref(reference):
            raise ValueError(
                f"script selection references missing {reference}; clear it or "
                "select a surviving/replacement entity"
            )


def _object_hash(value: Any) -> str:
    """Stable-enough in-process digest for non-serialized mesh working data."""

    try:
        payload = pickle.dumps(value, protocol=5)
    except (TypeError, ValueError, pickle.PickleError) as error:
        raise ValueError(f"script mesh data is not copyable: {error}") from error
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _safe_copy(value: Any) -> Any:
    try:
        return deepcopy(value)
    except BaseException:
        # ``result`` is a convenience for a console, never document state.  A
        # representation is safer than leaking the mutable working project.
        return repr(value)
