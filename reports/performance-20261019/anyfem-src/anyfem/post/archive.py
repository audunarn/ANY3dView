"""Optional native ANYpost archive consumer and standalone retained GPU viewer.

Existing ANYfem result sidecars and solver result objects remain separate APIs.
Importing this module does not import ANYpost, Tk, or a concrete viewer backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Lock, RLock, get_ident
from typing import Any, Callable


class PreparedArchive:
    """Single-use reader and snapshots prepared before owner-thread attachment.

    Call ``close()`` to discard an unused preparation. Once attached, reader
    ownership transfers to ArchiveSession and only that session closes it.
    """

    def __init__(self, reader: Any, metadata: dict[str, Any], geometry: Any, scalar_proofs: Any, *, cases_are_summaries: bool = False) -> None:
        self._lock = Lock()
        self._cases_are_summaries = cases_are_summaries
        self._reader = reader
        self._metadata = metadata
        self._geometry = geometry
        self._scalar_proofs = scalar_proofs

    def _claim(self):
        with self._lock:
            if self._reader is None:
                raise RuntimeError("prepared archive has been attached or discarded")
            result = self._reader, self._metadata, self._geometry, self._scalar_proofs
            self._reader = self._metadata = self._geometry = self._scalar_proofs = None
            return result

    def close(self) -> None:
        with self._lock:
            if self._reader is not None:
                self._reader.close()
                self._reader = self._metadata = self._geometry = self._scalar_proofs = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ArchiveSession:
    """Own a reader and one retained result mesh on the caller's UI thread.

    ``viewer`` may be an ANY3dView viewer or an ANYfem viewport exposing its
    viewer as ``canvas``. Supply the current model/mesh identities when attaching
    to an existing engineering model. A standalone archive viewer can omit them;
    the archive's own nonempty identities are still validated.

    ``dispatch(callback)`` must queue callbacks on this session's owner thread.
    The default is the viewer's thread-safe ``submit_update`` queue. Queued
    callbacks are retired on close, and an incorrect dispatcher fails before a
    callback can mutate the viewer. ``view`` exposes load errors and diagnostics.
    """

    def __init__(
        self,
        path: str | Path,
        viewer: Any,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        *,
        expected_model_identity: str | None = None,
        expected_mesh_identity: str | None = None,
        cache_bytes: int = 2 * 1024**3,
        working_bytes: int = 256 * 1024**2,
        on_presented: Callable[..., None] | None = None,
    ) -> None:
        self._owner = get_ident()
        self._closed = False
        self._resources_closed = False
        self._sample_map_key: tuple[str, str] | None = None
        self._sample_lock = RLock()
        self._sample_generation = 0
        self._sample_loading = set()
        self._sample_associations = None
        self._sample_ordinals = None
        self._sample_indices = None
        self._sample_index = None
        self._sample_maximum = 0
        self._sample_reads = 0
        self._sample_sorts = 0
        self._sample_selector_builds = 0
        self._sample_cache_owner = f"anyfem-native-samples:{id(self)}"
        self._native_scalar_proofs: dict[tuple[str, str], bool] = {}
        self._native_shape_quantities: set[str] = set()
        self._prepared_cases = self._prepared_summaries = self._prepared_descriptors = None
        def fail(error):
            if isinstance(path, PreparedArchive):
                path.close()
            raise error
        target = viewer if callable(getattr(viewer, "add_mesh_arrays", None)) else getattr(viewer, "canvas", None)
        if target is None or not callable(getattr(target, "add_mesh_arrays", None)):
            fail(TypeError("viewer must expose add_mesh_arrays, directly or through canvas"))
        dispatcher = dispatch if dispatch is not None else getattr(target, "submit_update", None)
        if not callable(dispatcher):
            fail(TypeError("an owner-thread dispatcher is required"))
        # Prefer explicitly supplied identities, then identities on the host.
        model_id = expected_model_identity if expected_model_identity is not None else getattr(viewer, "model_identity", None)
        mesh_id = expected_mesh_identity if expected_mesh_identity is not None else getattr(viewer, "mesh_identity", None)
        for name, value in (("model", model_id), ("mesh", mesh_id)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                fail(ValueError(f"expected {name} identity must be a nonempty string"))
        try:
            from anypost import open_results
            from anypost.presentation import RetainedResultView
        except ImportError as error:
            fail(ImportError("native result viewing needs ANYpost; install ANYfem[post]"))

        self.viewer = target
        prepared = path if isinstance(path, PreparedArchive) else None
        if prepared is not None:
            prepared_cases_are_summaries = prepared._cases_are_summaries
            self.reader, metadata, geometry, scalar_proofs = prepared._claim()
        else:
            self.reader = open_results(
                path,
                cache_bytes=cache_bytes,
                working_bytes=working_bytes,
                expected_model_identity=model_id,
                expected_mesh_identity=mesh_id,
            )
        try:
            if prepared is None:
                summary_method = getattr(self.reader, "metadata_summary", None)
                metadata = summary_method() if callable(summary_method) else self.reader.metadata()
            self._validate_archive_identities(metadata, model_id, mesh_id)
            self.model_identity = metadata["model_identity"]
            self.mesh_identity = metadata["mesh_identity"]
            # The immutable source header has already been copied once. Keep
            # only a compact axis-admission proof, never the per-entity shapes.
            if prepared is None:
                self._remember_native_scalar_proofs(metadata.get("descriptors", ()))
            else:
                self._native_scalar_proofs = scalar_proofs
                self._native_shape_quantities.update(quantity for quantity, _ in scalar_proofs)

            def publish(callback: Callable[[], None]) -> None:
                def owned_callback() -> None:
                    self._assert_owner()
                    if not self._closed:
                        callback()

                if not self._closed:
                    dispatcher(owned_callback)

            options = {} if on_presented is None else {"on_presented": on_presented}
            if prepared is not None:
                if prepared_cases_are_summaries:
                    self._prepared_summaries = tuple(metadata["cases"])
                else:
                    self._prepared_cases = tuple(metadata["cases"])
                self._prepared_descriptors = tuple(metadata["descriptors"])
                options.update(geometry=geometry, descriptors=self._prepared_descriptors)
            self._dispatch = publish
            self.view = RetainedResultView(self.reader, target, publish, **options)
            descriptors=getattr(self.view,"_descriptors",None)
            if descriptors is not None:self._remember_native_scalar_proofs(descriptors.values())
        except BaseException:
            self._closed = True
            self.reader.close()
            raise

    @staticmethod
    def _validate_archive_identities(metadata, model_id=None, mesh_id=None):
        for key, expected in (("model_identity", model_id), ("mesh_identity", mesh_id)):
            actual = metadata.get(key)
            if not isinstance(actual, str) or not actual.strip():
                raise ValueError(f"archive has no valid {key}")
            if expected is not None and actual != expected:
                raise ValueError(f"archive {key} does not match the current model")

    @classmethod
    def prepare(cls, path, *, expected_model_identity=None, expected_mesh_identity=None,
                cache_bytes=2 * 1024**3, working_bytes=256 * 1024**2) -> PreparedArchive:
        """Open and prefetch on an I/O executor before creating an active UI."""
        from anypost import open_results
        reader = open_results(path, cache_bytes=cache_bytes, working_bytes=working_bytes,
                              expected_model_identity=expected_model_identity,
                              expected_mesh_identity=expected_mesh_identity)
        try:
            summary_method = getattr(reader, "metadata_summary", None)
            cases_are_summaries = callable(summary_method)
            metadata = summary_method() if cases_are_summaries else reader.metadata()
            if cases_are_summaries and metadata.pop("cases_complete", False):
                cases_are_summaries = False
            cls._validate_archive_identities(metadata, expected_model_identity, expected_mesh_identity)
            # Native headers contain both tables, so no second descriptor copy.
            if "cases" not in metadata:
                case_method = getattr(reader, "case_summaries", None) if cases_are_summaries else None
                metadata["cases"] = case_method() if callable(case_method) else reader.cases()
            if "descriptors" not in metadata:
                metadata["descriptors"] = reader.components()
            geometry = reader.geometry()
            proofs = cls._native_scalar_proofs_from(metadata["descriptors"])
            return PreparedArchive(reader, metadata, geometry, proofs, cases_are_summaries=cases_are_summaries)
        except BaseException:
            reader.close()
            raise

    def _assert_owner(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError("archive session must run on its owner thread")

    def _assert_open(self) -> None:
        self._assert_owner()
        if self._closed:
            raise RuntimeError("archive session has been closed")

    @property
    def handle(self) -> Any:
        """The retained MeshHandle; component switches preserve its topology."""
        self._assert_open()
        return self.view.handle

    def cases(self) -> tuple[dict[str, Any], ...]:
        self._assert_open()
        if self._prepared_cases is not None:
            return tuple(json.loads(json.dumps(record)) for record in self._prepared_cases)
        return self.reader.cases()

    def case_metadata(self, case_id: str) -> dict[str, Any]:
        self._assert_open()
        method = getattr(self.reader, "case_metadata", None)
        if callable(method):
            return method(case_id)
        for record in self.cases():
            if record["case_id"] == case_id:
                return json.loads(json.dumps(record))
        raise ValueError("unknown case identity")

    def case_summaries(self) -> tuple[dict[str, Any], ...]:
        self._assert_open()
        if self._prepared_summaries is not None:
            return tuple(json.loads(json.dumps(record)) for record in self._prepared_summaries)
        method = getattr(self.reader, "case_summaries", None)
        if callable(method):
            return method()
        names = ("case_id", "status", "certified", "route", "input_index", "active_indices")
        return tuple({key: json.loads(json.dumps(record[key])) for key in names if key in record} for record in self.cases())

    def components(self, quantity_id: str | None = None) -> tuple[dict[str, Any], ...]:
        self._assert_open()
        if self._prepared_descriptors is not None:
            return tuple(dict(record) for record in self._prepared_descriptors
                         if quantity_id is None or record["quantity_id"] == quantity_id)
        return self.reader.components(quantity_id)

    def request_component(
        self, case_id: str, quantity_id: str, component: str, selection: Any = None
    ) -> int:
        """Queue latest-request-wins loading; ambiguous entity samples refuse."""
        self._assert_open()
        with self._sample_lock:self._sample_generation+=1
        return self.view.request_component(case_id, quantity_id, component, selection)

    def _clear_sample_map(self) -> None:
        with self._sample_lock:
            self._sample_generation+=1
            self._sample_map_key = None
            self._sample_associations = None
            self._sample_ordinals = None
            self._sample_indices = None
            self._sample_index = None
            self._sample_maximum = 0
        self._account_samples()

    def _sample_resident_bytes(self) -> int:
        return sum(array.nbytes for array in (self._sample_associations, self._sample_ordinals, self._sample_indices) if array is not None)

    def _account_samples(self) -> None:
        accounting = getattr(self.reader, "set_external_cache_bytes", None)
        if accounting is not None:
            accounting(self._sample_cache_owner, self._sample_resident_bytes())

    def _check_sample_cancel(self, cancel) -> None:
        if self._closed or (cancel is not None and cancel.is_set()):
            raise CancelledError("native sample selection was retired")

    @staticmethod
    def _native_scalar_proofs_from(descriptors, quantity_id=None):
        proofs = {}
        for descriptor in descriptors:
            quantity = descriptor.get("quantity_id", quantity_id)
            component = descriptor.get("component")
            if not isinstance(quantity, str) or not isinstance(component, str):
                continue
            metadata = descriptor.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            scalar = True
            for shapes in (descriptor.get("element_shapes"), metadata.get("element_shapes")):
                if isinstance(shapes, dict) and any(
                    isinstance(shape, (list, tuple)) and len(shape) > 1
                    for shape in shapes.values()
                ):
                    scalar = False
                    break
            proofs[(quantity, component)] = scalar
        return proofs

    def _remember_native_scalar_proofs(self, descriptors, quantity_id=None) -> None:
        proofs = self._native_scalar_proofs_from(descriptors, quantity_id)
        self._native_scalar_proofs.update(proofs)
        self._native_shape_quantities.update(quantity for quantity, _ in proofs)
        if quantity_id is not None:
            self._native_shape_quantities.add(quantity_id)

    def _check_sample_generation(self, generation, cancel=None):
        self._check_sample_cancel(cancel)
        if generation!=self._sample_generation:
            raise CancelledError("native sample map generation was superseded")

    def _install_sample_state(self, changes, byte_count, generation, cancel=None):
        # Never hold the sample lock while accounting may wait on archive I/O.
        with self._sample_lock:self._check_sample_generation(generation,cancel)
        accounting=getattr(self.reader,"set_external_cache_bytes",None)
        if accounting is not None:accounting(self._sample_cache_owner,byte_count)
        try:
            with self._sample_lock:
                self._check_sample_generation(generation,cancel)
                for name,value in changes.items():setattr(self,name,value)
        except BaseException:
            with self._sample_lock:current=self._sample_resident_bytes()
            if accounting is not None:accounting(self._sample_cache_owner,current)
            raise

    def _prepare_sample_map(self, quantity_id: str, component: str, cancel=None, generation=None) -> None:
        """Read/build outside the state lock; install only the current request."""
        key=(quantity_id,component)
        descriptors=getattr(self.view,"_descriptors",None)
        if quantity_id not in self._native_shape_quantities and descriptors is None:
            # Older/custom views lack a header snapshot: copy once outside
            # the sample-state lock, with cooperative checks on both sides.
            self._check_sample_cancel(cancel)
            records=self.reader.components(quantity_id)
            self._check_sample_cancel(cancel)
        else:records=descriptors.values() if descriptors is not None else ()
        with self._sample_lock:
            generation=self._sample_generation if generation is None else generation
            self._check_sample_generation(generation,cancel)
            if quantity_id not in self._native_shape_quantities:
                self._remember_native_scalar_proofs(records,quantity_id)
            if not self._native_scalar_proofs.get(key,True):
                from anypost import UnavailableFieldError
                raise UnavailableFieldError("native sample ordinals require scalar rank components without extra component axes; choose explicit flat indices")
            if key==self._sample_map_key:return
            old_ids=self._sample_associations
            resident=self._sample_resident_bytes()
        import numpy as np
        from anypost import WorkingSetError
        kwargs={"cancel_event":cancel} if getattr(self.reader,"supports_cancellation",False) is True else {}
        ids=self.reader.associations(quantity_id,component,**kwargs)
        self._check_sample_cancel(cancel)
        with self._sample_lock:
            self._sample_reads+=1
            self._check_sample_generation(generation,cancel)
        if not isinstance(ids,np.ndarray) or ids.ndim!=1 or ids.dtype.kind not in "iu":
            raise ValueError("native samples require a one-dimensional integer entity association")
        count=len(ids);limit=self.reader.working_bytes
        compare=ids.nbytes+resident+count+256
        if compare>limit:raise WorkingSetError(f"native sample comparison needs at most {compare} working bytes; cap is {limit}")
        if old_ids is not None and np.array_equal(ids,old_ids):
            with self._sample_lock:
                self._check_sample_generation(generation,cancel);self._sample_map_key=key
            return
        working=count*80+resident+256
        if working>limit:raise WorkingSetError(f"native sample map needs at most {working} working bytes; cap is {limit}")
        if count:
            order=np.argsort(ids,kind="stable");ordered=ids[order]
            change=np.empty(count,dtype=bool);change[0]=True
            np.not_equal(ordered[1:],ordered[:-1],out=change[1:])
            starts=np.where(change,np.arange(count,dtype=np.int64),0);np.maximum.accumulate(starts,out=starts)
            grouped=np.arange(count,dtype=np.int64);grouped-=starts;grouped+=1
            maximum=int(grouped.max());ordinals=np.empty(count,dtype=np.int64);ordinals[order]=grouped
        else:maximum=0;ordinals=np.empty(0,dtype=np.int64)
        ordinals.flags.writeable=False;ids.flags.writeable=False
        self._install_sample_state(dict(_sample_associations=ids,_sample_ordinals=ordinals,
            _sample_indices=None,_sample_index=None,_sample_maximum=maximum,_sample_map_key=key),
            ids.nbytes+ordinals.nbytes,generation,cancel)
        if count:
            with self._sample_lock:self._sample_sorts+=1

    def _synchronous_sample_guard(self,key,ordinal=None):
        with self._sample_lock:
            if (self._sample_loading or getattr(self.view,"io_active",False)) and (key!=self._sample_map_key or (ordinal is not None and ordinal!=self._sample_index)):
                raise RuntimeError("native sample map is loading; use request_native_sample")

    def sample_count(self, quantity_id: str, component: str) -> int:
        """Cached counts never wait on worker I/O; uncached requests are explicit."""
        self._assert_open();self._synchronous_sample_guard((quantity_id,component))
        self._prepare_sample_map(quantity_id,component)
        with self._sample_lock:return self._sample_maximum

    @staticmethod
    def _validate_sample_index(sample_index: int) -> None:
        import numpy as np
        if isinstance(sample_index, (bool, np.bool_)) or not isinstance(sample_index, (int, np.integer)) or sample_index < 1:
            raise ValueError("native sample ordinal must be a positive integer")

    def _select_native_sample(self, sample_index: int, cancel=None, generation=None):
        import numpy as np
        from anypost import WorkingSetError
        with self._sample_lock:
            generation=self._sample_generation if generation is None else generation
            self._check_sample_generation(generation,cancel)
            if sample_index>self._sample_maximum:raise ValueError(f"native sample ordinal exceeds maximum {self._sample_maximum}")
            if sample_index==self._sample_index:return self._sample_indices
            ids=self._sample_associations;ordinals=self._sample_ordinals;resident=self._sample_resident_bytes()
        working=resident+len(ordinals)*9+256
        if working>self.reader.working_bytes:raise WorkingSetError(f"native sample selection needs at most {working} working bytes; cap is {self.reader.working_bytes}")
        selection=np.flatnonzero(ordinals==sample_index);selection.flags.writeable=False
        self._install_sample_state(dict(_sample_indices=selection,_sample_index=sample_index),
            ids.nbytes+ordinals.nbytes+selection.nbytes,generation,cancel)
        with self._sample_lock:self._sample_selector_builds+=1
        return selection

    def sample_selection(self, quantity_id: str, component: str, sample_index: int):
        """Return kth occurrences, never average; pending uncached work refuses."""
        self._assert_open();self._validate_sample_index(sample_index)
        self._synchronous_sample_guard((quantity_id,component),sample_index)
        self._prepare_sample_map(quantity_id,component)
        return self._select_native_sample(sample_index)

    @property
    def native_sample_diagnostics(self) -> dict[str, int]:
        """Compact cache evidence; query after a request completes."""
        self._assert_owner()
        with self._sample_lock:
            return {"association_reads": self._sample_reads, "sorts": self._sample_sorts,
                    "selector_builds": self._sample_selector_builds,
                    "resident_bytes": self._sample_resident_bytes(), "maximum": self._sample_maximum}

    def request_native_sample(
        self, case_id: str, quantity_id: str, component: str, ordinal: int,
        *, on_samples_ready: Callable[[int], None] | None = None,
    ) -> int:
        """Load a native ordinal and discover its range on the sole I/O worker.

        ``on_samples_ready(maximum)`` runs on the owning thread only for the
        latest request. It reports actual association-table counts; no GUI
        archive read, implicit averaging or invented count is needed.
        """
        self._assert_open()
        self._validate_sample_index(ordinal)
        if on_samples_ready is not None and not callable(on_samples_ready):
            raise TypeError("on_samples_ready must be callable")
        request = {}
        with self._sample_lock:
            self._sample_generation+=1;generation=self._sample_generation

        def selection_loader(cancel):
            with self._sample_lock:self._sample_loading.add(generation)
            try:
                self._prepare_sample_map(quantity_id,component,cancel,generation)
                with self._sample_lock:
                    self._check_sample_generation(generation,cancel);maximum=self._sample_maximum
                selection=self._select_native_sample(ordinal,cancel,generation)
                if on_samples_ready is not None:
                    def publish_options():
                        if self._sample_generation==generation and self.view.latest_request_id==request["id"]:
                            on_samples_ready(maximum)
                    self._dispatch(publish_options)
                return selection
            finally:
                with self._sample_lock:self._sample_loading.discard(generation)

        serial = self.view.request_component(case_id, quantity_id, component, selection_loader=selection_loader)
        request["id"] = serial
        return serial

    def close(self) -> None:
        """Retire publication and loading before releasing the reader.

        If view retirement fails (for example an I/O timeout), the reader is
        preserved and ``close`` can be retried. Requests remain retired.
        """
        self._assert_owner()
        if self._resources_closed:
            return
        self._closed = True
        with self._sample_lock:self._sample_generation+=1
        self.view.close()
        self._clear_sample_map()
        with self._sample_lock:
            self._native_scalar_proofs.clear()
            self._native_shape_quantities.clear()
        self._prepared_cases = self._prepared_summaries = self._prepared_descriptors = None
        self.reader.close()
        self._resources_closed = True

    def __enter__(self) -> ArchiveSession:
        self._assert_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    """Inspect metadata or open a real Tk/ModernGL archive viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--list", action="store_true", help="print identities, cases and components without reading fields")
    parser.add_argument("--model-identity")
    parser.add_argument("--mesh-identity")
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--quantity")
    parser.add_argument("--component")
    parser.add_argument("--indices", help="explicit comma-separated association row indices")
    parser.add_argument("--sample", type=int, help="1-based native sample ordinal per entity; GUI defaults visibly to Sample 1")
    args = parser.parse_args(argv)
    if args.list:
        from anypost import open_results
        with open_results(args.archive, expected_model_identity=args.model_identity, expected_mesh_identity=args.mesh_identity) as reader:
            metadata_method = getattr(reader, "metadata_summary", None)
            cases_method = getattr(reader, "case_summaries", None)
            metadata = metadata_method() if callable(metadata_method) else reader.metadata()
            cases = cases_method() if callable(cases_method) else reader.cases()
            print(json.dumps({"metadata": metadata, "cases": cases, "components": reader.components()}, indent=2))
        return 0

    # Finish archive work before Tk exists; an active UI never waits for HDF I/O.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="archive-prepare") as executor:
        prepared = executor.submit(ArchiveSession.prepare, args.archive,
                                   expected_model_identity=args.model_identity,
                                   expected_mesh_identity=args.mesh_identity).result()
    try:
        return _show_archive(args, prepared)
    finally:
        prepared.close()


def _show_archive(args, prepared: PreparedArchive) -> int:

    import tkinter as tk
    from tkinter import ttk
    from any3dview.gpu import Any3DView

    root = tk.Tk()
    root.title(f"ANYfem archive — {args.archive.name}")
    root.geometry("1000x720")
    toolbar = ttk.Frame(root)
    toolbar.pack(fill="x")
    viewer = Any3DView(root, width=1000, height=650)
    viewer.pack(fill="both", expand=True)
    status = tk.StringVar(value="Select a case and component")
    ttk.Label(root, textvariable=status).pack(fill="x")
    session: ArchiveSession | None = None
    try:
        def show_presented(record: Any) -> None:
            status.set(f"Displayed: {record.case_id} — {record.quantity_id} / {record.component}")

        session = ArchiveSession(prepared, viewer, expected_model_identity=args.model_identity, expected_mesh_identity=args.mesh_identity, on_presented=show_presented)
        # Failed/uncertified cases remain discoverable through --list, but cannot
        # be presented as engineering fields. The reader enforces this again.
        cases = [row for row in session.case_summaries() if row.get("certified") is True and row.get("status") in {"completed", "complete", "converged", "success", "succeeded", "ok"}]
        components = [row for row in session.components() if row.get("availability") != "unavailable"]
        if not cases or not components:
            raise ValueError("archive has no completed certified case/component to display")
        case_ids = [str(row["case_id"]) for row in cases]
        labels = [f'{row["quantity_id"]} / {row["component"]} [{row["unit"]}] ({row["basis"]}, {row["sampling"]})' for row in components]
        case_choice = ttk.Combobox(toolbar, state="readonly", values=case_ids, width=28)
        field_choice = ttk.Combobox(toolbar, state="readonly", values=labels, width=44)
        sample_label = ttk.Label(toolbar, text="Native sample:")
        sample_choice = ttk.Combobox(toolbar, state="readonly", width=16)
        case_choice.pack(side="left", padx=4, pady=4)
        field_choice.pack(side="left", padx=4, pady=4)
        sample_label.pack(side="left", padx=4)
        sample_choice.pack(side="left", padx=4, pady=4)
        case_choice.current(case_ids.index(args.case_id) if args.case_id is not None else 0)
        field_index = next((i for i, row in enumerate(components) if row["quantity_id"] == args.quantity and row["component"] == args.component), None)
        if (args.quantity is None) != (args.component is None):
            raise ValueError("--quantity and --component must be supplied together")
        if args.quantity is not None and field_index is None:
            raise ValueError("requested component is unavailable")
        field_choice.current(field_index if field_index is not None else 0)
        selection = [int(value) for value in args.indices.split(",")] if args.indices is not None else None
        if args.indices is not None and args.sample is not None:
            raise ValueError("--indices and --sample are alternative explicit selections")
        sample_component = None

        def request(*_: Any) -> None:
            nonlocal sample_component
            row = components[field_choice.current()]
            try:
                selected_rows = selection
                if selected_rows is None:
                    key = (row["quantity_id"], row["component"])
                    if key != sample_component:
                        ordinal = args.sample if sample_component is None and args.sample is not None else 1
                        if ordinal < 1:
                            raise ValueError("native sample ordinal must be positive")
                        # The worker discovers the actual range; this is the
                        # explicit requested ordinal, not an invented count.
                        sample_choice.configure(state="readonly", values=("All samples", f"Sample {ordinal}"))
                        sample_choice.set(f"Sample {ordinal}")
                        sample_component = key
                    chosen = sample_choice.get()
                    if chosen == "All samples":
                        serial = session.request_component(case_choice.get(), *key)
                    else:
                        ordinal = int(chosen.removeprefix("Sample "))

                        def sample_options(maximum: int) -> None:
                            if key != sample_component:
                                return
                            if maximum * 128 > session.reader.working_bytes:
                                sample_choice.configure(state="disabled", values=("Range too large",))
                                sample_choice.set("Range too large")
                                status.set("Native sample range exceeds menu budget; use explicit --indices")
                                return
                            current = sample_choice.get()
                            choices = ["All samples"] + [f"Sample {index}" for index in range(1, maximum + 1)]
                            sample_choice.configure(values=choices)
                            sample_choice.set(current if current in choices else "Choose native sample")

                        serial = session.request_native_sample(case_choice.get(), *key, ordinal, on_samples_ready=sample_options)
                else:
                    sample_choice.configure(state="disabled", values=("Explicit indices",))
                    sample_choice.set("Explicit indices")
                    serial = session.request_component(case_choice.get(), row["quantity_id"], row["component"], selected_rows)
                status.set(f"Requested {serial}: {case_choice.get()} — {labels[field_choice.current()]}")
            except Exception as error:
                status.set(f"Refused: {error}")

        def poll_error() -> None:
            error = getattr(session.view, "error", None)
            if error is not None:
                status.set(f"Refused: {error}")
            if not session._closed:
                root.after(100, poll_error)

        case_choice.bind("<<ComboboxSelected>>", request)
        field_choice.bind("<<ComboboxSelected>>", request)
        sample_choice.bind("<<ComboboxSelected>>", request)
        viewer.fit_to_scene()
        request()
        poll_error()

        def close() -> None:
            session.close()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", close)
        root.mainloop()
    finally:
        if session is not None:
            session.close()
        try:
            root.destroy()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
