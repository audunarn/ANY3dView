"""Source adapters for the default ANYpost query engine.

The provider extracts source-owned values; it does not compute element physics.
A reader belongs to one query snapshot, so mutable legacy ShapeView objects are
observed afresh on the next query rather than retaining stale engine cache data.
"""
from __future__ import annotations
import numpy as np
from contextvars import ContextVar
from contextlib import contextmanager
_RESULT_ACCESS = ContextVar("anyfem_result_access",default=None)

@contextmanager
def result_access_scope(access):
    token=_RESULT_ACCESS.set(access)
    try:yield
    finally:_RESULT_ACCESS.reset(token)
from anypost import open_memory_results

CACHE_BYTES = 2 * 1024**3
WORKING_BYTES = 128 * 1024**2

def source_case_metadata(shape,analysis_status=None):
    """Separate source analysis status from established data-case readiness."""
    from .results import LinearSolution,ImportedSolution
    raw=getattr(shape,"raw_result",None)
    explicit=getattr(shape,"status",None) or getattr(raw,"status",None)
    selected=explicit or analysis_status or "unknown"
    actual=str(getattr(selected,"value",selected)).strip().lower()
    ready=isinstance(shape,(LinearSolution,ImportedSolution))
    status=actual if explicit or analysis_status else ("completed" if ready else "unknown")
    origin="imported stored data" if isinstance(shape,ImportedSolution) else "returned ANYfem linear data" if isinstance(shape,LinearSolution) else "source stored snapshot"
    return dict(status=status,analysis_status=str(analysis_status or actual),source_status=str(explicit or "unknown"),
                certified=False,source_frame_availability="stored_source_no_commitment_claim",provenance={"data_readiness":origin})

class ShapeProvider:
    provider_work_bytes = 0

    def __init__(self, shape, name, *, stresses=None, working_bytes=WORKING_BYTES, cancel_event=None):
        from .fields import DISPLACEMENT_FIELDS, PLASTICITY_FIELDS, _SURFACE_COMPONENTS, field_unit, recover
        from .results import ImportedSolution
        from .solver_data import resolve_solution_quantity
        from anypost.errors import WorkingSetError,RequestCancelled
        def check():
            if cancel_event is not None and cancel_event.is_set():raise RequestCancelled("source extraction cancelled")
        def admit(bytes_):
            check()
            if bytes_>working_bytes:raise WorkingSetError("source extraction exceeds query working budget")
        def extracted_values(ids,get):
            values=[]
            for index,identifier in enumerate(ids):
                if index%1024==0:check()
                values.append(get(int(identifier)))
            return np.asarray(values,dtype=np.float64)
        def source_shape(value):
            if isinstance(value,np.ndarray):return value.shape
            if isinstance(value,(list,tuple)):
                if not value:return (0,)
                inner=source_shape(value[0])
                for index in range(1,len(value)):
                    if index%1024==0:check()
                    if source_shape(value[index])!=inner:raise ValueError("ragged source component")
                return (len(value),)+inner
            return ()
        check()
        self.name = name
        self.unit = field_unit(name)
        self.missing = ()
        self.reduction_label = None
        self.vector = name == "magnitude"
        imported = getattr(shape, "fields", None)
        if isinstance(imported, dict) and name in imported:
            field = imported[name]
            admit(len(field.values)*64+1024)
            self.location = "element" if field.per_element else "node"
            self.ids = np.asarray(sorted(field.values), dtype=np.int64)
            self.values = extracted_values(self.ids,field.values.__getitem__)
            self.unit, self.missing = field.unit, field.missing
            self.vector = False
        elif name in DISPLACEMENT_FIELDS:
            admit(len(shape.built.mesh.nodes)*(128 if self.vector else 80)+1024)
            if not isinstance(shape, ImportedSolution):
                resolve_solution_quantity(shape, "displacement")
            self.location = "node"
            self.ids = np.asarray(sorted(shape.built.mesh.nodes), dtype=np.int64)
            self.values = np.asarray(shape.translations() if self.vector else shape.component(name), dtype=np.float64)
            check()
        elif name in PLASTICITY_FIELDS:
            resolved = resolve_solution_quantity(shape, "equivalent_plastic_strain")
            data = resolved.data
            if not data:
                raise ValueError(f"field {name!r} has no committed element values")
            admit(len(data)*80+1024)
            self.location = "element"
            self.ids = np.asarray([int(i) for i, v in data.items() if np.isfinite(float(v))], dtype=np.int64)
            self.values = extracted_values(self.ids,data.__getitem__)
            if not len(self.ids):
                raise ValueError(f"field {name!r} has no finite committed element values")
            mesh=shape.built.mesh
            self.missing=tuple(sorted((set(mesh.shells)|set(mesh.beams))-set(int(i) for i in self.ids)))
            self.unit = str(resolved.descriptor.unit or self.unit)
            self.reduction_label = str(resolved.descriptor.metadata.get("reduction", "max integration point/layer/fibre"))
        else:
            if isinstance(imported, dict) and imported:
                raise ValueError(f"unknown field {name!r} for imported result")
            mesh = shape.built.mesh
            admit((len(mesh.shells)+len(mesh.beams))*64+1024)
            all_ids = sorted(mesh.shells) + sorted(mesh.beams)
            # Mechanical recovery is owned by ANYsolver and runs on the
            # source worker before opening the engine snapshot. Its memory is
            # not represented as an ANYpost query-workspace guarantee.
            recovered = recover(shape) if stresses is None else stresses
            check()
            resolved = resolve_solution_quantity(shape, "stress", recovered=recovered)
            self.unit = str(resolved.descriptor.unit or self.unit)
            self.basis = str(getattr(resolved.descriptor,"basis","source"))
            from .fields import STRESS_FIELDS, reported_fields
            if name not in STRESS_FIELDS and name not in reported_fields(recovered, shape):
                raise ValueError(f"unknown field {name!r}: source recovery does not report it")
            rows, ids, missing = [], [], []
            row_bytes=0;width=0
            for eid in all_ids:
                check()
                record = recovered.element_stresses.get(eid)
                raw = None if record is None else record.get(name)
                if raw is None and record is not None and name in _SURFACE_COMPONENTS:
                    side, _, component = name.partition("_")
                    if record.get("membrane_" + component) is not None and record.get("bending_" + component) is not None:
                        pair_shape=np.broadcast_shapes(source_shape(record["membrane_"+component]),source_shape(record["bending_"+component]))
                        count=int(np.prod(pair_shape))
                        admit(len(all_ids)*64+row_bytes+(len(rows)+1)*(max(width,count,1)*16+128)+count*16)
                        from anysolver_results import shell_surface_component
                        raw = shell_surface_component(record, side, component)
                if raw is None:
                    missing.append(eid)
                    continue
                count=int(np.prod(source_shape(raw)))
                next_width=max(width,count)
                admit(len(all_ids)*64+row_bytes+(len(rows)+1)*(max(next_width,1)*16+128)+count*16)
                row = np.asarray(raw, dtype=np.float64).reshape(-1)
                row_bytes+=row.nbytes;width=next_width
                ids.append(eid)
                rows.append(row)
            self.location = "element"
            self.ids = np.asarray(ids, dtype=np.int64)
            self.missing = tuple(missing)
            width = max((len(row) for row in rows), default=0)
            snapshot_size = len(rows)*width*8 + len(rows)*16
            if max(snapshot_size*2, snapshot_size+row_bytes+len(all_ids)*64) > working_bytes:
                from anypost.errors import WorkingSetError
                raise WorkingSetError("native sample snapshot exceeds result working budget")
            self.values = np.full((len(rows), width), np.nan, dtype=np.float64)
            self.counts = np.asarray([len(row) for row in rows], dtype=np.int64)
            for index, row in enumerate(rows):
                check()
                self.values[index, :len(row)] = row
            rows.clear()
            row = raw = None
        snapshot_bytes=self.values.nbytes+self.ids.nbytes+getattr(self,"counts",np.empty(0)).nbytes
        if snapshot_bytes*2 > working_bytes:
            from anypost.errors import WorkingSetError
            raise WorkingSetError("source field snapshot exceeds result working budget")
        check()
        # Live readers own an alias-isolated revision snapshot. Mutable public
        # compatibility calls create another snapshot for their next query.
        self.values=np.frombuffer(self.values.tobytes(order="C"),dtype=np.float64).reshape(self.values.shape)
        self.ids=np.frombuffer(self.ids.tobytes(),dtype=np.int64)
        if hasattr(self,"counts"):self.counts=np.frombuffer(self.counts.tobytes(),dtype=np.int64)
        self.identity = f"anyfem-shape:{id(shape)}"
        self.case_metadata=source_case_metadata(shape)
        self.source_status=self.case_metadata["status"]
        self.frame_value=getattr(shape,"value",None)
        self.frame_label=str(getattr(shape,"label","source snapshot"))

    def metadata(self):
        descriptor = dict(quantity_id=self.name, components=(self.name,), location=self.location,
                          ids=self.ids, unit=self.unit, basis=getattr(self,"basis","source"), value_shape=self.values.shape[1:],
                          provider_work_bytes=0,availability="stored")
        if hasattr(self, "counts"):
            descriptor["sample_counts"] = self.counts
        if self.vector:
            descriptor["vector_components"] = ("ux", "uy", "uz")
        return dict(model_identity=self.identity, provenance={"source": "ANYfem source adapter"},
                    cases=({"id": "0", "label":self.frame_label,**self.case_metadata,
                            "frame_value":self.frame_value},), descriptors=(descriptor,))

    def read_field(self, case_id, quantity_id, component, selection=None, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            from anypost.errors import RequestCancelled
            raise RequestCancelled("source query cancelled")
        return self.values if selection is None else self.values[selection]


def evaluate_shape_field(shape, name="magnitude", *, reduction="mean", stresses=None, cancel_event=None, working_bytes=WORKING_BYTES):
    from .fields import Field, REDUCTIONS
    if reduction not in REDUCTIONS:
        raise ValueError(f"unknown reduction {reduction!r}")
    if cancel_event is not None and cancel_event.is_set():
        from anypost.errors import RequestCancelled
        raise RequestCancelled("field query cancelled")
    provider = ShapeProvider(shape, name, stresses=stresses, working_bytes=working_bytes,cancel_event=cancel_event)
    with open_memory_results(provider, cache_bytes=CACHE_BYTES, working_bytes=working_bytes,
                             expected_model_identity=provider.identity) as reader:
        values = _reader_values(reader, provider, name, reduction, cancel_event)
    mapping = {int(i): float(v) for i, v in zip(provider.ids, values)}
    return Field(name=name, unit=provider.unit,
                 node_values=mapping if provider.location == "node" else {},
                 element_values=mapping if provider.location == "element" else {},
                 missing=provider.missing, reduction=provider.reduction_label or reduction)


def read_sidecar_field(path, key, shape, frame=None, *, reader=None):
    """Retain the historical frame/component array contract through ANYpost."""
    from anypost import open_results
    from contextlib import nullcontext
    with (open_results(path, cache_bytes=CACHE_BYTES, working_bytes=WORKING_BYTES) if reader is None else nullcontext(reader)) as reader:
        descriptors = reader.components(key)
        cases = reader.cases()
        component_shape=tuple(descriptors[0]["component_shape"])
        unframed_vector=len(shape)==1 and component_shape==tuple(shape)
        case_ids=descriptors[0].get("case_ids",tuple(c["case_id"] for c in cases))
        count=len(case_ids)
        selected=(0,) if unframed_vector else range(count) if frame is None else (range(count)[frame] if isinstance(frame,slice) else (range(count)[frame],))
        output_count=int(np.prod(shape)) if unframed_vector else len(selected)*int(np.prod(shape[1:]))
        if output_count*24>WORKING_BYTES:
            from anypost.errors import WorkingSetError
            raise WorkingSetError("aggregate stored field conversion exceeds working budget; select a local frame")
        rows = []
        for index in selected:
            pieces = [reader.read_stored_field(d.get("case_ids", tuple(c["case_id"] for c in cases))[index], key, d["component"]) for d in descriptors]
            pieces = [np.asarray(v).reshape(component_shape) for v in pieces]
            row_shape=tuple(shape[1:]) if not unframed_vector else tuple(shape)
            row = np.stack(pieces, axis=-1) if component_shape+(len(pieces),)==row_shape else pieces[0]
            rows.append(row)
        if unframed_vector:
            values=rows[0]
            return values if frame is None else values[frame]
        if isinstance(frame, (int, np.integer)):
            return rows[0]
        return np.asarray(rows).reshape((len(rows),)+tuple(shape[1:]))


def _reader_values(reader, provider, name, reduction, cancel_event=None):
    method = "magnitude" if provider.vector else (reduction if provider.values.ndim > 1 else None)
    width = max(1, int(np.prod(provider.values.shape[1:])))
    count = len(provider.ids)
    chunk = max(1, min(32768, (reader.working_bytes - count * 8 - 16384) // (width * 64 + 128)))
    output = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        output[start:end] = reader.read_stored_field("0", name, name, slice(start, end), reduction=method,
                                             nan_policy="propagate", empty_policy="zero", cancel_event=cancel_event)
    return output


def retained_recovery_bytes(shape):
    """Measure retained recovery arrays/containers, excluding the FE model."""
    import sys
    from collections.abc import Mapping
    root = getattr(shape,"_stress",None)
    seen=set()
    def size(value):
        if id(value) in seen:return 0
        seen.add(id(value))
        if isinstance(value,np.ndarray):
            base=value
            while isinstance(base.base,np.ndarray):base=base.base
            if base is not value:return sys.getsizeof(value)+size(base)
            return sys.getsizeof(value)+(0 if value.flags.owndata else value.nbytes)
        if isinstance(value,Mapping):return sys.getsizeof(value)+sum(size(k)+size(v) for k,v in value.items())
        if isinstance(value,(tuple,list)):return sys.getsizeof(value)+sum(size(v) for v in value)
        if hasattr(value,"__dict__"):
            return sys.getsizeof(value)+size({k:v for k,v in vars(value).items() if k not in {"model","mesh","built","fe_model"}})
        return sys.getsizeof(value) if value is not None else 0
    return size(root)


class LiveResultAccess:
    """Bounded per-revision native snapshots and retained ANYpost field cache.

    The active component can use the free aggregate cache ceiling. Switching
    components retains engine readers; revision replacement retires the worker
    before closing them. Public mutable ShapeView evaluation uses a fresh reader.
    """
    def __init__(self, *, cache_bytes=CACHE_BYTES, working_bytes=WORKING_BYTES, analysis_status=None):
        self.cache_bytes, self.working_bytes = cache_bytes, working_bytes
        from collections import OrderedDict
        self._readers = OrderedDict()
        self._closed = False
        self.analysis_status = analysis_status
        self._publication_bytes = 0
        self._pending_publication_bytes = 0
        from threading import RLock
        self._cache_lock=RLock()
        self._recovery_resident = {}

    def field(self, shape, name, *, reduction="mean", cancel_event=None):
        with self._cache_operation(cancel_event=cancel_event):
            return self._field(shape,name,reduction=reduction,cancel_event=cancel_event)

    @contextmanager
    def _cache_operation(self,*,wait=True,cancel_event=None):
        from anypost.errors import WorkingSetError
        acquired=self._cache_lock.acquire(blocking=False)
        while wait and not acquired:
            if cancel_event is not None and cancel_event.is_set():
                from anypost.errors import RequestCancelled
                raise RequestCancelled("result query cancelled while waiting for cache ownership")
            acquired=self._cache_lock.acquire(timeout=.01)
        if not acquired:
            raise WorkingSetError("result cache operation overlaps active query; owner must retry after retirement")
        try:yield
        finally:self._cache_lock.release()

    def _field(self, shape, name, *, reduction="mean", cancel_event=None):
        from .fields import Field
        if self._closed:
            raise RuntimeError("result access is closed")
        if reduction not in {"mean","min","max","max_abs"}:raise ValueError(f"unknown reduction: {reduction}")
        key = (id(shape), name)
        item = self._readers.pop(key, None)
        if item is None:
            previous_recovery=getattr(shape,"_stress",None)
            try:
                provider = ShapeProvider(shape, name, working_bytes=self.working_bytes,cancel_event=cancel_event)
            except BaseException:
                current_recovery=getattr(shape,"_stress",None)
                if previous_recovery is None and current_recovery is not None:
                    # This extraction created it; a failed snapshot cannot turn
                    # owned recovery into apparently caller-provided residency.
                    self._recovery_resident[id(shape)]=(shape,current_recovery,retained_recovery_bytes(shape),True)
                    self._release_recovery(id(shape))
                raise
            provider.case_metadata=source_case_metadata(shape,self.analysis_status)
            provider.source_status=provider.case_metadata["status"]
            current_recovery=getattr(shape,"_stress",None)
            old=self._recovery_resident.get(id(shape))
            owned=(old is not None and old[3] and old[1] is current_recovery) or (previous_recovery is None and current_recovery is not None)
            self._recovery_resident[id(shape)]=(shape,current_recovery,retained_recovery_bytes(shape),owned)
            required = provider.values.nbytes + provider.ids.nbytes + getattr(provider, "counts", np.empty(0)).nbytes
            while self._readers and self.cache_bytes - self._publication_bytes - self._pending_publication_bytes - sum(value[2] for value in self._recovery_resident.values()) - sum(v[2].diagnostics["cache_resident_bytes"] for v in self._readers.values()) < required * 2 + 16384:
                old_key, (_, _, old_reader) = self._readers.popitem(last=False)
                old_reader.close()
                if old_key[0] != id(shape) and not any(key[0] == old_key[0] for key in self._readers):
                    self._release_recovery(old_key[0])
            available = self.cache_bytes - self._publication_bytes - self._pending_publication_bytes - sum(value[2] for value in self._recovery_resident.values()) - sum(v[2].diagnostics["cache_resident_bytes"] for v in self._readers.values())
            if available < required * 2 + 16384:
                self._release_recovery(id(shape))
                from anypost.errors import WorkingSetError
                raise WorkingSetError("retained source field/recovery exceeds aggregate cache ceiling")
            reader = None
            try:
                reader = open_memory_results(provider, cache_bytes=available, working_bytes=self.working_bytes, expected_model_identity=provider.identity)
                reader.set_external_cache_bytes("source_snapshot", required)
            except BaseException:
                if reader is not None:reader.close()
                self._release_recovery(id(shape))
                raise
            item = (shape, provider, reader)
        _, provider, reader = item
        available = self.cache_bytes - self._publication_bytes - self._pending_publication_bytes - sum(value[2] for value in self._recovery_resident.values()) - sum(v[2].diagnostics["cache_resident_bytes"] for v in self._readers.values())
        from anypost.errors import WorkingSetError
        while True:
            available=self.cache_bytes-self._publication_bytes-self._pending_publication_bytes-sum(v[2] for v in self._recovery_resident.values())-sum(v[2].diagnostics["cache_resident_bytes"] for v in self._readers.values())
            try:
                if available<0:raise WorkingSetError("publication/recovery exceeds cache budget")
                reader.resize_cache(available)
                break
            except WorkingSetError:
                if not self._readers:
                    reader.close()
                    if not any(k[0]==key[0] for k in self._readers):self._release_recovery(key[0])
                    raise
                old_key,(_,_,old_reader)=self._readers.popitem(last=False)
                old_reader.close()
                if old_key[0]!=key[0] and not any(k[0]==old_key[0] for k in self._readers):self._release_recovery(old_key[0])
        self._readers[key] = item
        values = _reader_values(reader, provider, name, reduction, cancel_event)
        mapping = {int(i): float(v) for i, v in zip(provider.ids, values)}
        return Field(name=name, unit=provider.unit,
                     node_values=mapping if provider.location == "node" else {},
                     element_values=mapping if provider.location == "element" else {},
                     missing=provider.missing, reduction=provider.reduction_label or reduction)

    def reserve_publication(self,field,scalar_buffer_bytes=0):
        with self._cache_operation(wait=False):
            return self._reserve_publication(field,scalar_buffer_bytes)

    def _reserve_publication(self,field,scalar_buffer_bytes=0):
        """Reserve candidate/old published fields before changing the viewport."""
        import sys
        from anypost.errors import WorkingSetError
        mappings=(field.node_values,field.element_values)
        required=sys.getsizeof(field)+sum(sys.getsizeof(m)+sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in m.items()) for m in mappings)+sys.getsizeof(field.missing)+scalar_buffer_bytes
        # Candidate dictionaries and conversion arrays coexist until accepted.
        if required*2 > self.working_bytes:raise WorkingSetError("field publication conversion exceeds working cap")
        while self._readers and required+self._publication_bytes+sum(v[2].diagnostics["cache_resident_bytes"] for v in self._readers.values())+sum(v[2] for v in self._recovery_resident.values())>self.cache_bytes:
            old_key,(_,_,reader)=self._readers.popitem(last=False);reader.close()
            if not any(key[0]==old_key[0] for key in self._readers):self._release_recovery(old_key[0])
        if required+self._publication_bytes+sum(v[2] for v in self._recovery_resident.values())>self.cache_bytes:
            raise WorkingSetError("old and candidate field publication exceed cache ceiling")
        self._pending_publication_bytes=required
        return required

    def commit_publication(self,reservation):
        self._publication_bytes=reservation
        self._pending_publication_bytes=0

    def discard_publication(self):
        """Owner-only release of an unaccepted candidate reservation."""
        self._pending_publication_bytes=0

    def _release_recovery(self, shape_id):
        entry=self._recovery_resident.pop(shape_id,None)
        if entry is not None:
            shape,recovery,bytes_,owned=entry
            if owned and getattr(shape,"_stress",None) is recovery:
                shape._stress=None

    def release_shape(self,shape):
        with self._cache_operation():
            for key in tuple(self._readers):
                if key[0]==id(shape):self._readers.pop(key)[2].close()
            self._release_recovery(id(shape))

    def resize_cache_budget(self,budget):
        with self._cache_operation():
            self.cache_bytes=int(budget)
            def resident():
                return self._publication_bytes+self._pending_publication_bytes+sum(v[2] for v in self._recovery_resident.values())+sum(v[2].diagnostics["cache_resident_bytes"] for v in self._readers.values())
            while self._readers and resident()>self.cache_bytes:
                key,(_,_,reader)=self._readers.popitem(last=False);reader.close()
                if not any(k[0]==key[0] for k in self._readers):self._release_recovery(key[0])
            if resident()>self.cache_bytes:
                from anypost.errors import WorkingSetError
                raise WorkingSetError("retained publication/recovery exceeds envelope cache budget")

    def close(self):
        with self._cache_operation():self._close()

    def _close(self):
        self._closed = True
        for shape, provider, reader in self._readers.values():
            reader.close()
        self._readers.clear()
        for shape_id in tuple(self._recovery_resident):self._release_recovery(shape_id)


class EnvelopeProvider:
    """Stream one authoritative shape at a time into ANYpost's envelope."""
    provider_work_bytes = 0
    def __init__(self, shapes, name, reduction, cancel_event=None, analysis_status=None,access=None):
        from .fields import DISPLACEMENT_FIELDS, field_unit
        self.shapes, self.name, self.reduction = shapes, name, reduction
        self.analysis_status=analysis_status
        imported = [getattr(shape,"fields",{}).get(name) for shape in shapes]
        locations = {"element" if field.per_element else "node" for field in imported if field is not None}
        if len(locations)>1:raise ValueError("envelope source locations disagree")
        self.location = next(iter(locations)) if locations else ("node" if name in DISPLACEMENT_FIELDS else "element")
        mesh = shapes[0].built.mesh
        self.ids = np.asarray(sorted(mesh.nodes) if self.location == "node" else sorted(mesh.shells) + sorted(mesh.beams), dtype=np.int64)
        self.unit = field_unit(name)
        self.current = None
        self.present_any=np.zeros(len(self.ids),dtype=bool)
        self.cancel_event = cancel_event
        self.access=access
        # One conservative 128MiB equation: leave outer metadata/envelope/
        # normalization scratch, then declare the WHOLE callback allowance.
        # Nested extraction gets only the remainder after Field dictionaries,
        # scalar conversion and presence arrays. It is not physics workspace.
        self.outer_reserved = len(self.ids)*256 + len(shapes)*4096 + 65536
        self.provider_work_bytes = max(0,WORKING_BYTES-self.outer_reserved)
        self.conversion_reserved = len(self.ids)*192 + 16384
        self.inner_working = self.provider_work_bytes-self.conversion_reserved
    def metadata(self):
        return dict(model_identity=f"anyfem-envelope:{id(self.shapes)}",
                    cases=tuple(dict(id=str(i), input_index=i,**source_case_metadata(shape,self.analysis_status)) for i,shape in enumerate(self.shapes)),
                    descriptors=(dict(quantity_id=self.name, components=(self.name,), location=self.location,
                                      ids=self.ids, value_shape=(), unit=self.unit, provider_work_bytes=self.provider_work_bytes),))
    def field_descriptor(self, case_id, quantity_id, component):
        from anypost.errors import RequestCancelled, WorkingSetError
        if self.cancel_event is not None and self.cancel_event.is_set():raise RequestCancelled("envelope cancelled")
        if self.inner_working <= 0:raise WorkingSetError("envelope source conversion exceeds working budget")
        shape=self.shapes[int(case_id)]
        self.access.release_shape(shape)
        original_working=self.access.working_bytes
        try:
            self.access.working_bytes=min(original_working,self.inner_working)
            field=self.access.field(shape,self.name,reduction=self.reduction,cancel_event=self.cancel_event)
        finally:
            self.access.working_bytes=original_working
            self.access.release_shape(shape)
        values = np.full(len(self.ids), np.nan, dtype=np.float64)
        mapping=field.values
        if field.per_element != (self.location=="element"):raise ValueError("envelope source location changed")
        present = np.asarray([int(i) in mapping for i in self.ids], dtype=bool)
        for index, eid in enumerate(self.ids):
            if present[index]:
                values[index] = mapping[int(eid)]
        self.present_any |= present
        self.current = (case_id, values)
        self.unit = field.unit
        # read_field returns the already prepared array; its transport/copies
        # are admitted separately by the core, without repeating callback work.
        return dict(present=present,provider_work_bytes=0)
    def read_field(self, case_id, quantity_id, component, selection=None, cancel_event=None):
        if self.current is None or self.current[0] != case_id:
            self.field_descriptor(case_id, quantity_id, component)
        values = self.current[1]
        return values if selection is None else values[selection]


def evaluate_envelope(solution, name="magnitude", *, mode="max_abs", reduction="mean", cancel_event=None):
    access=_RESULT_ACCESS.get()
    if access is None:
        return _evaluate_envelope(solution,name,mode=mode,reduction=reduction,cancel_event=cancel_event)
    with access._cache_operation(cancel_event=cancel_event):
        return _evaluate_envelope(solution,name,mode=mode,reduction=reduction,cancel_event=cancel_event)


def _evaluate_envelope(solution, name="magnitude", *, mode="max_abs", reduction="mean", cancel_event=None):
    from .fields import Field
    from .extract import Envelope
    shapes = getattr(solution, "shapes", None)
    shapes = [solution] if shapes is None else shapes
    if not shapes:
        return Envelope(Field(name + " envelope"), mode, {}, 0)
    access=_RESULT_ACCESS.get()
    owns_access=access is None
    if access is None:access=LiveResultAccess(cache_bytes=CACHE_BYTES-WORKING_BYTES)
    original_budget=access.cache_bytes
    provider = EnvelopeProvider(shapes, name, reduction,cancel_event,getattr(solution,"status",None),access=access)
    try:
        access.resize_cache_budget(min(original_budget,CACHE_BYTES-WORKING_BYTES))
        with open_memory_results(provider, cache_bytes=WORKING_BYTES, working_bytes=WORKING_BYTES) as reader:
            result = reader.read_envelope(name, name, mode=mode, nan_policy="propagate",cancel_event=cancel_event)
    finally:
        if owns_access:access.close()
        else:access.resize_cache_budget(original_budget)
    values, origin = {}, {}
    for eid, value, cid, present in zip(provider.ids, result.values, result.case_ids,provider.present_any):
        if present:values[int(eid)]=float(value)
        if cid is not None and str(cid):origin[int(eid)]=int(cid)
    field = Field(name=name + " envelope", unit=provider.unit,
                  node_values=values if provider.location == "node" else {},
                  element_values=values if provider.location == "element" else {}, reduction=reduction)
    return Envelope(field, mode, origin, len(shapes),tuple(result.admitted_case_ids),tuple(result.failed_case_ids))


def query_solution_quantity(solution, quantity_id):
    """Canonical source extraction followed by ANYpost transport admission."""
    from .solver_data import resolve_solution_quantity
    from types import SimpleNamespace
    from anypost.errors import WorkingSetError
    shapes=getattr(solution,"shapes",()) or ()
    if shapes and quantity_id in {"displacement","mode_shape"}:
        required=sum(np.asarray(shape.displacements).nbytes for shape in shapes)
        if required*3+4096 > WORKING_BYTES:
            raise WorkingSetError("requested live quantity needs selective source access")
    resolved=resolve_solution_quantity(solution,quantity_id)
    class Provider:
        provider_work_bytes=0
        def metadata(self):return dict(model_identity=f"anyfem-quantity:{id(solution)}",cases=(),descriptors=())
        def query(self):return resolved.data
        def read_field(self,*args,**kwargs):
            from anypost.errors import UnavailableFieldError
            raise UnavailableFieldError("query-only source adapter has no field descriptors")
    with open_memory_results(Provider(),cache_bytes=0,working_bytes=WORKING_BYTES) as reader:
        values=reader.query()
    return SimpleNamespace(data=values,descriptor=resolved.descriptor)
