"""Headless ANYfem adapter for the incremental native meshing runtime."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable, Mapping

from anygeometry import EntityHandle, EntityRef, extract_model_closure

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.meshing_view import GeometryMeshingView

from .native_meshing import (
    CertificationMode,
    CoalescedChangeSet,
    DirtyComponentResolution,
    MeshBackend,
    MeshGenerationRequest,
    NativeMeshResult,
    NativeMeshSettings,
    NativeMeshingRuntime,
)
from .structural_preparation import (
    prepare_structural_connectivity,
    remap_mesh_to_source,
    source_work_mapping,
)

__all__ = [
    "GeometryComponentSnapshot",
    "NativeProjectMeshingSession",
    "create_native_meshing_session",
]


@dataclass(frozen=True, slots=True)
class GeometryComponentSnapshot:
    """Detached component closure plus the source revision that created it."""

    geometry: Any
    component: EntityHandle
    source_geometry: Any
    source_component: EntityHandle
    model_id: Any
    revision: int
    source_to_work: Mapping[EntityHandle, EntityHandle]
    work_to_source: Mapping[EntityHandle, EntityHandle]
    closure: Any

    def assert_current(self) -> None:
        if self.source_geometry.model_id != self.model_id:
            raise RuntimeError("geometry model identity changed during mesh generation")
        if int(self.source_geometry.revision) != int(self.revision):
            raise RuntimeError(
                "geometry changed during mesh generation; stale result rejected"
            )


def _key(value: Any) -> tuple[str, int] | None:
    if hasattr(value, "kind") and hasattr(value, "id"):
        return str(value.kind), int(value.id)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return str(value[0]), int(value[1])
    return None


class NativeProjectMeshingSession:
    """Own event hooks, component resolution, generation and publication."""

    def __init__(
        self,
        project: Any,
        settings: NativeMeshSettings | None = None,
        *,
        max_background_jobs: int = 2,
    ) -> None:
        self.project = project
        self.geometry = project.geometry
        self._native_backend = project.set_native_triangulation_backend(
            project.native_triangulation_backend
        )
        resolved = settings or project.native_mesh_settings
        if resolved is None:
            if project.target_size is None:
                raise ValueError(
                    "native meshing needs persisted settings or project.target_size"
                )
            resolved = NativeMeshSettings.create(
                float(project.target_size), element_order=project.element_order
            )
        project.set_native_mesh_settings(resolved)
        self._lock = RLock()
        self._audit_reports: dict[int, Any] = {}
        self.runtime = NativeMeshingRuntime(
            resolved,
            resolve_dirty_components=self.resolve_dirty_components,
            generate_component=self.generate_component,
            capture_component=self.capture_component,
            max_background_jobs=max_background_jobs,
        )
        self.geometry.add_change_hook(self.on_geometry_change)
        self._closed = False

    def _handle(self, kind: str, identifier: int) -> EntityHandle:
        return EntityHandle(self.geometry.model_id, str(kind), int(identifier))

    def all_components(self) -> tuple[EntityHandle, ...]:
        view = GeometryMeshingView(self.geometry)
        handles: list[EntityHandle] = []
        covered_faces: set[int] = set()
        for sheet_id in sorted(view.sheets):
            handles.append(self.geometry.handle("sheet", int(sheet_id)))
            covered_faces.update(view.faces_for_sheet(int(sheet_id)))
        for face_id in sorted(set(view.faces) - covered_faces):
            handles.append(self.geometry.handle("face", int(face_id)))
        for member_id in sorted(view.members):
            handles.append(self.geometry.handle("member", int(member_id)))
        return tuple(handles)

    def capture_component(self, component: EntityHandle) -> GeometryComponentSnapshot:
        if component.model_id != self.geometry.model_id:
            raise ValueError("component handle belongs to another geometry model")
        # Intersections are relationships between components.  Capturing only
        # the requested component cannot qualify a beam/shell or plate/plate
        # neighbour, so each immutable component job receives the complete
        # structural closure and still publishes only its requested component.
        closure = extract_model_closure(
            self.geometry,
            self.all_components(),
            include_structural_closure=True,
            include_features=False,
        )
        return GeometryComponentSnapshot(
            geometry=closure.working_model,
            component=closure.source_to_work[component],
            source_geometry=self.geometry,
            source_component=component,
            model_id=closure.source_model_id,
            revision=closure.source_revision,
            source_to_work=closure.source_to_work,
            work_to_source=closure.work_to_source,
            closure=closure,
        )

    def _components_for_key(
        self, view: GeometryMeshingView, kind: str, identifier: int
    ) -> tuple[EntityHandle, ...]:
        result: set[EntityHandle] = set()
        if kind == "sheet" and identifier in view.sheets:
            result.add(self.geometry.handle(kind, identifier))
        elif kind == "member" and identifier in view.members:
            result.add(self.geometry.handle(kind, identifier))
        elif kind == "face" and identifier in view.faces:
            sheets = view.sheets_for_face(identifier)
            if sheets:
                result.update(self.geometry.handle("sheet", item) for item in sheets)
            else:
                result.add(self.geometry.handle("face", identifier))
        elif kind == "edge" and identifier in view.edges:
            result.update(
                self.geometry.handle("sheet", item)
                for item in view.sheets_using_edge(identifier)
            )
            result.update(
                self.geometry.handle("member", item)
                for item in view.members_using_edge(identifier)
            )
            for face_id in self.geometry.faces_using_edge(identifier):
                if not view.sheets_for_face(face_id):
                    result.add(self.geometry.handle("face", face_id))
        elif kind == "vertex" and identifier in view.vertices:
            for edge_id in self.geometry.edges_using_vertex(identifier):
                result.update(self._components_for_key(view, "edge", int(edge_id)))
        return tuple(sorted(result, key=lambda item: (item.kind, item.id)))

    def resolve_dirty_components(
        self, changes: CoalescedChangeSet
    ) -> DirtyComponentResolution:
        view = GeometryMeshingView(self.geometry)
        dirty: set[EntityHandle] = set()
        removed: set[EntityHandle] = set()
        conservative = bool(
            changes.document_settings_changed
            or changes.feature_history_changed
            or changes.ownership_changes
            or changes.member_changes
            or changes.attachment_changes
        )
        for value in changes.changed:
            parsed = _key(value)
            if parsed is None:
                conservative = True
                continue
            kind, identifier = parsed
            resolved = self._components_for_key(view, kind, identifier)
            if resolved:
                dirty.update(resolved)
            elif value not in changes.removed:
                conservative = True
        for value in changes.removed:
            parsed = _key(value)
            if parsed is not None and parsed[0] in {"sheet", "member", "face"}:
                removed.add(self._handle(*parsed))
        if conservative:
            dirty.update(self.all_components())
        dirty.difference_update(removed)
        order = lambda item: (item.kind, item.id)
        return DirtyComponentResolution(
            tuple(sorted(dirty, key=order)), tuple(sorted(removed, key=order))
        )

    def on_geometry_change(self, change_set: Any) -> int:
        report = self.geometry.audit_changed_region(change_set)
        with self._lock:
            self._audit_reports[int(change_set.revision_after)] = report
            if len(self._audit_reports) > 32:
                for revision in sorted(self._audit_reports)[:-32]:
                    del self._audit_reports[revision]
        return self.runtime.on_geometry_change(change_set)

    def _audit_diagnostics(
        self, changes: CoalescedChangeSet | None
    ) -> tuple[str, ...]:
        if changes is None:
            return ("interactive preflight; no ChangeSet audit requested",)
        with self._lock:
            reports = tuple(
                self._audit_reports[revision]
                for revision in sorted(self._audit_reports)
                if changes.revision_before < revision <= changes.revision_after
            )
        if not reports:
            return ("changed-region audit report unavailable; result is not certified",)
        return tuple(
            "changed-region audit "
            f"scope={getattr(getattr(report, 'scope', None), 'value', getattr(report, 'scope', 'unknown'))} "
            f"certifiable={bool(getattr(report, 'certifiable', False))}"
            for report in reports
        )

    def generate_component(self, request: MeshGenerationRequest) -> NativeMeshResult:
        request.cancellation.raise_if_cancelled("hybrid component start")
        snapshot = request.snapshot
        if not isinstance(snapshot, GeometryComponentSnapshot):
            raise TypeError("native generator requires GeometryComponentSnapshot")
        snapshot.assert_current()
        geometry = snapshot.geometry
        preparation = prepare_structural_connectivity(
            geometry,
            source_model_id=str(snapshot.model_id),
            source_revision=snapshot.revision,
            cancellation_check=request.cancellation.raise_if_cancelled,
        )
        preparation.source_to_working = source_work_mapping(snapshot.closure)
        view = GeometryMeshingView(geometry)
        component = snapshot.component
        face_ids: tuple[int, ...] = ()
        member_ids: tuple[int, ...] | None = ()
        if component.kind == "sheet":
            face_ids = view.faces_for_sheet(component.id)
        elif component.kind == "member":
            member_ids = (component.id,)
        elif component.kind == "face":
            current_faces = tuple(
                geometry.resolve_ref(EntityRef("face", component.id))
            )
            face_ids = tuple(item.id for item in current_faces) or (component.id,)
        else:
            raise ValueError(
                f"unsupported native mesh component kind {component.kind!r}"
            )

        target_size = float(request.settings.target_size)
        parameters: dict[str, Any] = request.settings.to_dict()["parameters"]
        if "native_backend" in parameters:
            raise ValueError(
                "native_backend is a project-level setting, not a mesh parameter"
            )
        for control in sorted(request.controls, key=lambda item: item.control_id):
            if control.target_size is not None:
                target_size = min(target_size, float(control.target_size))
            control_parameters = dict(control.parameters)
            if "native_backend" in control_parameters:
                raise ValueError(
                    "native_backend is a project-level setting, not a control "
                    f"parameter ({control.control_id!r})"
                )
            parameters.update(control_parameters)
        backend = getattr(request.backend, "value", request.backend)
        strategy = {
            "automatic": "auto",
            "mapped": "mapped",
            "native": "native",
            "quad_first": "native",
        }[str(backend)]
        if str(backend) == "quad_first":
            from .quad_first import effective_quad_options

            quad_options = effective_quad_options(
                parameters.get("quad_options")
            )
            native_backend = "python"
        else:
            native_backend = self._native_backend
        supported = {
            key: value
            for key, value in parameters.items()
            if key in {"recombine", "overlap_policy"}
        }
        if str(backend) == "quad_first":
            supported["quad_options"] = quad_options
            supported["layout_policy"] = parameters.get("layout_policy", "existing")
        strict = request.certification_mode is CertificationMode.STRICT
        generated = generate_hybrid_mesh_result(
            geometry,
            target_size=target_size,
            strategy=strategy,
            face_ids=face_ids,
            member_ids=member_ids,
            order=request.settings.element_order,
            certification_mode="strict" if strict else "none",
            cancellation_check=request.cancellation.raise_if_cancelled,
            native_backend=native_backend,
            structural_preparation={
                "automatic_face_connections": False,
                "automatic_member_connections": False,
                "automatic_member_sheet_connections": False,
                "declare_missing_owners": True,
            },
            mutation_policy="working_copy",
            **supported,
        )
        request.cancellation.raise_if_cancelled("hybrid component completion")
        snapshot.assert_current()
        diagnostics = list(self._audit_diagnostics(request.changes))
        if all(
            hasattr(generated.mesh, name)
            for name in (
                "elements_of_face",
                "elements_of_edge",
                "nodes_of_edge",
                "offset_nodes_of_edge",
                "node_of_vertex",
                "grid_of_face",
                "thickness_of_face",
            )
        ):
            remap_mesh_to_source(generated.mesh, snapshot.closure)
        else:
            # NativeMeshingRuntime deliberately permits lightweight backend
            # payloads for predictors/adapters.  Only a real Mesh carries
            # source associations that can be remapped.
            diagnostics.append(
                "structural association remap skipped for non-Mesh backend payload"
            )
        diagnostics.append(
            "structural preparation: "
            f"{preparation.created_count} declared connection(s), "
            f"working revision {preparation.working_revision}"
        )
        diagnostics.extend(
            f"face {face_id}: {route}"
            for face_id, route in sorted(generated.strategy_by_face.items())
        )
        return NativeMeshResult(
            generated.mesh,
            valid=True,
            certified=bool(generated.certifiable),
            diagnostics=tuple(diagnostics),
        )

    def request_remesh(
        self, components: EntityHandle | Iterable[EntityHandle] | None = None
    ) -> tuple[EntityHandle, ...]:
        selected = self.all_components() if components is None else components
        return self.runtime.request_remesh(selected)

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self.geometry.remove_change_hook(self.on_geometry_change)
        self.runtime.shutdown(wait=wait)

    def __enter__(self) -> "NativeProjectMeshingSession":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()


def create_native_meshing_session(
    project: Any,
    settings: NativeMeshSettings | None = None,
    *,
    max_background_jobs: int = 2,
) -> NativeProjectMeshingSession:
    return NativeProjectMeshingSession(
        project, settings, max_background_jobs=max_background_jobs
    )
