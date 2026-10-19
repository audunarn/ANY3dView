"""Headless, invocation-local composition; no solve or workflow publication.

The mesher derives production authority. Application intent is independently
bound by the detached remapper. Neither a persisted record nor this result is
authority for a later execution: that execution needs a fresh composition.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import numpy as np

from .prepared_mesh_remap import (
    DetachedPreparedMeshRemap, PreparedMeshRemapError, _snapshot,
    remap_prepared_mesh_to_project,
)
from .prepared_reference_scope import _project_inputs, validate_prepared_project_reference_scope


class PreparedMeshPreassemblyError(PreparedMeshRemapError):
    """A cancelled, changed, unsupported or refused candidate composition."""


@dataclass(frozen=True)
class PreparedMeshPreassembly:
    source_binding: DetachedPreparedMeshRemap
    mesh: object
    production_record: object
    element_ids: tuple[int, ...]
    field_inventory: tuple[tuple[str, int], ...]
    load_case_name: str
    load_case_id: str
    preassembly_checked: bool = True
    solver_executed: bool = False
    publication_qualified: bool = False


def _copy_project_for_preassembly(project):
    # Geometry owns its immutable topology containers and detached copying.
    geometry = project.geometry.clone(preserve_identity=True)
    return deepcopy(project, {id(project.geometry): geometry})


def preassemble_prepared_project_mesh(
    project, prepared_geometry, mesh, manifest, *, closure,
    association_receipt, association_component, association_registry,
    cell_current_faces, association_namespace='prepared_current',
    load_case='default', cancellation_check=None, current_project=None,
):
    """Derive fresh source S3 authority and check the existing FE build adapter.

    ``current_project`` optionally returns the caller's currently owned Project;
    replacement refuses even when the new Project has identical serialized data.
    No Project, input mesh, cache, mesh record or GUI state is published here.
    The returned mesh may be inspected, but later use needs fresh live validation.
    """
    from anymesher import (
        prepare_source_bound_qualified_s3_mesh,
        validate_prepared_current_component_associations,
        QUALIFIED_S3_FORMULATION_ID,
    )
    from anymesher.mesh import Mesh
    from .solve import build_fe_model

    def refuse(code, scope, detail):
        raise PreparedMeshPreassemblyError(code, scope, detail)

    def check(phase):
        if cancellation_check is not None and cancellation_check(phase):
            refuse('cancelled', 'preassembly', 'cancellation requested')
        if current_project is not None and current_project() is not project:
            refuse('project_replaced', 'preassembly', 'current Project identity changed')
        return False

    declared = {item.name for item in fields(Mesh)}

    def schema(candidate):
        if type(candidate) is not Mesh or set(vars(candidate)) != declared:
            refuse('unsupported_mesh_field', 'preassembly', 'canonical complete Mesh schema required')

    check('FEM prepared mesh preassembly start')
    if type(load_case) is not str or load_case not in project.load_cases:
        refuse('unwitnessed_load_case', 'preassembly', 'select a named case from the witnessed Project')
    if project.shell_formulation_policy.s3 != 'e4-pl-s3-v2d':
        refuse('unsupported_formulation', 'preassembly', 'qualified S3 composition requires the V2D policy')
    selected_case_id = project.load_cases[load_case].id
    schema(mesh)
    input_snapshot = _snapshot(mesh)
    receipt_snapshot = _snapshot(association_receipt)
    binding = remap_prepared_mesh_to_project(
        project, prepared_geometry, mesh, manifest, closure=closure,
        association_namespace=association_namespace,
        association_receipt=association_receipt,
        association_component=association_component,
        association_registry=association_registry,
        cell_current_faces=cell_current_faces, cancellation_check=check,
    )
    source_snapshot = _snapshot(binding.mesh)
    current_snapshot = _snapshot(binding.current_mesh)
    check('FEM source S3 production qualification')
    try:
        qualified, record = prepare_source_bound_qualified_s3_mesh(
            binding.mesh, project.geometry, prepared_geometry, binding.current_mesh,
            association_receipt, association_component, association_registry,
            cell_current_faces, cancellation_check=check,
        )
    except PreparedMeshPreassemblyError:
        raise
    except (ValueError, TypeError) as error:
        refuse('production_refused', 'source S3', str(error))
    schema(qualified)
    # Only the freshly derived owner production record is a new Mesh value.
    if (qualified.structural_preparation != {'qualified_s3': record}
            or any(_snapshot(getattr(qualified, name)) != _snapshot(getattr(binding.mesh, name))
                   for name in declared - {'structural_preparation'})):
        refuse('output_changed', 'source S3', 'qualification changed a field outside its production record')
    qualified_snapshot = _snapshot(qualified)
    record_snapshot = _snapshot(record)
    check('FEM source S3 preassembly')
    # Project.validate() normalizes legacy assignments. Inspect a detached copy
    # rather than mutate the authored Project, even on a refused build.
    project_snapshot = _project_inputs(project)
    build_project = _copy_project_for_preassembly(project)
    if (build_project is project or build_project.geometry is project.geometry
            or _project_inputs(build_project) != project_snapshot):
        refuse('project_copy_changed', 'FE build', 'detached Project copy differs from witnessed input')
    try:
        built = build_fe_model(build_project, qualified, load_case=load_case)
    except (ValueError, TypeError) as error:
        refuse('preassembly_refused', 'FE build', str(error))
    element_ids = tuple(sorted(built.fe_model.mesh.elements))
    if element_ids != tuple(sorted(qualified.shells)):
        refuse('preassembly_scope_changed', 'FE build', 'built element IDs differ from the qualified mesh')
    if any(getattr(built.fe_model.mesh.elements[cell], 'formulation_id', None)
           != QUALIFIED_S3_FORMULATION_ID for cell in qualified.tris):
        refuse('preassembly_formulation_changed', 'FE build', 'built T3 formulation differs from qualified S3')
    inventory = tuple((item.name, len(value) if hasattr(value, '__len__') else int(value is not None))
                      for item in fields(Mesh) for value in (getattr(qualified, item.name),))
    artifact = PreparedMeshPreassembly(binding, qualified, record, element_ids, inventory,
                                      load_case, selected_case_id)
    # Construct and inspect all outputs before callback-bearing final validation.
    check('FEM prepared mesh preassembly final')
    try:
        validate_prepared_current_component_associations(
            prepared_geometry, association_receipt, association_component, mesh,
            association_registry, cell_current_faces, cancellation_check=check,
        )
        # Evaluate replacement callbacks before snapshots and the final Project guard.
        check('FEM prepared mesh preassembly output')
        for candidate in (mesh, binding.mesh, binding.current_mesh, qualified):
            schema(candidate)
        if (_snapshot(mesh) != input_snapshot
                or _snapshot(association_receipt) != receipt_snapshot
                or _snapshot(binding.mesh) != source_snapshot
                or _snapshot(binding.current_mesh) != current_snapshot
                or _snapshot(qualified) != qualified_snapshot
                or _snapshot(record) != record_snapshot):
            refuse('input_or_output_changed', 'preassembly', 'callback, copy or FE build changed staged values')
        for node_id, value in qualified.nodes.items():
            if any(np.shares_memory(np.asarray(value), np.asarray(candidate.nodes[node_id]))
                   for candidate in (mesh, binding.mesh, binding.current_mesh)):
                refuse('output_not_detached', 'nodes', 'qualified coordinates alias prior stage storage')
        validate_prepared_project_reference_scope(
            project, prepared_geometry, manifest, closure=closure,
        )
    except PreparedMeshPreassemblyError:
        raise
    except (ValueError, TypeError) as error:
        refuse('invalid_binding', 'project/owner', str(error))
    # No copying, payload inspection or supplied callbacks follow the final guard.
    return artifact
