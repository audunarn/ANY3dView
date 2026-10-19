"""Detached, fail-closed application reference binding; never admits a solver.

Current associations retain their separate provenance. Absence of an original
ancestor of one entity kind is proved against the complete witnessed inventory;
it never means that no edge/face/structural/application reference can exist.
"""
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral
import numpy as np
from anygeometry.entities import EntityRef
from anygeometry.identity import EntityHandle, ResolutionStatus
from .prepared_reference_scope import (
    PreparedReferenceScopeError, PreparedProjectClosure,
    validate_prepared_project_reference_scope,
)


class PreparedMeshRemapError(PreparedReferenceScopeError):
    def __init__(self, code, scope, detail):
        self.code, self.scope = code, scope
        super().__init__(f'{code}: {scope}: {detail}')


@dataclass(frozen=True)
class AssociationBinding:
    field: str
    key: int | tuple
    source_ids: tuple[int, ...]
    proof: str
    namespace: str = 'prepared_current'


@dataclass(frozen=True)
class ProjectReferenceBinding:
    key: str
    category: str
    source_targets: tuple
    current_targets: tuple
    cell_ids: tuple[int, ...]
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class SourceOccurrenceBinding:
    source_sheet_id: int
    source_face_use_id: int
    source_face_id: int
    current_face_use_ids: tuple[int, ...]
    cell_ids: tuple[int, ...]
    source_orientation: int
    current_orientations: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DetachedPreparedMeshRemap:
    mesh: object
    current_mesh: object
    current_receipt: object
    associations: tuple[AssociationBinding, ...]
    references: tuple[ProjectReferenceBinding, ...]
    occurrences: tuple[SourceOccurrenceBinding, ...]
    field_inventory: tuple[tuple[str, int], ...]
    source_manifest: object
    source_witness: object
    source_reference_transfer_qualified: bool = True
    solver_admitted: bool = False
    publication_qualified: bool = False


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return ('array', value.dtype.str, value.shape, value.tobytes())
    if is_dataclass(value):
        return (type(value), tuple((f.name, _snapshot(getattr(value,f.name))) for f in fields(value)))
    if isinstance(value, dict):
        return tuple((k, _snapshot(v)) for k,v in sorted(value.items(), key=lambda pair:repr(pair[0])))
    if isinstance(value, (tuple,list)):
        return tuple(_snapshot(item) for item in value)
    return value


def _refuse(code, scope, detail):
    raise PreparedMeshRemapError(code, scope, detail)


def _ids(rows, scope):
    result=[]
    for row in rows:
        key=row['id']
        if isinstance(key,bool) or not isinstance(key,Integral) or key in result:
            _refuse('invalid_inventory',scope,'owner IDs must be unique integers')
        result.append(int(key))
    return tuple(result)


def _owner_maps(project, prepared, manifest):
    """Exhaustive original-of-kind ownership, not an empty-ancestry heuristic."""
    original=manifest.authored_document
    stores={'vertex':'vertices','edge':'edges','face':'faces','sheet':'sheets','face_use':'face_uses'}
    inverse={kind:{} for kind in stores}
    for kind,source,initial in manifest.source_to_prepared_ids:
        if kind not in inverse:continue
        if initial in inverse[kind]:
            _refuse('ambiguous_source',kind,'multiple source entities map to one original work entity')
        inverse[kind][initial]=source
    reverse={kind:{} for kind in ('vertex','edge','face','sheet')}
    original_descendants={kind:{} for kind in reverse}
    for kind,store in stores.items():
        document=original if kind in ('vertex','edge','face') else original['structural']
        original_ids=_ids(document[store],kind)
        if set(inverse[kind]) != set(original_ids):
            _refuse('incomplete_source_inventory',kind,'witnessed source map must cover the complete original work inventory')
        if kind=='face_use':continue  # Generic handle resolution omits sibling occurrences.
        for initial in original_ids:
            resolved=prepared.resolve_handle(EntityHandle(prepared.model_id,kind,initial))
            if resolved.status not in (ResolutionStatus.ACTIVE,ResolutionStatus.REPLACED,ResolutionStatus.DELETED):
                _refuse('unresolved_owner',f'{kind}:{initial}','owner resolution is unavailable')
            leaves=[]
            for handle in resolved.resolved:
                if handle.kind!=kind or handle.model_id!=prepared.model_id:
                    _refuse('wrong_namespace',f'{kind}:{initial}','owner resolution changed kind/model')
                leaves.append(handle.id)
            if len(set(leaves))!=len(leaves):
                _refuse('invalid_inventory',f'{kind}:{initial}','owner resolution repeated descendants')
            original_descendants[kind][initial]=tuple(leaves)
            for current in leaves:reverse[kind].setdefault(current,[]).append(initial)
    # Face ancestry must also agree with the dedicated owner preparation proof.
    for initial,children in manifest.owner_scope.face_preimages.face_descendants:
        if set(original_descendants['face'].get(initial,()))!=set(children):
            _refuse('face_scope_mismatch',f'face:{initial}','handle and dedicated owner face scope disagree')
    return inverse,reverse,original_descendants


def _reference_bindings(manifest,mesh):
    bindings=[]
    for record in manifest.references:
        cells=set();nodes=set()
        for ref in record.current_prepared_targets:
            if ref.kind=='face':
                if ref.id not in mesh.elements_of_face:
                    _refuse('missing_reference_scope',record.key,f'current face {ref.id} is not meshed')
                selected=mesh.elements_of_face[ref.id];cells.update(selected)
                nodes.update(n for cell in selected for n in mesh.shells[cell])
            elif ref.kind=='edge':
                if ref.id not in mesh.nodes_of_edge:
                    _refuse('missing_reference_scope',record.key,f'current edge {ref.id} lacks a chain')
                nodes.update(mesh.nodes_of_edge[ref.id])
            elif ref.kind=='vertex':
                if ref.id not in mesh.node_of_vertex:
                    _refuse('missing_reference_scope',record.key,f'current vertex {ref.id} lacks a node')
                nodes.add(mesh.node_of_vertex[ref.id])
            else:_refuse('unsupported_reference',record.key,ref.kind)
        bindings.append(ProjectReferenceBinding(record.key,record.category,
            record.source_targets,record.current_prepared_targets,tuple(sorted(cells)),tuple(sorted(nodes))))

    return bindings


def remap_prepared_mesh_to_project(project, prepared_geometry, mesh, manifest, *,
        closure, association_namespace='prepared_current', association_receipt,
        association_component, association_registry, cell_current_faces,
        cancellation_check=None):
    """Return a detached source Mesh plus complete CURRENT provenance/accounting.

    Supports only scopes accepted by the live versioned mesher receipt. This
    function neither changes a Project nor publishes/admits the returned mesh.
    All source loads/properties remain in the unchanged Project; each manifest
    reference is bound separately to its exact current cells/nodes.
    """
    if type(closure) is not PreparedProjectClosure:
        _refuse('missing_witness','closure','requires this process\'s witnessed Project extraction')
    if association_namespace!='prepared_current':
        _refuse('wrong_namespace','mesh','only explicit prepared_current input is supported')
    try:
        from anymesher import validate_prepared_current_component_associations
        from anymesher.mesh import Mesh
    except ImportError as error:
        raise PreparedMeshRemapError('missing_capability','mesher',str(error)) from error
    if type(mesh) is not Mesh:
        _refuse('unsupported_mesh','mesh','requires the canonical Mesh schema')
    declared={field.name for field in fields(Mesh)}
    def validate_instance_schema(candidate=mesh):
        if type(candidate) is not Mesh:
            _refuse('unsupported_mesh','mesh','requires the canonical Mesh schema')
        actual=set(vars(candidate))
        unknown=actual-declared;missing=declared-actual
        if unknown or missing:
            _refuse('unsupported_mesh_field','mesh',
                f'unknown instance fields {sorted(unknown)}; missing fields {sorted(missing)}')
    validate_instance_schema()
    unsupported=('beams','couplings','elements_of_edge','elements_of_member',
        'nodes_of_member','offset_nodes_of_edge','grid_of_face','block_grids_of_face',
        'structural_preparation','hybrid_diagnostics','automatic_beam_connections',
        'automatic_shell_connections','automatic_intersections','seeding',
        'thickness_of_face','activity')
    for name in unsupported:
        value=getattr(mesh,name)
        present=value is not None if name=='seeding' else (len(value)>0 if hasattr(value,'__len__') else bool(value))
        if present:_refuse('unsupported_association',name,'nonempty field is outside the qualified shell source-binding scope')
    before=_snapshot(mesh)

    def validate(output_check=None):
        if cancellation_check is not None and cancellation_check('FEM source reference binding'):
            _refuse('cancelled','remap','cancellation requested')
        try:
            validate_instance_schema()
            validate_prepared_project_reference_scope(project,prepared_geometry,manifest,closure=closure)
            validate_prepared_current_component_associations(prepared_geometry,
                association_receipt,association_component,mesh,association_registry,
                cell_current_faces,cancellation_check=cancellation_check)
            validate_instance_schema()
            if association_receipt.association_namespace!='prepared_current':
                _refuse('wrong_namespace','receipt','requires CURRENT associations')
            if _snapshot(mesh)!=before:
                _refuse('input_changed','mesh','staged input changed during binding')
            if output_check is not None:
                output_check()
            # Owner cancellation callbacks may change application input while
            # returning False; payload inspection can also invoke user code.
            # No copying, inspection or supplied callbacks follow this check.
            validate_prepared_project_reference_scope(project,prepared_geometry,manifest,closure=closure)
        except PreparedMeshRemapError:raise
        except ValueError as error:
            raise PreparedMeshRemapError('invalid_binding','project/owner',str(error)) from error

    validate()
    inverse,reverse,descendants=_owner_maps(project,prepared_geometry,manifest)
    receipt=association_receipt
    result=deepcopy(mesh)
    result.geometry_model_id=project.geometry.model_id
    result.geometry_revision=project.geometry.revision
    accounting=[]

    def source_ids(kind,current):
        return tuple(sorted(inverse[kind][original] for original in reverse[kind].get(current,())))

    def map_cells(field,kind):
        made={}
        for current,cells in getattr(mesh,field).items():
            owners=source_ids(kind,current)
            if len(owners)!=1:
                _refuse('missing_or_ambiguous_source',f'{field}/{current}','requires one exact source owner')
            made.setdefault(owners[0],[]).extend(cells)
            accounting.append(AssociationBinding(field,current,owners,'complete witnessed original inventory + owner descendants'))
        for source,cells in made.items():
            if len(cells)!=len(set(cells)):
                _refuse('overlapping_cells',f'{field}/{source}','duplicate cells across current descendants')
            made[source]=sorted(cells)  # Membership set, never an edge traversal.
        setattr(result,field,made)

    map_cells('elements_of_face','face')
    map_cells('elements_of_sheet','sheet')
    source_chains={}
    for root,original,forward,current_edges,chain in receipt.source_boundary_chains:
        source=inverse['edge'].get(original)
        if source is None:
            _refuse('missing_source_chain',f'edge:{original}','source original boundary edge is not witnessed')
        if set(current_edges)!=set(descendants['edge'].get(original,())):
            _refuse('incomplete_source_chain',f'edge:{original}','chain does not cover exact owner descendants')
        ordered=tuple(chain) if forward else tuple(reversed(chain))
        if source in source_chains and source_chains[source]!=ordered:
            _refuse('conflicting_source_chain',f'edge:{source}','multiple occurrences disagree on traversal')
        source_chains[source]=ordered
        accounting.append(AssociationBinding('source_boundary_chains',(root,original),(source,),
            'owner original edge traversal and orientation','prepared_original'))
    for current,chain in mesh.nodes_of_edge.items():
        owners=source_ids('edge',current)
        if any(source not in source_chains for source in owners):
            _refuse('missing_source_chain',f'edge:{current}','original edge lacks a qualified ordered chain')
        accounting.append(AssociationBinding('nodes_of_edge',current,owners,
            'source boundary traversal' if owners else 'complete original EDGE inventory/resolution; retained CURRENT, not no-reference'))
    result.nodes_of_edge={source:list(chain) for source,chain in source_chains.items()}
    for current,chain in receipt.joint_chains:
        accounting.append(AssociationBinding('joint_chains',current,source_ids('edge',current),
            'qualified current joint chain; primitive EDGE ancestry separately proved; CURRENT provenance retained'))
    for root,rows in getattr(receipt,'created_material_uv_by_root',()):
        source=inverse['face'].get(root)
        if source is None:_refuse('missing_material_scope',f'root:{root}','created-node UV has no witnessed original face')
        accounting.append(AssociationBinding('created_material_uv_by_root',root,(source,),
            'exact qualified ORIGINAL-chart UV retained without transformation','prepared_original'))
    vertices={}
    preimages=dict(receipt.current_vertex_preimages)
    for current,node in mesh.node_of_vertex.items():
        originals=tuple(reverse['vertex'].get(current,()))
        if set(originals)!=set(preimages[current]):
            _refuse('vertex_scope_mismatch',f'vertex:{current}','receipt and complete original vertex proof disagree')
        owners=source_ids('vertex',current)
        if len(owners)>1:
            _refuse('ambiguous_source_vertex',f'vertex:{current}','multiple original vertices are outside this contract')
        for source in owners:
            if source in vertices and vertices[source]!=node:
                _refuse('split_source_vertex',f'vertex:{source}','one source vertex requires multiple nodes')
            vertices[source]=node
        accounting.append(AssociationBinding('node_of_vertex',current,owners,
            'unique original vertex' if owners else 'complete original VERTEX inventory/resolution; retained CURRENT, not no-reference'))
    result.node_of_vertex=vertices

    occurrence=[]
    face_use_cells=dict(receipt.elements_of_face_use)
    # Use the owner's live orientation enum. Serialized documents use strings;
    # interpreting those strings here would duplicate the owner's schema.
    current_uses_by_id=prepared_geometry.face_uses
    seen=set()
    for sheet,root,original_use,current_uses in receipt.occurrence_correspondence:
        source_sheet=inverse['sheet'].get(sheet)
        source_use=inverse['face_use'].get(original_use)
        source_face=inverse['face'].get(root)
        if None in (source_sheet,source_use,source_face):
            _refuse('missing_source_occurrence',f'face_use:{original_use}','all original Sheet/FaceUse/root owners are required')
        source_record=project.geometry.face_uses[source_use]
        if source_record.sheet_id!=source_sheet or source_record.face_id!=source_face:
            _refuse('occurrence_source_mismatch',f'face_use:{source_use}','source occurrence endpoints disagree with witnessed mapping')
        cells=[]
        for use in current_uses:
            if use in seen or use not in face_use_cells:
                _refuse('invalid_occurrence',f'face_use:{use}','occurrence missing or claimed twice')
            seen.add(use);cells.extend(face_use_cells[use])
            if int(current_uses_by_id[use].orientation)!=int(source_record.orientation):
                _refuse('occurrence_orientation_mismatch',f'face_use:{use}','source/current occurrence orientation changed')
            accounting.append(AssociationBinding('elements_of_face_use',use,(source_use,),'owner-qualified complete occurrence correspondence'))
        occurrence.append(SourceOccurrenceBinding(source_sheet,source_use,source_face,
            tuple(current_uses),tuple(sorted(cells)),int(source_record.orientation),
            tuple((use,int(current_uses_by_id[use].orientation)) for use in current_uses)))
    if seen!=set(face_use_cells):
        _refuse('unaccounted_occurrence','face_uses','receipt contains unbound occurrences')

    bindings=_reference_bindings(manifest,mesh)

    # Stage every detached copy before the last freshness guard. An admitted
    # ndarray subclass can run code during deepcopy, even with unchanged bytes.
    current_mesh=deepcopy(mesh)
    current_receipt=deepcopy(receipt)
    # All other Mesh fields are retained exactly. The live mesher schema proof
    # refuses unsupported nonempty association/metadata fields before this point.
    unchanged={f.name for f in fields(mesh)}-{
        'geometry_model_id','geometry_revision','elements_of_face','elements_of_sheet',
        'nodes_of_edge','node_of_vertex'}
    if any(_snapshot(getattr(result,name))!=_snapshot(getattr(mesh,name)) for name in unchanged):
        _refuse('output_changed','physical/unsupported fields','copy changed an unremapped field')
    expected_source=_snapshot(result)
    expected_receipt=_snapshot(receipt)
    def check_detached_output():
        for copied in (result,current_mesh):
            validate_instance_schema(copied)
            if any(np.shares_memory(np.asarray(value),np.asarray(mesh.nodes[key]))
                   for key,value in copied.nodes.items()):
                _refuse('output_not_detached','nodes','copied coordinates alias input storage')
        if (_snapshot(current_mesh)!=before or _snapshot(current_receipt)!=expected_receipt
                or _snapshot(result)!=expected_source):
            _refuse('output_changed','detached output','copy or callback changed validated output values')
    inventory=tuple((f.name,len(value) if hasattr(value,'__len__') else int(value is not None))
        for f in fields(mesh) for value in (getattr(mesh,f.name),))
    artifact=DetachedPreparedMeshRemap(result,current_mesh,current_receipt,
        tuple(accounting),tuple(bindings),tuple(occurrence),inventory,manifest,closure)
    validate(output_check=check_detached_output)
    return artifact
