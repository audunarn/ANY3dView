"""Read-only external reference visibility for authored-root mesh consumers.

This is a bound input inventory, not mesh permission, child membership proof,
coordinate remapping, or authority to mutate a project. Owner geometry proofs
remain transient. The persistence serializer is deliberately not used: it
normalizes assignments and would mutate the very inputs being witnessed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import UUID
from weakref import WeakKeyDictionary

import numpy as np
from anygeometry.entities import EntityRef
from anygeometry.features import FeatureOutputRef
from anygeometry.serialization import to_dict as geometry_to_dict

from .model.project import Project
from .model.regions import RegionDomain, RegionRegistry, RegionRef

__all__ = ["PreparedReferenceScopeError", "PreparedProjectReferenceScope",
           "ExternalProjectReference", "FaceReferenceScope", "PreparedProjectClosure",
           "extract_prepared_project_closure",
           "query_prepared_project_reference_scope",
           "validate_prepared_project_reference_scope",
           "assert_authored_root_face_scope"]


class PreparedReferenceScopeError(ValueError):
    """Missing, stale, unsupported or scope-widening application input."""


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PreparedProjectClosure:
    """Transient owner extraction witnessed by this application process.

    Construct through extract_prepared_project_closure, before preparation.
    Reconstructed or replaced instances do not carry extraction authority.
    """
    closure: object

    @property
    def working_model(self):
        return self.closure.working_model

    @property
    def source_to_work(self):
        return self.closure.source_to_work

    @property
    def work_to_source(self):
        return self.closure.work_to_source


_closure_bindings = WeakKeyDictionary()


def extract_prepared_project_closure(project, handles, *, include_structural_closure=True):
    """Extract an owner closure and bind its exact source, work and ID map.

    This creates a detached model; it never prepares or edits design geometry.
    Later external Project references may change, but source geometry may not.
    """
    from anygeometry import extract_model_closure
    inputs, source_document = _project_inputs(project)
    closure = extract_model_closure(project.geometry, handles,
        include_structural_closure=include_structural_closure, include_features=False)
    if _project_inputs(project) != (inputs, source_document):
        raise PreparedReferenceScopeError("project changed during closure extraction")
    basis = PreparedProjectClosure(closure)
    _closure_bindings[basis] = (source_document,
        _dump(geometry_to_dict(closure.working_model)),
        tuple(sorted(closure.source_to_work.items())),
        tuple(sorted(closure.work_to_source.items())))
    return basis


@dataclass(frozen=True, slots=True)
class FaceReferenceScope:
    authored_face_id: int
    current_face_ids: tuple[int, ...]
    whole_authored_root: bool
    required_boundary_edge_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExternalProjectReference:
    key: str
    category: str
    authority: str
    region_id: str | None
    record_json: str
    source_targets: tuple[EntityRef, ...]
    current_prepared_targets: tuple[EntityRef, ...]
    face_scopes: tuple[FaceReferenceScope, ...]


@dataclass(frozen=True, slots=True)
class PreparedProjectReferenceScope:
    schema_version: int
    document_id: str
    project_geometry_model_id: str
    project_geometry_revision: int
    project_checksum: str
    project_input_json: str
    source_geometry_document_json: str
    owner_scope: object
    # (kind, source ID, initial prepared ID). Structural occurrence mappings
    # are retained exactly; no face-use alias is derived from face membership.
    source_to_prepared_ids: tuple[tuple[str, int, int], ...]
    authored_face_sources: tuple[tuple[int, str, int], ...]
    references: tuple[ExternalProjectReference, ...]
    inventory_counts: tuple[tuple[str, int], ...]
    required_boundary_edge_ids: tuple[int, ...]
    required_vertex_ids: tuple[int, ...]
    manifest_checksum: str

    @property
    def authored_document(self):
        """Detached complete ORIGINAL owner input, including FaceUses/Sheets."""
        return self.owner_scope.authored_document

    @property
    def current_document(self):
        """Detached complete CURRENT owner input; no equivalence is asserted."""
        return self.owner_scope.current_document


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze(value):
    """Pure, type-bearing project snapshot; never adopts or repairs records."""
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__module__+"."+type(value).__qualname__, "value": _freeze(value.value)}
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise PreparedReferenceScopeError("non-finite project input")
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (UUID, Path, datetime)):
        return {type(value).__name__: str(value)}
    if isinstance(value, Fraction):
        return {"rational": [value.numerator, value.denominator]}
    if isinstance(value, np.ndarray):
        return {"array": _freeze(value.tolist()), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, RegionRegistry):
        return [_freeze(region) for region in sorted(value, key=lambda item:item.id)]
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__module__+"."+type(value).__qualname__,
                "fields": {field.name:_freeze(getattr(value,field.name))
                           for field in fields(value) if not field.name.startswith("_")}}
    if isinstance(value, Mapping):
        rows=[[_freeze(key),_freeze(item)] for key,item in value.items()]
        return {"mapping": sorted(rows,key=lambda row:_dump(row[0]))}
    if isinstance(value, (tuple,list)):
        return {"tuple" if isinstance(value,tuple) else "list": [_freeze(item) for item in value]}
    if isinstance(value, (set,frozenset)):
        return {"set": sorted((_freeze(item) for item in value),key=_dump)}
    raise PreparedReferenceScopeError(f"unsupported project input type: {type(value).__module__}.{type(value).__qualname__}")


def _project_inputs(project):
    if type(project) is not Project or project.mesh_only:
        raise PreparedReferenceScopeError("authored geometry reference scope requires a supported Project")
    declared={field.name for field in fields(Project)}
    unknown={name for name in vars(project) if not name.startswith("_") and name not in declared}
    if unknown:
        raise PreparedReferenceScopeError(f"unsupported external project fields: {sorted(unknown)}")
    inventoried = {"geometry", "supports", "masses", "imperfections", "refinements",
        "load_cases", "regions", "geometry_attachment_regions", "section_assignments",
        "face_sections", "edge_sections", "face_assignment_ids", "edge_assignment_ids",
        "output_requests", "seeding_overrides", "sheet_join_intents"}

    def refuse_uninventoried_refs(value, path):
        if isinstance(value, (EntityRef, FeatureOutputRef, RegionRef)):
            raise PreparedReferenceScopeError(f"unsupported reference-bearing input: {path}")
        if isinstance(value, np.ndarray):
            refuse_uninventoried_refs(value.tolist(), path + "/array")
            return
        if is_dataclass(value) and not isinstance(value, type):
            children = ((field.name, getattr(value, field.name)) for field in fields(value)
                        if not field.name.startswith("_"))
        elif isinstance(value, Mapping):
            children = ((str(key), (key, item)) for key, item in value.items())
        elif isinstance(value, (tuple, list, set, frozenset)):
            children = ((str(index), item) for index, item in enumerate(value))
        else:
            return
        for name, child in children:
            refuse_uninventoried_refs(child, path + "/" + name)

    for field in fields(Project):
        if field.name not in inventoried and not field.name.startswith("_"):
            refuse_uninventoried_refs(getattr(project, field.name), field.name)
    problems=project.sheet_join_intent_problems()
    if problems:raise PreparedReferenceScopeError("unresolved Sheet ownership intent: "+"; ".join(problems))
    content={field.name:_freeze(getattr(project,field.name)) for field in fields(Project)
             if field.name!="geometry" and not field.name.startswith("_")}
    document=_dump(geometry_to_dict(project.geometry))
    content["geometry"] = json.loads(document)
    return _dump(content),document


def _mapping(project, prepared, owner_scope, closure):
    if closure is None:
        if prepared is not project.geometry:
            raise PreparedReferenceScopeError("a detached prepared model requires its source ModelClosure")
        stores={"vertex":"vertices","edge":"edges","face":"faces","face_use":"face_uses",
                "sheet":"sheets","part":"parts","member":"members","coedge":"coedges",
                "member_edge_use":"member_edge_uses","junction":"junctions","attachment":"attachments"}
        mapping=tuple(sorted((kind,key,key) for kind,store in stores.items()
                             for key in getattr(prepared,store)))
        roots=tuple((root,"project_authored",root) for root in owner_scope.face_preimages.authored_face_ids)
        return mapping,roots
    from anygeometry.closure import ModelClosure
    from anygeometry.identity import ResolutionStatus
    if type(closure) is not PreparedProjectClosure or closure not in _closure_bindings:
        raise PreparedReferenceScopeError("detached preparation requires a witnessed project closure extraction")
    source_document, initial_document, forward, inverse = _closure_bindings[closure]
    closure = closure.closure
    if type(closure) is not ModelClosure or closure.working_model is not prepared:
        raise PreparedReferenceScopeError("wrong prepared closure")
    if (source_document != _dump(geometry_to_dict(project.geometry))
            or initial_document != _dump(owner_scope.authored_document)
            or forward != tuple(sorted(closure.source_to_work.items()))
            or inverse != tuple(sorted(closure.work_to_source.items()))):
        raise PreparedReferenceScopeError("closure extraction source, authored work or mapping changed")
    if closure.source_model_id!=project.geometry.model_id or closure.source_revision!=project.geometry.revision:
        raise PreparedReferenceScopeError("source closure is stale or belongs to another model")
    rows=[]
    for source,target in closure.source_to_work.items():
        if (source.model_id!=project.geometry.model_id or target.model_id!=prepared.model_id
                or source.kind!=target.kind or closure.work_to_source.get(target)!=source):
            raise PreparedReferenceScopeError("invalid bidirectional source closure mapping")
        if project.geometry.resolve_handle(source).status is not ResolutionStatus.ACTIVE:
            raise PreparedReferenceScopeError("source closure contains an inactive source handle")
        rows.append((source.kind,source.id,target.id))
    if len(rows)!=len(closure.work_to_source):
        raise PreparedReferenceScopeError("incomplete source closure mapping")
    mapping=tuple(sorted(rows))
    face_sources={target:source for kind,source,target in mapping if kind=="face"}
    try:roots=tuple((root,"project_current",face_sources[root])
                    for root in owner_scope.face_preimages.authored_face_ids)
    except KeyError as error:
        raise PreparedReferenceScopeError("an authored prepared root has no source face") from error
    return mapping,roots


def _inventory(project, prepared, owner_scope, mapping):
    from .model.attributes import Support,Mass,LoadCase,PointLoad,Pressure,LineLoad,SurfaceTraction
    from .model.imperfections import Imperfection
    from .mesh.refinement import Refinement
    from .model.sections import SectionAssignment
    from .model.records import OutputRequest
    from .model.ownership import SheetJoinIntent
    source=project.geometry
    lookup={(kind,key):target for kind,key,target in mapping}
    roots=dict(owner_scope.face_preimages.face_descendants)
    references=[]; counts={}; boundary=set(); vertices=set()

    def live(ref):
        if type(ref) is not EntityRef or ref.kind not in {"vertex","edge","face"}:
            raise PreparedReferenceScopeError("unsupported geometry reference")
        resolved=tuple(source.resolve_ref(ref))
        if not resolved:
            raise PreparedReferenceScopeError(f"missing geometry reference {ref}")
        for item in resolved:source.entity_ref(item.kind,item.id)
        return resolved

    def region_targets(identifier):
        region=project.regions[identifier]
        if region.domain is not RegionDomain.GEOMETRY:
            raise PreparedReferenceScopeError(f"mesh-scoped region {identifier} needs a separate association basis")
        candidates=tuple(source.entity_ref(kind,key) for kind,store in
                         (("vertex",source.vertices),("edge",source.edges),("face",source.faces)) for key in store)
        result=project.regions.resolve(identifier,geometry=source,candidates=candidates,
            feature_resolver=lambda anchor:source.features.resolve(anchor,source))
        return tuple(dict.fromkeys(item for ref in result for item in live(ref)))

    def targets(record):
        region=getattr(record,"region",None)
        if region is None and getattr(record,"ref",None) is not None:
            region=project.raw_geometry_attachment_region(record)
        if region is not None:
            if type(region) is not RegionRef:raise PreparedReferenceScopeError("unsupported application region reference")
            return region.id,region_targets(region.id)
        ref=getattr(record,"ref",None)
        if ref is None:
            raise PreparedReferenceScopeError("a scoped record has no supported reference")
        return None,live(ref)

    def add(key,category,record,refs,region_id=None,authority="canonical",allow_empty=False):
        if any(item.key == key for item in references):
            raise PreparedReferenceScopeError(f"duplicate application reference identity: {key}")
        refs=tuple(sorted(set(refs),key=lambda ref:(ref.kind,ref.id)))
        if not refs and not allow_empty:raise PreparedReferenceScopeError(f"empty application scope: {key}")
        current=[]
        for ref in refs:
            initial=lookup.get((ref.kind,ref.id))
            if initial is None:raise PreparedReferenceScopeError(f"application reference outside prepared closure: {key}: {ref}")
            made=tuple(prepared.resolve_ref(EntityRef(ref.kind,initial)))
            if not made:raise PreparedReferenceScopeError(f"unresolved prepared reference: {key}: {ref}")
            for item in made:prepared.entity_ref(item.kind,item.id)
            current.extend(made)
        current=tuple(sorted(set(current),key=lambda ref:(ref.kind,ref.id)))
        face_ids={ref.id for ref in current if ref.kind=="face"}; face_scopes=[]
        for root,children in sorted(roots.items()):
            chosen=face_ids.intersection(children)
            if not chosen:continue
            whole=chosen==set(children)
            edges=tuple(sorted({use.edge for face in chosen for loop in
                (prepared.faces[face].loop,*prepared.faces[face].holes) for use in loop})) if not whole else ()
            face_scopes.append(FaceReferenceScope(root,tuple(sorted(chosen)),whole,edges))
            if authority!="derived_cache":boundary.update(edges)
        if face_ids-{face for children in roots.values() for face in children}:
            raise PreparedReferenceScopeError("current face is outside complete authored preimages")
        if authority!="derived_cache":
            boundary.update(ref.id for ref in current if ref.kind=="edge")
            vertices.update(ref.id for ref in current if ref.kind=="vertex")
        references.append(ExternalProjectReference(key,category,authority,region_id,
                          _dump(_freeze(record)),refs,current,tuple(face_scopes)))

    def collection(name,records,expected):
        records=tuple(records);counts[name]=len(records)
        for index,record in enumerate(records):
            if type(record) is not expected:raise PreparedReferenceScopeError(f"unsupported {name} record type")
            coordinate=getattr(record,"coordinate_system_id","global")
            if coordinate not in project.coordinate_systems:raise PreparedReferenceScopeError(f"missing coordinate system {coordinate}")
            if name=="refinements" and record.ref is None:
                if record.center is None:raise PreparedReferenceScopeError("unbound refinement")
                refs=tuple(source.entity_ref("face",key) for key in source.faces)
                add(f"{name}/{index}",name,record,refs,allow_empty=True);continue
            region_id,refs=targets(record)
            add(f"{name}/{getattr(record,'id',index)}",name,record,refs,region_id)

    for name,expected in (("supports",Support),("masses",Mass),("imperfections",Imperfection),("refinements",Refinement)):
        collection(name,getattr(project,name),expected)
    counts["load_cases"]=len(project.load_cases)
    for key,case in sorted(project.load_cases.items()):
        if type(case) is not LoadCase:raise PreparedReferenceScopeError("unsupported load case")
        for name,expected in (("point_loads",PointLoad),("pressures",Pressure),("line_loads",LineLoad),("surface_tractions",SurfaceTraction)):
            collection(f"load_cases/{case.id}/{name}",getattr(case,name),expected)
        if case.gravity is not None:
            if case.gravity_coordinate_system_id not in project.coordinate_systems:raise PreparedReferenceScopeError("missing gravity coordinate system")
            add(f"load_cases/{case.id}/gravity","gravity",case,
                tuple(source.entity_ref("face",key) for key in source.faces),allow_empty=True)

    counts["regions"]=len(project.regions)
    for region in sorted(project.regions,key=lambda item:item.id):
        add(f"regions/{region.id}","region",region,region_targets(region.id),region.id,allow_empty=True)
    counts["geometry_attachment_regions"]=len(project.geometry_attachment_regions)
    for ref,region in project.geometry_attachment_regions.items():
        live(ref)
        if type(region) is not RegionRef:raise PreparedReferenceScopeError("unsupported raw attachment binding")
        add(f"geometry_attachment_regions/{ref.kind}/{ref.id}","raw_binding",(ref,region),region_targets(region.id),region.id)

    counts["section_assignments"]=len(project.section_assignments)
    assignment_targets={}
    for key,assignment in sorted(project.section_assignments.items()):
        if type(assignment) is not SectionAssignment or assignment.id!=key:raise PreparedReferenceScopeError("unsupported section assignment")
        store=project.plate_sections if assignment.kind=="plate" else project.beam_sections
        section=next((section for section in store.values() if section.id==assignment.section_id),None)
        if section is None or section.material not in project.materials:raise PreparedReferenceScopeError("missing section or material")
        refs=region_targets(assignment.region.id)
        expected="face" if assignment.kind=="plate" else "edge"
        if any(ref.kind!=expected for ref in refs):raise PreparedReferenceScopeError("wrong section assignment entity kind")
        assignment_targets[key]=set(refs)
        add(f"section_assignments/{key}","section_assignment",assignment,refs,assignment.region.id)
    for attribute,kind,sections,ids in (("face_sections","face",project.plate_sections,project.face_assignment_ids),
                                       ("edge_sections","edge",project.beam_sections,project.edge_assignment_ids)):
        cache=getattr(project,attribute);counts[attribute]=len(cache)
        if set(ids)-set(cache):raise PreparedReferenceScopeError("assignment IDs lack a section cache entry")
        for identifier,name in sorted(cache.items()):
            if name not in sections or sections[name].material not in project.materials:raise PreparedReferenceScopeError("unknown cached section or material")
            refs=live(EntityRef(kind,identifier));assignment_id=ids.get(identifier)
            authority="legacy"
            if assignment_id is not None:
                assignment=project.section_assignments.get(assignment_id)
                if (assignment is None or assignment.section_id!=sections[name].id
                        or not set(refs).issubset(assignment_targets[assignment_id])):
                    raise PreparedReferenceScopeError("section cache disagrees with canonical assignment")
                authority="derived_cache"
            add(f"{attribute}/{identifier}","section_cache",(identifier,name,assignment_id),refs,authority=authority)

    counts["output_requests"]=len(project.output_requests)
    for key,request in sorted(project.output_requests.items()):
        if type(request) is not OutputRequest or key!=request.id:raise PreparedReferenceScopeError("unsupported output request")
        if request.basis not in project.coordinate_systems:raise PreparedReferenceScopeError("missing output coordinate system")
        add(f"output_requests/{key}","output_request",request,region_targets(request.region.id),request.region.id)
    counts["seeding_overrides"]=len(project.seeding_overrides)
    for edge,divisions in sorted(project.seeding_overrides.items()):
        if type(divisions) is not int or divisions<=0:raise PreparedReferenceScopeError("unsupported seeding divisions")
        add(f"seeding_overrides/{edge}","seeding",(edge,divisions),live(EntityRef("edge",edge)))
    counts["sheet_join_intents"]=len(project.sheet_join_intents)
    for key,intent in sorted(project.sheet_join_intents.items()):
        if type(intent) is not SheetJoinIntent or key!=intent.id:raise PreparedReferenceScopeError("unsupported Sheet Join intent")
        refs=[]
        for anchor in intent.anchors:
            resolved=live(anchor) if type(anchor) is EntityRef else source.features.resolve(anchor,source)
            refs.extend(current for ref in resolved for current in live(ref))
        add(f"sheet_join_intents/{key}","sheet_join",intent,refs)
    return tuple(sorted(references,key=lambda item:item.key)),tuple(sorted(counts.items())),tuple(sorted(boundary)),tuple(sorted(vertices))


def _manifest_checksum(manifest):
    return sha256(_dump({field.name:_freeze(getattr(manifest,field.name))
                        for field in fields(manifest) if field.name!="manifest_checksum"}).encode()).hexdigest()


def _owner_apis():
    try:
        from anygeometry import query_prepared_model_scope,validate_prepared_model_scope_binding
    except ImportError as error:
        raise PreparedReferenceScopeError("ANYgeometry lacks the coordinated prepared-model-scope API") from error
    return query_prepared_model_scope,validate_prepared_model_scope_binding


def query_prepared_project_reference_scope(project, prepared_geometry, *, closure=None):
    """Capture supported FEM inputs and exact owner preparation, without edits.

    No absent container is accepted as a negative proof. Unknown records,
    unresolved regions and references outside a detached closure refuse. A
    current child scope stays current-child; it is never expanded to its root.
    """
    query_prepared_model_scope,validate_prepared_model_scope_binding=_owner_apis()
    inputs,document=_project_inputs(project)
    checksum=sha256(inputs.encode()).hexdigest()
    owner=query_prepared_model_scope(prepared_geometry)
    mapping,roots=_mapping(project,prepared_geometry,owner,closure)
    references,counts,boundary,vertices=_inventory(project,prepared_geometry,owner,mapping)
    manifest=PreparedProjectReferenceScope(1,project.document_id,str(project.geometry.model_id),
        project.geometry.revision,checksum,inputs,document,owner,mapping,roots,references,
        counts,boundary,vertices,"")
    from dataclasses import replace
    manifest=replace(manifest,manifest_checksum=_manifest_checksum(manifest))
    validate_prepared_model_scope_binding(prepared_geometry,owner)
    if _project_inputs(project)!=(inputs,document):raise PreparedReferenceScopeError("project changed during reference capture")
    return manifest


def validate_prepared_project_reference_scope(project, prepared_geometry, manifest, *, closure=None):
    """Revalidate immediately before publication; do not trust saved metadata."""
    _,validate_prepared_model_scope_binding=_owner_apis()
    if type(manifest) is not PreparedProjectReferenceScope or manifest.schema_version!=1:
        raise PreparedReferenceScopeError("unsupported prepared project reference manifest")
    if _manifest_checksum(manifest)!=manifest.manifest_checksum:
        raise PreparedReferenceScopeError("prepared project reference manifest was altered")
    if (manifest.document_id!=project.document_id or manifest.project_geometry_model_id!=str(project.geometry.model_id)
            or manifest.project_geometry_revision!=project.geometry.revision):
        raise PreparedReferenceScopeError("project reference manifest is stale or belongs to another project")
    inputs,document=_project_inputs(project)
    if inputs!=manifest.project_input_json or document!=manifest.source_geometry_document_json:
        raise PreparedReferenceScopeError("project inputs changed")
    if sha256(inputs.encode()).hexdigest()!=manifest.project_checksum:
        raise PreparedReferenceScopeError("project fingerprint was altered")
    validate_prepared_model_scope_binding(prepared_geometry,manifest.owner_scope)
    mapping,roots=_mapping(project,prepared_geometry,manifest.owner_scope,closure)
    if mapping!=manifest.source_to_prepared_ids or roots!=manifest.authored_face_sources:
        raise PreparedReferenceScopeError("source-to-prepared reference mapping changed")
    expected=_inventory(project,prepared_geometry,manifest.owner_scope,mapping)
    if expected!=(manifest.references,manifest.inventory_counts,manifest.required_boundary_edge_ids,manifest.required_vertex_ids):
        raise PreparedReferenceScopeError("project reference inventory was altered")
    validate_prepared_model_scope_binding(prepared_geometry,manifest.owner_scope)
    if _project_inputs(project)!=(inputs,document):raise PreparedReferenceScopeError("project changed during reference validation")


def assert_authored_root_face_scope(manifest, authored_face_id, current_face_ids):
    """Refuse root-only publication when a child-local face scope exists.

    This checks association intent only. Even success gives no quality,
    geometric coverage, retained-constraint or publication permission. A
    mesher supporting child scopes needs its own conforming cell membership.
    """
    if type(manifest) is not PreparedProjectReferenceScope or _manifest_checksum(manifest)!=manifest.manifest_checksum:
        raise PreparedReferenceScopeError("invalid prepared project reference manifest")
    children=dict(manifest.owner_scope.face_preimages.face_descendants).get(authored_face_id)
    if children is None or set(current_face_ids)!=set(children):
        raise PreparedReferenceScopeError("authored-root association cannot alias a current child or partial domain")
    for reference in manifest.references:
        if reference.authority=="derived_cache":continue
        if any(scope.authored_face_id==authored_face_id and not scope.whole_authored_root for scope in reference.face_scopes):
            raise PreparedReferenceScopeError(f"child-local face reference requires conforming child associations: {reference.key}")
