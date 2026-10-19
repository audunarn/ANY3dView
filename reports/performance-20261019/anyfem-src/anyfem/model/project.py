"""The project: geometry plus everything attached to it.

This is the headless API.  A script can build a complete model -- geometry,
sections, supports, loads -- and solve it without a GUI ever being imported.
That property is deliberate: it is what makes the preprocessor testable, it
gives the application its scripting console, and it is the seam a parametric
front-end would later call into.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from anymaterial import MaterialSpec
from anygeometry.closure import extract_model_closure
from anygeometry.entities import EntityRef
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel

from ..mesh.mapped import ELEMENT_ORDERS, Mesh, generate_mesh as generate_mapped_mesh
from anymesher.hybrid import generate_hybrid_mesh
from anymesher import check_mappable
if TYPE_CHECKING:
    from anymesher import MeshAutomationOptions
from ..mesh.refinement import Refinement
from ..mesh.seeding import Seeding
from ..native_meshing import NativeMeshSettings
from ..mesh_controls import MeshControls, StructuredMeshControls
from ..structural_preparation import (
    StructuralPreparationError,
    prepare_structural_connectivity,
    remap_mesh_to_source,
    source_work_mapping,
)
from .attributes import (
    Combination,
    LoadCase,
    Mass,
    PressureTable,
    Support,
    WetSurface,
)
from .imperfections import Imperfection
from .formulations import ShellFormulationPolicy
from .sections import BeamSection, PlateSection, SectionAssignment
from .coordinates import CoordinateSystem, GLOBAL_COORDINATES
from .records import (
    AnalysisDefinition,
    ArtifactRef,
    JobRecord,
    MeshRecord,
    OutputRequest,
)
from .regions import (
    GeometryGroupRef,
    ManualRegion,
    Region,
    RegionDomain,
    RegionRef,
    RegionRegistry,
)
from .units import UnitProfile, unit_profile
from .ownership import (
    SheetJoinIntent,
    infer_sheet_join_intents,
    reapply_sheet_join_intents,
    sheet_join_intent_problems,
)

__all__ = ["Project", "ProjectError"]


class ProjectError(ValueError):
    """Raised when a model is incomplete or inconsistent."""


_NATIVE_TRIANGULATION_BACKENDS = ("auto", "python", "native")


@dataclass
class Project:
    """A complete finite element model, before meshing."""

    name: str = "model"
    geometry: GeometryModel = field(default_factory=GeometryModel)
    materials: Dict[str, MaterialSpec] = field(default_factory=dict)
    material_ids: Dict[str, str] = field(default_factory=dict)
    plate_sections: Dict[str, PlateSection] = field(default_factory=dict)
    beam_sections: Dict[str, BeamSection] = field(default_factory=dict)
    face_sections: Dict[int, str] = field(default_factory=dict)
    edge_sections: Dict[int, str] = field(default_factory=dict)
    face_assignment_ids: Dict[int, str] = field(default_factory=dict)
    edge_assignment_ids: Dict[int, str] = field(default_factory=dict)
    # Canonical bindings.  The four dictionaries above remain materialized
    # compatibility views for established headless callers.
    section_assignments: Dict[str, SectionAssignment] = field(default_factory=dict)
    # Feature-independent structural ownership intent.  Geometry feature
    # replay may rebuild face topology/owners from its baseline; these exact
    # anchors deterministically restore explicit multi-face Sheets afterwards.
    sheet_join_intents: Dict[str, SheetJoinIntent] = field(default_factory=dict)
    supports: List[Support] = field(default_factory=list)
    masses: List[Mass] = field(default_factory=list)
    # Plates wetted by fluid; the target of wet pressures in a load case.
    wet_surfaces: List[WetSurface] = field(default_factory=list)
    # Reference points that stand for a set of lines, points or plates; the
    # target of connection loads in a load case.
    connections: List[Connection] = field(default_factory=list)
    load_cases: Dict[str, LoadCase] = field(default_factory=dict)
    combinations: Dict[str, Combination] = field(default_factory=dict)
    imperfections: List[Imperfection] = field(default_factory=list)
    refinements: List[Refinement] = field(default_factory=list)
    element_order: str = "linear"
    # Document-level registries.  They are additive to the established
    # headless fields above, so existing scripts remain source-compatible.
    document_id: str = field(default_factory=lambda: str(uuid4()))
    units: UnitProfile = field(default_factory=unit_profile)
    coordinate_systems: Dict[str, CoordinateSystem] = field(
        default_factory=lambda: {"global": GLOBAL_COORDINATES}
    )
    regions: RegionRegistry = field(default_factory=RegionRegistry)
    output_requests: Dict[str, OutputRequest] = field(default_factory=dict)
    analyses: Dict[str, AnalysisDefinition] = field(default_factory=dict)
    mesh_records: Dict[str, MeshRecord] = field(default_factory=dict)
    jobs: Dict[str, JobRecord] = field(default_factory=dict)
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)
    # Submodel definitions and runs (``anyfem.model.submodels``).  Outside the
    # numerical model hash: adding or editing one stales no job of this model.
    submodels: Dict[str, Any] = field(default_factory=dict)
    submodel_runs: Dict[str, Any] = field(default_factory=dict)
    target_size: float | None = None
    seeding_overrides: Dict[int, int] = field(default_factory=dict)
    native_mesh_settings: NativeMeshSettings | None = None
    native_triangulation_backend: str = "auto"
    # Imported analyses can be mesh-native and therefore intentionally have
    # no fabricated geometry topology behind their group references.
    mesh_only: bool = False
    imported_format: str | None = None
    imported_semantics_artifact_id: str | None = None
    # Compatibility recovery is persisted document intent, not session state.
    # A v1--v6 file with the historic detached-feature corruption keeps its
    # exact topology as one checksummed frozen feature and archives the broken
    # records here for diagnostics.  Geometry editing may be disabled while
    # assignments and analyses remain usable; ``read_only_reason`` is reserved
    # for an unresolved/invalid recovery that must fail closed entirely.
    archived_feature_histories: List[Dict[str, Any]] = field(default_factory=list)
    compatibility_diagnostics: List[str] = field(default_factory=list)
    geometry_editing_disabled_reason: str | None = None
    read_only_reason: str | None = None
    # Appended so established positional Project arguments remain stable.
    shell_formulation_policy: ShellFormulationPolicy = field(
        default_factory=ShellFormulationPolicy.current_default
    )
    # Appended application binding for raw owner records without a region field.
    # Established positional arguments and owner record types stay unchanged.
    geometry_attachment_regions: Dict[EntityRef, RegionRef] = field(default_factory=dict)
    _singleton_region_cache: Dict[object, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _singleton_region_cache_size: int = field(
        default=-1, init=False, repr=False, compare=False
    )
    _last_mesh_preparation: Dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.document_id = str(self.document_id)
        if not isinstance(self.shell_formulation_policy, ShellFormulationPolicy):
            raise TypeError("shell_formulation_policy must be a ShellFormulationPolicy")
        self.set_native_triangulation_backend(self.native_triangulation_backend)
        self.coordinate_systems.setdefault("global", GLOBAL_COORDINATES)
        for name in self.materials:
            self.material_ids.setdefault(
                name, str(uuid5(NAMESPACE_URL, f"{self.document_id}:material:{name}"))
            )
        for case in self.load_cases.values():
            case._region_factory = self.singleton_region

    def infer_existing_sheet_join_intents(self) -> tuple[SheetJoinIntent, ...]:
        """Adopt losslessly replayable explicit multi-face owners only."""

        inferred = infer_sheet_join_intents(
            self.geometry, document_id=self.document_id
        )
        for identifier, intent in inferred.items():
            self.sheet_join_intents.setdefault(identifier, intent)
        return tuple(inferred.values())

    def add_sheet_join_intent(self, intent: SheetJoinIntent) -> SheetJoinIntent:
        """Register exact structural Sheet ownership intent by stable UUID."""

        if not isinstance(intent, SheetJoinIntent):
            raise TypeError("add_sheet_join_intent expects a SheetJoinIntent")
        if intent.id in self.sheet_join_intents:
            raise ProjectError(f"duplicate Sheet Join intent ID {intent.id!r}")
        self.sheet_join_intents[intent.id] = intent
        return intent

    def remove_sheet_join_intent(self, intent_id: str) -> SheetJoinIntent:
        try:
            return self.sheet_join_intents.pop(str(intent_id))
        except KeyError:
            raise ProjectError(
                f"no Sheet Join intent with ID {intent_id!r}"
            ) from None

    def reapply_sheet_join_intents(self) -> tuple[int, ...]:
        """Rebuild explicit multi-face ownership from exact persistent anchors."""

        return reapply_sheet_join_intents(
            self.geometry, self.sheet_join_intents.values()
        )

    def sheet_join_intent_problems(self) -> tuple[str, ...]:
        """Return read-only diagnostics for unsatisfied structural joins."""

        return sheet_join_intent_problems(
            self.geometry, self.sheet_join_intents.values()
        )

    def regenerate_geometry_features(self, registry=None):
        """Regenerate features and Sheet ownership as one atomic project edit."""
        from anygeometry.features import RegenerationReport
        from .feature_bindings import (
            _stage_feature_project, _rebind_feature_attachments,
            _attribute_snapshot, _restore_attributes,
        )
        working_project = _stage_feature_project(self)
        report = working_project._regenerate_geometry_features_detached(registry)
        if report.success:
            try:
                _rebind_feature_attachments(working_project, report.replacements, previous_geometry=self.geometry)
                after_attributes = _attribute_snapshot(working_project)
            except (GeometryError, ValueError, KeyError) as error:
                return RegenerationReport(False, report.features, (), diagnostic=str(error))
            self.geometry.restore_design(working_project.geometry.design_snapshot())
            _restore_attributes(self, after_attributes)
        return report

    def _regenerate_geometry_features_detached(self, registry=None):
        """Regenerate an already-detached project without another clone.

        Command staging owns the surrounding project clone. Keeping that fact
        explicit avoids serializing large generated geometry twice while the
        public method above remains atomic for direct callers.
        """

        from anygeometry.features import RegenerationReport

        if registry is None:
            from ..geometry_features import anyfem_feature_registry

            registry = anyfem_feature_registry()
        report = self.geometry.regenerate_features(registry)
        if not report.success:
            return report
        try:
            reapply_sheet_join_intents(
                self.geometry, self.sheet_join_intents.values()
            )
        except (GeometryError, ValueError) as error:
            return RegenerationReport(
                False,
                report.features,
                (),
                diagnostic=(
                    "feature regeneration could not restore structural Sheet "
                    f"ownership: {error}"
                ),
            )
        return report

    def singleton_region(
        self,
        ref: EntityRef,
        *,
        _output_anchors: Mapping[EntityRef, object] | None = None,
    ) -> RegionRef:
        """Return the hidden persistent region backing a legacy direct ref."""

        anchor: object = (
            (_output_anchors or {}).get(ref, ref)
            if _output_anchors is not None
            else self._feature_output_anchors().get(ref, ref)
        )
        self._refresh_singleton_region_cache()
        cached = self._singleton_region_cache.get(anchor)
        if cached is not None and cached in self.regions:
            return RegionRef(cached)
        anchor_key = (
            f"feature:{anchor.feature_id}:{anchor.output_key}:{anchor.kind}"
            if all(hasattr(anchor, name) for name in ("feature_id", "output_key", "kind"))
            else f"entity:{ref.kind}:{ref.id}"
        )
        region = Region(
            id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"{self.document_id}:singleton:{anchor_key}",
                )
            ),
            name=f"scope_{ref.kind}_{ref.id}",
            domain=RegionDomain.GEOMETRY,
            entity_kind=ref.kind,
            definition=ManualRegion((anchor,)),
            hidden=True,
        )
        self.regions.add(region)
        self._singleton_region_cache[anchor] = region.id
        self._singleton_region_cache_size = len(self.regions)
        return RegionRef(region.id)

    def adopt_geometry_attachment_regions(self) -> None:
        """Bind legacy raw records only through exact owner output provenance."""
        if self.mesh_only:
            return
        records = (*self.imperfections, *self.refinements)
        references = {item.ref for item in records if isinstance(getattr(item, "ref", None), EntityRef)}
        self.geometry_attachment_regions = {
            reference: region for reference, region in self.geometry_attachment_regions.items()
            if reference in references
        }
        anchors = self._feature_output_anchors()
        for item in records:
            reference = getattr(item, "ref", None)
            if not isinstance(reference, EntityRef) or reference in self.geometry_attachment_regions:
                continue
            current = self.geometry.resolve_ref(reference)
            if len(current) == 1 and current[0] in anchors:
                self.geometry_attachment_regions[reference] = self.singleton_region(current[0], _output_anchors=anchors)

    def raw_geometry_attachment_region(self, item) -> RegionRef | None:
        """Raw-record bindings never override another record sharing its cache."""
        if isinstance(item, (Imperfection, Refinement)):
            return self.geometry_attachment_regions.get(item.ref)
        return None

    def resolve_geometry_attachment(self, reference: EntityRef) -> tuple[EntityRef, ...]:
        """Resolve a raw authored cache through its durable scope or lineage."""
        if self.mesh_only:
            return (reference,)
        region = self.geometry_attachment_regions.get(reference)
        if region is None:
            return tuple(self.geometry.resolve_ref(reference))
        return tuple(self.regions.resolve(region.id, geometry=self.geometry,
            feature_resolver=lambda anchor: self.geometry.features.resolve(anchor, self.geometry)))

    def _feature_output_anchors(self) -> dict[EntityRef, object]:
        """Current topology-to-design map, built once by bulk assignments."""

        try:
            from anygeometry.features import FeatureOutputRef
        except ImportError:  # pragma: no cover - coordinated package floor
            return {}
        anchors: dict[EntityRef, object] = {}
        for feature in getattr(self.geometry.features, "records", ()):
            for key, output in feature.outputs.items():
                anchors[output] = FeatureOutputRef(
                    feature.feature_id, key, output.kind
                )
        return anchors

    def _refresh_singleton_region_cache(self) -> None:
        if self._singleton_region_cache_size == len(self.regions):
            return
        self._singleton_region_cache.clear()
        for region in self.regions:
            definition = region.definition
            if (
                region.hidden
                and isinstance(definition, ManualRegion)
                and len(definition.anchors) == 1
            ):
                self._singleton_region_cache.setdefault(
                    definition.anchors[0], region.id
                )
        self._singleton_region_cache_size = len(self.regions)

    def add_coordinate_system(self, system: CoordinateSystem) -> CoordinateSystem:
        if system.id in self.coordinate_systems:
            raise ProjectError(f"duplicate coordinate-system ID {system.id!r}")
        if any(item.name == system.name for item in self.coordinate_systems.values()):
            raise ProjectError(f"coordinate system {system.name!r} already exists")
        self.coordinate_systems[system.id] = system
        return system

    def add_analysis(self, analysis: AnalysisDefinition) -> AnalysisDefinition:
        if analysis.id in self.analyses:
            raise ProjectError(f"duplicate analysis ID {analysis.id!r}")
        missing = sorted(
            set(analysis.output_request_ids).difference(self.output_requests)
        )
        if missing:
            raise ProjectError(
                f"analysis {analysis.name!r} references missing output request(s) "
                f"{missing}"
            )
        self.analyses[analysis.id] = analysis
        return analysis

    def add_output_request(self, request: OutputRequest) -> OutputRequest:
        """Register one immutable, canonically region-scoped output request."""

        if request.id in self.output_requests:
            raise ProjectError(f"duplicate output-request ID {request.id!r}")
        if request.region.id not in self.regions:
            raise ProjectError(
                f"output request {request.label!r} references missing region "
                f"{request.region.id!r}"
            )
        self.output_requests[request.id] = request
        return request

    def update_output_request(self, request: OutputRequest) -> OutputRequest:
        """Replace one immutable request without changing its stable UUID."""

        if request.id not in self.output_requests:
            raise ProjectError(f"unknown output-request ID {request.id!r}")
        if request.region.id not in self.regions:
            raise ProjectError(
                f"output request {request.label!r} references missing region "
                f"{request.region.id!r}"
            )
        self.output_requests[request.id] = request
        return request

    def remove_output_request(
        self, request_id: str, *, cascade: bool = False
    ) -> OutputRequest:
        """Remove a request, rejecting or detaching dependent analyses."""

        identifier = str(request_id)
        try:
            request = self.output_requests[identifier]
        except KeyError:
            raise ProjectError(f"unknown output-request ID {identifier!r}") from None
        dependents = tuple(
            analysis
            for analysis in self.analyses.values()
            if identifier in analysis.output_request_ids
        )
        if dependents and not cascade:
            raise ProjectError(
                f"output request {request.label!r} is used by analysis/analyses "
                f"{[item.name for item in dependents]}; detach it first"
            )
        if cascade:
            for analysis in dependents:
                self.analyses[analysis.id] = replace(
                    analysis,
                    output_request_ids=tuple(
                        value
                        for value in analysis.output_request_ids
                        if value != identifier
                    ),
                )
        del self.output_requests[identifier]
        return request

    def output_request_provenance(self, analysis_id: str) -> tuple[dict, ...]:
        """Return requested metadata only; this never creates result fields."""

        try:
            analysis = self.analyses[str(analysis_id)]
        except KeyError:
            raise ProjectError(f"unknown analysis ID {analysis_id!r}") from None
        values: list[dict] = []
        for request_id in analysis.output_request_ids:
            try:
                request = self.output_requests[request_id]
            except KeyError:
                raise ProjectError(
                    f"analysis {analysis.name!r} references missing output request "
                    f"{request_id!r}"
                ) from None
            values.append(request.to_dict())
        return tuple(values)

    def add_mesh_record(self, mesh: MeshRecord) -> MeshRecord:
        if mesh.id in self.mesh_records:
            raise ProjectError(f"duplicate mesh record ID {mesh.id!r}")
        self.mesh_records[mesh.id] = mesh
        return mesh

    def add_job(self, job: JobRecord) -> JobRecord:
        if job.id in self.jobs:
            raise ProjectError(f"duplicate job ID {job.id!r}")
        self.jobs[job.id] = job
        return job

    def add_artifact(self, artifact: ArtifactRef) -> ArtifactRef:
        if artifact.id in self.artifacts:
            raise ProjectError(f"duplicate artifact ID {artifact.id!r}")
        self.artifacts[artifact.id] = artifact
        return artifact

    # ------------------------------------------------------------------
    # materials and sections
    # ------------------------------------------------------------------
    def add_material(self, material: MaterialSpec) -> MaterialSpec:
        self.materials[material.name] = material
        self.material_ids.setdefault(material.name, str(uuid4()))
        return material

    def add_plate_section(
        self, name: str, thickness: float, material: str, *, id: str | None = None
    ) -> PlateSection:
        self._require_material(material)
        section = PlateSection(
            name=name,
            thickness=thickness,
            material=material,
            **({"id": id} if id is not None else {}),
        )
        self.plate_sections[name] = section
        return section

    def add_beam_section(self, section: BeamSection) -> BeamSection:
        self._require_material(section.material)
        self.beam_sections[section.name] = section
        return section

    # ------------------------------------------------------------------
    # assignment
    # ------------------------------------------------------------------
    def _ensure_plate_ownership(self, face_ids: Iterable[int]) -> tuple[int, ...]:
        """Give unowned assigned faces persistent Part/Sheet topology.

        Each independently authored plate starts as its own Sheet.  Sharing a
        B-rep edge establishes geometric adjacency, not a promise that both
        faces have a coherent structural orientation.  Engineers can join
        compatible faces explicitly; assignment must never guess and create
        an invalid multi-face Sheet.
        """

        geometry = self.geometry
        owned_faces = {
            face_use.face_id for face_use in geometry.face_uses.values()
        }
        pending = tuple(
            sorted(
                {
                    int(face_id)
                    for face_id in face_ids
                    if int(face_id) not in owned_faces
                }
            )
        )
        if not pending:
            return ()

        created: list[tuple[int, int]] = []
        try:
            for face_id in pending:
                sheet_id = geometry.add_sheet(
                    (face_id,),
                    name=f"assigned plate {face_id}",
                )
                created.append((sheet_id, geometry.sheets[sheet_id].part_id))
        except BaseException:
            for sheet_id, part_id in reversed(created):
                if sheet_id in geometry.sheets:
                    geometry.remove_sheet(sheet_id)
                if part_id in geometry.parts:
                    part = geometry.parts[part_id]
                    if not part.sheet_ids and not part.member_ids:
                        geometry.remove_part(part_id)
            raise
        return tuple(sheet_id for sheet_id, _part_id in created)

    def _ensure_beam_ownership(self, edge_ids: Iterable[int]) -> tuple[int, ...]:
        """Give each unowned section-bearing edge one persistent Member."""

        geometry = self.geometry
        owned_edges = {
            use.edge_id for use in geometry.member_edge_uses.values()
        }
        pending = tuple(
            sorted(
                {
                    int(edge_id)
                    for edge_id in edge_ids
                    if int(edge_id) not in owned_edges
                }
            )
        )
        created: list[tuple[int, int]] = []
        try:
            for edge_id in pending:
                member_id = geometry.add_member(
                    (edge_id,), name=f"assigned beam {edge_id}"
                )
                created.append((member_id, geometry.members[member_id].part_id))
        except BaseException:
            for member_id, part_id in reversed(created):
                if member_id in geometry.members:
                    geometry.remove_member(member_id)
                if part_id in geometry.parts:
                    part = geometry.parts[part_id]
                    if not part.sheet_ids and not part.member_ids:
                        geometry.remove_part(part_id)
            raise
        return tuple(member_id for member_id, _part_id in created)

    def assign_plate(self, face_id: int, section: str) -> None:
        """Give a plate its thickness and material.

        This established ID-based API now creates a hidden singleton region.
        The dictionaries remain immediately readable, while the region-backed
        record is the durable source of truth.
        """

        self._assign_many((face_id,), "plate", section)

    def assign_plates(self, face_ids: Iterable[int], section: str) -> None:
        self._assign_many(face_ids, "plate", section)

    def assign_plate_group(self, group: str, section: str) -> None:
        """Assign a plate section to every face in an ANYgeometry group."""

        self.geometry_group(group, kind="face")
        self.assign_plate_region(
            self._geometry_group_region(group, "face"),
            section,
            name=f"{section} on {group}",
        )

    def assign_beam(self, edge_id: int, section: str) -> None:
        """Turn a line into a beam member through a singleton region."""

        reference = self.geometry.entity_ref("edge", edge_id)
        self._assign_singleton(reference, "beam", section)

    def assign_beams(self, edge_ids: Iterable[int], section: str) -> None:
        self._assign_many(edge_ids, "beam", section)

    def assign_beam_group(self, group: str, section: str) -> None:
        """Assign a beam section to every edge in an ANYgeometry group."""

        self.geometry_group(group, kind="edge")
        self.assign_beam_region(
            self._geometry_group_region(group, "edge"),
            section,
            name=f"{section} on {group}",
        )

    def assign_plate_region(
        self,
        region: RegionRef,
        section: str,
        *,
        name: str | None = None,
        id: str | None = None,
    ) -> SectionAssignment:
        """Assign a plate section to a reusable geometry region."""

        return self._add_section_assignment(
            kind="plate", region=region, section=section, name=name, id=id
        )

    def assign_beam_region(
        self,
        region: RegionRef,
        section: str,
        *,
        name: str | None = None,
        id: str | None = None,
    ) -> SectionAssignment:
        """Assign a beam section to a reusable geometry region."""

        return self._add_section_assignment(
            kind="beam", region=region, section=section, name=name, id=id
        )

    def add_section_assignment(
        self, assignment: SectionAssignment
    ) -> SectionAssignment:
        """Add an already constructed canonical binding atomically."""

        # Preserve direct edits made through the historical dictionaries
        # before composing a new canonical record with them.
        self.resolve_section_assignments(strict=True)
        if assignment.id in self.section_assignments:
            raise ProjectError(f"duplicate section assignment ID {assignment.id!r}")
        self._require_section_id(assignment.kind, assignment.section_id)
        self._require_assignment_region(assignment)
        previous = self._section_compatibility_snapshot()
        self.section_assignments[assignment.id] = assignment
        try:
            self.resolve_section_assignments(strict=True, adopt_legacy=False)
        except BaseException:
            self.section_assignments.pop(assignment.id, None)
            self._restore_section_compatibility(previous)
            raise
        return assignment

    def remove_section_assignment(self, assignment_id: str) -> SectionAssignment:
        """Remove one binding and rebuild the compatibility maps."""

        try:
            assignment = self.section_assignments.pop(str(assignment_id))
        except KeyError:
            raise ProjectError(
                f"no section assignment with ID {assignment_id!r}"
            ) from None
        self.resolve_section_assignments(strict=False, adopt_legacy=False)
        return assignment

    def resolve_section_assignments(
        self, *, strict: bool = False, adopt_legacy: bool = True
    ) -> tuple[str, ...]:
        """Resolve canonical bindings and refresh legacy topology maps.

        Records are processed by stable UUID, so even an invalid overlapping
        document materializes the same diagnostic view on every machine.  An
        overlap never silently applies last-writer-wins: the lower UUID is
        retained only for inspection and the diagnostic blocks meshing/solve.
        """

        if self.mesh_only and not self.section_assignments:
            # Imported neutral meshes historically carry synthetic owner IDs
            # rather than ANYgeometry entities.  Keep that established map
            # until a mesh-aware canonical assignment is explicitly present.
            return ()
        if adopt_legacy:
            self._adopt_legacy_section_maps()

        face_sections: dict[int, str] = {}
        edge_sections: dict[int, str] = {}
        face_ids: dict[int, str] = {}
        edge_ids: dict[int, str] = {}
        owners: dict[tuple[str, int], SectionAssignment] = {}
        problems: list[str] = []
        stores = {
            "face": self.geometry.faces,
            "edge": self.geometry.edges,
        }
        candidate_cache = {
            kind: tuple(EntityRef(kind, identifier) for identifier in store)
            for kind, store in stores.items()
        }

        for assignment in sorted(
            self.section_assignments.values(), key=lambda item: item.id
        ):
            expected_kind = "face" if assignment.kind == "plate" else "edge"
            try:
                section_name = self._section_name(
                    assignment.kind, assignment.section_id
                )
                region = self.regions[assignment.region.id]
                if region.domain is not RegionDomain.GEOMETRY:
                    raise ProjectError(
                        "mesh-scoped section assignments require an active "
                        "immutable mesh"
                    )
                if region.entity_kind != expected_kind:
                    raise ProjectError(
                        f"{assignment.kind} assignment expects a {expected_kind} "
                        f"region, not {region.entity_kind!r}"
                    )
                store = stores[expected_kind]
                resolved = self.regions.resolve(
                    assignment.region.id,
                    geometry=self.geometry,
                    candidates=candidate_cache[expected_kind],
                    feature_resolver=lambda anchor: self.geometry.features.resolve(
                        anchor, self.geometry
                    ),
                )
                references = tuple(dict.fromkeys(resolved))
                if not references:
                    raise ProjectError(f"region {region.name!r} is empty")
                for reference in references:
                    if not isinstance(reference, EntityRef):
                        raise ProjectError(
                            f"region {region.name!r} resolved a non-geometry target"
                        )
                    if reference.kind != expected_kind or reference.id not in store:
                        raise ProjectError(
                            f"region {region.name!r} resolved invalid {reference}"
                        )
            except (KeyError, TypeError, ValueError) as error:
                problems.append(
                    f"section assignment {assignment.name!r} is unresolved: {error}"
                )
                continue

            for reference in sorted(references, key=lambda item: item.id):
                key = (expected_kind, int(reference.id))
                previous = owners.get(key)
                if previous is not None:
                    problems.append(
                        f"section assignments {previous.name!r} and "
                        f"{assignment.name!r} overlap on {expected_kind} "
                        f"{reference.id}"
                    )
                    continue
                owners[key] = assignment
                if expected_kind == "face":
                    face_sections[reference.id] = section_name
                    face_ids[reference.id] = assignment.id
                else:
                    edge_sections[reference.id] = section_name
                    edge_ids[reference.id] = assignment.id

        self.face_sections.clear()
        self.face_sections.update(face_sections)
        self.edge_sections.clear()
        self.edge_sections.update(edge_sections)
        self.face_assignment_ids.clear()
        self.face_assignment_ids.update(face_ids)
        self.edge_assignment_ids.clear()
        self.edge_assignment_ids.update(edge_ids)

        if not problems and not self.mesh_only:
            try:
                self._ensure_plate_ownership(face_sections)
                self._ensure_beam_ownership(edge_sections)
            except GeometryError as error:
                problems.append(f"structural ownership is invalid: {error}")

        result = tuple(problems)
        if strict and result:
            raise ProjectError("invalid section assignments:\n  - " + "\n  - ".join(result))
        return result

    def _add_section_assignment(
        self,
        *,
        kind: str,
        region: RegionRef,
        section: str,
        name: str | None,
        id: str | None,
    ) -> SectionAssignment:
        section_id = self._section_id(kind, section)
        assignment = SectionAssignment(
            kind=kind,  # type: ignore[arg-type]
            section_id=section_id,
            region=region,
            name=name or f"{self._section_name(kind, section_id)} assignment",
            **({"id": id} if id is not None else {}),
        )
        return self.add_section_assignment(assignment)

    def _assign_singleton(
        self,
        reference: EntityRef,
        kind: str,
        section: str,
        *,
        resolve_existing: bool = True,
        materialize: bool = True,
        output_anchors: Mapping[EntityRef, object] | None = None,
        legacy_by_region: dict[tuple[str, str], SectionAssignment] | None = None,
    ) -> SectionAssignment:
        if resolve_existing:
            self.resolve_section_assignments(strict=False)
        section_id = self._section_id(kind, section)
        materialized_ids = (
            self.face_assignment_ids
            if reference.kind == "face"
            else self.edge_assignment_ids
        )
        materialized = self.section_assignments.get(
            materialized_ids.get(reference.id, "")
        )
        if (
            materialized is not None
            and materialized.kind == kind
            and materialized.section_id == section_id
        ):
            # A prior singleton may have expanded through explicit geometry
            # replacement lineage.  Reapplying the same section to one of its
            # descendants is an idempotent legacy operation, not an overlap.
            return materialized
        # The materialized assignment is authoritative for reassignment.  A
        # feature executor may normalize a historical semantic output key on
        # reopen (for example ``face`` to ``face/0``); looking up a freshly
        # generated singleton region in that case would create a second scope
        # over the same topology and fail with an overlap instead of replacing
        # the section selected by the engineer.
        materialized_singleton = (
            materialized
            if materialized is not None
            and materialized.legacy_singleton
            and materialized.kind == kind
            else None
        )
        region = (
            materialized_singleton.region
            if materialized_singleton is not None
            else self.singleton_region(reference, _output_anchors=output_anchors)
        )
        expected = "face" if kind == "plate" else "edge"
        if materialized_singleton is not None:
            existing = materialized_singleton
        elif legacy_by_region is not None:
            existing = legacy_by_region.get((kind, region.id))
        else:
            existing = next(
                (
                    item
                    for item in self.section_assignments.values()
                    if item.legacy_singleton
                    and item.kind == kind
                    and item.region == region
                ),
                None,
            )
        previous = self._section_compatibility_snapshot() if materialize else None
        old_record = existing
        if existing is None:
            assignment = SectionAssignment(
                kind=kind,  # type: ignore[arg-type]
                section_id=section_id,
                region=region,
                name=f"{section} on {expected} {reference.id}",
                legacy_singleton=True,
            )
        else:
            assignment = replace(
                existing,
                section_id=section_id,
                name=f"{section} on {expected} {reference.id}",
            )
        self.section_assignments[assignment.id] = assignment
        if legacy_by_region is not None:
            legacy_by_region[(kind, region.id)] = assignment
        if not materialize:
            return assignment
        try:
            self.resolve_section_assignments(strict=True, adopt_legacy=False)
        except BaseException:
            if old_record is None:
                self.section_assignments.pop(assignment.id, None)
            else:
                self.section_assignments[old_record.id] = old_record
            assert previous is not None
            self._restore_section_compatibility(previous)
            raise
        return assignment

    def _assign_many(
        self, identifiers: Iterable[int], kind: str, section: str
    ) -> None:
        entity_kind = "face" if kind == "plate" else "edge"
        references = tuple(
            dict.fromkeys(
                self.geometry.entity_ref(entity_kind, int(identifier))
                for identifier in identifiers
            )
        )
        if not references:
            return
        self._section_id(kind, section)
        self.resolve_section_assignments(strict=True)
        prior_assignments = dict(self.section_assignments)
        prior_regions = {region.id for region in self.regions}
        prior_maps = self._section_compatibility_snapshot()
        output_anchors = self._feature_output_anchors()
        legacy_by_region = {
            (item.kind, item.region.id): item
            for item in self.section_assignments.values()
            if item.legacy_singleton
        }
        try:
            for reference in references:
                self._assign_singleton(
                    reference,
                    kind,
                    section,
                    resolve_existing=False,
                    materialize=False,
                    output_anchors=output_anchors,
                    legacy_by_region=legacy_by_region,
                )
            self.resolve_section_assignments(strict=True, adopt_legacy=False)
            if kind == "plate":
                self._ensure_plate_ownership(reference.id for reference in references)
            else:
                self._ensure_beam_ownership(reference.id for reference in references)
        except BaseException:
            self.section_assignments.clear()
            self.section_assignments.update(prior_assignments)
            for region in tuple(self.regions):
                if region.id not in prior_regions:
                    self.regions.remove(region.id)
            self._singleton_region_cache_size = -1
            self._restore_section_compatibility(prior_maps)
            raise

    def _geometry_group_region(self, group: str, kind: str) -> RegionRef:
        anchor = GeometryGroupRef(group, kind)
        for region in self.regions:
            if (
                region.domain is RegionDomain.GEOMETRY
                and region.entity_kind == kind
                and isinstance(region.definition, ManualRegion)
                and region.definition.anchors == (anchor,)
            ):
                return RegionRef(region.id)
        region = Region(
            id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"{self.document_id}:geometry-group:{kind}:{group}",
                )
            ),
            name=f"scope_{group}_{kind}",
            domain=RegionDomain.GEOMETRY,
            entity_kind=kind,
            definition=ManualRegion((anchor,)),
            hidden=True,
        )
        self.regions.add(region)
        return RegionRef(region.id)

    def _adopt_legacy_section_maps(self) -> None:
        """Fold direct compatibility-dictionary edits into singleton records."""

        for kind, entity_kind, values, identifiers in (
            (
                "plate",
                "face",
                self.face_sections,
                self.face_assignment_ids,
            ),
            (
                "beam",
                "edge",
                self.edge_sections,
                self.edge_assignment_ids,
            ),
        ):
            store = (
                self.geometry.faces if entity_kind == "face" else self.geometry.edges
            )
            keys_by_assignment: dict[str, list[int]] = {}
            for key, assignment_id in identifiers.items():
                keys_by_assignment.setdefault(assignment_id, []).append(key)
            # Commands written against the historical dictionaries remove the
            # value but cannot know about the new canonical registry.  A cached
            # assignment UUID whose value disappeared is an explicit deletion.
            for assignment in tuple(self.section_assignments.values()):
                if assignment.kind != kind or not assignment.legacy_singleton:
                    continue
                keys = keys_by_assignment.get(assignment.id, ())
                present = [values[key] for key in keys if key in values]
                if keys and not present:
                    try:
                        section_name = self._section_name(
                            assignment.kind, assignment.section_id
                        )
                        region = self.regions[assignment.region.id]
                        assignment_targets = self.regions.resolve(
                            assignment.region.id,
                            geometry=self.geometry,
                            candidates=tuple(
                                EntityRef(entity_kind, identifier)
                                for identifier in store
                            ),
                            feature_resolver=lambda anchor: (
                                self.geometry.features.resolve(
                                    anchor, self.geometry
                                )
                            ),
                        )
                        carried = tuple(
                            target.id
                            for target in assignment_targets
                            if isinstance(target, EntityRef)
                            and target.kind == entity_kind
                            and values.get(target.id) == section_name
                        )
                    except (KeyError, TypeError, ValueError):
                        carried = ()
                    if carried:
                        for key in keys:
                            identifiers.pop(key, None)
                        for key in carried:
                            identifiers[key] = assignment.id
                        continue
                    self.section_assignments.pop(assignment.id, None)
                    for key in keys:
                        identifiers.pop(key, None)
                    continue
                if present:
                    names = set(present)
                    if len(names) == 1:
                        try:
                            section_id = self._section_id(kind, present[0])
                        except ProjectError:
                            continue
                        if assignment.section_id != section_id:
                            self.section_assignments[assignment.id] = replace(
                                assignment, section_id=section_id
                            )

            for identifier, section_name in sorted(tuple(values.items())):
                assignment_id = identifiers.get(identifier)
                assignment = (
                    self.section_assignments.get(assignment_id)
                    if assignment_id is not None
                    else None
                )
                try:
                    requested_section_id = self._section_id(kind, section_name)
                except ProjectError:
                    if assignment is not None:
                        # A section label was edited while the compatibility
                        # cache still held its former label.  The UUID-backed
                        # canonical record is authoritative in that case.
                        continue
                    raise
                if assignment is not None:
                    try:
                        assigned_name = self._section_name(
                            assignment.kind, assignment.section_id
                        )
                    except ProjectError:
                        assigned_name = None
                    if assigned_name == section_name:
                        continue
                if identifier not in store:
                    # A feature-backed singleton may still be cached under its
                    # previous materialized ID.  Its assignment UUID above is
                    # enough to retain and rematerialize it at the new output.
                    if assignment is not None and assignment.legacy_singleton:
                        continue
                    continue
                reference = EntityRef(entity_kind, int(identifier))
                region = self.singleton_region(reference)
                candidate_id = assignment_id or str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{self.document_id}:{entity_kind}-assignment:{identifier}",
                    )
                )
                if candidate_id in self.section_assignments:
                    candidate_id = str(uuid4())
                record = SectionAssignment(
                    id=candidate_id,
                    kind=kind,  # type: ignore[arg-type]
                    section_id=requested_section_id,
                    region=region,
                    name=f"{section_name} on {entity_kind} {identifier}",
                    legacy_singleton=True,
                )
                self.section_assignments[record.id] = record
                identifiers[int(identifier)] = record.id

    def _section_id(self, kind: str, value: str) -> str:
        sections = self.plate_sections if kind == "plate" else self.beam_sections
        if value in sections:
            return str(sections[value].id)
        for section in sections.values():
            if section.id == str(value):
                return str(section.id)
        label = "plate" if kind == "plate" else "beam"
        raise ProjectError(f"no {label} section named or identified by {value!r}")

    def _section_name(self, kind: str, section_id: str) -> str:
        sections = self.plate_sections if kind == "plate" else self.beam_sections
        for name, section in sections.items():
            if section.id == str(section_id):
                return name
        label = "plate" if kind == "plate" else "beam"
        raise ProjectError(
            f"{label} section UUID {section_id!r} does not exist"
        )

    def _require_section_id(self, kind: str, section_id: str) -> None:
        self._section_name(kind, section_id)

    def _require_assignment_region(self, assignment: SectionAssignment) -> Region:
        try:
            region = self.regions[assignment.region.id]
        except (KeyError, ValueError) as error:
            raise ProjectError(
                f"section assignment uses missing region {assignment.region.id!r}: "
                f"{error}"
            ) from None
        expected = "face" if assignment.kind == "plate" else "edge"
        if (
            region.domain is RegionDomain.GEOMETRY
            and region.entity_kind != expected
        ):
            raise ProjectError(
                f"{assignment.kind} assignment requires a {expected} region"
            )
        return region

    def _section_compatibility_snapshot(self) -> tuple[dict, dict, dict, dict]:
        return (
            dict(self.face_sections),
            dict(self.edge_sections),
            dict(self.face_assignment_ids),
            dict(self.edge_assignment_ids),
        )

    def _restore_section_compatibility(
        self, snapshot: tuple[dict, dict, dict, dict]
    ) -> None:
        for target, values in zip(
            (
                self.face_sections,
                self.edge_sections,
                self.face_assignment_ids,
                self.edge_assignment_ids,
            ),
            snapshot,
        ):
            target.clear()
            target.update(values)

    def geometry_group(
        self, name: str, *, kind: str | None = None
    ) -> tuple[EntityRef, ...]:
        """Resolve one semantic geometry group, optionally by entity kind."""

        if name not in self.geometry.groups:
            raise ProjectError(f"no geometry group named {name!r}")
        references = self.geometry.group(name)
        if kind is not None:
            references = tuple(item for item in references if item.kind == kind)
        if not references:
            detail = "entities" if kind is None else f"{kind} entities"
            raise ProjectError(f"geometry group {name!r} has no {detail}")
        return references

    @property
    def beam_edges(self) -> List[int]:
        return sorted(self.edge_sections)

    def beam_section_frame(self, edge_id: int) -> tuple[Any, Any, Any]:
        """Return unit beam-axis, local-y and local-z vectors for one line.

        Local z is the web direction. By default it follows the first attached
        plate normal; Front/Back and the section's axial rotation are then
        applied deterministically. An explicit ``web_direction`` remains the
        advanced override for beam-only models.
        """

        import numpy as np

        section = self.beam_section_of(int(edge_id))
        samples = np.asarray(
            self.geometry.sample_edge(int(edge_id), np.asarray((0.0, 1.0))),
            dtype=float,
        )
        axis = samples[-1] - samples[0]
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 0.0:
            raise ProjectError(f"beam line {edge_id} has zero length")
        axis /= axis_length

        if section.web_direction is not None:
            local_z = np.asarray(section.web_direction, dtype=float)
        else:
            face_ids = tuple(sorted(self.geometry.faces_using_edge(int(edge_id))))
            if not face_ids:
                midpoint = 0.5 * (samples[0] + samples[-1])
                attached: list[int] = []
                for face_id in sorted(self.geometry.faces):
                    _point, _uv, distance = self.geometry.project_to_face(
                        face_id, midpoint
                    )
                    bounds = self.geometry.conservative_face_bounds(face_id)
                    extent = (
                        1.0
                        if bounds is None
                        else float(
                            np.linalg.norm(
                                np.asarray(bounds[3:]) - np.asarray(bounds[:3])
                            )
                        )
                    )
                    tolerance = self.geometry.tolerance.effective_surface_residual(
                        extent
                    )
                    if float(distance) <= tolerance:
                        attached.append(int(face_id))
                face_ids = tuple(attached)
            if face_ids:
                local_z = np.asarray(
                    self.geometry.face_normal(face_ids[0], 0.5, 0.5), dtype=float
                )
            else:
                # A free beam still needs a stable frame. Choose the global
                # direction least parallel to its axis.
                candidates = np.eye(3)
                local_z = candidates[
                    int(np.argmin(np.abs(candidates @ axis)))
                ]

        local_z = local_z - float(local_z @ axis) * axis
        length = float(np.linalg.norm(local_z))
        if length <= 1.0e-12 and section.web_direction is None:
            # A beam that pierces a shell can have its midpoint on that shell
            # while its axis is parallel to the shell normal.  The shell is a
            # connection target, not a usable section-frame reference.  Fall
            # back to the same deterministic global direction as a free beam;
            # an explicit, invalid web direction remains an actionable error.
            candidates = np.eye(3)
            local_z = candidates[int(np.argmin(np.abs(candidates @ axis)))]
            local_z = local_z - float(local_z @ axis) * axis
            length = float(np.linalg.norm(local_z))
        if length <= 1.0e-12:
            raise ProjectError(
                f"beam section {section.name!r} web direction is parallel to line {edge_id}"
            )
        local_z /= length
        if section.attachment_side == "back":
            local_z *= -1.0
        angle = np.deg2rad(float(section.rotation_deg))
        if angle:
            local_z = (
                local_z * np.cos(angle)
                + np.cross(axis, local_z) * np.sin(angle)
            )
            local_z /= float(np.linalg.norm(local_z))
        local_y = np.cross(local_z, axis)
        local_y /= float(np.linalg.norm(local_y))
        return axis, local_y, local_z

    def beam_offset_vector(self, edge_id: int) -> tuple[float, float, float]:
        """Exact neutral-axis offset from the line attached to the plate."""

        import numpy as np

        section = self.beam_section_of(int(edge_id))
        if section.offset_mode == "centerline":
            return (0.0, 0.0, 0.0)
        _axis, local_y, local_z = self.beam_section_frame(int(edge_id))
        if section.offset_mode == "automatic":
            y, z = section.centroid_from_attachment()
            vector = y * local_y + z * local_z
        else:
            vector = float(section.eccentricity) * local_z
        return tuple(float(value) for value in np.asarray(vector, dtype=float))

    @property
    def beam_offsets(self) -> Dict[int, float | tuple[float, float, float]]:
        """Non-zero neutral-axis offsets for assigned beam lines.

        Historical unrotated manual offsets remain scalars. Automatic,
        back-side or rotated attachment uses ANYmesher's exact vector form.
        """

        offsets: Dict[int, float | tuple[float, float, float]] = {}
        for edge_id, name in self.edge_sections.items():
            if name not in self.beam_sections:
                continue
            section = self.beam_sections[name]
            if (
                section.offset_mode == "manual"
                and section.attachment_side == "front"
                and float(section.rotation_deg) == 0.0
            ):
                if section.eccentricity:
                    offsets[int(edge_id)] = float(section.eccentricity)
                continue
            vector = self.beam_offset_vector(int(edge_id))
            if any(value != 0.0 for value in vector):
                offsets[int(edge_id)] = vector
        return offsets

    # ------------------------------------------------------------------
    # supports and loads
    # ------------------------------------------------------------------
    def add_support(self, support: Support) -> Support:
        if not self.mesh_only:
            self._require_entity(support.ref)
        if support.region is None:
            support = replace(support, region=self.singleton_region(support.ref))
        self.supports.append(support)
        return support

    def add_symmetry(
        self,
        ref: EntityRef,
        normal: str | Sequence[float],
        *,
        antisymmetric: bool = False,
        tolerance: float = 1.0e-6,
    ) -> Support:
        """Restrain a point, line or plate as a symmetry plane.

        Checks that the entity actually *lies* in the plane before adding the
        support.  A symmetry condition on an edge that runs across the plane
        rather than along it restrains the wrong degrees of freedom everywhere
        it touches, and the model still solves and still looks reasonable, so
        the mistake would otherwise surface as a stiffness that is merely a bit
        wrong.

        ``antisymmetric`` swaps to the complementary set, for a load that is
        antisymmetric about the plane.
        """

        from .attributes import _AXES, _symmetry_axis, antisymmetry, symmetry

        self._require_entity(ref)
        axis = _symmetry_axis(normal)
        self._require_in_plane(ref, _AXES[axis], axis, tolerance)
        build = antisymmetry if antisymmetric else symmetry
        return self.add_support(build(ref, axis))

    def _require_in_plane(
        self, ref: EntityRef, index: int, axis: str, tolerance: float
    ) -> None:
        points = self._entity_points(ref)
        spread = float(points[:, index].max() - points[:, index].min())
        if spread > tolerance:
            raise ProjectError(
                f"{ref} does not lie in a plane normal to {axis}: its "
                f"{axis} coordinate varies by {spread:.6g} m (from "
                f"{points[:, index].min():.6g} to {points[:, index].max():.6g}). "
                "A symmetry condition applies to the entities lying *in* the "
                "plane, not to those crossing it."
            )

    def _entity_points(self, ref: EntityRef) -> "np.ndarray":
        """Points along an entity, enough to tell whether it is planar.

        Faces are sampled through ANYgeometry's authoritative surface.  Looking
        only at the boundary would miss a curved or ruled interior whose edge
        happens to lie in the requested symmetry plane.
        """

        import numpy as np

        geometry = self.geometry
        if ref.kind == "vertex":
            return np.asarray([geometry.vertex_position(ref.id)], dtype=float)
        if ref.kind == "edge":
            return np.asarray(
                geometry.sample_edge(ref.id, np.linspace(0.0, 1.0, 9)),
                dtype=float,
            )
        if ref.kind == "face":
            parameters = np.linspace(0.0, 1.0, 5)
            return np.asarray(
                [
                    geometry.face_point(ref.id, float(u), float(v))
                    for u in parameters
                    for v in parameters
                ],
                dtype=float,
            )
        raise ProjectError(f"cannot take points of a {ref.kind}")

    def add_mass(self, mass: Mass) -> Mass:
        if not self.mesh_only:
            self._require_entity(mass.ref)
        if mass.region is None:
            mass = replace(mass, region=self.singleton_region(mass.ref))
        self.masses.append(mass)
        return mass

    def add_wet_surface(self, surface: WetSurface) -> WetSurface:
        """Mark plates as wet so wet pressures act on them."""

        if not self.mesh_only:
            self._require_entity(surface.ref)
        if surface.region is None:
            surface = replace(surface, region=self.singleton_region(surface.ref))
        self.wet_surfaces.append(surface)
        return surface

    def add_connection(self, connection: Connection) -> Connection:
        """Tie the selected geometry to one reference point.

        A load applied at the reference point (``LoadCase.add_connection_load``)
        acts on everything the connection ties together.  Names are unique so
        a load can be addressed by the connection's name as well as its id.
        """

        if not self.mesh_only:
            self._require_entity(connection.ref)
        if any(item.name == connection.name for item in self.connections):
            raise ProjectError(
                f"a connection named {connection.name!r} already exists"
            )
        if connection.region is None:
            connection = replace(
                connection, region=self.singleton_region(connection.ref)
            )
        self.connections.append(connection)
        return connection

    def connection_for(self, key: str) -> Connection:
        """A connection by id or, failing that, by its (unique) name."""

        for item in self.connections:
            if item.id == key:
                return item
        for item in self.connections:
            if item.name == key:
                return item
        raise ProjectError(f"no connection with id or name {key!r}")

    def connection_targets(self, connection: Connection) -> tuple[EntityRef, ...]:
        """The geometry entities a connection ties, resolved through its region."""

        if self.mesh_only:
            return (connection.ref,)
        if connection.region is None:
            return tuple(self.geometry.resolve_ref(connection.ref))
        geometry = self.geometry
        region = self.regions[connection.region.id]
        stores = {
            "vertex": geometry.vertices,
            "edge": geometry.edges,
            "face": geometry.faces,
        }
        return tuple(
            dict.fromkeys(
                self.regions.resolve(
                    region.id,
                    geometry=geometry,
                    candidates=tuple(
                        EntityRef(region.entity_kind, identifier)
                        for identifier in stores.get(region.entity_kind, {})
                    ),
                    feature_resolver=lambda anchor: geometry.features.resolve(
                        anchor, geometry
                    ),
                )
            )
        )

    def connection_reference_point(self, connection: Connection) -> "np.ndarray":
        """Where a connection's reference point is, from the geometry alone.

        ``centroid`` is the length-weighted centre of the selected lines, the
        area-weighted centre of selected plates and the mean of selected
        points; ``bounding_box`` the centre of their bounding box. The point is
        independent of the mesh, so remeshing never moves it.
        """

        if connection.reference == "point":
            return np.asarray(connection.reference_point, dtype=float).copy()
        if self.mesh_only:
            raise ProjectError(
                "an imported mesh has no geometry to place a connection's "
                "reference point; give it an explicit point"
            )
        geometry = self.geometry
        targets = self.connection_targets(connection)
        if not targets:
            raise ProjectError(
                f"connection {connection.name!r} resolves to nothing"
            )
        if connection.reference == "bounding_box":
            points = np.vstack([self._entity_points(ref) for ref in targets])
            return 0.5 * (points.min(axis=0) + points.max(axis=0))
        total = 0.0
        moment = np.zeros(3)
        for ref in targets:
            if ref.kind == "vertex":
                weight, centre = 1.0, geometry.vertex_position(ref.id)
            elif ref.kind == "edge":
                weight = float(geometry.edge_length(ref.id))
                # Uniform parameter is uniform arc length on every edge type.
                samples = np.asarray(
                    geometry.sample_edge(ref.id, (np.arange(32) + 0.5) / 32.0),
                    dtype=float,
                )
                centre = samples.mean(axis=0)
            else:
                grid = np.linspace(0.0, 1.0, 17)
                points = np.asarray(
                    [
                        [geometry.face_point(ref.id, float(u), float(v)) for v in grid]
                        for u in grid
                    ],
                    dtype=float,
                )
                du = points[1:, :-1] - points[:-1, :-1]
                dv = points[:-1, 1:] - points[:-1, :-1]
                cells = np.linalg.norm(np.cross(du, dv), axis=-1)
                midpoints = 0.25 * (
                    points[1:, :-1] + points[:-1, :-1] + points[1:, 1:] + points[:-1, 1:]
                )
                weight = float(cells.sum())
                centre = (cells[..., None] * midpoints).sum(axis=(0, 1)) / max(
                    weight, 1.0e-300
                )
            total += weight
            moment += weight * np.asarray(centre, dtype=float)
        if total <= 0.0:
            raise ProjectError(
                f"connection {connection.name!r} selects nothing with a size"
            )
        return moment / total

    def add_load_case_series(
        self,
        name: str,
        times: Sequence[float],
        *,
        template: str | LoadCase | None = None,
        builder: Callable[[LoadCase, float, int], None] | None = None,
        parameters: Mapping[str, float] | None = None,
        digits: int | None = None,
    ) -> List[LoadCase]:
        """Create one load case per time step, for a quasi-static sequence.

        Each case is named ``"<name>_<step>"`` (zero-padded), carries its
        ``time`` and the shared ``parameters``, and is evaluated by its own
        equations -- so a pressure expression in ``t`` yields a different load
        at every step. The loads of each case come from ``template`` (an
        existing case, copied) and/or ``builder(case, time, step)``, called
        once per new case to add loads that depend on the step.

        Because stiffness does not change between steps, the cases can be
        solved together with one factorization
        (:func:`anyfem.solve.solve_linear_static_many`).
        """

        times = [float(t) for t in times]
        if not times:
            raise ProjectError("a load case series needs at least one time step")
        if not all(np.isfinite(times)):
            raise ProjectError("load case series times must be finite")
        if template is None and builder is None:
            raise ProjectError(
                "a load case series needs a template case or a builder"
            )
        source = None
        if template is not None:
            source = (
                template
                if isinstance(template, LoadCase)
                else self.load_cases.get(template)
            )
            if source is None:
                raise ProjectError(f"no load case named {template!r}")
        width = digits if digits is not None else len(str(len(times) - 1))
        names = [f"{name}_{index:0{width}d}" for index in range(len(times))]
        clashes = [item for item in names if item in self.load_cases]
        if clashes:
            raise ProjectError(
                f"load case(s) {clashes[:3]} already exist; remove them or "
                "choose another series name"
            )
        created: List[LoadCase] = []
        try:
            for index, (case_name, time) in enumerate(zip(names, times)):
                case = self.load_case(case_name)
                if source is not None:
                    case.point_loads[:] = list(source.point_loads)
                    case.pressures[:] = list(source.pressures)
                    case.line_loads[:] = list(source.line_loads)
                    case.surface_tractions[:] = list(source.surface_tractions)
                    case.wet_pressures[:] = list(source.wet_pressures)
                    case.connection_loads[:] = list(source.connection_loads)
                    case.gravity = (
                        None if source.gravity is None else np.array(source.gravity)
                    )
                    case.gravity_coordinate_system_id = (
                        source.gravity_coordinate_system_id
                    )
                    case.follower_pressure = source.follower_pressure
                    for key, value in source.parameters.items():
                        case.set_parameter(key, value)
                for key, value in (parameters or {}).items():
                    case.set_parameter(key, value)
                case.set_time(time)
                case.series = name
                case.step = index
                case.number = None  # a step is numbered by its step
                created.append(case)
                if builder is not None:
                    builder(case, time, index)
        except BaseException:
            for case in created:
                self.load_cases.pop(case.name, None)
            raise
        return created

    def load_case(self, name: str = "default") -> LoadCase:
        """Fetch a load case, creating it on first use."""

        if name not in self.load_cases:
            self.load_cases[name] = LoadCase(
                name=name,
                _region_factory=self.singleton_region,
                number=1
                + max(
                    (
                        item.number or 0
                        for item in self.load_cases.values()
                        if item.series is None
                    ),
                    default=0,
                ),
            )
        case = self.load_cases[name]
        case._region_factory = self.singleton_region
        return case

    # ------------------------------------------------------------------
    # load series: shared loads with envelopes, one derived step per time
    # ------------------------------------------------------------------
    @staticmethod
    def _series_times(times: Sequence[float]) -> tuple[float, ...]:
        try:
            values = tuple(float(item) for item in times)
        except (TypeError, ValueError):
            raise ProjectError("series times must be numbers") from None
        if not values:
            raise ProjectError("a load series needs at least one time step")
        if not all(np.isfinite(values)):
            raise ProjectError("series times must be finite")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ProjectError("series times must increase")
        return values

    @staticmethod
    def series_step_names(name: str, count: int) -> List[str]:
        """``name_000`` ... for ``count`` steps, zero-padded to a common width."""

        width = len(str(max(count - 1, 0)))
        return [f"{name}_{index:0{width}d}" for index in range(count)]

    def add_load_series(
        self,
        name: str,
        times: Sequence[float],
        *,
        parameters: Mapping[str, float] | None = None,
    ) -> LoadCase:
        """A load case that covers a whole time history, with one step per time.

        Loads added to the returned case (point, line, pressure, wet pressure,
        connection ...) are stored **once**.  Each load may carry an
        :class:`~anyloads.Envelope` saying from which case or time to which it
        acts and how its strength varies, so a load that spans all of a 600
        step record is one object, and a pulse between steps 20 and 300 is one
        object with a window.  The steps (``name_000`` ...) are derived cases
        that take the definition's loads evaluated at their own time; they are
        what is solved and are never edited.  The definition itself is not
        solvable.
        """

        times = self._series_times(times)
        if name in self.load_cases:
            raise ProjectError(f"a load case named {name!r} already exists")
        case = self.load_case(name)
        case.times = times
        try:
            for key, value in (parameters or {}).items():
                case.set_parameter(key, value)
            self.expand_load_series(name)
        except BaseException:
            self.remove_load_series(name)
            raise
        return case

    def series_children(self, name: str) -> List[LoadCase]:
        """The derived steps of a load series definition, in step order."""

        children = [
            case for case in self.load_cases.values() if case.parent == name
        ]
        return sorted(children, key=lambda case: case.step or 0)

    def expand_load_series(self, name: str) -> List[LoadCase]:
        """(Re)create the derived steps of a definition from its times."""

        parent = self.load_cases.get(name)
        if parent is None or parent.times is None:
            raise ProjectError(f"{name!r} is not a load series definition")
        for child in self.series_children(name):
            del self.load_cases[child.name]
        names = self.series_step_names(name, len(parent.times))
        clashes = [item for item in names if item in self.load_cases]
        if clashes:
            raise ProjectError(
                f"load case(s) {clashes[:3]} already exist; rename them or the series"
            )
        children = []
        for index, (step_name, time) in enumerate(zip(names, parent.times)):
            child = LoadCase(
                name=step_name,
                id=str(uuid5(NAMESPACE_URL, f"{parent.id}:step:{index}")),
                parent=name,
                series=name,
                step=index,
                time=time,
                number=index + 1,
                _region_factory=self.singleton_region,
            )
            self.load_cases[step_name] = child
            children.append(child)
        return children

    def remove_load_series(self, name: str) -> List[LoadCase]:
        """Remove a definition and its steps; returns everything removed."""

        removed = list(self.series_children(name))
        for child in removed:
            self.load_cases.pop(child.name, None)
        parent = self.load_cases.pop(name, None)
        return ([parent] if parent is not None else []) + removed

    def resolve_case(self, case: LoadCase) -> LoadCase:
        """The case to build from: a step becomes its definition's loads at its time.

        Any other case is returned as it is.  The result of resolving a step is
        a view: it shares the definition's load lists and must not be edited.
        """

        if case.parent is None:
            return case
        parent = self.load_cases.get(case.parent)
        if parent is None or parent.times is None:
            raise ProjectError(
                f"load case {case.name!r} belongs to the load series "
                f"{case.parent!r}, which no longer exists"
            )
        from copy import copy

        view = copy(parent)
        view.name = case.name
        view.id = case.id
        view.time = case.time
        view.step = case.step
        view.number = case.number
        view.series = parent.name
        view.parent = None
        view.times = None
        view._window_times = parent.times
        view.parameters = {**parent.parameters, **case.parameters}
        return view

    def add_combination(
        self, name: str, factors: Mapping[str, float]
    ) -> Combination:
        """Define a factored sum of load cases."""

        unknown = sorted(set(factors) - set(self.load_cases))
        if unknown:
            raise ProjectError(
                f"combination {name!r} references undefined load case(s) "
                f"{unknown}"
            )
        combination = Combination(name=name, factors=dict(factors))
        self.combinations[name] = combination
        return combination

    def add_imperfection(self, imperfection: Imperfection) -> Imperfection:
        self._require_entity(imperfection.ref)
        self.imperfections.append(imperfection)
        return imperfection

    # ------------------------------------------------------------------
    # meshing controls
    # ------------------------------------------------------------------
    def add_refinement(self, refinement: Refinement) -> Refinement:
        """Ask for smaller elements in one region."""

        if refinement.ref is not None:
            self._require_entity(refinement.ref)
        self.refinements.append(refinement)
        return refinement

    def set_element_order(self, order: str) -> str:
        """Choose Q4 and 2-node beams, or Q8 and 3-node beams."""

        if order not in ELEMENT_ORDERS:
            raise ProjectError(
                f"unknown element order {order!r}; expected one of "
                f"{', '.join(ELEMENT_ORDERS)}"
            )
        self.element_order = order
        return order

    def set_native_mesh_settings(
        self, settings: NativeMeshSettings | None
    ) -> NativeMeshSettings | None:
        """Persist native controls after enforcing model-bound handle identity."""

        if settings is not None:
            if "native_backend" in dict(settings.parameters):
                raise ProjectError(
                    "native_backend is a project-level setting; remove it from "
                    "NativeMeshSettings.parameters"
                )
            for control in settings.controls:
                if "native_backend" in dict(control.parameters):
                    raise ProjectError(
                        "native_backend is a project-level setting; remove it from "
                        f"control {control.control_id!r} parameters"
                    )
            foreign = [
                handle
                for handle in settings.handles
                if handle.model_id != self.geometry.model_id
            ]
            if foreign:
                raise ProjectError(
                    "native mesh controls contain an entity handle from another "
                    f"geometry model: {foreign[0]}"
                )
        self.native_mesh_settings = settings
        return settings

    def set_native_triangulation_backend(self, backend: str) -> str:
        """Set the constrained-triangulation implementation selector."""

        if not isinstance(backend, str) or backend not in _NATIVE_TRIANGULATION_BACKENDS:
            expected = ", ".join(_NATIVE_TRIANGULATION_BACKENDS)
            raise ProjectError(
                f"native triangulation backend must be one of {expected}; got {backend!r}"
            )
        self.native_triangulation_backend = backend
        return backend

    # ------------------------------------------------------------------
    # meshing
    # ------------------------------------------------------------------
    def generate_mesh(
        self,
        target_size: float | None = None,
        *,
        overrides: Mapping[int, int] | None = None,
        seeding: Seeding | None = None,
        refinements: Iterable[Refinement] | None = None,
        order: str | None = None,
        strategy: str | None = None,
        structure_preference: str | None = None,
        quality_policy: Mapping[str, float] | None = None,
        certification_mode: str | None = None,
        mesh_controls: MeshControls | None = None,
        structured_controls: StructuredMeshControls | None = None,
        quad_options: object | None = None,
        layout_policy: str | None = None,
        automation: MeshAutomationOptions | None = None,
        change_set: object | None = None,
        cancellation_check: Callable[[str], None] | None = None,
        execution_policy: object | None = None,
        pool_lease: object | None = None,
    ) -> Mesh:
        """Mesh every plate, plus every line carrying a beam.

        ``refinements`` and ``order`` default to the project's own, so a script
        that sets them once does not have to repeat them at every call, and a
        one-off comparison can still override them here.

        ``execution_policy`` is execution-only (see :mod:`anyfem.mesh_execution`):
        it selects component-parallel meshing through ANYmesh's public parallel
        API for the whole-project hybrid route, is never serialized into the
        project and never affects any mesh hash.  ``pool_lease`` may carry an
        ANYmesh ``ParallelPoolLease`` for scripts that keep a warm pool; it is
        used in-process only and never transported into the Qt workbench.
        """

        self._last_mesh_execution: dict[str, object] = {
            "route": "serial",
            "workers": None,
            "reason": "selected meshing path runs serially",
        }
        from ..mesh_execution import MeshExecutionPolicy

        if execution_policy is not None and not isinstance(
            execution_policy, MeshExecutionPolicy
        ):
            raise ProjectError(
                "execution_policy must be a MeshExecutionPolicy "
                "(see anyfem.mesh_execution)"
            )

        if self.read_only_reason is not None:
            raise ProjectError(
                "geometry cannot be meshed while compatibility recovery is "
                f"blocked: {self.read_only_reason}"
            )
        ownership_problems = self.sheet_join_intent_problems()
        if ownership_problems:
            raise ProjectError(
                "structural Sheet ownership intent is unresolved:\n  - "
                + "\n  - ".join(ownership_problems)
            )
        feature_problems = self._feature_materialization_problems()
        if feature_problems:
            raise ProjectError(
                "geometry cannot be meshed:\n  - " + "\n  - ".join(feature_problems)
            )
        # Beam membership is a meshing input, so resolve region-backed
        # assignments before asking the mesher for its owner list.  Existing
        # but unresolved or overlapping bindings are unsafe and fail closed;
        # a project with no assignments at all may still be meshed for setup.
        self.resolve_section_assignments(strict=True)
        if not self.geometry.faces and not self.edge_sections:
            raise ProjectError(
                "nothing to mesh: the model has no plates and no beams"
            )
        settings = self.native_mesh_settings
        resolved_size = (
            float(target_size)
            if target_size is not None
            else (
                float(settings.target_size)
                if settings is not None
                else (None if self.target_size is None else float(self.target_size))
            )
        )
        if resolved_size is None:
            raise ProjectError(
                "mesh target size is not set; pass target_size or store native mesh settings"
            )
        settings_backend = (
            None if settings is None else getattr(settings.backend, "value", settings.backend)
        )
        resolved_strategy = strategy or {
            None: "auto",
            "automatic": "auto",
            "mapped": "mapped",
            "native": "native",
            "quad_first": "quad_first",
        }.get(settings_backend, str(settings_backend))
        if resolved_strategy == "quad_first":
            from ..quad_first import effective_quad_options
            from anymesher.quad.public_integration import QuadPublicUnsupported

            if quad_options is None and settings is not None:
                quad_options = settings.to_dict()["parameters"].get("quad_options")
            quad_options = effective_quad_options(quad_options)
            if layout_policy is None:
                layout_policy = (
                    settings.to_dict()["parameters"].get("layout_policy", "existing")
                    if settings is not None else "existing"
                )
            if layout_policy not in ("existing", "adaptive"):
                raise ProjectError("layout_policy must be 'existing' or 'adaptive'")
            boundary_beams = [
                int(edge_id) for edge_id in self.beam_edges
                if self.geometry.faces_using_edge(int(edge_id))
            ]
            if boundary_beams:
                raise QuadPublicUnsupported(
                    "quad-first source-boundary edge beam ownership is unsupported: "
                    + ", ".join(str(edge_id) for edge_id in sorted(boundary_beams))
                )
            resolved_strategy = "native"
        elif quad_options is not None:
            raise ProjectError("quad_options requires strategy='quad_first'")
        elif layout_policy not in (None, "existing"):
            raise ProjectError("adaptive layout requires strategy='quad_first'")
        resolved_order = order or (
            self.element_order if settings is None else settings.element_order
        )
        parameters = {} if settings is None else dict(settings.parameters)
        controls = mesh_controls or MeshControls.from_settings(settings)
        structured_controls = (
            structured_controls
            or StructuredMeshControls.from_settings(settings)
        )
        resolved_certification = certification_mode or controls.certification_mode
        supported_parameters = {
            key: value
            for key, value in parameters.items()
            if key in {"recombine", "overlap_policy"}
        }
        supported_parameters["recombine"] = controls.recombine
        # Mapped-only runs must not activate the opt-in native path.
        if resolved_strategy != "mapped":
            native_options = controls.native_options()
            if native_options is not None:
                supported_parameters["native_options"] = native_options
        stored_quality = {
            key.removeprefix("mesh_quality_"): value
            for key, value in parameters.items()
            if key.startswith("mesh_quality_")
        }
        structure_options = structured_controls.owner_options(
            preference=(
                structure_preference
                if structure_preference is not None
                else parameters.get("structure_preference", "balanced")
            ),
            quality_policy=(
                quality_policy
                if quality_policy is not None
                else stored_quality
            ),
        )
        native_backend = self.set_native_triangulation_backend(
            self.native_triangulation_backend
        )
        if quad_options is not None:
            native_backend = "python"
        source_geometry = self.geometry
        source_members = {
            member_id
            for edge_id in self.beam_edges
            for member_id in source_geometry.members_using_edge(edge_id)
        }
        closure_handles = tuple(
            [
                source_geometry.handle("face", identifier)
                for identifier in sorted(source_geometry.faces)
            ]
            + [
                source_geometry.handle("edge", identifier)
                for identifier in self.beam_edges
            ]
            + [
                source_geometry.handle("sheet", identifier)
                for identifier in sorted(source_geometry.sheets)
            ]
            + [
                source_geometry.handle("member", identifier)
                for identifier in sorted(source_members)
            ]
        )
        closure = extract_model_closure(
            source_geometry,
            closure_handles,
            include_structural_closure=True,
            include_features=False,
        )
        working = closure.working_model
        working_member_ids = tuple(
            sorted(
                closure.source_to_work[
                    source_geometry.handle("member", identifier)
                ].id
                for identifier in source_members
                if source_geometry.handle("member", identifier)
                in closure.source_to_work
            )
        )
        from .._meshing_runtime import collect_geometry_runtime, merge_geometry_runtime

        try:
            with collect_geometry_runtime() as preparation_runtime:
                preparation = prepare_structural_connectivity(
                    working,
                    source_model_id=str(closure.source_model_id),
                    source_revision=closure.source_revision,
                    member_ids=working_member_ids,
                    cancellation_check=cancellation_check,
                )
        except StructuralPreparationError as error:
            raise ProjectError(f"structural preparation failed: {error}") from None
        preparation.source_to_working = source_work_mapping(closure)

        def descendants(kind: str, identifier: int) -> tuple[EntityRef, ...]:
            source = source_geometry.handle(kind, identifier)
            initial = closure.source_to_work.get(source)
            if initial is None:
                return ()
            resolved = tuple(
                working.resolve_ref(EntityRef(kind, initial.id))
            )
            if not resolved and initial.id in {
                "vertex": working.vertices,
                "edge": working.edges,
                "face": working.faces,
            }[kind]:
                resolved = (EntityRef(kind, initial.id),)
            return resolved

        working_beam_edges = tuple(
            sorted(
                {
                    item.id
                    for source_id in self.beam_edges
                    for item in descendants("edge", source_id)
                }
            )
        )
        working_offsets = {
            item.id: offset
            for source_id, offset in self.beam_offsets.items()
            for item in descendants("edge", source_id)
        }
        source_overrides = (
            self.seeding_overrides if overrides is None else dict(overrides)
        )
        working_overrides: dict[int, int] = {}
        for source_id, divisions in source_overrides.items():
            made = descendants("edge", int(source_id))
            if not made:
                continue
            lengths = [working.edge_length(item.id) for item in made]
            total = sum(lengths)
            remaining = max(int(divisions), len(made))
            for index, (item, length) in enumerate(zip(made, lengths)):
                if index == len(made) - 1:
                    share = remaining
                else:
                    share = max(
                        1,
                        round(
                            int(divisions)
                            * (length / total if total > 0.0 else 1.0 / len(made))
                        ),
                    )
                    share = min(share, remaining - (len(made) - index - 1))
                working_overrides[item.id] = int(share)
                remaining -= int(share)

        source_refinements = (
            self.refinements if refinements is None else list(refinements)
        )
        working_refinements: list[Refinement] = []
        for refinement in source_refinements:
            if refinement.ref is None:
                working_refinements.append(refinement)
                continue
            made = tuple(dict.fromkeys(target for source in self.resolve_geometry_attachment(refinement.ref)
                                       for target in descendants(source.kind, source.id)))
            working_refinements.extend(
                replace(refinement, ref=item) for item in made
            )

        already_structured = bool(working.faces) and all(
            check_mappable(working, face_id).ok
            for face_id in working.faces
        )

        working_seeding = seeding
        if seeding is not None:
            if preparation.created_count:
                # Newly imprinted edges need a fresh globally compatible seed.
                working_seeding = None
                preparation.diagnostics.append(
                    "precomputed seeding was recomputed after structural imprint"
                )
            else:
                working_seeding = Seeding(
                    divisions=dict(seeding.divisions),
                    sweeps=int(seeding.sweeps),
                    classes=dict(seeding.classes),
                    size_field=None,
                )

        mapped_fast_path = (
            automation is None
            and resolved_strategy != "native"
            and quad_options is None
            and already_structured
            and len(working.faces) > 1
            and not working_beam_edges
            and preparation.created_count == 0
        )
        if mapped_fast_path:
            self._last_mesh_execution = {
                "route": "serial",
                "workers": None,
                "reason": "the mapped fast path has no parallel route",
            }
            if cancellation_check is not None:
                cancellation_check("mapped generation start")
            mesh = generate_mapped_mesh(
                working,
                target_size=resolved_size,
                overrides=working_overrides,
                face_ids=tuple(sorted(working.faces)),
                seeding=working_seeding,
                refinements=working_refinements,
                order=resolved_order,
            )
            for sheet_id, sheet in working.sheets.items():
                element_ids = {
                    int(element_id)
                    for face_use_id in sheet.face_use_ids
                    for element_id in mesh.elements_of_face.get(
                        working.face_uses[face_use_id].face_id, ()
                    )
                }
                if element_ids:
                    mesh.elements_of_sheet[int(sheet_id)] = sorted(element_ids)
            mesh.structural_preparation = {
                "schema": "anyfem.prepared-mapped-closure",
                "status": "already_structured",
            }
            mesh.hybrid_diagnostics = {
                "requested_target_size": resolved_size,
                "strategy_by_face": {
                    int(face_id): "mapped" for face_id in working.faces
                },
                "triangulation_backend_by_face": {},
                "geometry_model_id": str(working.model_id),
                "geometry_revision": int(working.revision),
                "certification_mode": str(resolved_certification),
                "certifiable": True,
                "preflight_count": 0,
                "structured_layout_status": "already_structured",
                "structured_plan_hash": None,
                "structured_quality": None,
                "structural_preparation_hash": None,
                "qualified_s3_preparation": None,
                "reused_prepared_working_copy": True,
                "phase_seconds": {},
                "completed_phases": ["mapped_generation"],
            }
            if cancellation_check is not None:
                cancellation_check("mapped generation complete")
        else:
            meshing_parameters = dict(
                target_size=resolved_size,
                strategy=resolved_strategy,
                overrides=working_overrides,
                beam_edges=working_beam_edges,
                beam_offsets=working_offsets,
                member_ids=working_member_ids,
                face_ids=tuple(sorted(working.faces)),
                seeding=working_seeding,
                refinements=working_refinements,
                order=resolved_order,
                certification_mode=resolved_certification,
                change_set=change_set,
                cancellation_check=cancellation_check,
                native_backend=native_backend,
                quad_options=quad_options,
                layout_policy=layout_policy or "existing",
                structural_preparation={
                    "automatic_face_connections": False,
                    "automatic_member_connections": False,
                    "automatic_member_sheet_connections": False,
                    "declare_missing_owners": True,
                },
                mutation_policy="working_copy",
                qualified_s3=(
                    self.shell_formulation_policy.s3 == "e4-pl-s3-v2d"
                    and resolved_order == "linear"
                ),
                structured_options=(
                    # ANYmesher's structured-layout seed solution is
                    # intentionally authoritative for its generated blocks.
                    # A local size zone is a stronger, explicit user request,
                    # so retain the existing topology and let refinement-aware
                    # global seeding solve it.
                    None
                    if resolved_strategy == "native" or working_refinements
                    else structure_options
                ),
                **supported_parameters,
            )
            if automation is None:
                from ..mesh_execution import (
                    ParallelMeshUnsupported,
                    parallel_mesh_api,
                    resolve_policy,
                    resolve_route,
                    run_parallel_hybrid_mesh,
                )

                policy = resolve_policy(execution_policy)
                probed = parallel_mesh_api() if policy.requests_parallel else None
                route = resolve_route(
                    policy,
                    strategy=resolved_strategy,
                    change_set=change_set,
                    quad_options=quad_options,
                    automation=None,
                    strict=resolved_certification == "strict",
                    warm_pool=pool_lease is not None,
                    api_probe=lambda: probed,
                )
                self._last_mesh_execution = {
                    "route": route.route,
                    "workers": route.workers,
                    "reason": route.reason,
                }
                mesh = None
                if route.parallel:
                    try:
                        mesh = run_parallel_hybrid_mesh(
                            working,
                            meshing_parameters,
                            route,
                            probed,
                            pool_lease=pool_lease,
                        )
                    except ParallelMeshUnsupported as error:
                        # Refused before anything was submitted: the ordinary
                        # serial hybrid mesher takes over and the reason is
                        # recorded.  A failure after submission is the
                        # mesher's own typed error and is never retried here.
                        self._last_mesh_execution = {
                            "route": "serial",
                            "workers": None,
                            "reason": f"parallel route refused before dispatch: {error}",
                        }
                        mesh = None
                if mesh is None:
                    mesh = generate_hybrid_mesh(working, **meshing_parameters)
            else:
                self._last_mesh_execution = {
                    "route": "serial",
                    "workers": None,
                    "reason": "automatic mesh recovery runs serially",
                }
                try:
                    from anymesher import generate_automatic_mesh_result
                except ImportError as error:
                    raise ProjectError(
                        "automatic mesh recovery requires an ANYmesher installation "
                        "with the MeshAutomationOptions owner API"
                    ) from error
                mesh = generate_automatic_mesh_result(
                    working,
                    automation=automation,
                    fallback_structured_options=(
                        None if working_refinements else structure_options
                    ),
                    **meshing_parameters,
                ).mesh
        owner_parallel = getattr(mesh, "hybrid_diagnostics", {}).get("parallel")
        if isinstance(owner_parallel, Mapping) and "used" in owner_parallel:
            used = bool(owner_parallel["used"])
            runtime = owner_parallel.get("runtime", {})
            self._last_mesh_execution.update(
                route="parallel" if used else "serial",
                workers=runtime.get("workers") if used else None,
                reason=owner_parallel.get("reason", self._last_mesh_execution.get("reason", "")),
                parallel={key: value for key, value in owner_parallel.items() if key != "runtime"},
            )
        if all(
            hasattr(mesh, name)
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
            mesh.automatic_intersections = sum(
                1
                for item in preparation.connections
                if item.first.startswith("face:")
                and item.second.startswith("face:")
                and not item.reused
            )
            mesh.automatic_beam_connections = max(
                int(getattr(mesh, "automatic_beam_connections", 0)),
                sum(
                    1
                    for item in preparation.connections
                    if (
                        item.first.startswith("member:")
                        or item.second.startswith("member:")
                    )
                    and not item.reused
                )
            )
            remap_mesh_to_source(mesh, closure)
        mesh_preparation = getattr(mesh, "structural_preparation", {})
        preparation_payload = preparation.to_dict()
        if isinstance(mesh_preparation, Mapping) and "qualified_s3" in mesh_preparation:
            preparation_payload["qualified_s3"] = mesh_preparation["qualified_s3"]
        merge_geometry_runtime(mesh, preparation_runtime)
        hybrid_diagnostics = getattr(mesh, "hybrid_diagnostics", {})
        if isinstance(hybrid_diagnostics, Mapping):
            runtime = hybrid_diagnostics.get("runtime")
            if isinstance(runtime, Mapping):
                self._last_mesh_execution["meshing"] = deepcopy(dict(runtime))
            automation_record = hybrid_diagnostics.get("automation")
            if isinstance(automation_record, Mapping):
                preparation_payload["automation"] = dict(automation_record)
            structured_layout = hybrid_diagnostics.get("structured_layout")
            if isinstance(structured_layout, Mapping):
                # Keep the exact detached plan, source-to-working handle map,
                # acceptance decision and regularity metrics beside the
                # structural intersection preparation in the mesh artifact.
                preparation_payload["structured_layout"] = dict(
                    structured_layout
                )
        self._last_mesh_preparation = preparation_payload
        return mesh

    def native_meshing_session(
        self,
        settings: NativeMeshSettings | None = None,
        *,
        max_background_jobs: int = 2,
    ):
        """Create an incremental, stale-safe native meshing session."""

        from ..native_meshing_backend import create_native_meshing_session

        return create_native_meshing_session(
            self,
            settings,
            max_background_jobs=max_background_jobs,
        )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _feature_materialization_problems(self) -> list[str]:
        problems: list[str] = []
        for record in self.geometry.features.records:
            if record.suppressed:
                continue
            if record.state == "invalid":
                problems.append(
                    record.diagnostic
                    or f"feature {record.feature_id} materialization is invalid"
                )
            elif record.materialization_checksum is not None:
                # Validate every checksummed materialization, not only records
                # already labelled ``frozen``.  A live future-feature record
                # can retain an older state label, and direct output edits must
                # still fail before meshing rather than bypassing persistence.
                diagnostic = self.geometry.features.validate_materialization(
                    record, self.geometry
                )
                if diagnostic is not None:
                    problems.append(diagnostic)
        return problems

    def validate(
        self, *, require_loads: bool = True, require_supports: bool = True
    ) -> None:
        """Fail closed on an incomplete model, naming what is missing.

        Not every analysis needs the same things: a free-free modal analysis
        has no supports and no loads by design, so the caller says what this
        run actually requires.
        """

        problems: List[str] = self._feature_materialization_problems()
        problems.extend(self.sheet_join_intent_problems())
        if self.read_only_reason is not None:
            problems.append(
                "compatibility recovery is blocking solve submission: "
                f"{self.read_only_reason}"
            )
        problems.extend(self.resolve_section_assignments(strict=False))

        stores = {
            "vertex": self.geometry.vertices,
            "edge": self.geometry.edges,
            "face": self.geometry.faces,
        }

        def missing(ref: EntityRef) -> bool:
            store = stores.get(ref.kind)
            return store is None or ref.id not in store

        def scope_problem(item) -> str | None:
            region_ref = getattr(item, "region", None) or self.raw_geometry_attachment_region(item)
            if region_ref is None:
                return (
                    f"references missing {item.ref}" if missing(item.ref) and not self.geometry.resolve_ref(item.ref) else None
                )
            try:
                region = self.regions[region_ref.id]
                direct_ref = getattr(item, "ref", None)
                # Preserve the long-standing diagnostic for a legacy direct
                # reference.  Named, feature-output and query regions retain
                # the richer canonical unresolved-region wording below.
                if (
                    direct_ref is not None
                    and
                    region.hidden
                    and isinstance(region.definition, ManualRegion)
                    and region.definition.anchors == (direct_ref,)
                    and missing(direct_ref)
                    and not self.geometry.resolve_ref(direct_ref)
                ):
                    return f"references missing {direct_ref}"
                candidates = ()
                if region.domain is RegionDomain.GEOMETRY and not self.mesh_only:
                    store = stores.get(region.entity_kind, {})
                    candidates = tuple(
                        EntityRef(region.entity_kind, identifier)
                        for identifier in store
                    )
                resolved = self.regions.resolve(
                    region_ref.id,
                    geometry=None if self.mesh_only else self.geometry,
                    candidates=candidates,
                    feature_resolver=(
                        None
                        if self.mesh_only
                        else lambda anchor: self.geometry.features.resolve(
                            anchor, self.geometry
                        )
                    ),
                )
            except (KeyError, TypeError, ValueError) as error:
                return f"uses unresolved region {region_ref.id!r}: {error}"
            if not resolved:
                return f"uses unresolved or empty region {region.name!r}"
            return None

        def coordinate_problem(item) -> str | None:
            identifier = getattr(item, "coordinate_system_id", "global")
            if identifier not in self.coordinate_systems:
                return f"uses coordinate system {identifier!r}, which does not exist"
            return None

        for face_id, section in self.face_sections.items():
            if face_id not in self.geometry.faces:
                problems.append(
                    f"plate assignment references missing face {face_id}"
                )
            if section not in self.plate_sections:
                problems.append(
                    f"face {face_id} uses undefined plate section {section!r}"
                )
        for edge_id, section in self.edge_sections.items():
            if edge_id not in self.geometry.edges:
                problems.append(
                    f"beam assignment references missing edge {edge_id}"
                )
            if section not in self.beam_sections:
                problems.append(
                    f"edge {edge_id} uses undefined beam section {section!r}"
                )

        for label, items in (
            ("support", self.supports),
            ("mass", self.masses),
            ("wet surface", self.wet_surfaces),
            ("connection", self.connections),
            ("imperfection", self.imperfections),
        ):
            for item in items:
                scoped = scope_problem(item)
                if scoped is not None:
                    problems.append(f"{label} {item.name!r} {scoped}")
                coordinates = coordinate_problem(item)
                if coordinates is not None:
                    problems.append(f"{label} {item.name!r} {coordinates}")
        for refinement in self.refinements:
            if refinement.ref is not None:
                scoped = scope_problem(refinement)
                if scoped is not None:
                    problems.append(f"refinement {refinement.name!r} {scoped}")
        for case_name, case in self.load_cases.items():
            for label, loads in (
                ("point load", case.point_loads),
                ("pressure", case.pressures),
                ("line load", case.line_loads),
                ("surface traction", case.surface_tractions),
            ):
                for index, load in enumerate(loads):
                    scoped = scope_problem(load)
                    if scoped is not None:
                        problems.append(
                            f"load case {case_name!r} {label} {index} {scoped}"
                        )
                    coordinates = coordinate_problem(load)
                    if coordinates is not None:
                        problems.append(
                            f"load case {case_name!r} {label} {index} {coordinates}"
                        )
            problems.extend(self._wet_pressure_problems(case_name, case))
            problems.extend(self._envelope_problems(case_name, case))
            for load in case.connection_loads:
                if not any(item.id == load.connection for item in self.connections):
                    problems.append(
                        f"load case {case_name!r} connection load {load.name!r} "
                        f"names a connection that does not exist ({load.connection!r})"
                    )
            if (
                case.gravity is not None
                and case.gravity_coordinate_system_id not in self.coordinate_systems
            ):
                problems.append(
                    f"load case {case_name!r} uses missing gravity coordinate "
                    f"system {case.gravity_coordinate_system_id!r}"
                )
        for name, combination in self.combinations.items():
            unknown = sorted(set(combination.factors) - set(self.load_cases))
            if unknown:
                problems.append(
                    f"combination {name!r} references undefined load case(s) "
                    f"{unknown}"
                )

        for request in self.output_requests.values():
            scoped = scope_problem(request)
            if scoped is not None:
                problems.append(f"output request {request.label!r} {scoped}")
            if (
                request.basis
                not in ("global", "local", "element", "material", "cylindrical")
                and request.basis not in self.coordinate_systems
            ):
                problems.append(
                    f"output request {request.label!r} uses missing result basis "
                    f"{request.basis!r}"
                )

        for analysis in self.analyses.values():
            missing_requests = sorted(
                set(analysis.output_request_ids).difference(self.output_requests)
            )
            if missing_requests:
                problems.append(
                    f"analysis {analysis.name!r} references missing output "
                    f"request(s) {missing_requests}"
                )
            for request_id in analysis.output_request_ids:
                request = self.output_requests.get(request_id)
                if request is None:
                    continue
                for diagnostic in request.problems_for_analysis(analysis.type):
                    problems.append(
                        f"analysis {analysis.name!r} output request "
                        f"{request.label!r}: {diagnostic}"
                    )
            if analysis.output_requests:
                problems.append(
                    f"analysis {analysis.name!r} still contains legacy output "
                    "request data without a complete canonical region; edit it "
                    "into a typed output request before solving"
                )
        if self.element_order not in ELEMENT_ORDERS:
            problems.append(
                f"unknown element order {self.element_order!r}; expected one "
                f"of {', '.join(ELEMENT_ORDERS)}"
            )

        unsectioned = sorted(set(self.geometry.faces) - set(self.face_sections))
        if unsectioned:
            problems.append(
                f"plates without a section: {unsectioned}. Assign a plate "
                "section, or the solver has no thickness to use."
            )

        for name, section in self.plate_sections.items():
            if section.material not in self.materials:
                problems.append(
                    f"plate section {name!r} uses undefined material "
                    f"{section.material!r}"
                )
        for name, beam in self.beam_sections.items():
            if beam.material not in self.materials:
                problems.append(
                    f"beam section {name!r} uses undefined material "
                    f"{beam.material!r}"
                )

        if require_supports and not self.supports:
            problems.append(
                "the model has no supports; a linear static solve would be "
                "singular"
            )

        # A nonzero prescribed translation or rotation is an imposed action:
        # it produces reactions and internal forces through the affine
        # constraint RHS even when the assembled external load vector is zero.
        # Zero-valued supports remain restraints and must not make an unloaded
        # static model appear loaded.
        has_prescribed_motion = any(
            value != 0.0
            for item in self.supports
            for value in item.constraints.values()
        )
        if require_loads and not has_prescribed_motion and (
            not self.load_cases
            or all(case.is_empty() for case in self.load_cases.values())
        ):
            problems.append(
                "the model has no loads or nonzero prescribed displacement/rotation"
            )

        if problems:
            raise ProjectError(
                f"model {self.name!r} is not ready to solve:\n  - "
                + "\n  - ".join(problems)
            )

    def _envelope_problems(self, case_name: str, case: LoadCase) -> List[str]:
        """Static checks on the envelopes of a case's loads; geometry-free."""

        problems: List[str] = []
        if case.parent is not None:
            parent = self.load_cases.get(case.parent)
            if parent is None or parent.times is None:
                problems.append(
                    f"load case {case_name!r} belongs to the load series "
                    f"{case.parent!r}, which no longer exists"
                )
            return problems
        known = set(case.parameters)
        for collection in (
            "point_loads", "pressures", "line_loads", "surface_tractions",
            "wet_pressures", "connection_loads",
        ):
            for load in getattr(case, collection):
                envelope = getattr(load, "envelope", None)
                if envelope is None:
                    continue
                unknown = sorted(envelope.names() - known)
                if unknown:
                    problems.append(
                        f"load case {case_name!r} {getattr(load, 'name', collection)!r}: "
                        f"the envelope reads {unknown}; set them as parameters of the case"
                    )
                if envelope.uses_time and case.times is None and case.time is None:
                    problems.append(
                        f"load case {case_name!r} {getattr(load, 'name', collection)!r}: "
                        "the envelope varies with time but the case has none; make the "
                        "case a load series"
                    )
        return problems

    def _wet_pressure_problems(self, case_name: str, case: LoadCase) -> List[str]:
        """Static checks on a case's wet pressures; geometry-free."""

        # Names an equation reads (parameters, time) are checked when a case is
        # built, not here: a template case may legitimately lack the time and
        # parameters that only its generated series supplies.
        problems: List[str] = []
        if not case.wet_pressures:
            return problems
        for load in case.wet_pressures:
            if load.table is not None:
                problem = self._table_problem(load.table)
                if problem is not None:
                    problems.append(
                        f"load case {case_name!r} wet pressure {load.name!r}: "
                        f"{problem}"
                    )
        known = {surface.id for surface in self.wet_surfaces}
        if not known:
            problems.append(
                f"load case {case_name!r} has wet pressure but the model has "
                "no wet surfaces; mark the wetted plates first"
            )
        for load in case.wet_pressures:
            unknown = sorted(set(load.surfaces) - known)
            if unknown and known:
                problems.append(
                    f"load case {case_name!r} wet pressure {load.name!r} "
                    f"names unknown wet surface(s) {unknown}"
                )
        return problems

    @staticmethod
    def _table_problem(table: PressureTable) -> str | None:
        """Why a pressure table cannot be used, from the file alone."""

        from pathlib import Path

        from anyloads import PressureTableError, read_table

        path = Path(table.path).expanduser()
        if not path.is_file():
            return f"pressure table {table.path} does not exist"
        try:
            read_table(table)
        except PressureTableError as error:
            return str(error)
        return None

    def untrusted_code(self) -> List[tuple[str, str, str]]:
        """``(case, pressure name, code)`` for pressure code not yet trusted.

        Code that came from a project file is inert until the user approves
        it; callers that are about to run a solve ask first.
        """

        from anyloads import is_trusted

        found: List[tuple[str, str, str]] = []
        seen: set[str] = set()
        for case_name, case in self.load_cases.items():
            for load in case.wet_pressures:
                if load.code is not None and load.code not in seen and not is_trusted(load.code):
                    seen.add(load.code)
                    found.append((case_name, load.name, load.code))
        return found

    def add_load_cases_from_table(
        self,
        name: str,
        table: PressureTable,
        *,
        surfaces: Sequence[str] = (),
        scale: float = 1.0,
    ) -> List[LoadCase]:
        """One load case per case identifier in a pressure table.

        A time-keyed table gives cases at those times (the pressure follows
        each case's time, so ``interpolate_time`` tables also work at other
        times); a number- or name-keyed table pins each case to its own rows. A
        table without identifiers is one map and makes one case.
        """

        keys = table.keys()
        if table.key is None:
            keys = [None]
        if not keys:
            raise ProjectError("the pressure table has no rows")
        width = len(str(len(keys) - 1))
        names = [f"{name}_{index:0{width}d}" for index in range(len(keys))]
        clashes = [item for item in names if item in self.load_cases]
        if clashes:
            raise ProjectError(
                f"load case(s) {clashes[:3]} already exist; remove them or "
                "choose another name"
            )
        created: List[LoadCase] = []
        try:
            for index, (case_name, key) in enumerate(zip(names, keys)):
                case = self.load_case(case_name)
                created.append(case)
                case.series = name
                case.step = index
                case.number = None
                if table.key == "time":
                    case.set_time(key)
                    case.add_wet_pressure(
                        table=table, surfaces=surfaces, scale=scale,
                        name=f"{name} table",
                    )
                else:
                    case.add_wet_pressure(
                        table=table,
                        table_case=key,
                        surfaces=surfaces,
                        scale=scale,
                        name=f"{name} table",
                    )
        except BaseException:
            for case in created:
                self.load_cases.pop(case.name, None)
            raise
        return created

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def plate_section_of(self, face_id: int) -> PlateSection:
        try:
            return self.plate_sections[self.face_sections[int(face_id)]]
        except KeyError:
            raise ProjectError(f"face {face_id} has no plate section") from None

    def beam_section_of(self, edge_id: int) -> BeamSection:
        try:
            return self.beam_sections[self.edge_sections[int(edge_id)]]
        except KeyError:
            raise ProjectError(f"edge {edge_id} has no beam section") from None

    def _require_material(self, name: str) -> MaterialSpec:
        try:
            return self.materials[name]
        except KeyError:
            raise ProjectError(f"no material named {name!r}") from None

    def _require_entity(self, ref: EntityRef) -> None:
        self.geometry.entity_ref(ref.kind, ref.id)

    # ------------------------------------------------------------------
    # convenience references
    # ------------------------------------------------------------------
    def face(self, face_id: int) -> EntityRef:
        return self.geometry.entity_ref("face", face_id)

    def edge(self, edge_id: int) -> EntityRef:
        return self.geometry.entity_ref("edge", edge_id)

    def point(self, vertex_id: int) -> EntityRef:
        return self.geometry.entity_ref("vertex", vertex_id)
