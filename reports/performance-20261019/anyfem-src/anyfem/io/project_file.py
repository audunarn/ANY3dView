"""Saving and loading a project.

The file stores the model, not its consequences: geometry, sections, supports,
loads, combinations and imperfections.  The mesh and the results are left out
because they are regenerable, and storing them would make a saved file go
stale the moment the model changed.

**Entity IDs are part of the data.**  Loads, supports and sections reference
geometry by ID, so a load cycle that renumbered anything would silently
re-target them.  The ID counters are saved too, so entities created after a
reload cannot collide with entities that already exist.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from anygeometry.entities import EntityRef
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.serialization import from_dict as geometry_from_dict
from anygeometry.serialization import to_dict as geometry_to_dict

from anyloads import PressureTable
from ..model.attributes import (
    LoadCase,
    Mass,
    Support,
    WetPressure,
    WetSurface,
)
from ..model.connections import CONNECTION_DOF_NAMES, Connection, ConnectionLoad
from ..model.regions import RegionRef
from ..mesh.refinement import Refinement
from ..model.imperfections import Imperfection
from ..model.materials import Material
from ..model.project import Project
from ..model.formulations import ShellFormulationPolicy
from ..model.ownership import SheetJoinIntent
from ..model.sections import BeamSection, SectionAssignment
from ..model.coordinates import CoordinateSystem, GLOBAL_COORDINATES
from ..model.records import (
    AnalysisDefinition,
    ArtifactRef,
    JobRecord,
    MeshRecord,
    OutputRequest,
)
from ..model.regions import RegionRegistry, region_from_dict
from ..model.units import UnitProfile, unit_profile
from ..native_meshing import NativeMeshSettings

__all__ = [
    "FORMAT_VERSION",
    "ProjectFileError",
    "load_project",
    "project_from_dict",
    "project_to_dict",
    "save_project",
]

FORMAT_VERSION = 8
# Wet surfaces, wet pressures and load-case series are written as format 9 and
# only then, so a file that does not use them stays readable by older ANYfem,
# while one that does is refused by a reader that would silently drop its loads.
WET_LOADS_FORMAT = 9
# Connections (a reference point tied to selected geometry, and the loads
# applied there) are format 10, again only when a project uses them.
CONNECTIONS_FORMAT = 10
# Load series definitions and load envelopes (a window and a factor over time)
# are format 11, again only when used.
ENVELOPES_FORMAT = 11
# Submodel definitions and runs are format 12, again only when a project has
# them.
SUBMODELS_FORMAT = 12
MAX_READ_FORMAT = SUBMODELS_FORMAT
SUFFIX = ".anyfem"
_NATIVE_TRIANGULATION_BACKENDS = ("auto", "python", "native")


class ProjectFileError(ValueError):
    """Raised when a project file cannot be read."""


def _native_backend_value(value: Any, context: str) -> str:
    if not isinstance(value, str) or value not in _NATIVE_TRIANGULATION_BACKENDS:
        expected = ", ".join(_NATIVE_TRIANGULATION_BACKENDS)
        raise ProjectFileError(f"{context} must be one of {expected}")
    return value


def _nested_native_backend_path(value: Any, path: str) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key == "native_backend":
                return child
            found = _nested_native_backend_path(item, child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _nested_native_backend_path(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _without_legacy_native_backends(
    value: Any, path: str
) -> tuple[Any, tuple[tuple[str, Any], ...]]:
    """Copy a legacy native-settings tree while extracting every selector."""

    if isinstance(value, Mapping):
        made: dict[Any, Any] = {}
        found: list[tuple[str, Any]] = []
        for key, item in value.items():
            child = f"{path}.{key}"
            if key == "native_backend":
                found.append((child, item))
                continue
            cleaned, nested = _without_legacy_native_backends(item, child)
            made[key] = cleaned
            found.extend(nested)
        return made, tuple(found)
    if isinstance(value, list):
        made_list: list[Any] = []
        found = []
        for index, item in enumerate(value):
            cleaned, nested = _without_legacy_native_backends(
                item, f"{path}[{index}]"
            )
            made_list.append(cleaned)
            found.extend(nested)
        return made_list, tuple(found)
    if isinstance(value, tuple):
        made_tuple: list[Any] = []
        found = []
        for index, item in enumerate(value):
            cleaned, nested = _without_legacy_native_backends(
                item, f"{path}[{index}]"
            )
            made_tuple.append(cleaned)
            found.extend(nested)
        return tuple(made_tuple), tuple(found)
    return value, ()


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def project_to_dict(project: Project) -> Dict[str, Any]:
    """The whole model as plain data."""

    ownership_problems = project.sheet_join_intent_problems()
    if ownership_problems:
        raise ProjectFileError(
            "structural Sheet ownership intent is unresolved:\n  - "
            + "\n  - ".join(ownership_problems)
        )
    # Fold any direct edits through the historical assignment dictionaries
    # into canonical region-backed records before taking the persisted view.
    project.resolve_section_assignments(strict=False)
    project.adopt_geometry_attachment_regions()
    geometry = project.geometry
    native_backend = project.set_native_triangulation_backend(
        project.native_triangulation_backend
    )
    native_settings = (
        None
        if project.native_mesh_settings is None
        else project.native_mesh_settings.to_dict()
    )
    nested_backend = _nested_native_backend_path(
        native_settings, "meshing.native"
    )
    if nested_backend is not None:
        raise ProjectFileError(
            f"{nested_backend} is non-canonical; use meshing.native_backend"
        )
    return {
        **({"geometry_attachment_regions": [
            {"ref": _ref(reference), "region": binding.id}
            for reference, binding in sorted(project.geometry_attachment_regions.items(), key=lambda item: (item[0].kind, item[0].id))
        ]} if project.geometry_attachment_regions else {}),
        "anyfem": {
            "schema": "anyfem.project",
            "format": (
                SUBMODELS_FORMAT
                if project.submodels or project.submodel_runs
                else ENVELOPES_FORMAT
                if _uses_envelopes(project)
                else CONNECTIONS_FORMAT
                if _uses_connections(project)
                else WET_LOADS_FORMAT
                if _uses_wet_loads(project)
                else FORMAT_VERSION
            ),
            "document_id": project.document_id,
            "artifact_root": f"{project.name}.anyfem-data",
        },
        "name": project.name,
        # Imported solver models are deliberately mesh-native.  Keeping this
        # state explicit prevents an empty geometry model from being mistaken
        # for an unfinished modelled project when the document is reopened.
        "mesh_only": bool(project.mesh_only),
        "imported_format": project.imported_format,
        "imported_semantics_artifact_id": (
            project.imported_semantics_artifact_id
        ),
        "compatibility": {
            "archived_feature_histories": deepcopy(
                project.archived_feature_histories
            ),
            "diagnostics": list(project.compatibility_diagnostics),
            "geometry_editing_disabled_reason": (
                project.geometry_editing_disabled_reason
            ),
            "read_only_reason": project.read_only_reason,
        },
        "units": project.units.to_dict(),
        "shell_formulations": project.shell_formulation_policy.to_dict(),
        "coordinate_systems": [
            system.to_dict()
            for system in sorted(
                project.coordinate_systems.values(), key=lambda item: item.id
            )
        ],
        "regions": project.regions.to_list(),
        "output_requests": [
            item.to_dict()
            for item in sorted(
                project.output_requests.values(), key=lambda value: value.id
            )
        ],
        "geometry": geometry_to_dict(geometry),
        "materials": [
            {**material.to_dict(), "id": project.material_ids.get(material.name)}
            for material in _by_name(project.materials)
        ],
        "plate_sections": [
            {
                "id": section.id,
                "name": section.name,
                "thickness": section.thickness,
                "material": section.material,
            }
            for section in _by_name(project.plate_sections)
        ],
        "beam_sections": [
            {
                "id": section.id,
                "name": section.name,
                "profile": section.profile,
                "material": section.material,
                "web_height": section.web_height,
                "web_thickness": section.web_thickness,
                "flange_width": section.flange_width,
                "flange_thickness": section.flange_thickness,
                "web_direction": (
                    None
                    if section.web_direction is None
                    else _vector(section.web_direction)
                ),
                "eccentricity": section.eccentricity,
                "offset_mode": section.offset_mode,
                "attachment_side": section.attachment_side,
                "rotation_deg": section.rotation_deg,
            }
            for section in _by_name(project.beam_sections)
        ],
        "face_sections": {str(k): v for k, v in sorted(project.face_sections.items())},
        "edge_sections": {str(k): v for k, v in sorted(project.edge_sections.items())},
        "assignment_ids": {
            "faces": {
                str(key): value
                for key, value in sorted(project.face_assignment_ids.items())
            },
            "edges": {
                str(key): value
                for key, value in sorted(project.edge_assignment_ids.items())
            },
        },
        "assignments": {
            "sections": [
                item.to_dict()
                for item in sorted(
                    project.section_assignments.values(),
                    key=lambda value: value.id,
                )
            ]
        },
        "ownership": {
            "sheet_joins": [
                item.to_dict()
                for item in sorted(
                    project.sheet_join_intents.values(),
                    key=lambda value: value.id,
                )
            ]
        },
        "supports": [
            {
                "id": support.id,
                "name": support.name,
                "ref": _ref(support.ref),
                "region": _region_ref(support.region),
                "coordinate_system_id": support.coordinate_system_id,
                "constraints": {
                    key: float(value) for key, value in support.constraints.items()
                },
            }
            for support in project.supports
        ],
        "masses": [
            {
                "id": mass.id,
                "name": mass.name,
                "ref": _ref(mass.ref),
                "value": mass.value,
                "region": _region_ref(mass.region),
                "distribution_policy": mass.distribution_policy,
            }
            for mass in project.masses
        ],
        **(
            {
                "wet_surfaces": [
                    _wet_surface_to_dict(surface) for surface in project.wet_surfaces
                ]
            }
            if project.wet_surfaces
            else {}
        ),
        **(
            {
                "connections": [
                    _connection_to_dict(connection)
                    for connection in project.connections
                ]
            }
            if project.connections
            else {}
        ),
        "load_cases": [
            _load_case_to_dict(case, numbered=_uses_wet_loads(project))
            for case in _by_name(project.load_cases)
        ],
        "combinations": [
            {"id": item.id, "name": item.name, "factors": dict(item.factors)}
            for item in _by_name(project.combinations)
        ],
        "imperfections": [
            {
                "id": item.id,
                "name": item.name,
                "ref": _ref(item.ref),
                "kind": item.kind,
                "amplitude": item.amplitude,
                "direction": _vector(item.direction),
                "waves": [int(item.waves[0]), int(item.waves[1])],
                "axes": [int(item.axes[0]), int(item.axes[1])],
            }
            for item in project.imperfections
        ],
        # Meshing controls. They do not change the model, only the mesh made
        # from it -- but a project reopened without them meshes differently
        # from the one that was saved, which is not something a file format
        # should let happen quietly.
        "meshing": {
            "element_order": project.element_order,
            "target_size": project.target_size,
            "native_backend": native_backend,
            "native": native_settings,
            "seeding_overrides": {
                str(key): int(value)
                for key, value in sorted(project.seeding_overrides.items())
            },
            "refinements": [
                {
                    "name": item.name,
                    "size": float(item.size),
                    "radius": float(item.radius),
                    "growth": float(item.growth),
                    "ref": None if item.ref is None else _ref(item.ref),
                    "center": (
                        None if item.center is None else _vector(item.center)
                    ),
                }
                for item in project.refinements
            ],
        },
        "analyses": [
            item.to_dict()
            for item in sorted(project.analyses.values(), key=lambda value: value.id)
        ],
        "mesh_records": [
            item.to_dict()
            for item in sorted(project.mesh_records.values(), key=lambda value: value.id)
        ],
        "jobs": [
            item.to_dict()
            for item in sorted(project.jobs.values(), key=lambda value: value.id)
        ],
        "artifacts": [
            item.to_dict()
            for item in sorted(project.artifacts.values(), key=lambda value: value.id)
        ],
        **({
            "submodels": [
                item.to_dict()
                for item in sorted(project.submodels.values(), key=lambda value: value.id)
            ],
            "submodel_runs": [
                item.to_dict()
                for item in sorted(project.submodel_runs.values(), key=lambda value: value.id)
            ],
        } if project.submodels or project.submodel_runs else {}),
    }


def save_project(project: Project, path: str | Path) -> Path:
    """Atomically write a project, preserving the prior valid file."""

    destination = Path(path)
    if not destination.suffix:
        destination = destination.with_suffix(SUFFIX)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = project_to_dict(project)
    document["anyfem"]["artifact_root"] = f"{destination.name}-data"
    payload = json.dumps(document, indent=2, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Prove the complete bytes are readable before replacing a user's
        # last valid project.
        project_from_dict(json.loads(temporary.read_text(encoding="utf-8")))
        if destination.exists():
            backup = destination.with_suffix(destination.suffix + ".bak")
            backup.write_bytes(destination.read_bytes())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
def project_from_dict(data: Mapping[str, Any]) -> Project:
    """Rebuild a project and report malformed serialized data consistently."""

    if not isinstance(data, Mapping):
        raise ProjectFileError("an ANYfem project must be a JSON object")
    try:
        return _project_from_dict(data)
    except ProjectFileError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        OverflowError,
    ) as error:
        detail = str(error) or type(error).__name__
        raise ProjectFileError(f"invalid ANYfem project data: {detail}") from None


def _project_from_dict(data: Mapping[str, Any]) -> Project:
    """Rebuild a project from plain data, IDs and all."""

    header = data.get("anyfem")
    if not isinstance(header, Mapping) or "format" not in header:
        raise ProjectFileError(
            "this does not look like an ANYfem project file: no format header"
        )
    version = int(header["format"])
    if version < 1:
        raise ProjectFileError(f"unsupported ANYfem project format {version}")
    if version > MAX_READ_FORMAT:
        raise ProjectFileError(
            f"the file is format {version} but this ANYfem reads up to "
            f"{MAX_READ_FORMAT}; upgrade ANYfem to open it"
        )

    document_id = str(header.get("document_id", ""))
    if not document_id:
        # Legacy files had no document identity.  Derive it from their exact
        # persisted semantics so repeated v1-v3 migrations produce identical
        # hidden regions and assignment UUIDs instead of fresh random IDs.
        legacy_payload = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        document_id = str(
            uuid5(NAMESPACE_URL, f"anyfem:legacy-document:{legacy_payload}")
        )
    if version >= 4:
        mesh_only_value = data.get("mesh_only", False)
        if not isinstance(mesh_only_value, bool):
            raise ProjectFileError("mesh_only must be true or false")
        mesh_only = mesh_only_value
        imported_format = _optional_text(
            data.get("imported_format"), "imported_format"
        )
        imported_artifact_id = _optional_text(
            data.get("imported_semantics_artifact_id"),
            "imported_semantics_artifact_id",
        )
    else:
        # Versions 1--3 predate imported-document persistence.  Their empty
        # geometry still means an ordinary model, exactly as it did then.
        mesh_only = False
        imported_format = None
        imported_artifact_id = None
    compatibility_data = data.get("compatibility", {})
    if not isinstance(compatibility_data, Mapping):
        raise ProjectFileError("compatibility must be a JSON object")
    archived_histories = compatibility_data.get(
        "archived_feature_histories", ()
    )
    compatibility_diagnostics = compatibility_data.get("diagnostics", ())
    if not isinstance(archived_histories, (list, tuple)) or not all(
        isinstance(item, Mapping) for item in archived_histories
    ):
        raise ProjectFileError(
            "compatibility.archived_feature_histories must be a list of objects"
        )
    if not isinstance(compatibility_diagnostics, (list, tuple)) or not all(
        isinstance(item, str) for item in compatibility_diagnostics
    ):
        raise ProjectFileError("compatibility.diagnostics must be a list of strings")
    geometry_editing_disabled_reason = _optional_text(
        compatibility_data.get("geometry_editing_disabled_reason"),
        "compatibility.geometry_editing_disabled_reason",
    )
    read_only_reason = _optional_text(
        compatibility_data.get("read_only_reason"),
        "compatibility.read_only_reason",
    )
    units_data = data.get("units")
    units = (
        UnitProfile.from_dict(units_data)
        if isinstance(units_data, Mapping)
        else unit_profile()
    )
    shell_formulations_data = data.get("shell_formulations")
    if shell_formulations_data is None:
        if version >= 8:
            raise ProjectFileError("format 8 shell_formulations is required")
        shell_formulation_policy = ShellFormulationPolicy.legacy_compatible()
        compatibility_diagnostics = list(compatibility_diagnostics)
        compatibility_diagnostics.append(
            "project predates persisted shell formulation policy; Q4 and TRI3 "
            "remain explicit legacy formulations and are not upgraded implicitly"
        )
    elif isinstance(shell_formulations_data, Mapping):
        try:
            shell_formulation_policy = ShellFormulationPolicy.from_dict(
                shell_formulations_data
            )
        except ValueError as error:
            raise ProjectFileError(f"shell_formulations: {error}") from None
    else:
        raise ProjectFileError("shell_formulations must be an object")

    geometry_data = data.get("geometry", {})
    if not isinstance(geometry_data, Mapping):
        raise ProjectFileError("geometry must be a JSON object")
    if geometry_data.get("schema") == "anygeometry":
        try:
            geometry = geometry_from_dict(geometry_data)
        except GeometryError as error:
            raise ProjectFileError(f"geometry: {error}") from None
        project = Project(
            name=str(data.get("name", "model")),
            geometry=geometry,
            units=units,
            shell_formulation_policy=shell_formulation_policy,
            mesh_only=mesh_only,
            imported_format=imported_format,
            imported_semantics_artifact_id=imported_artifact_id,
            archived_feature_histories=[
                deepcopy(dict(item)) for item in archived_histories
            ],
            compatibility_diagnostics=list(compatibility_diagnostics),
            geometry_editing_disabled_reason=geometry_editing_disabled_reason,
            read_only_reason=read_only_reason,
            **({"document_id": document_id} if document_id else {}),
        )
    else:
        # Formats 1 and 2 embedded the original ANYmesher-era topology schema.
        # Keep reading those files, then write the owner codec on the next save.
        project = Project(
            name=str(data.get("name", "model")),
            units=units,
            shell_formulation_policy=shell_formulation_policy,
            mesh_only=mesh_only,
            imported_format=imported_format,
            imported_semantics_artifact_id=imported_artifact_id,
            archived_feature_histories=[
                deepcopy(dict(item)) for item in archived_histories
            ],
            compatibility_diagnostics=list(compatibility_diagnostics),
            geometry_editing_disabled_reason=geometry_editing_disabled_reason,
            read_only_reason=read_only_reason,
            **({"document_id": document_id} if document_id else {}),
        )
        _geometry_from_dict(project.geometry, geometry_data)

    _recover_detached_feature_corruption(
        project, geometry_data, source_format=version
    )

    ownership_data = data.get("ownership")
    if version < 7:
        if ownership_data is not None:
            raise ProjectFileError(
                "ownership is only valid in project format 7 or newer"
            )
        # Infer once, after legacy feature recovery has established the final
        # persistent output identities.  Constructor-time inference could
        # otherwise retain FeatureOutputRefs into an archived corrupt history.
        project.sheet_join_intents.clear()
        project.infer_existing_sheet_join_intents()
    else:
        if ownership_data is None:
            raise ProjectFileError("format 7 ownership is required")
        if not isinstance(ownership_data, Mapping):
            raise ProjectFileError("ownership must be a JSON object")
        if "sheet_joins" not in ownership_data:
            raise ProjectFileError("format 7 ownership.sheet_joins is required")
        joins_data = ownership_data["sheet_joins"]
        if not isinstance(joins_data, (list, tuple)):
            raise ProjectFileError("ownership.sheet_joins must be a list")
        # The v7 registry is authoritative, including an explicit empty list.
        project.sheet_join_intents.clear()
        for index, entry in enumerate(joins_data):
            try:
                intent = SheetJoinIntent.from_dict(entry)
                project.add_sheet_join_intent(intent)
            except (TypeError, ValueError) as error:
                raise ProjectFileError(
                    f"ownership.sheet_joins[{index}]: {error}"
                ) from None
        try:
            project.reapply_sheet_join_intents()
        except (GeometryError, ValueError) as error:
            raise ProjectFileError(
                "ownership.sheet_joins cannot be resolved exactly: "
                f"{error}"
            ) from None

    coordinate_data = data.get("coordinate_systems", ())
    if coordinate_data:
        if not isinstance(coordinate_data, list):
            raise ProjectFileError("coordinate_systems must be a list")
        project.coordinate_systems.clear()
        for entry in coordinate_data:
            system = CoordinateSystem.from_dict(entry)
            if system.id in project.coordinate_systems:
                raise ProjectFileError(
                    f"duplicate coordinate-system ID {system.id!r}"
                )
            project.coordinate_systems[system.id] = system
        project.coordinate_systems.setdefault("global", GLOBAL_COORDINATES)

    regions_data = data.get("regions", ())
    if regions_data:
        if not isinstance(regions_data, list):
            raise ProjectFileError("regions must be a list")
        project.regions = RegionRegistry(region_from_dict(entry) for entry in regions_data)

    output_request_data = data.get("output_requests", ())
    attachment_data = data.get("geometry_attachment_regions", [])
    if not isinstance(attachment_data, list):
        raise ProjectFileError("geometry_attachment_regions must be a list")
    for index, entry in enumerate(attachment_data):
        context = f"geometry_attachment_regions[{index}]"
        try:
            from anygeometry.features import FeatureOutputRef
            from ..model.regions import ManualRegion, RegionDomain
            reference = _ref_from(entry["ref"])
            binding = _region_ref_from(project, entry["region"], context + ".region")
            if binding is None or reference in project.geometry_attachment_regions:
                raise ValueError("missing or duplicate binding")
            region = project.regions[binding.id]
            anchors = region.definition.anchors if isinstance(region.definition, ManualRegion) else ()
            if (region.domain is not RegionDomain.GEOMETRY or region.entity_kind != reference.kind
                    or len(anchors) != 1 or not isinstance(anchors[0], FeatureOutputRef)
                    or anchors[0].kind != reference.kind):
                raise ValueError("binding must name one same-kind feature output")
            project.geometry.features.get(anchors[0].feature_id)
            project.geometry_attachment_regions[reference] = binding
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectFileError(f"{context}: {error}") from None
    if output_request_data:
        if not isinstance(output_request_data, list):
            raise ProjectFileError("output_requests must be a list")
        for index, entry in enumerate(output_request_data):
            if not isinstance(entry, Mapping):
                raise ProjectFileError(
                    f"output_requests[{index}] must be a JSON object"
                )
            try:
                project.add_output_request(OutputRequest.from_dict(entry))
            except (TypeError, ValueError) as error:
                raise ProjectFileError(
                    f"output_requests[{index}]: {error}"
                ) from None

    for entry in data.get("materials", ()):
        if "constants" in entry or "symmetry" in entry:
            material = Material.from_dict(entry)
        else:
            # Version 1 stored isotropic constants at the top level and a DNV
            # descriptor as a three-item JSON list.
            material = Material(
                name=str(entry["name"]),
                elastic_modulus=float(entry["elastic_modulus"]),
                poisson_ratio=float(entry["poisson_ratio"]),
                density=float(entry.get("density", 0.0)),
                yield_stress=float(entry.get("yield_stress", 0.0)),
                hardening=entry.get("hardening"),
            )
        project.add_material(material)
        if entry.get("id"):
            project.material_ids[material.name] = str(entry["id"])
        else:
            project.material_ids[material.name] = str(
                uuid5(NAMESPACE_URL, f"{project.document_id}:material:{material.name}")
            )
    for entry in data.get("plate_sections", ()):
        project.add_plate_section(
            name=str(entry["name"]),
            thickness=float(entry["thickness"]),
            material=str(entry["material"]),
            id=None if entry.get("id") is None else str(entry["id"]),
        )
    for entry in data.get("beam_sections", ()):
        project.add_beam_section(
            BeamSection(
                id=str(entry.get("id")) if entry.get("id") else str(uuid4()),
                name=str(entry["name"]),
                profile=str(entry["profile"]),
                material=str(entry["material"]),
                web_height=float(entry.get("web_height", 0.0)),
                web_thickness=float(entry.get("web_thickness", 0.0)),
                flange_width=float(entry.get("flange_width", 0.0)),
                flange_thickness=float(entry.get("flange_thickness", 0.0)),
                web_direction=entry.get("web_direction"),
                eccentricity=float(entry.get("eccentricity", 0.0)),
                offset_mode=str(entry.get("offset_mode", "manual")),
                attachment_side=str(entry.get("attachment_side", "front")),
                rotation_deg=float(entry.get("rotation_deg", 0.0)),
            )
        )

    for face_id, section in data.get("face_sections", {}).items():
        _restore_section_assignment(
            project, "face", face_id, section, "face_sections"
        )
    for edge_id, section in data.get("edge_sections", {}).items():
        _restore_section_assignment(
            project, "edge", edge_id, section, "edge_sections"
        )
    assignment_ids = data.get("assignment_ids", {})
    if isinstance(assignment_ids, Mapping):
        faces = assignment_ids.get("faces", {})
        edges = assignment_ids.get("edges", {})
        if isinstance(faces, Mapping):
            for key, value in faces.items():
                identifier = int(key)
                if identifier in project.face_sections:
                    project.face_assignment_ids[identifier] = str(value)
        if isinstance(edges, Mapping):
            for key, value in edges.items():
                identifier = int(key)
                if identifier in project.edge_sections:
                    project.edge_assignment_ids[identifier] = str(value)
    for identifier in project.face_sections:
        project.face_assignment_ids.setdefault(
            identifier,
            str(uuid5(NAMESPACE_URL, f"{project.document_id}:face-assignment:{identifier}")),
        )
    for identifier in project.edge_sections:
        project.edge_assignment_ids.setdefault(
            identifier,
            str(uuid5(NAMESPACE_URL, f"{project.document_id}:edge-assignment:{identifier}")),
        )

    assignments_data = data.get("assignments", {})
    if assignments_data is None:
        assignments_data = {}
    if not isinstance(assignments_data, Mapping):
        raise ProjectFileError("assignments must be a JSON object")
    section_assignment_data = assignments_data.get(
        "sections", data.get("section_assignments", ())
    )
    if not isinstance(section_assignment_data, list) and not isinstance(
        section_assignment_data, tuple
    ):
        raise ProjectFileError("assignments.sections must be a list")
    if section_assignment_data:
        for index, entry in enumerate(section_assignment_data):
            if not isinstance(entry, Mapping):
                raise ProjectFileError(
                    f"assignments.sections[{index}] must be a JSON object"
                )
            assignment = SectionAssignment.from_dict(entry)
            if assignment.id in project.section_assignments:
                raise ProjectFileError(
                    f"duplicate section assignment ID {assignment.id!r}"
                )
            try:
                project._require_section_id(  # type: ignore[attr-defined]
                    assignment.kind, assignment.section_id
                )
                project._require_assignment_region(  # type: ignore[attr-defined]
                    assignment
                )
            except ValueError as error:
                raise ProjectFileError(
                    f"assignments.sections[{index}]: {error}"
                ) from None
            project.section_assignments[assignment.id] = assignment
        # Canonical records win over their redundant compatibility cache.  An
        # unresolved feature output remains represented and diagnostic; it is
        # never replaced with the old numeric target.
        project.resolve_section_assignments(strict=False, adopt_legacy=False)
    elif not project.mesh_only:
        # Sequential v1-v3 migration (and early v4 files): preserve established
        # IDs when present, otherwise derive deterministic UUIDs and singleton
        # regions from the document identity and topology reference.
        project.resolve_section_assignments(strict=False, adopt_legacy=True)

    for entry in data.get("supports", ()):
        support = Support(
            id=str(entry.get("id")) if entry.get("id") else str(uuid4()),
            name=entry["name"],
            ref=_existing_ref(project, entry["ref"], "support.ref", region=entry.get("region")),
            constraints={
                key: float(value)
                for key, value in entry["constraints"].items()
            },
            region=_region_ref_from(project, entry.get("region"), "support.region"),
            coordinate_system_id=str(entry.get("coordinate_system_id", "global")),
        )
        if project.mesh_only or not project.geometry.resolve_ref(support.ref):
            project.supports.append(support)
        else:
            project.add_support(support)
    for entry in data.get("masses", ()):
        mass = Mass(
            id=str(entry.get("id")) if entry.get("id") else str(uuid4()),
            ref=_existing_ref(project, entry["ref"], "mass.ref", region=entry.get("region")),
            value=float(entry["value"]),
            name=entry.get("name", "mass"),
            region=_region_ref_from(project, entry.get("region"), "mass.region"),
            distribution_policy=str(entry.get("distribution_policy", "total_distributed")),
        )
        if project.mesh_only or not project.geometry.resolve_ref(mass.ref):
            project.masses.append(mass)
        else:
            project.add_mass(mass)

    for entry in data.get("connections", ()):
        try:
            connection = Connection(
                id=str(entry.get("id")) if entry.get("id") else str(uuid4()),
                name=str(entry.get("name", "connection")),
                ref=_existing_ref(
                    project, entry["ref"], "connection.ref", region=entry.get("region")
                ),
                region=_region_ref_from(
                    project, entry.get("region"), "connection.region"
                ),
                kind=str(entry.get("kind", "rigid")),
                dofs=tuple(entry.get("dofs", CONNECTION_DOF_NAMES)),
                reference=str(entry.get("reference", "centroid")),
                reference_point=entry.get("reference_point"),
                weighting=str(entry.get("weighting", "tributary")),
                stiffness=entry.get("stiffness"),
                tie=str(entry.get("tie", "rigid")),
            )
        except ValueError as error:
            raise ProjectFileError(f"connections: {error}") from None
        if project.mesh_only or not project.geometry.resolve_ref(connection.ref):
            project.connections.append(connection)
        else:
            project.add_connection(connection)

    for entry in data.get("wet_surfaces", ()):
        surface = WetSurface(
            id=str(entry.get("id")) if entry.get("id") else str(uuid4()),
            name=str(entry.get("name", "wet surface")),
            ref=_existing_ref(
                project, entry["ref"], "wet_surface.ref", region=entry.get("region")
            ),
            region=_region_ref_from(
                project, entry.get("region"), "wet_surface.region"
            ),
            fluid_side=str(entry.get("fluid_side", "positive_normal")),
            interior_point=entry.get("interior_point"),
        )
        if project.mesh_only or not project.geometry.resolve_ref(surface.ref):
            project.wet_surfaces.append(surface)
        else:
            project.add_wet_surface(surface)

    for entry in data.get("load_cases", ()):
        # The steps of a load series are derived from its definition's times;
        # the entries written for them only keep other readers and hashes whole.
        if entry.get("parent") is None:
            _load_case_from_dict(project, entry)
    for case in [item for item in project.load_cases.values() if item.times is not None]:
        try:
            project.expand_load_series(case.name)
        except ProjectError as error:
            raise ProjectFileError(f"load series {case.name!r}: {error}") from None
    for entry in data.get("combinations", ()):
        combination = project.add_combination(
            name=str(entry["name"]),
            factors={str(k): float(v) for k, v in entry["factors"].items()},
        )
        if entry.get("id"):
            project.combinations[combination.name] = replace(
                combination, id=str(entry["id"])
            )
    for index, entry in enumerate(data.get("imperfections", ())):
        imperfection = Imperfection(
            ref=_existing_ref(
                project, entry["ref"], "imperfection.ref", raw_attachment=True
            ),
            kind=entry.get("kind", "auto"),
            amplitude=entry.get("amplitude"),
            direction=tuple(entry.get("direction", (0.0, 0.0, 1.0))),
            waves=tuple(entry.get("waves", (1, 1))),
            axes=tuple(entry.get("axes", (0, 1))),
            name=entry.get("name", "imperfection"),
            id=str(
                entry.get("id")
                or uuid5(
                    NAMESPACE_URL,
                    f"{project.document_id}:imperfection:{index}",
                )
            ),
        )
        if project.mesh_only or imperfection.ref in project.geometry_attachment_regions:
            project.imperfections.append(imperfection)
        else:
            project.add_imperfection(imperfection)

    meshing = data.get("meshing")
    if version >= 6 and not isinstance(meshing, Mapping):
        raise ProjectFileError(f"format {version} meshing must be an object")
    native_backend = "python"
    if isinstance(meshing, Mapping):
        if version >= 6:
            if "native_backend" not in meshing:
                raise ProjectFileError(
                    "format 6 meshing.native_backend is required"
                )
            native_backend = _native_backend_value(
                meshing["native_backend"], "meshing.native_backend"
            )
        elif "native_backend" in meshing:
            raise ProjectFileError(
                "meshing.native_backend is only valid in format 6 or newer"
            )
        # Absent in files written before meshing controls existed, which is
        # what the defaults are for.
        project.set_element_order(str(meshing.get("element_order", "linear")))
        target_size = meshing.get("target_size")
        project.target_size = None if target_size is None else float(target_size)
        overrides = meshing.get("seeding_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ProjectFileError("meshing.seeding_overrides must be an object")
        project.seeding_overrides = {
            int(key): int(value) for key, value in overrides.items()
        }
        native_settings = meshing.get("native")
        if native_settings is not None:
            if not isinstance(native_settings, Mapping):
                raise ProjectFileError("meshing.native must be an object or null")
            nested_backend = _nested_native_backend_path(
                native_settings, "meshing.native"
            )
            if version >= 6 and nested_backend is not None:
                raise ProjectFileError(
                    f"{nested_backend} is non-canonical in format 6; "
                    "use meshing.native_backend only"
                )
            native_settings = dict(native_settings)
            if version < 6:
                native_settings, legacy_backends = _without_legacy_native_backends(
                    native_settings, "meshing.native"
                )
                decoded_backends = tuple(
                    (path, _native_backend_value(value, path))
                    for path, value in legacy_backends
                )
                distinct = {value for _path, value in decoded_backends}
                if len(distinct) > 1:
                    detail = ", ".join(
                        f"{path}={value!r}" for path, value in decoded_backends
                    )
                    raise ProjectFileError(
                        f"conflicting legacy native_backend values: {detail}"
                    )
                if decoded_backends:
                    native_backend = decoded_backends[0][1]
            project.set_native_mesh_settings(
                NativeMeshSettings.from_dict(native_settings)
            )
        for entry in meshing.get("refinements", ()):
            center = entry.get("center")
            reference = entry.get("ref")
            refinement = Refinement(
                size=float(entry["size"]),
                radius=float(entry.get("radius", 0.0)),
                growth=float(entry.get("growth", 1.5)),
                ref=(
                    None
                    if reference is None
                    else _existing_ref(project, reference, "refinement.ref", raw_attachment=True)
                ),
                center=None if center is None else tuple(center),
                name=entry.get("name", "refinement"),
            )
            if project.mesh_only or refinement.ref in project.geometry_attachment_regions:
                project.refinements.append(refinement)
            else:
                project.add_refinement(refinement)
    project.set_native_triangulation_backend(native_backend)

    for index, entry in enumerate(data.get("analyses", ())):
        if not isinstance(entry, Mapping):
            raise ProjectFileError(f"analyses[{index}] must be a JSON object")
        analysis = AnalysisDefinition.from_dict(entry)
        analysis = _migrate_legacy_output_requests(project, analysis)
        project.add_analysis(analysis)
    for entry in data.get("mesh_records", ()):
        project.add_mesh_record(MeshRecord.from_dict(entry))
    for entry in data.get("jobs", ()):
        project.add_job(JobRecord.from_dict(entry))
    for entry in data.get("artifacts", ()):
        project.add_artifact(ArtifactRef.from_dict(entry))
    if (
        project.imported_semantics_artifact_id is not None
        and project.imported_semantics_artifact_id not in project.artifacts
    ):
        raise ProjectFileError(
            "imported_semantics_artifact_id names no project artifact: "
            f"{project.imported_semantics_artifact_id!r}"
        )
    if project.imported_semantics_artifact_id is not None:
        imported_artifact = project.artifacts[
            project.imported_semantics_artifact_id
        ]
        if imported_artifact.kind != "mesh":
            raise ProjectFileError(
                "imported_semantics_artifact_id must name a mesh artifact"
            )
    _read_submodels(project, data, version)
    _finalize_compatibility_recovery(project)
    return project


def _read_submodels(project: Project, data: Mapping[str, Any], version: int) -> None:
    """Submodel definitions and runs (format 12).

    A structurally valid run is kept even when its definition or an artifact it
    names is gone: its manifest stays readable and staleness reports it.
    """

    from ..model.submodels import SubmodelDefinition, SubmodelRecordError, SubmodelRun

    if version < SUBMODELS_FORMAT:
        if "submodels" in data or "submodel_runs" in data:
            raise ProjectFileError(
                f"format {version} files cannot hold submodels (format {SUBMODELS_FORMAT})"
            )
        return
    for key, record_type, registry in (
        ("submodels", SubmodelDefinition, project.submodels),
        ("submodel_runs", SubmodelRun, project.submodel_runs),
    ):
        entries = data.get(key, [])
        if not isinstance(entries, list):
            raise ProjectFileError(f"{key} must be a list")
        for index, entry in enumerate(entries):
            try:
                record = record_type.from_dict(entry, f"{key}[{index}]")
            except (SubmodelRecordError, ProjectFileError) as error:
                raise ProjectFileError(f"{key}[{index}]: {error}") from None
            if record.id in registry:
                raise ProjectFileError(f"{key}[{index}]: duplicate id {record.id!r}")
            registry[record.id] = record


def _recover_detached_feature_corruption(
    project: Project,
    geometry_data: Mapping[str, Any],
    *,
    source_format: int,
) -> None:
    """Freeze the exact last-good topology of affected v1--v6 documents.

    Older ANYfem commands appended a feature and then modified the detached
    ``FeatureRecord`` returned by ANYgeometry.  The live history therefore
    retained a known executable feature in ``pending``/``active`` state with
    no outputs even though valid topology had already been created.  Replaying
    that history destroys design identity.  Recovery archives the records and
    binds every *exact active* entity to one checksummed unknown frozen feature;
    it never follows replacement lineage or searches by geometry.
    """

    if source_format > 6:
        return
    from anygeometry.features import builtin_feature_registry

    registry = builtin_feature_registry()
    corrupt = tuple(
        record
        for record in project.geometry.features.records
        if (
            not record.suppressed
            and registry.has(record.kind)
            and not record.outputs
            and str(record.state) in {"pending", "active", "ok"}
        )
    )
    if not corrupt:
        return

    raw_history = geometry_data.get("features", {})
    project.archived_feature_histories.append(
        {
            "source_format": int(source_format),
            "reason": "detached FeatureRecord outputs were never committed",
            "history": deepcopy(
                dict(raw_history) if isinstance(raw_history, Mapping) else {}
            ),
        }
    )
    history = project.geometry.features
    prior_next_id = history.next_id
    history.restore({"baseline": None, "records": (), "next_id": prior_next_id})
    outputs = {
        f"{kind}/{identifier}": EntityRef(kind, identifier)
        for kind, store in (
            ("vertex", project.geometry.vertices),
            ("edge", project.geometry.edges),
            ("face", project.geometry.faces),
        )
        for identifier in sorted(store)
    }
    reason = (
        "Recovered a legacy detached-feature history as exact frozen Base "
        "Geometry. Geometry feature editing is disabled; assignments, meshing "
        "and solving remain available after target validation."
    )
    project.geometry_editing_disabled_reason = reason
    project.compatibility_diagnostics.append(reason)
    if not outputs:
        project.read_only_reason = (
            "legacy feature recovery found no last-good topology to validate"
        )
        project.compatibility_diagnostics.append(project.read_only_reason)
        return
    try:
        history.adopt_frozen(
            project.geometry,
            kind="anyfem.frozen.base_geometry",
            name="Recovered Base Geometry",
            parameters={
                "source_format": int(source_format),
                "archived_feature_count": len(corrupt),
            },
            outputs=outputs,
            diagnostic=reason,
        )
    except (GeometryError, TypeError, ValueError) as error:
        project.read_only_reason = (
            "legacy feature recovery could not validate the exact topology: "
            f"{error}"
        )
        project.compatibility_diagnostics.append(project.read_only_reason)


def _finalize_compatibility_recovery(project: Project) -> None:
    """Fail closed when a recovered frozen topology has unresolved targets."""

    if project.geometry_editing_disabled_reason is None:
        return
    problems = list(project.geometry.validate_topology())
    problems.extend(project._feature_materialization_problems())  # type: ignore[attr-defined]
    referenced_regions: set[str] = {
        assignment.region.id for assignment in project.section_assignments.values()
    }
    scoped_items = [
        *project.supports,
        *project.masses,
        *project.wet_surfaces,
        *project.connections,
        *project.output_requests.values(),
    ]
    for case in project.load_cases.values():
        scoped_items.extend(case.point_loads)
        scoped_items.extend(case.pressures)
        scoped_items.extend(case.line_loads)
        scoped_items.extend(case.surface_tractions)
    referenced_regions.update(
        item.region.id
        for item in scoped_items
        if getattr(item, "region", None) is not None
    )
    for region_id in sorted(referenced_regions):
        try:
            region = project.regions[region_id]
            if getattr(region.domain, "value", region.domain) != "geometry":
                continue
            candidates = tuple(
                EntityRef(region.entity_kind, identifier)
                for identifier in {
                    "vertex": project.geometry.vertices,
                    "edge": project.geometry.edges,
                    "face": project.geometry.faces,
                }[region.entity_kind]
            )
            resolved = project.regions.resolve(
                region_id,
                geometry=project.geometry,
                candidates=candidates,
                feature_resolver=lambda anchor: project.geometry.features.resolve(
                    anchor, project.geometry
                ),
            )
            if not resolved:
                problems.append(f"required region {region.name!r} is empty")
        except (KeyError, TypeError, ValueError, GeometryError) as error:
            problems.append(f"required region {region_id!r} is unresolved: {error}")
    if problems:
        detail = "; ".join(dict.fromkeys(str(item) for item in problems))
        project.read_only_reason = (
            "recovered legacy geometry has unresolved or invalid required "
            f"targets: {detail}"
        )
        if project.read_only_reason not in project.compatibility_diagnostics:
            project.compatibility_diagnostics.append(project.read_only_reason)


def load_project(path: str | Path) -> Project:
    """Read a project from a file."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ProjectFileError(f"cannot read {source}: {error}") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProjectFileError(f"{source} is not valid JSON: {error}") from None
    return project_from_dict(data)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _migrate_legacy_output_requests(
    project: Project, analysis: AnalysisDefinition
) -> AnalysisDefinition:
    """Upgrade only legacy requests that carry complete, explicit intent.

    A pre-v4 dictionary such as ``{"stress": True}`` says nothing about the
    required scope or result location.  It remains in the compatibility field
    and validation blocks a solve until an engineer resolves it; migration
    never invents a node/element scope.  Complete entries receive UUID5 IDs so
    repeated migration of the same file is byte-for-byte deterministic.
    """

    legacy = dict(analysis.output_requests)
    if not legacy:
        return analysis

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    structured = (
        ("quantity_keys" in legacy or "quantities" in legacy)
        and ("region" in legacy or "region_id" in legacy)
        and "location" in legacy
    )
    if structured:
        candidates.append(("request", legacy))
    else:
        raw_requests = legacy.get("requests")
        if isinstance(raw_requests, list):
            for index, value in enumerate(raw_requests):
                if isinstance(value, Mapping):
                    candidates.append((f"requests:{index}", value))
        for key, value in legacy.items():
            if key == "requests" or not isinstance(value, Mapping):
                continue
            entry = dict(value)
            entry.setdefault("quantity_keys", (str(key),))
            entry.setdefault("label", str(key).replace("_", " ").title())
            candidates.append((f"key:{key}", entry))

    migrated_markers: set[str] = set()
    identifiers = list(analysis.output_request_ids)
    for marker, value in candidates:
        entry = dict(value)
        if not (
            (entry.get("quantity_keys") is not None or entry.get("quantities") is not None)
            and (entry.get("region") is not None or entry.get("region_id") is not None)
            and entry.get("location") is not None
        ):
            continue
        canonical = json.dumps(
            entry, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        entry.setdefault(
            "id",
            str(
                uuid5(
                    NAMESPACE_URL,
                    f"{project.document_id}:output-request:{analysis.id}:"
                    f"{marker}:{canonical}",
                )
            ),
        )
        entry.setdefault("label", f"{analysis.name} output")
        try:
            request = OutputRequest.from_dict(entry)
        except (TypeError, ValueError):
            # Preserve malformed/incomplete legacy data for an explicit user
            # decision.  The analysis validator will report that it remains.
            continue
        if request.region.id not in project.regions:
            continue
        existing = project.output_requests.get(request.id)
        if existing is None:
            project.add_output_request(request)
        elif existing.semantic_dict() != request.semantic_dict():
            raise ProjectFileError(
                f"legacy output request {request.id!r} conflicts with the "
                "typed output-request registry"
            )
        if request.id not in identifiers:
            identifiers.append(request.id)
        migrated_markers.add(marker)

    if structured and "request" in migrated_markers:
        remaining: dict[str, Any] = {}
    else:
        remaining = dict(legacy)
        for marker in migrated_markers:
            if marker.startswith("key:"):
                remaining.pop(marker.split(":", 1)[1], None)
        if "requests" in remaining:
            values = remaining.get("requests")
            if isinstance(values, list):
                incomplete = [
                    value
                    for index, value in enumerate(values)
                    if f"requests:{index}" not in migrated_markers
                ]
                if incomplete:
                    remaining["requests"] = incomplete
                else:
                    remaining.pop("requests", None)
    return replace(
        analysis,
        output_request_ids=tuple(identifiers),
        output_requests=remaining,
    )


def _by_name(mapping):
    return [mapping[key] for key in sorted(mapping)]


def _vector(values) -> List[float]:
    return [float(value) for value in np.asarray(values, dtype=float).ravel()]


def _ref(ref: EntityRef) -> Dict[str, Any]:
    return {"kind": ref.kind, "id": int(ref.id)}


def _region_ref(reference: RegionRef | None) -> str | None:
    return None if reference is None else reference.id


def _region_ref_from(
    project: Project, value: object, context: str
) -> RegionRef | None:
    if value is None:
        return None
    identifier = str(value.get("id")) if isinstance(value, Mapping) else str(value)
    if identifier not in project.regions:
        raise ProjectFileError(f"{context} references missing region {identifier!r}")
    return RegionRef(identifier)


def _ref_from(data: Mapping[str, Any]) -> EntityRef:
    if not isinstance(data, Mapping):
        raise ValueError("an entity reference must be an object")
    kind = str(data["kind"])
    if kind not in ("vertex", "edge", "face"):
        raise ValueError(
            f"unknown entity kind {kind!r}; expected vertex, edge or face"
        )
    raw_id = data["id"]
    if isinstance(raw_id, bool):
        raise ValueError("an entity reference ID must be a positive integer")
    identifier = int(raw_id)
    if identifier <= 0 or (
        isinstance(raw_id, float) and not raw_id.is_integer()
    ):
        raise ValueError("an entity reference ID must be a positive integer")
    return EntityRef(kind, identifier)  # type: ignore[arg-type]


def _existing_ref(
    project: Project, data: Mapping[str, Any], context: str, *, region=None, raw_attachment=False
) -> EntityRef:
    """Decode a legacy scalar cache through exact topology lineage.

    Region UUIDs are the canonical scope in current projects.  The scalar
    ``ref`` remains for headless compatibility and may name a predecessor that
    was split after the region was created.  Materialise one stable active
    representative through ANYgeometry's explicit replacement history only;
    the region still retains the complete descendant set.  No proximity
    matching is permitted.
    """

    try:
        ref = _ref_from(data)
        if project.mesh_only:
            # Imported groups use the EntityRef vocabulary but intentionally
            # have no ANYgeometry entity behind them.  Their existence is
            # proved against the restored mesh association later.
            return ref
        binding = project.geometry_attachment_regions.get(ref) if raw_attachment and region is None else None
        if binding is not None:
            from ..model.feature_bindings import _resolve_attachment_scope
            targets, inactive = _resolve_attachment_scope(project, binding.id)
            if not inactive and (not targets or any(target.kind != ref.kind for target in targets)):
                raise ValueError("raw attachment scope is missing or incompatible")
            return ref
        try:
            return project.geometry.entity_ref(ref.kind, ref.id)
        except (KeyError, ValueError):
            resolved = project.geometry.resolve_ref(ref)
            if resolved:
                return sorted(resolved, key=lambda item: (item.kind, item.id))[0]
            if region is not None:
                from ..model.feature_bindings import _resolve_attachment_scope
                region_ref = _region_ref_from(project, region, context + ".region")
                _targets, inactive = _resolve_attachment_scope(project, region_ref.id)
                if inactive and project.regions[region_ref.id].entity_kind == ref.kind:
                    return ref
            raise
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectFileError(f"{context}: {error}") from None


def _optional_text(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProjectFileError(f"{context} must be a non-empty string or null")
    return value


def _restore_section_assignment(
    project: Project,
    kind: str,
    raw_id: object,
    raw_section: object,
    context: str,
) -> None:
    try:
        if isinstance(raw_id, bool):
            raise ValueError
        identifier = int(raw_id)
    except (TypeError, ValueError):
        raise ProjectFileError(f"{context} IDs must be positive integers") from None
    if identifier <= 0:
        raise ProjectFileError(f"{context} IDs must be positive integers")
    section = str(raw_section)
    available = project.plate_sections if kind == "face" else project.beam_sections
    if section not in available:
        label = "plate" if kind == "face" else "beam"
        raise ProjectFileError(f"no {label} section named {section!r}")
    if not project.mesh_only:
        try:
            project.geometry.entity_ref(kind, identifier)
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectFileError(f"{context}: {error}") from None
    target = project.face_sections if kind == "face" else project.edge_sections
    target[identifier] = section


def _geometry_from_dict(geometry: GeometryModel, data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping):
        raise ProjectFileError("geometry must be a JSON object")
    # Formats 1 and 2 predate the ANYgeometry owner codec.  Recreate their
    # records through public model operations in one transaction: the entity
    # stores are intentionally immutable views in ANYgeometry 0.2.1.
    with geometry.transaction():
        _populate_legacy_geometry(geometry, data)


def _set_legacy_next_id(
    geometry: GeometryModel, kind: str, identifier: int
) -> None:
    state = geometry.id_state()
    state[kind] = int(identifier)
    geometry.restore_id_state(state)


def _populate_legacy_geometry(
    geometry: GeometryModel, data: Mapping[str, Any]
) -> None:
    vertices = sorted(data.get("vertices", ()), key=lambda item: int(item["id"]))
    for entry in vertices:
        vertex_id = int(entry["id"])
        if vertex_id <= 0 or vertex_id in geometry.vertices:
            raise ProjectFileError(
                f"geometry.vertices[{vertex_id}].id must be unique and positive"
            )
        position = np.asarray(entry["position"], dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ProjectFileError(
                f"geometry.vertices[{vertex_id}].position needs three finite "
                "components"
            )
        _set_legacy_next_id(geometry, "vertex", vertex_id)
        made = geometry.add_point(*position)
        if made != vertex_id:  # pragma: no cover - owner allocator contract
            raise ProjectFileError(
                f"geometry.vertices[{vertex_id}] could not preserve its ID"
            )

    edges = sorted(data.get("edges", ()), key=lambda item: int(item["id"]))
    for entry in edges:
        curve_data = entry.get("curve", {"kind": "line"})
        edge_id = int(entry["id"])
        if edge_id <= 0 or edge_id in geometry.edges:
            raise ProjectFileError(
                f"geometry.edges[{edge_id}].id must be unique and positive"
            )
        start = int(entry["start"])
        end = int(entry["end"])
        for field, vertex_id in (("start", start), ("end", end)):
            if vertex_id not in geometry.vertices:
                raise ProjectFileError(
                    f"geometry.edges[{edge_id}].{field} references missing "
                    f"vertex {vertex_id}"
                )
        if start == end:
            raise ProjectFileError(
                f"geometry.edges[{edge_id}] needs two distinct end vertices"
            )
        curve_kind = curve_data.get("kind", "line")
        if curve_kind == "arc":
            via = int(curve_data["via"])
            if via not in geometry.vertices:
                raise ProjectFileError(
                    f"geometry.edges[{edge_id}].curve.via references missing "
                    f"vertex {via}"
                )
            if len({start, via, end}) != 3:
                raise ProjectFileError(
                    f"geometry.edges[{edge_id}] arc needs three distinct vertices"
                )
            make_edge = lambda: geometry.add_arc(start, via, end)
        elif curve_kind == "line":
            make_edge = lambda: geometry.add_line(start, end)
        else:
            raise ProjectFileError(
                f"geometry.edges[{edge_id}].curve.kind {curve_kind!r} is unknown"
            )
        _set_legacy_next_id(geometry, "edge", edge_id)
        try:
            made = make_edge()
        except GeometryError as error:
            raise ProjectFileError(f"geometry.edges[{edge_id}]: {error}") from None
        if made != edge_id:  # pragma: no cover - owner allocator contract
            raise ProjectFileError(
                f"geometry.edges[{edge_id}] could not preserve its ID"
            )

    faces = sorted(data.get("faces", ()), key=lambda item: int(item["id"]))
    for entry in faces:
        face_id = int(entry["id"])
        if face_id <= 0 or face_id in geometry.faces:
            raise ProjectFileError(
                f"geometry.faces[{face_id}].id must be unique and positive"
            )
        loop_items: list[tuple[int, bool]] = []
        for item in entry["loop"]:
            if len(item) != 2 or not isinstance(item[1], bool):
                raise ProjectFileError(
                    f"geometry.faces[{face_id}].loop entries need an edge ID "
                    "and a boolean direction"
                )
            edge_id = int(item[0])
            if edge_id not in geometry.edges:
                raise ProjectFileError(
                    f"geometry.faces[{face_id}].loop references missing edge "
                    f"{edge_id}"
                )
            loop_items.append((edge_id, item[1]))
        loop = tuple(loop_items)
        if len(loop) < 4:
            raise ProjectFileError(
                f"geometry.faces[{face_id}].loop needs at least four edges"
            )

        def start_vertex(item: tuple[int, bool]) -> int:
            edge = geometry.edges[item[0]]
            return edge.start if item[1] else edge.end

        def end_vertex(item: tuple[int, bool]) -> int:
            edge = geometry.edges[item[0]]
            return edge.end if item[1] else edge.start

        for current, following in zip(loop, loop[1:] + loop[:1]):
            if end_vertex(current) != start_vertex(following):
                raise ProjectFileError(
                    f"geometry.faces[{face_id}].loop is not continuous at "
                    f"edge {following[0]}"
                )
        corners = tuple(int(corner) for corner in entry["corners"])
        if (
            len(corners) != 4
            or len(set(corners)) != 4
            or any(not 0 <= corner < len(loop) for corner in corners)
            or tuple(sorted(corners)) != corners
        ):
            raise ProjectFileError(
                f"geometry.faces[{face_id}].corners must be four distinct loop "
                "positions in order"
            )
        # add_face orients its first supplied edge forward.  Rotate a valid
        # legacy loop to a forward edge so its persisted normal is retained.
        pivot = next((index for index, item in enumerate(loop) if item[1]), 0)
        oriented_loop = loop[pivot:] + loop[:pivot]
        oriented_corners = tuple(
            sorted((corner - pivot) % len(loop) for corner in corners)
        )
        _set_legacy_next_id(geometry, "face", face_id)
        try:
            made = geometry.add_face(
                [edge_id for edge_id, _forward in oriented_loop],
                corners=oriented_corners,
            )
        except GeometryError as error:
            raise ProjectFileError(f"geometry.faces[{face_id}]: {error}") from None
        if made != face_id:  # pragma: no cover - owner allocator contract
            raise ProjectFileError(
                f"geometry.faces[{face_id}] could not preserve its ID"
            )
        restored_loop = tuple(
            (item.edge, item.forward) for item in geometry.faces[face_id].loop
        )
        if restored_loop != oriented_loop:
            raise ProjectFileError(
                f"geometry.faces[{face_id}].loop orientation cannot be "
                "preserved by the public geometry owner API"
            )

    counters = data.get("next_id")
    if counters:
        saved_state = {str(k): int(v) for k, v in counters.items()}
        state = geometry.id_state()
        for kind, store in (
            ("vertex", geometry.vertices),
            ("edge", geometry.edges),
            ("face", geometry.faces),
        ):
            minimum = max(store, default=0) + 1
            if kind not in saved_state or saved_state[kind] < minimum:
                raise ProjectFileError(
                    f"geometry.next_id.{kind} must be at least {minimum}"
                )
            state[kind] = saved_state[kind]
        geometry.restore_id_state(state)
    else:
        # An older file without counters: continue past whatever it holds, so
        # a new entity can never collide with a saved one.
        state = geometry.id_state()
        state.update(
            vertex=max(geometry.vertices, default=0) + 1,
            edge=max(geometry.edges, default=0) + 1,
            face=max(geometry.faces, default=0) + 1,
        )
        geometry.restore_id_state(state)


def _uses_envelopes(project: Project) -> bool:
    return any(
        case.times is not None
        or any(
            getattr(load, "envelope", None) is not None
            for collection in (
                case.point_loads, case.pressures, case.line_loads,
                case.surface_tractions, case.wet_pressures, case.connection_loads,
            )
            for load in collection
        )
        for case in project.load_cases.values()
    )


def _envelope_entry(load: Any) -> Dict[str, Any]:
    envelope = getattr(load, "envelope", None)
    return {} if envelope is None else {"envelope": envelope.to_dict()}


def _envelope_from(entry: Mapping[str, Any], context: str) -> Any:
    from anyloads import Envelope, LoadError

    if entry.get("envelope") is None:
        return None
    try:
        return Envelope.from_dict(entry["envelope"])
    except LoadError as error:
        raise ProjectFileError(f"{context}.envelope: {error}") from None


def _uses_connections(project: Project) -> bool:
    return bool(project.connections) or any(
        case.connection_loads for case in project.load_cases.values()
    )


def _connection_to_dict(connection: Connection) -> Dict[str, Any]:
    return {
        "id": connection.id,
        "name": connection.name,
        "ref": _ref(connection.ref),
        "region": _region_ref(connection.region),
        "kind": connection.kind,
        "dofs": list(connection.dofs),
        "reference": connection.reference,
        "reference_point": (
            None
            if connection.reference_point is None
            else _vector(connection.reference_point)
        ),
        "weighting": connection.weighting,
        **(
            {"stiffness": list(connection.stiffness), "tie": connection.tie}
            if connection.kind == "spring"
            else {}
        ),
    }


def _uses_wet_loads(project: Project) -> bool:
    return bool(project.wet_surfaces) or any(
        case.wet_pressures or case.time is not None or case.parameters
        or case.series is not None
        for case in project.load_cases.values()
    )


def _wet_surface_to_dict(surface: WetSurface) -> Dict[str, Any]:
    return {
        "id": surface.id,
        "name": surface.name,
        "ref": _ref(surface.ref),
        "region": _region_ref(surface.region),
        "fluid_side": surface.fluid_side,
        "interior_point": (
            None if surface.interior_point is None else _vector(surface.interior_point)
        ),
    }


def _wet_pressure_to_dict(load: WetPressure) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "id": load.id,
        "name": load.name,
        "scale": float(load.scale),
        "surfaces": list(load.surfaces),
    }
    if load.value is not None:
        entry["value"] = float(load.value)
    elif load.expression is not None:
        entry["expression"] = load.expression
    elif load.code is not None:
        entry["code"] = load.code
        if load.code_vectorized:
            entry["code_vectorized"] = True
    elif load.table is not None:
        entry["table"] = load.table.to_dict()
        if load.table_case is not None:
            entry["table_case"] = load.table_case
    else:
        entry["mapping_key"] = load.mapping_key
        entry["mapping"] = {
            str(key): float(value) for key, value in sorted(load.mapping.items())
        }
    entry.update(_envelope_entry(load))
    return entry


def _load_case_to_dict(case: LoadCase, *, numbered: bool = False) -> Dict[str, Any]:
    return {
        **({"number": int(case.number)} if numbered and case.number is not None else {}),
        **(
            {"wet_pressures": [_wet_pressure_to_dict(i) for i in case.wet_pressures]}
            if case.wet_pressures
            else {}
        ),
        **(
            {
                "connection_loads": [
                    {
                        "id": load.id,
                        "name": load.name,
                        "connection": load.connection,
                        "force": _vector(load.force),
                        "moment": _vector(load.moment),
                        **_envelope_entry(load),
                    }
                    for load in case.connection_loads
                ]
            }
            if case.connection_loads
            else {}
        ),
        **({"times": [float(t) for t in case.times]} if case.times is not None else {}),
        **({"parent": case.parent} if case.parent is not None else {}),
        **({"time": float(case.time)} if case.time is not None else {}),
        **({"parameters": dict(sorted(case.parameters.items()))} if case.parameters else {}),
        **({"series": case.series} if case.series is not None else {}),
        **({"step": int(case.step)} if case.step is not None else {}),
        "id": case.id,
        "name": case.name,
        "follower_pressure": bool(case.follower_pressure),
        "gravity": None if case.gravity is None else _vector(case.gravity),
        "gravity_coordinate_system_id": case.gravity_coordinate_system_id,
        "point_loads": [
            {
                "id": load.id,
                "ref": _ref(load.ref),
                "region": _region_ref(load.region),
                "force": _vector(load.force),
                "moment": _vector(load.moment),
                "coordinate_system_id": load.coordinate_system_id,
                "distribution_policy": load.distribution_policy,
                **_envelope_entry(load),
            }
            for load in case.point_loads
        ],
        "pressures": [
            {
                "id": load.id, "ref": _ref(load.ref),
                "region": _region_ref(load.region), "value": float(load.value),
                **_envelope_entry(load),
            }
            for load in case.pressures
        ],
        "line_loads": [
            {
                "id": load.id, "ref": _ref(load.ref),
                "region": _region_ref(load.region),
                "coordinate_system_id": load.coordinate_system_id,
                "force_per_length": _vector(load.force_per_length),
                **_envelope_entry(load),
            }
            for load in case.line_loads
        ],
        "surface_tractions": [
            {
                "id": load.id, "ref": _ref(load.ref),
                "region": _region_ref(load.region),
                "coordinate_system_id": load.coordinate_system_id,
                "traction": _vector(load.traction),
                **_envelope_entry(load),
            }
            for load in case.surface_tractions
        ],
    }


def _load_case_from_dict(project: Project, data: Mapping[str, Any]) -> LoadCase:
    case = project.load_case(str(data["name"]))
    if data.get("id"):
        case.id = str(data["id"])
    case.gravity_coordinate_system_id = str(
        data.get("gravity_coordinate_system_id", "global")
    )
    follower = data.get("follower_pressure", False)
    if not isinstance(follower, bool):
        raise ProjectFileError(
            f"load_cases[{case.name!r}].follower_pressure must be true or false"
        )
    case.follower_pressure = follower
    gravity = data.get("gravity")
    if gravity is None:
        case.gravity = None
    else:
        vector = np.asarray(gravity, dtype=float)
        if vector.shape != (3,):
            raise ProjectFileError(
                f"load_cases[{case.name!r}].gravity needs three finite components"
            )
        case.set_acceleration(*vector)

    for index, entry in enumerate(data.get("point_loads", ())):
        load = case.add_point_load(
            ref=_existing_ref(
                project,
                entry["ref"],
                f"load_cases[{case.name!r}].point_loads[{index}].ref",
                region=entry.get("region"),
            ),
            force=entry["force"],
            moment=entry["moment"],
            region=_region_ref_from(
                project, entry.get("region"),
                f"load_cases[{case.name!r}].point_loads[{index}].region",
            ),
            coordinate_system_id=str(entry.get("coordinate_system_id", "global")),
            distribution_policy=str(entry.get("distribution_policy", "per_target")),
            envelope=_envelope_from(entry, f"load_cases[{case.name!r}].point_loads[{index}]"),
        )
        if entry.get("id"):
            case.point_loads[-1] = replace(load, id=str(entry["id"]))
    for index, entry in enumerate(data.get("pressures", ())):
        load = case.add_pressure(
            ref=_existing_ref(
                project,
                entry["ref"],
                f"load_cases[{case.name!r}].pressures[{index}].ref",
                region=entry.get("region"),
            ),
            value=entry["value"],
            region=_region_ref_from(
                project, entry.get("region"),
                f"load_cases[{case.name!r}].pressures[{index}].region",
            ),
            envelope=_envelope_from(entry, f"load_cases[{case.name!r}].pressures[{index}]"),
        )
        if entry.get("id"):
            case.pressures[-1] = replace(load, id=str(entry["id"]))
    for index, entry in enumerate(data.get("connection_loads", ())):
        try:
            load = ConnectionLoad(
                id=str(entry["id"]) if entry.get("id") else str(uuid4()),
                name=str(entry.get("name", "connection load")),
                connection=str(entry["connection"]),
                force=entry.get("force", (0.0, 0.0, 0.0)),
                moment=entry.get("moment", (0.0, 0.0, 0.0)),
                envelope=_envelope_from(entry, f"load_cases[{case.name!r}].connection_loads[{index}]"),
            )
        except (KeyError, ValueError) as error:
            raise ProjectFileError(
                f"load_cases[{case.name!r}].connection_loads[{index}]: {error}"
            ) from None
        case.connection_loads.append(load)
    wet_context = f"load_cases[{case.name!r}]"
    if data.get("times") is not None:
        try:
            case.times = project._series_times(data["times"])
        except Exception as error:
            raise ProjectFileError(f"{wet_context}.times: {error}") from None
    if data.get("time") is not None:
        case.set_time(data["time"])
    parameters = data.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ProjectFileError(f"{wet_context}.parameters must be an object")
    for key, value in parameters.items():
        case.set_parameter(str(key), value)
    case.series = None if data.get("series") is None else str(data["series"])
    case.step = None if data.get("step") is None else int(data["step"])
    if data.get("number") is not None:
        case.number = int(data["number"])
    elif case.series is not None:
        case.number = None  # a series step is numbered by its step
    for index, entry in enumerate(data.get("wet_pressures", ())):
        try:
            load = WetPressure(
                id=str(entry["id"]) if entry.get("id") else str(uuid4()),
                name=str(entry.get("name", "wet pressure")),
                value=entry.get("value"),
                expression=entry.get("expression"),
                mapping=entry.get("mapping"),
                mapping_key=str(entry.get("mapping_key", "face")),
                surfaces=tuple(entry.get("surfaces", ())),
                scale=entry.get("scale", 1.0),
                code=entry.get("code"),
                code_vectorized=bool(entry.get("code_vectorized", False)),
                table=(
                    None
                    if entry.get("table") is None
                    else PressureTable.from_dict(entry["table"])
                ),
                table_case=entry.get("table_case"),
                envelope=_envelope_from(entry, f"{wet_context}.wet_pressures[{index}]"),
            )
        except ValueError as error:
            raise ProjectFileError(
                f"{wet_context}.wet_pressures[{index}]: {error}"
            ) from None
        case.wet_pressures.append(load)
    for index, entry in enumerate(data.get("line_loads", ())):
        load = case.add_line_load(
            ref=_existing_ref(
                project,
                entry["ref"],
                f"load_cases[{case.name!r}].line_loads[{index}].ref",
                region=entry.get("region"),
            ),
            force_per_length=entry["force_per_length"],
            region=_region_ref_from(
                project, entry.get("region"),
                f"load_cases[{case.name!r}].line_loads[{index}].region",
            ),
            coordinate_system_id=str(entry.get("coordinate_system_id", "global")),
            envelope=_envelope_from(entry, f"load_cases[{case.name!r}].line_loads[{index}]"),
        )
        if entry.get("id"):
            case.line_loads[-1] = replace(load, id=str(entry["id"]))
    for index, entry in enumerate(data.get("surface_tractions", ())):
        load = case.add_surface_traction(
            ref=_existing_ref(
                project,
                entry["ref"],
                f"load_cases[{case.name!r}].surface_tractions[{index}].ref",
                region=entry.get("region"),
            ),
            traction=entry["traction"],
            region=_region_ref_from(
                project, entry.get("region"),
                f"load_cases[{case.name!r}].surface_tractions[{index}].region",
            ),
            coordinate_system_id=str(entry.get("coordinate_system_id", "global")),
            envelope=_envelope_from(entry, f"load_cases[{case.name!r}].surface_tractions[{index}]"),
        )
        if entry.get("id"):
            case.surface_tractions[-1] = replace(load, id=str(entry["id"]))
    return case
