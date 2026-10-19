"""Toolkit-neutral constructors shared by desktop definition tasks."""
from __future__ import annotations
from typing import Iterable, Mapping, Sequence
import numpy as np
from anygeometry.entities import EntityRef
from ..model.coordinates import CoordinateSystem
from ..model.records import OutputRequest
from ..model.regions import BooleanRegion, ElementFaceRef, ManualRegion, MeshEntityRef as RegionMeshEntityRef, Region, RegionDomain
from ..model.units import UnitProfile
from ..selection import MeshEntityRef as SelectionMeshEntityRef, mode_label

MAX_REGION_OPERANDS = 1000

_UNIT_CHOICES: Mapping[str, tuple[str, ...]] = {
    "length": ("m", "cm", "mm"),
    "force": ("N", "kN", "MN"),
    "pressure": ("Pa", "kPa", "MPa", "GPa"),
    "mass": ("kg", "t"),
    "time": ("s", "ms"),
    "angle": ("deg", "rad"),
    "moment": ("N*m", "N*mm", "kN*m"),
    "line_load": ("N/m", "N/mm", "kN/m"),
    "density": ("kg/m3", "t/m3"),
    "acceleration": ("m/s2", "mm/s2"),
}

_UNIT_LABELS = {
    "length": "Length",
    "force": "Force",
    "pressure": "Pressure / stress",
    "mass": "Mass",
    "time": "Time",
    "angle": "Angle",
    "moment": "Moment",
    "line_load": "Line load",
    "density": "Density",
    "acceleration": "Acceleration",
}


def _feature_anchor(project, ref: EntityRef):
    """Prefer persistent feature-output identity when one owns ``ref``."""

    history = getattr(project.geometry, "features", None)
    for feature in reversed(tuple(getattr(history, "records", ()))):
        for output_key, output in feature.outputs.items():
            if output == ref:
                try:
                    from anygeometry.features import FeatureOutputRef
                except ImportError:  # pragma: no cover - coordinated package floor
                    return ref
                return FeatureOutputRef(
                    feature.feature_id, str(output_key), str(output.kind)
                )
    return ref


def region_from_selection(
    project,
    name: str,
    items: Iterable[EntityRef | SelectionMeshEntityRef],
    mode: str,
    *,
    mesh_id: str | None = None,
) -> Region:
    """Build one manual region from a homogeneous commercial selection.

    Geometry owners are upgraded to feature-output anchors when possible.
    Mesh owners are converted to the persisted, mesh-UUID-bound region types.
    """

    selected = tuple(items)
    if not selected:
        raise ValueError("select at least one entity before creating a region")

    if mode in ("vertex", "edge", "face"):
        if any(not isinstance(item, EntityRef) or item.kind != mode for item in selected):
            raise ValueError(
                f"a {mode_label(mode).lower()} region needs only "
                f"{mode_label(mode).lower()} geometry selections"
            )
        anchors = tuple(_feature_anchor(project, item) for item in selected)
        return Region(
            name=name,
            domain=RegionDomain.GEOMETRY,
            entity_kind=mode,
            definition=ManualRegion(anchors),
        )

    if mode not in ("node", "element", "element_face"):
        raise ValueError(f"{mode!r} is not a region-capable selection filter")
    if not mesh_id:
        raise ValueError("generate or import a mesh before creating a mesh region")
    if any(
        not isinstance(item, SelectionMeshEntityRef) or item.kind != mode
        for item in selected
    ):
        raise ValueError(
            f"a {mode_label(mode).lower()} region needs only matching mesh selections"
        )

    mesh_anchors: list[RegionMeshEntityRef | ElementFaceRef] = []
    for item in selected:
        if item.kind == "element_face":
            mesh_anchors.append(
                ElementFaceRef(
                    str(mesh_id),
                    int(item.element_id),
                    int(item.local_face),
                )
            )
        else:
            mesh_anchors.append(
                RegionMeshEntityRef(str(mesh_id), item.kind, int(item.id))
            )
    return Region(
        name=name,
        domain=RegionDomain.MESH,
        entity_kind=mode,
        definition=ManualRegion(mesh_anchors),
        mesh_id=str(mesh_id),
    )


def boolean_region(
    name: str,
    operation: str,
    operands: Iterable[Region],
) -> Region:
    """Create a type-safe Boolean region definition."""

    regions = tuple(operands)
    if len(regions) < 2:
        raise ValueError("select at least two regions for a Boolean operation")
    if len({item.id for item in regions}) != len(regions):
        raise ValueError("a Boolean region cannot use the same operand twice")
    first = regions[0]
    incompatible = [
        item.name
        for item in regions[1:]
        if item.domain != first.domain
        or item.entity_kind != first.entity_kind
        or item.mesh_id != first.mesh_id
    ]
    if incompatible:
        raise ValueError(
            "Boolean operands must share the same domain, entity type and mesh; "
            f"incompatible: {', '.join(incompatible)}"
        )
    definition = BooleanRegion(
        operation=str(operation).lower(),  # type: ignore[arg-type]
        region_ids=tuple(item.id for item in regions),
    )
    return Region(
        name=name,
        domain=first.domain,
        entity_kind=first.entity_kind,
        definition=definition,
        mesh_id=first.mesh_id,
    )


def coordinate_system_from_values(
    name: str,
    kind: str,
    origin: Sequence[str | float],
    axis: Sequence[str | float],
    reference: Sequence[str | float],
    profile: UnitProfile,
) -> CoordinateSystem:
    """Parse a coordinate-system task using the active display units."""

    if len(origin) != 3 or len(axis) != 3 or len(reference) != 3:
        raise ValueError("origin, axis and reference each need three components")
    origin_si = tuple(profile.parse(value, "length") for value in origin)

    def direction(values: Sequence[str | float], label: str) -> tuple[float, ...]:
        try:
            parsed = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            raise ValueError(f"{label} needs three numeric components") from None
        if not np.all(np.isfinite(parsed)):
            raise ValueError(f"{label} needs finite components")
        return parsed

    normalized_kind = str(kind).strip().lower()
    return CoordinateSystem(
        name=str(name).strip(),
        kind=normalized_kind,  # type: ignore[arg-type]
        origin=origin_si,
        axis=direction(axis, "axis"),
        reference=direction(reference, "reference direction"),
    )


def unit_profile_from_values(
    name: str, units: Mapping[str, str]
) -> UnitProfile:
    """Validate a custom profile from Details form values."""

    if not str(name).strip():
        raise ValueError("custom unit profile needs a name")
    return UnitProfile(
        name=str(name).strip(),
        units={dimension: str(units[dimension]) for dimension in UnitProfile.REQUIRED},
    )


def output_request_from_values(
    label: str,
    quantities: str | Iterable[str],
    region_id: str,
    location: str,
    *,
    recovery: str = "native",
    reduction: str = "none",
    basis: str = "global",
    frame_policy: str = "all",
    frame_indices: str | Iterable[int] = (),
) -> OutputRequest:
    """Build a typed request from compact Details-form values."""

    if isinstance(quantities, str):
        keys = tuple(
            value for value in quantities.replace(",", " ").split() if value
        )
    else:
        keys = tuple(str(value).strip() for value in quantities if str(value).strip())
    indices = (
        tuple(int(value) for value in frame_indices.replace(",", " ").split())
        if isinstance(frame_indices, str) else tuple(frame_indices)
    )
    return OutputRequest(
        label=str(label).strip(),
        quantity_keys=keys,
        region=str(region_id),  # OutputRequest canonicalizes to RegionRef.
        location=str(location).strip().lower(),
        recovery=str(recovery).strip(),
        reduction=str(reduction).strip(),
        basis=str(basis).strip(),
        frame_policy=str(frame_policy).strip().lower(),
        frame_indices=indices,
    )


def _next_name(prefix: str, names: Iterable[str]) -> str:
    occupied = {str(name).casefold() for name in names}
    index = 1
    while f"{prefix}-{index}".casefold() in occupied:
        index += 1
    return f"{prefix}-{index}"
