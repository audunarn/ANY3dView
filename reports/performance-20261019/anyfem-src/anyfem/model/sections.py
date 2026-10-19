"""Sections: plate thickness, and the typical structural beam profiles.

Beam section properties are resolved through the solver's own
``StiffenerCrossSection.from_geometry`` so area, inertias and torsion constant
cannot diverge from the conventions the beam elements assume.  Only the
profiles that function actually implements are offered; anything else would
silently fall through to a bare-web section, so it is rejected instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Sequence
from uuid import uuid4

import numpy as np
from anysolver import StiffenerCrossSection

from .regions import RegionRef

__all__ = [
    "BeamSection",
    "ATTACHMENT_SIDES",
    "OFFSET_MODES",
    "PLATE_PROFILES",
    "PlateSection",
    "PROFILES",
    "SectionAssignment",
]

# Profile names accepted by StiffenerCrossSection.from_geometry.  "Flatbar"
# uses (b, tf) as the bar width and thickness; the others use the full
# (hw, tw, b, tf) web/flange set.
PROFILES: tuple[str, ...] = ("Flatbar", "T-bar", "Angle", "L-bulb")
PLATE_PROFILES = PROFILES  # backwards-friendly alias
OFFSET_MODES: tuple[str, ...] = ("manual", "automatic", "centerline")
ATTACHMENT_SIDES: tuple[str, ...] = ("front", "back")


def _uuid() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class PlateSection:
    """A plate thickness and its material."""

    name: str
    thickness: float
    material: str
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ValueError(
                f"plate section {self.name!r}: thickness must be finite and positive"
            )


@dataclass(frozen=True)
class BeamSection:
    """A typical structural profile carried on a line.

    ``web_direction`` prescribes the beam's local z axis -- the direction the
    web stands in.  Without it the solver falls back to a heuristic, which
    leaves asymmetric sections unconstrained, so it is worth setting for
    angles and bulbs.

    ``offset_mode`` controls the neutral-axis location. ``automatic`` derives
    the actual centroid from the profile rectangles, ``manual`` retains the
    historical explicit eccentricity, and ``centerline`` shares plate nodes.
    ``attachment_side`` and ``rotation_deg`` place the section relative to the
    plate attachment line and are used by meshing, solving and visualization.
    """

    name: str
    profile: str
    material: str
    web_height: float = 0.0
    web_thickness: float = 0.0
    flange_width: float = 0.0
    flange_thickness: float = 0.0
    web_direction: Sequence[float] | None = None
    eccentricity: float = 0.0
    offset_mode: Literal["manual", "automatic", "centerline"] = "manual"
    attachment_side: Literal["front", "back"] = "front"
    rotation_deg: float = 0.0
    id: str = field(default_factory=_uuid)

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(
                f"beam section {self.name!r}: unknown profile {self.profile!r}. "
                f"Supported profiles are {', '.join(PROFILES)}."
            )
        dimensions = {
            "web_height": self.web_height,
            "web_thickness": self.web_thickness,
            "flange_width": self.flange_width,
            "flange_thickness": self.flange_thickness,
        }
        invalid = [
            name
            for name, value in dimensions.items()
            if not np.isfinite(value) or value < 0.0
        ]
        if invalid:
            raise ValueError(
                f"beam section {self.name!r}: dimensions must be finite and "
                f"non-negative; invalid {', '.join(invalid)}"
            )
        if not np.isfinite(self.eccentricity):
            raise ValueError(
                f"beam section {self.name!r}: eccentricity must be finite"
            )
        if self.offset_mode not in OFFSET_MODES:
            raise ValueError(
                f"beam section {self.name!r}: offset_mode must be one of "
                f"{', '.join(OFFSET_MODES)}"
            )
        if self.attachment_side not in ATTACHMENT_SIDES:
            raise ValueError(
                f"beam section {self.name!r}: attachment_side must be front or back"
            )
        if not np.isfinite(self.rotation_deg):
            raise ValueError(
                f"beam section {self.name!r}: rotation_deg must be finite"
            )
        if self.profile == "Flatbar":
            if self.flange_width <= 0.0 or self.flange_thickness <= 0.0:
                raise ValueError(
                    f"beam section {self.name!r}: a flat bar needs a positive "
                    "flange_width (bar width) and flange_thickness"
                )
        else:
            if self.web_height <= 0.0 or self.web_thickness <= 0.0:
                raise ValueError(
                    f"beam section {self.name!r}: profile {self.profile} needs a "
                    "positive web_height and web_thickness"
                )
        if self.web_direction is not None:
            try:
                direction = np.asarray(self.web_direction, dtype=float)
            except (TypeError, ValueError):
                raise ValueError(
                    f"beam section {self.name!r}: web_direction needs three "
                    "finite components"
                ) from None
            if (
                direction.shape != (3,)
                or not np.all(np.isfinite(direction))
                or float(np.linalg.norm(direction)) == 0.0
            ):
                raise ValueError(
                    f"beam section {self.name!r}: web_direction needs three "
                    "finite, non-zero components"
                )
            object.__setattr__(
                self, "web_direction", tuple(float(value) for value in direction)
            )

    def profile_rectangles(self) -> tuple[tuple[float, float, float, float], ...]:
        """Idealized rectangles as ``(y, z, width_y, height_z)``.

        Coordinates are measured from the attachment line on the plate. They
        use the same profile convention as ANYmesher's section-property
        calculation, so the displayed solid and the numerical section agree.
        """

        hw, tw = float(self.web_height), float(self.web_thickness)
        b, tf = float(self.flange_width), float(self.flange_thickness)
        if self.profile == "T-bar":
            return ((0.0, hw / 2.0, tw, hw), (0.0, hw + tf / 2.0, b, tf))
        if self.profile in ("Angle", "L-bulb"):
            return (
                (tw / 2.0, hw / 2.0, tw, hw),
                (b / 2.0, hw + tf / 2.0, b, tf),
            )
        return ((0.0, tf / 2.0, b, tf),)

    def centroid_from_attachment(self) -> tuple[float, float]:
        """Local ``(y, z)`` centroid measured from the plate attachment line."""

        rectangles = self.profile_rectangles()
        areas = np.asarray([width * height for _y, _z, width, height in rectangles])
        area = float(np.sum(areas))
        return (
            float(sum(a * item[0] for a, item in zip(areas, rectangles)) / area),
            float(sum(a * item[1] for a, item in zip(areas, rectangles)) / area),
        )

    def centered_profile_rectangles(
        self,
    ) -> tuple[tuple[float, float, float, float], ...]:
        """Profile rectangles around the beam neutral-axis line."""

        cy, cz = self.centroid_from_attachment()
        return tuple(
            (y - cy, z - cz, width, height)
            for y, z, width, height in self.profile_rectangles()
        )

    def properties(
        self, *, orientation: Sequence[float] | None = None
    ) -> Dict[str, float]:
        """Cross-section properties in the solver's own dictionary form."""

        section = StiffenerCrossSection.from_geometry(
            self.profile,
            float(self.web_height),
            float(self.web_thickness),
            float(self.flange_width),
            float(self.flange_thickness),
        )
        properties: Dict[str, float] = {
            "area": float(section.area),
            "Iy": float(section.Iy),
            "Iz": float(section.Iz),
            "J": float(section.J),
            "shear_factor_y": float(section.shear_factor_y),
            "shear_factor_z": float(section.shear_factor_z),
            "c_y": float(section.c_y),
            "c_z": float(section.c_z),
            "torsion_modulus": float(section.torsion_modulus),
        }
        resolved_orientation = (
            self.web_direction if orientation is None else orientation
        )
        if resolved_orientation is not None:
            properties["orientation"] = np.asarray(resolved_orientation, dtype=float)
        return properties


@dataclass(frozen=True)
class SectionAssignment:
    """A stable section-to-region binding.

    The section UUID and region UUID are the durable references.  ``name`` is
    deliberately only an editable label; changing it cannot change numerical
    meaning or invalidate another object.  ``legacy_singleton`` identifies the
    compatibility records created by :meth:`Project.assign_plate` and
    :meth:`Project.assign_beam`.  It is persisted so direct mutations of the
    historical ``face_sections``/``edge_sections`` dictionaries can be folded
    back into the canonical record without treating a named multi-entity
    assignment as a collection of unrelated topology assignments.
    """

    kind: Literal["plate", "beam"]
    section_id: str
    region: RegionRef
    name: str = "Section assignment"
    id: str = field(default_factory=_uuid)
    legacy_singleton: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("plate", "beam"):
            raise ValueError(f"unknown section assignment kind {self.kind!r}")
        if not str(self.section_id).strip():
            raise ValueError("section assignment needs a section UUID")
        if not str(self.id).strip():
            raise ValueError("section assignment needs a UUID")
        if not str(self.name).strip():
            raise ValueError("section assignment needs a name")
        object.__setattr__(self, "section_id", str(self.section_id))
        object.__setattr__(self, "id", str(self.id))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "section_id": self.section_id,
            "region": self.region.id,
            "legacy_singleton": bool(self.legacy_singleton),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SectionAssignment":
        region = data.get("region")
        if region is None:
            raise ValueError("section assignment needs a region UUID")
        identifier = data.get("id")
        return cls(
            id=_uuid() if identifier is None else str(identifier),
            name=str(data.get("name", "Section assignment")),
            kind=str(data.get("kind", "")),  # type: ignore[arg-type]
            section_id=str(data.get("section_id", "")),
            region=RegionRef(str(region)),
            legacy_singleton=bool(data.get("legacy_singleton", False)),
        )


def rectangular_bar(
    name: str, width: float, thickness: float, material: str
) -> BeamSection:
    """Convenience builder for a plain rectangular bar."""

    return BeamSection(
        name=name,
        profile="Flatbar",
        material=material,
        flange_width=width,
        flange_thickness=thickness,
    )
