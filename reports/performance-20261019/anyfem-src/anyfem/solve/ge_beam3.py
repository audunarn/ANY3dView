"""Explicit historical GE-B3 and production B3-GE consumer adapters."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from anysolver.beam_sections import GeneralizedBeamSection
from anysolver.elements import create_element
from anysolver.ge_beam3_element import (
    GE_BEAM3_QUALIFIED_FORMULATION_ID, GeometricallyExactBeam3D3NElement,
)


SCHEMA = "anyfem.ge-beam3-explicit-opt-in-v1"
B3_GE_SCHEMA = "anyfem.b3-ge-native-opt-in-v2"
CONSUMER_POLICY = {
    "enabled": True,
    "selector": "ge-beam3",
    "formulation_id": GE_BEAM3_QUALIFIED_FORMULATION_ID,
}
B3_GE_CONSUMER_POLICY = {
    "enabled": True,
    "selector": "b3-ge",
    "native_profile_id": "GE_BEAM3_NATIVE_OWNED_WORKFLOWS_V1",
    "explicit_opt_in": True,
    "legacy_b3_default": True,
}


def _matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (6, 6) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 6x6 matrix")
    if not np.array_equal(result, result.T):
        raise ValueError(f"{label} must be exactly symmetric")
    np.linalg.cholesky(result)
    return result.copy()


@dataclass(frozen=True)
class GeBeam3OptIn:
    """Complete physical authority required to construct one GE-B3 family."""

    section_stiffness: Any
    section_mass_per_length: Any
    reference_orientation: Sequence[float]
    section_name: str = "ANYfem explicit GE-B3"
    contact_radius: float = 0.0

    def __post_init__(self) -> None:
        stiffness = _matrix(self.section_stiffness, "section stiffness")
        mass = _matrix(self.section_mass_per_length, "section mass")
        orientation = np.asarray(self.reference_orientation, dtype=float)
        if orientation.shape != (3,) or not np.isfinite(orientation).all() or np.linalg.norm(orientation) <= 0.:
            raise ValueError("reference_orientation must be a finite nonzero three-vector")
        if type(self.section_name) is not str or not self.section_name.strip():
            raise ValueError("section_name must be nonempty text")
        if not np.isfinite(self.contact_radius) or self.contact_radius < 0.:
            raise ValueError("contact_radius must be finite and non-negative")
        stiffness.setflags(write=False); mass.setflags(write=False); orientation = orientation.copy(); orientation.setflags(write=False)
        object.__setattr__(self, "section_stiffness", stiffness)
        object.__setattr__(self, "section_mass_per_length", mass)
        object.__setattr__(self, "reference_orientation", orientation)

    @property
    def policy(self) -> dict[str, object]:
        return dict(CONSUMER_POLICY)

    def build(self, element_id: int, node_ids: Sequence[int], material_name: str) -> GeometricallyExactBeam3D3NElement:
        section = GeneralizedBeamSection(
            self.section_stiffness, mass_matrix=self.section_mass_per_length,
            name=self.section_name,
        )
        made = create_element(
            "ge-beam3", int(element_id), list(node_ids), str(material_name),
            cross_section={"contact_radius": float(self.contact_radius)},
            section=section, reference_orientation=self.reference_orientation,
        )
        if type(made) is not GeometricallyExactBeam3D3NElement or made.formulation_id != GE_BEAM3_QUALIFIED_FORMULATION_ID:
            raise RuntimeError("ANYfem GE-B3 adapter resolved an unexpected formulation")
        return made

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "consumer_policy": self.policy,
            "section_stiffness": self.section_stiffness.tolist(),
            "section_mass_per_length": self.section_mass_per_length.tolist(),
            "reference_orientation": self.reference_orientation.tolist(),
            "section_name": self.section_name,
            "contact_radius": float(self.contact_radius),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeBeam3OptIn":
        required = {"schema", "consumer_policy", "section_stiffness", "section_mass_per_length",
                    "reference_orientation", "section_name", "contact_radius"}
        if type(data) is not dict or set(data) != required or data["schema"] != SCHEMA:
            raise ValueError("strict ANYfem GE-B3 opt-in record required")
        if data["consumer_policy"] != CONSUMER_POLICY:
            raise ValueError("qualified GE-B3 consumer policy required")
        return cls(data["section_stiffness"], data["section_mass_per_length"],
                   data["reference_orientation"], data["section_name"], data["contact_radius"])


@dataclass(frozen=True)
class B3GENativeOptIn:
    """Exact native definition graph selected explicitly as B3-GE."""

    definitions: Sequence[Any]

    def __post_init__(self) -> None:
        from anysolver._ge_beam3_native_definition import NativeBeamDefinition

        rows = tuple(self.definitions)
        if not rows or any(type(row) is not NativeBeamDefinition for row in rows):
            raise ValueError("one or more exact native B3-GE definitions required")
        detached = tuple(
            NativeBeamDefinition.from_bytes(bytes(row.raw), expected_sha256=row.sha256)
            for row in rows
        )
        object.__setattr__(self, "definitions", detached)

    @property
    def policy(self) -> dict[str, object]:
        return dict(B3_GE_CONSUMER_POLICY)

    def create_analysis(self, boundaries: Sequence[Any], *, retained_refinement: bool = False):
        from anysolver import b3_ge

        return b3_ge.create_analysis(
            b3_ge.SELECTOR,
            self.definitions,
            tuple(boundaries),
            retained_refinement=retained_refinement,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": B3_GE_SCHEMA,
            "consumer_policy": self.policy,
            "definitions": [
                {
                    "raw_base64": base64.b64encode(row.raw).decode("ascii"),
                    "sha256": row.sha256,
                }
                for row in self.definitions
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "B3GENativeOptIn":
        if (
            type(data) is not dict
            or set(data) != {"schema", "consumer_policy", "definitions"}
            or data["schema"] != B3_GE_SCHEMA
            or data["consumer_policy"] != B3_GE_CONSUMER_POLICY
            or type(data["definitions"]) is not list
        ):
            raise ValueError("strict ANYfem B3-GE native opt-in record required")
        rows = []
        from anysolver._ge_beam3_native_definition import NativeBeamDefinition

        for item in data["definitions"]:
            if type(item) is not dict or set(item) != {"raw_base64", "sha256"}:
                raise ValueError("strict B3-GE definition binding required")
            if (
                type(item["raw_base64"]) is not str
                or len(item["raw_base64"]) > 2_800_000
                or type(item["sha256"]) is not str
                or len(item["sha256"]) != 64
            ):
                raise ValueError("bounded B3-GE definition binding required")
            try:
                raw = base64.b64decode(item["raw_base64"], validate=True)
            except Exception as exc:
                raise ValueError("invalid B3-GE definition encoding") from exc
            rows.append(NativeBeamDefinition.from_bytes(raw, expected_sha256=item["sha256"]))
        return cls(tuple(rows))


__all__ = ["GeBeam3OptIn", "B3GENativeOptIn", "SCHEMA", "B3_GE_SCHEMA"]
