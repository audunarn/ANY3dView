"""Persisted application policy for shell formulation selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from anysolver import QUALIFIED_Q4_FORMULATION_ID, S3_V2D_FORMULATION_ID

__all__ = ["ShellFormulationPolicy"]


_Q4 = {"legacy", "e4-pl"}
_S3 = {"legacy-s3", "e4-pl-s3-v2d"}
_HIGHER = {"legacy"}


def _token(value: object, *, field: str, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"shell formulation {field} must be text")
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in allowed:
        raise ValueError(
            f"unsupported shell formulation {field}={value!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return normalized


@dataclass(frozen=True)
class ShellFormulationPolicy:
    """Topology-specific identities used to construct solver shell elements.

    Current models select the qualified Q4 and V2D S3 formulations. Historical
    models remain explicit legacy through :meth:`legacy_compatible`; loading a
    file never infers the current policy when formulation authority is absent.
    """

    q4: str = "e4-pl"
    s3: str = "e4-pl-s3-v2d"
    higher_order: str = "legacy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "q4", _token(self.q4, field="q4", allowed=_Q4))
        object.__setattr__(self, "s3", _token(self.s3, field="s3", allowed=_S3))
        object.__setattr__(
            self,
            "higher_order",
            _token(self.higher_order, field="higher_order", allowed=_HIGHER),
        )

    @classmethod
    def current_default(cls) -> "ShellFormulationPolicy":
        return cls(q4="e4-pl", s3="e4-pl-s3-v2d")

    @classmethod
    def legacy_compatible(cls) -> "ShellFormulationPolicy":
        return cls(q4="legacy", s3="legacy-s3")

    @classmethod
    def qualified_q4_only(cls) -> "ShellFormulationPolicy":
        """Return the transition policy used before V2D S3 activation."""

        return cls(q4="e4-pl", s3="legacy-s3")

    def for_node_count(self, node_count: int) -> str:
        if node_count == 3:
            return self.s3
        if node_count == 4:
            return self.q4
        if node_count in {6, 8}:
            return self.higher_order
        raise ValueError(f"unsupported shell topology with {node_count} nodes")

    def formulation_id_for_node_count(self, node_count: int) -> str:
        token = self.for_node_count(node_count)
        if node_count == 3:
            return (
                S3_V2D_FORMULATION_ID
                if token == "e4-pl-s3-v2d"
                else "LEGACY_SHELL_ELEMENT_TRI3"
            )
        if node_count == 4:
            return (
                QUALIFIED_Q4_FORMULATION_ID
                if token == "e4-pl"
                else "LEGACY_SHELL_ELEMENT_Q4"
            )
        return "LEGACY_SHELL_ELEMENT_HIGHER_ORDER"

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": "anyfem.shell-formulation-policy-v1",
            "q4": self.q4,
            "s3": self.s3,
            "higher_order": self.higher_order,
            "q4_formulation_id": self.formulation_id_for_node_count(4),
            "s3_formulation_id": self.formulation_id_for_node_count(3),
            "higher_order_formulation_id": self.formulation_id_for_node_count(8),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ShellFormulationPolicy":
        if not isinstance(data, Mapping):
            raise ValueError("shell_formulations must be an object")
        expected = {
            "schema",
            "q4",
            "s3",
            "higher_order",
            "q4_formulation_id",
            "s3_formulation_id",
            "higher_order_formulation_id",
        }
        if set(data) != expected:
            raise ValueError(
                "shell_formulations must contain exactly "
                + ", ".join(sorted(expected))
            )
        if data["schema"] != "anyfem.shell-formulation-policy-v1":
            raise ValueError("unsupported shell formulation policy schema")
        made = cls(
            q4=data["q4"],  # type: ignore[arg-type]
            s3=data["s3"],  # type: ignore[arg-type]
            higher_order=data["higher_order"],  # type: ignore[arg-type]
        )
        expected_ids = made.to_dict()
        for key in (
            "q4_formulation_id",
            "s3_formulation_id",
            "higher_order_formulation_id",
        ):
            if data[key] != expected_ids[key]:
                raise ValueError(f"shell formulation identity mismatch for {key}")
        return made
