"""Explicit, SG1-compatible quad-first application settings.

The presence of a :class:`QuadMeshingOptions` value is the owner's opt-in
signal.  None continues to mean the historical meshing route.
"""

from __future__ import annotations

from anymesher.quad.options import QuadMeshingOptions


def sg1_quad_options() -> QuadMeshingOptions:
    """Return the tested SG1 option set, including owner resource defaults."""

    return QuadMeshingOptions(
        orientation="cross_4theta",
        quality_model="shape_jacobian",
        line_search="safeguarded",
    )


def effective_quad_options(value: QuadMeshingOptions | dict | None) -> QuadMeshingOptions:
    """Resolve a requested quad-first route, rejecting malformed options."""

    if value is None:
        return sg1_quad_options()
    return QuadMeshingOptions.coerce(value)


def sg1_compatible(value: QuadMeshingOptions) -> bool:
    return value == sg1_quad_options()
