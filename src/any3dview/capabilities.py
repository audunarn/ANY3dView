"""Read-only backend feature descriptions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ViewerCapabilities:
    gpu: bool = False
    dynamic_arrays: bool = False
    node_scalar_field: bool = False
    element_scalar_field: bool = False
    shader_deformation: bool = False
    active_element_mask: bool = False
    incremental_chunks: bool = False
    integer_picking: bool = False
    through_selection: bool = False
    clipping_planes: bool = False
    transparency: bool = False
    geometry_changeset: bool = False
    software_fallback: bool = False


CORE_CAPABILITIES = ViewerCapabilities(
    dynamic_arrays=True,
    active_element_mask=True,
    incremental_chunks=True,
    through_selection=True,
    clipping_planes=True,
)
