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
    legacy_primitives: bool = False
    text_hud: bool = False
    legends: bool = False
    camera_controls: bool = False
    work_plane_projection: bool = False
    hover_selection: bool = False
    region_selection: bool = False
    lasso_selection: bool = False
    animation: bool = False
    image_capture: bool = False
    line_occlusion: bool = False
    stippled_transparency: bool = False


CORE_CAPABILITIES = ViewerCapabilities(
    dynamic_arrays=True,
    active_element_mask=True,
    incremental_chunks=True,
    through_selection=True,
    clipping_planes=True,
)
