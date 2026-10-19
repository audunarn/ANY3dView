"""Turning a model into draw instructions.

The scene builder produces plain data -- polygons, polylines, markers, each
carrying the tag of the entity it came from.  The viewport executes it against
the 3D canvas.  Keeping the two apart means what gets drawn can be tested
without a display, and it is the reason a picked tag always resolves back to a
geometry entity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TypeVar

import numpy as np
from anygeometry.curves import Straight
from anygeometry.entities import EntityRef
from anygeometry.model import GeometryModel
from anygeometry.operations import surface_point
from anygeometry.surfaces import Plane

from ..mesh.mapped import Mesh
from ..model.project import Project
from ..selection import MeshEntityRef, entity_tag

__all__ = [
    "Arrow",
    "Sphere",
    "FacePatch",
    "PointMarker",
    "Polyline",
    "Scene",
    "RESULT_COLORMAPS",
    "COLOR_IMPERFECTION",
    "build_geometry_scene",
    "build_mesh_scene",
    "build_attribute_overlay",
    "build_imperfection_overlay",
    "build_collision_overlay",
    "build_result_scene",
    "build_persisted_result_scene",
    "entity_sample_points",
    "face_display_polygons",
    "face_normal",
    "GeometryDisplayResolution",
    "geometry_display_resolution",
    "geometry_characteristic_size",
]

# Faces are drawn as a coarse Coons tessellation.  Straight-sided plates need
# only one quad, but curved ones need enough to read as curved.
DISPLAY_DIVISIONS = 8


@dataclass(frozen=True)
class GeometryDisplayResolution:
    """One full and interaction display resolution for model geometry.

    Structural cylinders/cones commonly contain many narrow curved faces.
    Tessellating each such face at the normal 8 x 8 preview resolution turns a
    few hundred engineering panels into tens of thousands of render polygons.
    The resolution is deliberately renderer-neutral so the viewport can swap
    to the interaction scene during pan/orbit without altering model intent.
    """

    divisions: int
    curve_samples: int
    interaction_divisions: int
    interaction_curve_samples: int


def geometry_display_resolution(
    geometry: GeometryModel,
    detail: str = "Auto",
) -> GeometryDisplayResolution:
    """Choose bounded display resolution for the current geometry.

    ``Auto`` preserves the high-quality preview for a small number of curved
    faces, then reduces *per-face* display tessellation as a revolved model
    gains structural segments.  Flat four-sided plates remain one polygon via
    :func:`_flat_four_edge_polygon`, so this specifically addresses curved
    geometry rather than reducing ordinary plate-model readability.
    """

    policies = {
        "Fine": (8, 24, 2, 8),
        "Balanced": (4, 16, 1, 6),
        "Fast": (2, 8, 1, 4),
    }
    try:
        normalized = str(detail).strip().title()
    except AttributeError as error:  # defensive for scripted display settings
        raise ValueError("geometry detail must be text") from error
    if normalized != "Auto":
        try:
            return GeometryDisplayResolution(*policies[normalized])
        except KeyError as error:
            raise ValueError(
                "geometry detail must be Auto, Fine, Balanced or Fast"
            ) from error

    curved_faces = sum(
        1
        for face in geometry.faces.values()
        if face.support_surface is not None
        and not isinstance(face.support_surface, Plane)
    )
    if curved_faces <= 64:
        return GeometryDisplayResolution(*policies["Fine"])
    if curved_faces <= 256:
        return GeometryDisplayResolution(*policies["Balanced"])
    return GeometryDisplayResolution(*policies["Fast"])

COLOR_PLATE = "#7ba7cc"
COLOR_LINE = "#37474f"
COLOR_BEAM = "#c1663a"
COLOR_POINT = "#263238"
COLOR_MESH_FILL = "#93b7c9"
COLOR_MESH_EDGE = "#4a6572"
COLOR_SUPPORT = "#2e7d32"
COLOR_LOAD = "#c62828"
COLOR_IMPERFECTION = "#8e24aa"

RESULT_COLORMAPS: Dict[str, tuple[tuple[float, str], ...]] = {
    "Cool-warm": (
        (0.00, "#3b4cc0"), (0.25, "#7396f5"), (0.50, "#e8e8e8"),
        (0.75, "#f49a7b"), (1.00, "#b40426"),
    ),
    "Viridis": (
        (0.00, "#440154"), (0.25, "#3b528b"), (0.50, "#21918c"),
        (0.75, "#5ec962"), (1.00, "#fde725"),
    ),
    "Plasma": (
        (0.00, "#0d0887"), (0.25, "#7e03a8"), (0.50, "#cc4778"),
        (0.75, "#f89540"), (1.00, "#f0f921"),
    ),
    "Turbo": (
        (0.00, "#30123b"), (0.20, "#466be3"), (0.40, "#1bcfd4"),
        (0.60, "#71fc6a"), (0.80, "#faba39"), (1.00, "#7a0403"),
    ),
    "Grayscale": ((0.00, "#101010"), (1.00, "#f5f5f5")),
}

SceneOwner = EntityRef | MeshEntityRef


@dataclass
class FacePatch:
    """One or more polygons drawn as a single tagged batch."""

    ref: Optional[SceneOwner]
    polygons: List[np.ndarray]
    colors: List[str]
    outline: str = ""
    # A tessellated geometry face has one semantic owner for the whole batch.
    # A mesh patch additionally has one element (and element-face) owner per
    # polygon.  Keeping those bindings as scene data lets ANYtk3D retain one
    # efficient face batch without losing element-level selection.
    owners: tuple[SceneOwner, ...] = ()
    polygon_owners: Optional[List[tuple[SceneOwner, ...]]] = None
    # Per-polygon visibility can differ from collapsed-feature pick ownership.
    polygon_visibility_owners: Optional[List[tuple[SceneOwner, ...]]] = None

    def __post_init__(self) -> None:
        if self.polygon_visibility_owners is not None and len(self.polygon_visibility_owners) != len(self.polygons):
            raise ValueError("polygon_visibility_owners must have one entry per face polygon")
        if self.polygon_owners is not None and len(self.polygon_owners) != len(
            self.polygons
        ):
            raise ValueError(
                "polygon_owners must have one entry per face polygon"
            )

    @property
    def tag(self) -> str:
        return "" if self.ref is None else entity_tag(self.ref)

    def owners_for_polygon(self, index: int) -> tuple[SceneOwner, ...]:
        if self.polygon_owners is not None:
            return self.polygon_owners[index]
        if self.owners:
            return self.owners
        return () if self.ref is None else (self.ref,)


@dataclass
class Polyline:
    """A 3D polyline: a modelled line, a beam, or an annotation."""

    ref: Optional[SceneOwner]
    points: np.ndarray
    color: str = COLOR_LINE
    width: int = 2
    owners: tuple[SceneOwner, ...] = ()
    # Keep structural centre-lines legible when they are coplanar with a
    # shell.  Depth clipping otherwise makes one connected beam appear as
    # disconnected alternating segments.
    draw_overlay: bool = False
    # Display annotations can share geometric owners without being model axes.
    annotation: bool = False

    @property
    def tag(self) -> str:
        return "" if self.ref is None else entity_tag(self.ref)

    @property
    def pick_owners(self) -> tuple[SceneOwner, ...]:
        if self.owners:
            return self.owners
        return () if self.ref is None else (self.ref,)


@dataclass
class PointMarker:
    ref: Optional[SceneOwner]
    position: np.ndarray
    color: str = COLOR_POINT
    size: float = 0.0
    owners: tuple[SceneOwner, ...] = ()

    @property
    def tag(self) -> str:
        return "" if self.ref is None else entity_tag(self.ref)

    @property
    def pick_owners(self) -> tuple[SceneOwner, ...]:
        if self.owners:
            return self.owners
        return () if self.ref is None else (self.ref,)


@dataclass
class Sphere:
    """A solid sphere annotation, for an impacting body."""

    centre: np.ndarray
    radius: float
    color: str = "#455a64"
    opacity: float = 1.0

    @property
    def tag(self) -> str:
        return ""


@dataclass
class Scene:
    """Everything to draw for one view of the model."""

    faces: List[FacePatch] = field(default_factory=list)
    lines: List[Polyline] = field(default_factory=list)
    points: List[PointMarker] = field(default_factory=list)
    arrows: List["Arrow"] = field(default_factory=list)
    spheres: List["Sphere"] = field(default_factory=list)
    legend: Optional[Dict[str, object]] = None
    result_field: object = None
    result_mesh: object = None
    result_colormap: object = None
    result_scalar_range: object = None

    def copy(self) -> "Scene":
        """Return independent draw lists while sharing coordinate payloads.

        Geometry caches reuse large NumPy polygon arrays. Overlays only extend
        the outer lists, so this avoids mutating the cache without duplicating
        the coordinate buffers.
        """

        return Scene(
            faces=list(self.faces),
            lines=list(self.lines),
            points=list(self.points),
            arrows=list(self.arrows),
            spheres=list(self.spheres),
            legend=None if self.legend is None else dict(self.legend),
            result_field=self.result_field, result_mesh=self.result_mesh,
            result_colormap=self.result_colormap, result_scalar_range=self.result_scalar_range,
        )

    def tags(self) -> List[str]:
        return [
            item.tag
            for group in (self.faces, self.lines, self.points)
            for item in group
            if item.tag
        ]

    def merge(self, other: "Scene") -> "Scene":
        """Draw another scene on top of this one."""

        self.faces.extend(other.faces)
        self.lines.extend(other.lines)
        self.points.extend(other.points)
        self.arrows.extend(other.arrows)
        self.spheres.extend(other.spheres)
        if other.legend is not None:
            self.legend = other.legend
        return self

    def bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Axis-aligned bounds of everything in the scene."""

        low: Optional[np.ndarray] = None
        high: Optional[np.ndarray] = None

        def include(values) -> None:
            nonlocal low, high
            try:
                array = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                for value in values:
                    include(value)
                return
            if array.size == 0:
                return
            points = array.reshape(-1, 3)
            local_low = points.min(axis=0)
            local_high = points.max(axis=0)
            low = local_low if low is None else np.minimum(low, local_low)
            high = local_high if high is None else np.maximum(high, local_high)

        for patch in self.faces:
            include(patch.polygons)
        for line in self.lines:
            include(line.points)
        for marker in self.points:
            include(marker.position)
        for arrow in self.arrows:
            include(arrow.start)
            include(arrow.end)
        for sphere in self.spheres:
            offset = np.full(3, sphere.radius)
            include(sphere.centre - offset)
            include(sphere.centre + offset)
        if low is None or high is None:
            return None
        return low, high

    def characteristic_size(self) -> float:
        """A length to scale markers and symbols against."""

        extent = self.bounds()
        if extent is None:
            return 1.0
        span = float(np.linalg.norm(extent[1] - extent[0]))
        return span if span > 0.0 else 1.0


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def _face_grid(
    geometry: GeometryModel,
    face_id: int,
    parameters: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Evaluate one square UV grid through ANYgeometry's batch API."""

    values = np.asarray(parameters, dtype=float)
    u, v = np.meshgrid(values, values, indexing="ij")
    uv = np.column_stack((u.reshape(-1), v.reshape(-1)))
    return geometry.evaluate_face_many(face_id, uv).reshape(
        (len(values), len(values), 3)
    )


def face_display_polygons(
    geometry: GeometryModel, face_id: int, divisions: int = DISPLAY_DIVISIONS
) -> List[np.ndarray]:
    """Tessellate a face for display, using the same blend the mesher uses.

    A plate is therefore never drawn as a different surface from the one that
    gets meshed.
    """

    face = geometry.faces[face_id]
    if (
        isinstance(face.surface, Plane)
        and face.parameterization is None
        and not face.holes
        and len(face.loop) == 3
        and all(isinstance(geometry.edges[item.edge].curve, Straight) for item in face.loop)
    ):
        # A Plane describes the support beyond the face boundary.  Sampling
        # its square UV domain would draw the missing half of this triangle.
        return [
            np.asarray([
                geometry.vertex_position(geometry.oriented_start_vertex(item))
                for item in face.loop
            ])
        ]
    if not face.corners and face.surface is None and face.parameterization is None and not face.holes and len(face.loop) == 4 and all(isinstance(geometry.edges[item.edge].curve, Straight) for item in face.loop):
        from shapely.geometry import Polygon
        from shapely.ops import triangulate
        points = np.asarray([geometry.vertex_position(geometry.oriented_start_vertex(item)) for item in face.loop])
        origin = points[0]
        _, _, axes = np.linalg.svd(points - origin)
        scale = max(float(np.linalg.norm(points - origin, axis=1).max()), 1e-12)
        if float(np.abs((points - origin) @ axes[2]).max()) > 1e-9 * scale:
            raise ValueError(f"neutral face {face_id} boundary is not planar")
        projected = (points - origin) @ axes[:2].T
        polygon = Polygon(projected)
        boundary_vertices = {tuple(uv): point for uv, point in zip(projected, points)}
        if not polygon.is_valid or polygon.area <= 1e-12 * scale * scale:
            raise ValueError(f"neutral face {face_id} has an invalid display boundary")
        polygons = []
        normal = np.sum(np.cross(points - origin, np.roll(points - origin, -1, axis=0)), axis=0)
        for triangle in triangulate(polygon):
            if not polygon.covers(triangle):continue
            # Delaunay uses input vertices. Preserve their authoritative XYZ
            # exactly instead of introducing inverse-projection roundoff.
            vertices = np.asarray([boundary_vertices[tuple(uv)] for uv in triangle.exterior.coords[:-1]])
            if np.dot(np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]),normal) < 0:vertices=vertices[::-1]
            polygons.append(vertices)
        return polygons
    if face.holes:
        # ANYgeometry owns the trim loops; Shapely supplies a constrained
        # planar triangulation which is then mapped back to the authoritative
        # structural surface.  Filtering with ``covers`` prevents Delaunay
        # triangles from bridging a hole or concave outer trim.
        from shapely.geometry import Polygon
        from shapely.ops import triangulate

        loops = geometry.face_trim_loops_uv(
            face_id, curve_samples=max(17, divisions * 4 + 1)
        )
        trimmed = Polygon(loops[0], holes=loops[1:])
        if not trimmed.is_valid or trimmed.is_empty:
            raise ValueError(f"face {face_id} has an invalid display trim")
        polygons = []
        for triangle in triangulate(trimmed):
            if not trimmed.covers(triangle):
                continue
            uv = list(triangle.exterior.coords)[:-1]
            polygons.append(
                np.asarray(
                    [
                        surface_point(geometry, face, float(u), float(v))
                        for u, v in uv
                    ]
                )
            )
        return polygons

    parameters = np.linspace(0.0, 1.0, divisions + 1)
    grid = _face_grid(geometry, face_id, parameters)

    polygons: List[np.ndarray] = []
    for i in range(divisions):
        for j in range(divisions):
            polygons.append(
                np.array(
                    [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
                )
            )
    return polygons


def _flat_four_edge_polygon(
    geometry: GeometryModel, face_id: int
) -> Optional[np.ndarray]:
    """Return one authoritative display quad, or require tessellation."""

    face = geometry.faces[face_id]
    if face.holes:
        return None
    neutral_quad = not face.corners and len(face.loop) == 4 and face.surface is None and face.parameterization is None
    if len(face.corners) != 4 and not neutral_quad:
        return None
    sides = tuple((item,) for item in face.loop) if neutral_quad else face.sides()
    if any(len(side) != 1 for side in sides):
        return None
    if any(
        not isinstance(geometry.edges[side[0].edge].curve, Straight)
        for side in sides
    ):
        return None

    # Straight topology does not prove a flat explicit surface (a ruled
    # surface can have straight carrier edges).  Sample through ANYgeometry so
    # the GUI fast path cannot flatten a surface the geometry owner keeps
    # curved.
    parameters = (0.0, 0.5, 1.0)
    sampled = np.asarray(
        [
            geometry.face_point(face_id, u, v)
            for u in parameters
            for v in parameters
        ]
    ) if not neutral_quad else np.asarray([
        geometry.vertex_position(geometry.oriented_start_vertex(item)) for item in face.loop
    ])
    corners = sampled if neutral_quad else sampled[[0, 6, 8, 2]]
    first = corners[1] - corners[0]
    second = corners[2] - corners[0]
    normal = np.cross(first, second)
    normal_length = float(np.linalg.norm(normal))
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        float(np.linalg.norm(corners[3] - corners[0])),
        1.0e-12,
    )
    if normal_length <= 1.0e-12 * scale * scale:
        return None
    distances = np.abs((sampled - corners[0]) @ normal) / normal_length
    if neutral_quad and any(np.dot(np.cross(corners[(index+1)%4]-corners[index],corners[(index+2)%4]-corners[(index+1)%4]),normal) < -1e-12 * scale**4 for index in range(4)):
        return None
    return corners if float(np.max(distances)) <= 1.0e-9 * scale else None


def _beam_section_polygons(section, points, reference_z) -> List[np.ndarray]:
    """Sweep the actual idealized section rectangles along a beam polyline."""

    coordinates = np.asarray(points, dtype=float)
    reference_z = np.asarray(reference_z, dtype=float)
    polygons: List[np.ndarray] = []
    for start, end in zip(coordinates, coordinates[1:]):
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length <= 0.0:
            continue
        axis /= length
        local_z = reference_z - float(reference_z @ axis) * axis
        z_length = float(np.linalg.norm(local_z))
        if z_length <= 1.0e-12:
            fallback = np.eye(3)[int(np.argmin(np.abs(np.eye(3) @ axis)))]
            local_z = fallback - float(fallback @ axis) * axis
            z_length = float(np.linalg.norm(local_z))
        local_z /= z_length
        local_y = np.cross(local_z, axis)
        local_y /= float(np.linalg.norm(local_y))
        for y, z, width, height in section.centered_profile_rectangles():
            if width <= 0.0 or height <= 0.0:
                continue
            offsets = (
                (y - width / 2.0) * local_y + (z - height / 2.0) * local_z,
                (y + width / 2.0) * local_y + (z - height / 2.0) * local_z,
                (y + width / 2.0) * local_y + (z + height / 2.0) * local_z,
                (y - width / 2.0) * local_y + (z + height / 2.0) * local_z,
            )
            first = [start + offset for offset in offsets]
            last = [end + offset for offset in offsets]
            polygons.extend(
                (
                    np.asarray(first),
                    np.asarray(last[::-1]),
                    np.asarray((first[0], first[1], last[1], last[0])),
                    np.asarray((first[1], first[2], last[2], last[1])),
                    np.asarray((first[2], first[3], last[3], last[2])),
                    np.asarray((first[3], first[0], last[0], last[3])),
                )
            )
    return polygons


def _add_beam_section_patch(
    scene: Scene,
    section,
    points,
    reference_z,
    *,
    color: str,
    ref: Optional[SceneOwner],
    owners: tuple[SceneOwner, ...] = (),
) -> bool:
    polygons = _beam_section_polygons(section, points, reference_z)
    if not polygons:
        return False
    scene.faces.append(
        FacePatch(
            ref=ref,
            polygons=polygons,
            colors=[color] * len(polygons),
            outline=COLOR_MESH_EDGE,
            owners=owners,
        )
    )
    return True


def build_geometry_scene(
    project: Project,
    *,
    divisions: int = DISPLAY_DIVISIONS,
    curve_samples: int = 24,
    show_points: bool = True,
    show_beam_sections: bool = True,
    entity_owners: Optional[Mapping[EntityRef, int]] = None,
    exposed_feature_ids: Iterable[int] = (),
) -> Scene:
    """Draw user geometry plus the surfaces of generated features.

    Generator topology remains exact in the geometry model.  Its auxiliary
    points and lines are presentation details and are omitted until the owning
    feature is explicitly exploded.  Assigned beam edges remain visible
    because they represent structural members rather than construction lines.
    """

    geometry = project.geometry
    scene = Scene()
    if entity_owners is None:
        from anygeometry import feature_entity_owners

        entity_owners = feature_entity_owners(geometry)
    exposed = {int(identifier) for identifier in exposed_feature_ids}

    collapsed_face_patches: dict[int, FacePatch] = {}
    collapsed_face_owners: dict[int, list[EntityRef]] = {}
    for face_id in sorted(geometry.faces):
        face_ref = EntityRef("face", face_id)
        feature_id = entity_owners.get(face_ref)
        collapsed_feature = feature_id is not None and feature_id not in exposed
        flat_polygon = _flat_four_edge_polygon(geometry, face_id)
        polygons = (
            [flat_polygon]
            if flat_polygon is not None
            else face_display_polygons(
                geometry,
                face_id,
                min(divisions, 2) if collapsed_feature else divisions,
            )
        )
        if collapsed_feature:
            # A collapsed generator is one semantic surface.  Retain all exact
            # face owners on that surface so selecting it applies to the
            # complete cylinder/cone/panel; Explode restores per-face picking.
            patch = collapsed_face_patches.get(feature_id)
            if patch is None:
                patch = FacePatch(
                    ref=face_ref,
                    polygons=[],
                    colors=[],
                    outline="",
                    polygon_visibility_owners=[],
                )
                collapsed_face_patches[feature_id] = patch
                collapsed_face_owners[feature_id] = []
                scene.faces.append(patch)
            patch.polygons.extend(polygons)
            patch.colors.extend([COLOR_PLATE] * len(polygons))
            patch.polygon_visibility_owners.extend([(face_ref,)] * len(polygons))
            collapsed_face_owners[feature_id].append(face_ref)
            continue
        scene.faces.append(
            FacePatch(
                ref=face_ref,
                polygons=polygons,
                colors=[COLOR_PLATE] * len(polygons),
                outline="",
            )
        )
    for feature_id, patch in collapsed_face_patches.items():
        patch.owners = tuple(collapsed_face_owners[feature_id])

    for edge_id in sorted(geometry.edges):
        is_beam = edge_id in project.edge_sections
        feature_id = entity_owners.get(EntityRef("edge", edge_id))
        if feature_id is not None and feature_id not in exposed and not is_beam:
            continue
        samples = (
            2
            if isinstance(geometry.edges[edge_id].curve, Straight)
            else curve_samples
        )
        points = geometry.sample_edge(
            edge_id, np.linspace(0.0, 1.0, samples)
        )
        rendered_section = False
        if is_beam and show_beam_sections:
            section = project.beam_section_of(edge_id)
            _axis, _local_y, local_z = project.beam_section_frame(edge_id)
            points = points + np.asarray(project.beam_offset_vector(edge_id))
            rendered_section = _add_beam_section_patch(
                scene,
                section,
                points,
                local_z,
                color=COLOR_BEAM,
                ref=EntityRef("edge", edge_id),
            )
        scene.lines.append(
            Polyline(
                ref=EntityRef("edge", edge_id),
                points=points,
                color=COLOR_BEAM if is_beam else COLOR_LINE,
                width=(3 if rendered_section else 4) if is_beam else 2,
                draw_overlay=is_beam,
            )
        )

    if show_points:
        from ..presentation.points import displayed_geometry_point_ids
        for vertex_id in sorted(displayed_geometry_point_ids(geometry)):
            feature_id = entity_owners.get(EntityRef("vertex", vertex_id))
            if feature_id is not None and feature_id not in exposed:
                continue
            scene.points.append(
                PointMarker(
                    ref=EntityRef("vertex", vertex_id),
                    position=geometry.vertex_position(vertex_id),
                )
            )

    return scene


# ----------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------
def build_mesh_scene(
    project: Project, mesh: Mesh, *, show_beam_sections: bool = True
) -> Scene:
    """Draw a retained mesh with geometry and FE ownership in one scene.

    Shells remain batched by their owning geometry plate for rendering and
    legacy tag picking.  Each polygon nevertheless carries its own mesh
    element and element-face references for the commercial selection index.
    Nodes are emitted as marker data and the viewport submits all of them via
    one :meth:`ANYtk3D.add_markers` call.
    """

    scene = Scene()
    shells = mesh.shells
    grouped_shells: set[int] = set()

    for face_id, element_ids in sorted(mesh.elements_of_face.items()):
        available = [
            int(element_id)
            for element_id in element_ids
            if int(element_id) in shells
        ]
        polygons = [
            np.array([mesh.nodes[node] for node in mesh.corners_of(element_id)])
            for element_id in available
        ]
        if not polygons:
            continue
        grouped_shells.update(available)
        geometry_ref = (
            EntityRef("face", face_id)
            if face_id in project.geometry.faces
            else None
        )
        polygon_owners = [
            tuple(
                owner
                for owner in (
                    geometry_ref,
                    MeshEntityRef("element", element_id),
                    MeshEntityRef("element_face", (element_id, 0)),
                )
                if owner is not None
            )
            for element_id in available
        ]
        scene.faces.append(
            FacePatch(
                ref=geometry_ref,
                polygons=polygons,
                colors=[COLOR_MESH_FILL] * len(polygons),
                outline=COLOR_MESH_EDGE,
                polygon_owners=polygon_owners,
            )
        )

    # Imported meshes are allowed to have incomplete geometry associations.
    # They must still be visible and selectable directly as mesh entities.
    ungrouped_shells = sorted(set(shells) - grouped_shells)
    if ungrouped_shells:
        scene.faces.append(
            FacePatch(
                ref=None,
                polygons=[
                    np.array(
                        [mesh.nodes[node] for node in mesh.corners_of(element_id)]
                    )
                    for element_id in ungrouped_shells
                ],
                colors=[COLOR_MESH_FILL] * len(ungrouped_shells),
                outline=COLOR_MESH_EDGE,
                polygon_owners=[
                    (
                        MeshEntityRef("element", element_id),
                        MeshEntityRef("element_face", (element_id, 0)),
                    )
                    for element_id in ungrouped_shells
                ],
            )
        )

    grouped_beams: set[int] = set()
    for edge_id, element_ids in sorted(mesh.elements_of_edge.items()):
        for element_id in element_ids:
            if element_id not in mesh.beams:
                continue
            grouped_beams.add(int(element_id))
            # Through every node, so a quadratic beam draws its curvature.
            beam_nodes = mesh.beams[element_id]
            geometry_ref = (
                EntityRef("edge", edge_id)
                if edge_id in project.geometry.edges
                else None
            )
            owners = tuple(
                owner
                for owner in (
                    geometry_ref,
                    MeshEntityRef("element", int(element_id)),
                )
                if owner is not None
            )
            rendered_section = False
            if show_beam_sections and edge_id in project.edge_sections:
                section = project.beam_section_of(edge_id)
                _axis, _local_y, local_z = project.beam_section_frame(edge_id)
                rendered_section = _add_beam_section_patch(
                    scene,
                    section,
                    np.asarray([mesh.nodes[node] for node in beam_nodes]),
                    local_z,
                    color=COLOR_BEAM,
                    ref=geometry_ref,
                    owners=owners,
                )
            scene.lines.append(
                Polyline(
                    ref=geometry_ref,
                    points=np.array([mesh.nodes[node] for node in beam_nodes]),
                    color=COLOR_BEAM,
                    width=1 if rendered_section else 4,
                    draw_overlay=True,
                    owners=owners,
                )
            )

    for element_id in sorted(set(mesh.beams) - grouped_beams):
        span = mesh.beams[element_id]
        scene.lines.append(
            Polyline(
                ref=None,
                points=np.array([mesh.nodes[node] for node in span]),
                color=COLOR_BEAM,
                width=4,
                draw_overlay=True,
                owners=(MeshEntityRef("element", int(element_id)),),
            )
        )

    vertex_of_node = {
        int(node_id): int(vertex_id)
        for vertex_id, node_id in mesh.node_of_vertex.items()
        if vertex_id in project.geometry.vertices
    }
    from ..presentation.points import USER_POINT_GROUP, displayed_geometry_point_ids
    point_ids = displayed_geometry_point_ids(project.geometry) if USER_POINT_GROUP in project.geometry.groups else None
    for node_id in sorted(mesh.nodes):
        vertex_id = vertex_of_node.get(int(node_id))
        # An explicit user-point policy also omits generated FE-node markers;
        # the mesh elements remain rendered and selectable as before.
        if point_ids is not None and vertex_id not in point_ids:
            continue
        geometry_ref = (
            None if vertex_id is None else EntityRef("vertex", vertex_id)
        )
        mesh_ref = MeshEntityRef("node", int(node_id))
        scene.points.append(
            PointMarker(
                ref=mesh_ref,
                position=np.asarray(mesh.nodes[node_id], dtype=float),
                owners=tuple(
                    owner
                    for owner in (geometry_ref, mesh_ref)
                    if owner is not None
                ),
            )
        )

    return scene


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------
def build_result_scene(
    shape,
    *,
    component: str = "magnitude",
    field: Optional[str] = None,
    scale: float = 1.0,
    colormap: Optional[Sequence[tuple[float, str]]] = None,
    limits: Optional[tuple[float, float]] = None,
    values: Optional[object] = None,
    display_units: str = "SI (m / Pa)",
    show_nodes: bool = False,
    show_beam_sections: bool = True,
) -> Scene:
    """Draw the deformed shape, coloured by any field.

    ``field`` names what to colour by -- a displacement component or a stress
    component.  A per-node field is averaged onto each element for shading; a
    per-element field such as von Mises colours each element with its own
    value, which is where it actually lives.  ``values`` accepts a
    pre-computed :class:`~anyfem.post.fields.Field`, so an envelope can be
    drawn the same way a single shape is.
    """

    from ..post.fields import Field, evaluate_field, field_unit

    mesh = shape.built.mesh
    name = field or component
    if values is None:
        resolved = evaluate_field(shape, name)
    elif isinstance(values, Field):
        resolved = values
        name = resolved.name
    else:
        raise TypeError("values must be a Field")

    ordered_nodes = sorted(mesh.nodes)
    try:
        positions = shape.deformed_positions(scale)
    except KeyError:
        # Imported stress-only files have no translation field.  They remain
        # valid contour results and are displayed on the undeformed mesh.
        positions = mesh.node_positions()
    deformed = {node: positions[index] for index, node in enumerate(ordered_nodes)}
    # Lightweight imported/test result adapters historically expose only a
    # mesh on ``built``.  Geometry ownership is additive: keep those adapters
    # renderable and simply omit a geometry PickOwner when no project exists.
    built_project = getattr(shape.built, "project", None)
    geometry = getattr(built_project, "geometry", None)

    from .result_display import unit_transform

    value_scale, display_unit = unit_transform(resolved.unit, display_units)
    finite_values = [
        float(value)
        for value in resolved.values.values()
        if np.isfinite(float(value))
    ]
    has_values = bool(finite_values)
    if limits is None:
        if has_values:
            raw_low, raw_high = min(finite_values), max(finite_values)
            if raw_high <= raw_low:
                raw_high = raw_low + 1.0
        else:
            raw_low, raw_high = 0.0, 1.0
        low, high = raw_low * value_scale, raw_high * value_scale
    else:
        low, high = limits
    if high <= low:
        high = low + 1.0
    span = high - low

    def colour(value: float) -> str:
        return _ramp_color((value * value_scale - low) / span, colormap)

    def nodal_average(node_ids: Sequence[int]) -> Optional[float]:
        """Average only a complete finite nodal value set.

        A result quantity may deliberately cover only part of a mixed mesh.
        Missing values are unavailable, not zero, and must not make result
        browsing fail merely because an adjacent element uses another result
        location or formulation.
        """

        samples = []
        for node_id in node_ids:
            value = resolved.node_values.get(int(node_id))
            if value is None or not np.isfinite(float(value)):
                return None
            samples.append(float(value))
        return float(np.mean(samples)) if samples else None

    scene = Scene()
    shells = mesh.shells
    for face_id, element_ids in sorted(mesh.elements_of_face.items()):
        polygons: List[np.ndarray] = []
        colors: List[str] = []
        rendered_ids: List[int] = []
        for element_id in element_ids:
            if element_id not in shells:
                continue
            nodes = shells[element_id]
            polygons.append(
                np.array(
                    [deformed[node] for node in mesh.corners_of(element_id)]
                )
            )
            rendered_ids.append(int(element_id))
            if resolved.per_element:
                value = resolved.element_values.get(element_id)
            else:
                value = nodal_average(nodes)
            colors.append(
                COLOR_MESH_FILL
                if value is None or not np.isfinite(float(value))
                else colour(float(value))
            )
        if polygons:
            geometry_ref = (
                EntityRef("face", face_id)
                if geometry is not None and face_id in geometry.faces
                else None
            )
            scene.faces.append(
                FacePatch(
                    ref=geometry_ref,
                    polygons=polygons,
                    colors=colors,
                    outline="",
                    polygon_owners=[
                        tuple(
                            owner
                            for owner in (
                                geometry_ref,
                                MeshEntityRef("element", element_id),
                                MeshEntityRef(
                                    "element_face", (element_id, 0)
                                ),
                            )
                            if owner is not None
                        )
                        for element_id in rendered_ids
                    ],
                )
            )

    for edge_id, element_ids in sorted(mesh.elements_of_edge.items()):
        for element_id in element_ids:
            if element_id not in mesh.beams:
                continue
            beam_nodes = mesh.beams[element_id]
            if resolved.per_element:
                value = resolved.element_values.get(element_id)
                line_colour = (
                    COLOR_BEAM
                    if value is None or not np.isfinite(float(value))
                    else colour(float(value))
                )
            else:
                value = nodal_average(beam_nodes)
                line_colour = (
                    COLOR_BEAM if value is None else colour(float(value))
                )
            geometry_ref = (
                EntityRef("edge", edge_id)
                if geometry is not None and edge_id in geometry.edges
                else None
            )
            owners = tuple(
                owner
                for owner in (
                    geometry_ref,
                    MeshEntityRef("element", int(element_id)),
                )
                if owner is not None
            )
            rendered_section = False
            if (
                show_beam_sections
                and built_project is not None
                and edge_id in built_project.edge_sections
            ):
                section = built_project.beam_section_of(edge_id)
                _axis, _local_y, local_z = built_project.beam_section_frame(edge_id)
                rendered_section = _add_beam_section_patch(
                    scene,
                    section,
                    np.asarray([deformed[node] for node in beam_nodes]),
                    local_z,
                    color=line_colour,
                    ref=geometry_ref,
                    owners=owners,
                )
            scene.lines.append(
                Polyline(
                    ref=geometry_ref,
                    points=np.array([deformed[node] for node in beam_nodes]),
                    color=line_colour,
                    width=1 if rendered_section else 4,
                    draw_overlay=True,
                    owners=owners,
                )
            )

    if show_nodes:
        scene.points.extend(
            PointMarker(
                ref=MeshEntityRef("node", int(node)),
                position=np.asarray(deformed[node], dtype=float),
                color=COLOR_POINT,
                owners=(MeshEntityRef("node", int(node)),),
            )
            for node in ordered_nodes
        )

    unit = display_unit or field_unit(name)
    legend_colors = (
        [colour(float(value) / value_scale) for value in np.linspace(low, high, 5)]
        if has_values
        else []
    )
    scene.legend = {
        "levels": list(np.linspace(low, high, 5)) if has_values else [],
        "unit": unit,
        "title": (
            _component_title(name)
            if has_values
            else f"{_component_title(name)} (unavailable)"
        ),
        "colors": legend_colors,
    }
    scene.result_field, scene.result_mesh = resolved, mesh
    if not resolved.per_element:
        scene.legend["title"] += " · element mean for display"
    scene.result_colormap = tuple(colormap or _DEFAULT_RAMP)
    scene.result_scalar_range = (low / value_scale, high / value_scale)
    return scene


def read_persisted_display_field(dataset,field_key,*,frame=0,component=None):
    from ..post.fields import Field
    stored = dataset.field(field_key)
    descriptor = stored.descriptor
    local_frame=_persisted_frame(stored,frame)
    if component is not None and component not in descriptor.components:
        raise ValueError(f"{descriptor.label} has no saved component {component!r}")
    if component is not None:
        values = stored.read_component(component,local_frame)
        scalar = _artifact_scalar_rows(values,(),None,descriptor.provenance.get("scalar_sample_axes"))
    else:
        values = stored.read(local_frame)
        scalar = _artifact_scalar_rows(values,descriptor.components,None,descriptor.provenance.get("scalar_sample_axes"))
    location = descriptor.location
    if location not in ("node", "element", "integration_point"):
        raise ValueError(f"{descriptor.label} is not a spatial contour quantity")
    table_kind = "node" if location == "node" else "element"
    table_key = f"{field_key}_{table_kind}_ids"
    if table_key not in dataset.table_keys:
        raise ValueError(f"{descriptor.label} has no persisted {table_kind} association")
    identifiers = np.asarray(dataset.table(table_key), dtype=int).reshape(-1)
    if len(scalar) != len(identifiers):
        raise ValueError(
            f"{descriptor.label} has {len(scalar)} values for {len(identifiers)} IDs"
        )
    mapping = {
        int(identifier): float(value)
        for identifier, value in zip(identifiers, scalar)
        if np.isfinite(value)
    }
    field = Field(
        name=field_key,
        unit=descriptor.unit,
        node_values=mapping if table_kind == "node" else {},
        element_values=mapping if table_kind == "element" else {},
        reduction="absolute single component" if component is None and len(descriptor.components)==1 and values.ndim>1 else descriptor.reduction,
    )

    return field,descriptor


def build_persisted_result_scene(
    project: Project,
    mesh: Mesh,
    dataset,
    field_key: str,
    *,
    frame: int = 0,
    component: str | None = None,
    scale: float = 1.0,
    limits: Optional[tuple[float, float]] = None,
    colormap: Optional[Sequence[tuple[float, str]]] = None,
    display_units: str = "SI (m / Pa)",
    show_nodes: bool = False,
    show_beam_sections: bool = True,
) -> Scene:
    """Render one lazily-read artifact field without inventing quantities."""

    from ..post.fields import Field

    field,descriptor = read_persisted_display_field(dataset,field_key,frame=frame,component=component)
    ordered_nodes = sorted(mesh.nodes)
    positions = np.asarray([mesh.nodes[node] for node in ordered_nodes], dtype=float)
    if "displacement" in dataset.field_keys and scale != 0.0:
        displacement = dataset.field("displacement")
        displacement_key = "displacement_node_ids"
        if displacement_key in dataset.table_keys:
            displacement_values = np.asarray(
                displacement.read(_persisted_frame(displacement,frame)),
                dtype=float,
            )
            displacement_ids = np.asarray(
                dataset.table(displacement_key), dtype=int
            ).reshape(-1)
            labels=tuple(displacement.descriptor.components)
            if displacement_values.ndim >= 2 and {"ux","uy","uz"}<=set(labels):
                translation_indices=[labels.index(label) for label in ("ux","uy","uz")]
                by_node = {
                    int(node): np.take(row,translation_indices,axis=-1).reshape(-1, 3)[0]
                    for node, row in zip(displacement_ids, displacement_values)
                }
                positions = np.asarray(
                    [
                        mesh.nodes[node]
                        + scale * by_node.get(int(node), np.zeros(3))
                        for node in ordered_nodes
                    ],
                    dtype=float,
                )

    shape = SimpleNamespace(
        built=SimpleNamespace(mesh=mesh, project=project),
        deformed_positions=lambda _scale=1.0: positions,
    )
    scene = build_result_scene(
        shape,
        field=field_key,
        scale=1.0,
        limits=limits,
        values=field,
        colormap=colormap,
        display_units=display_units,
        show_nodes=show_nodes,
        show_beam_sections=show_beam_sections,
    )
    _,_,scene.legend=persisted_display_metadata(field,descriptor,component,limits=limits,display_units=display_units,colormap=colormap)
    return scene


def _artifact_frame(shape: Sequence[int], frame: int) -> int | None:
    if not shape:
        return None
    return min(max(int(frame), 0), int(shape[0]) - 1)

def _persisted_frame(stored,frame):
    resolver=getattr(stored,"root_frame_index",None)
    return resolver(frame) if callable(resolver) else _artifact_frame(stored.shape,frame)


def _artifact_scalar_rows(values, components, component, sample_axes=None):
    from anypost import vector_magnitude, absolute_sample_max
    array = np.asarray(values, dtype=float)
    if array.ndim <= 1:
        return array.reshape(-1)
    if component and component in components:
        array = np.take(array, components.index(component), axis=-1)
    elif components and array.shape[-1] == len(components):
        labels=("ux","uy","uz") if {"ux","uy","uz"}<=set(components) else tuple(components)
        indices=[components.index(label) for label in labels]
        array = vector_magnitude(np.take(array,indices,axis=-1), components=labels)
    return absolute_sample_max(array,nan_policy="omit") if array.ndim > 1 else array


def result_display_metadata(field, *, limits=None, display_units="SI (m / Pa)", colormap=None):
    """Legend/range only; no geometry or owner reconstruction."""
    from .result_display import unit_transform
    factor, unit = unit_transform(field.unit, display_units)
    values = np.asarray(list(field.values.values()),dtype=float)
    finite = values[np.isfinite(values)]
    if limits is None:
        raw_low,raw_high=(float(finite.min()),float(finite.max())) if len(finite) else (0.,1.)
        if raw_high<=raw_low:raw_high=raw_low+1.
        low,high=raw_low*factor,raw_high*factor
    else:low,high=limits
    if high <= low:high=low+1.
    palette=tuple(colormap or _DEFAULT_RAMP)
    levels=list(np.linspace(low,high,5)) if len(finite) else []
    colors=[_ramp_color((v-low)/(high-low),palette) for v in levels]
    title=_component_title(field.name)+( " (unavailable)" if not len(finite) else "")
    if not field.per_element:title+=" · element mean for display"
    return (low/factor,high/factor),palette,dict(levels=levels,colors=colors,unit=unit or field.unit,title=title)

def persisted_display_metadata(field,descriptor,component=None,**options):
    range_,palette,legend=result_display_metadata(field,**options)
    suffix=f" · {component}" if component is not None else " · absolute scalar" if field.reduction=="absolute single component" else ""
    unavailable=" (unavailable)" if not any(np.isfinite(v) for v in field.values.values()) else ""
    legend["title"]=descriptor.label+suffix+unavailable
    if not field.per_element:legend["title"]+=" · element mean for display"
    return range_,palette,legend


def _nodal_values(solution, component: str) -> np.ndarray:
    if component == "magnitude":
        return np.linalg.norm(solution.translations(), axis=1)
    return np.abs(solution.component(component))


_FIELD_TITLES = {
    "magnitude": "Displacement",
    "ux": "Displacement X",
    "uy": "Displacement Y",
    "uz": "Displacement Z",
    "rx": "Rotation X",
    "ry": "Rotation Y",
    "rz": "Rotation Z",
    "von_mises": "von Mises",
    "equivalent_plastic_strain": "Equivalent plastic strain (PEEQ)",
}


def _component_title(name: str) -> str:
    if name in _FIELD_TITLES:
        return _FIELD_TITLES[name]
    return name.replace("_", " ")


_DEFAULT_RAMP = RESULT_COLORMAPS["Cool-warm"]


def _ramp_color(
    position: float, colormap: Optional[Sequence[tuple[float, str]]] = None
) -> str:
    stops = tuple(colormap) if colormap else _DEFAULT_RAMP
    value = min(max(float(position), 0.0), 1.0)
    for (low, low_color), (high, high_color) in zip(stops, stops[1:]):
        if value <= high:
            span = high - low
            local = 0.0 if span <= 0.0 else (value - low) / span
            return _blend(low_color, high_color, local)
    return stops[-1][1]


def _blend(first: str, second: str, amount: float) -> str:
    start = np.array([int(first[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    end = np.array([int(second[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    mixed = np.clip(start + (end - start) * amount, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*mixed)


# ----------------------------------------------------------------------
# loads and supports
# ----------------------------------------------------------------------
COLOR_MASS = "#6a1b9a"
COLOR_PRESSURE = "#1565c0"
COLOR_WET = "#0288d1"
# Sample points per side across a wet plate before a mesh exists.
_WET_GRID = 5
COLOR_MOMENT = "#ef6c00"
COLOR_ROTATION = "#00838f"

# Arrows are drawn at a fraction of the model size, not to scale with the
# load: a 10 kN arrow and a 10 MN arrow should both be readable.
ARROW_FRACTION = 0.10
SYMBOL_FRACTION = 0.022

# How many points along a line or across a plate carry a symbol.  Enough to
# read as "all along here", few enough not to bury the model.
_EDGE_SAMPLES = 3
_FACE_SAMPLES = 2

# Dense assignments stay visible without creating thousands of canvas items.
# Each attribute kind gets its own budget, so pressure cannot hide point loads
# or supports merely because it happens to be more numerous.
OVERLAY_SYMBOL_LIMIT = 256

_T = TypeVar("_T")


@dataclass
class Arrow:
    """An annotation arrow.  Never tagged, so it cannot be picked."""

    start: np.ndarray
    end: np.ndarray
    color: str = COLOR_LOAD

    @property
    def tag(self) -> str:
        return ""


def _limited(items: Sequence[_T], limit: int = OVERLAY_SYMBOL_LIMIT) -> List[_T]:
    """Keep a deterministic, evenly distributed subset of dense symbols."""

    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[0]] if limit == 1 else []
    last = len(items) - 1
    return [items[index * last // (limit - 1)] for index in range(limit)]


def _entity_symbol_samples(ref: EntityRef) -> int:
    """Number of placement points produced for one entity, without sampling."""

    return {
        "vertex": 1,
        "edge": _EDGE_SAMPLES,
        "face": _FACE_SAMPLES * _FACE_SAMPLES,
    }.get(ref.kind, 0)


def _budgeted_assignments(items: Sequence[_T]) -> List[_T]:
    """Distribute a symbol budget across assignments before sampling geometry.

    ``entity_sample_points`` is substantially more expensive for a curved face
    than constructing an Arrow.  Limiting the finished Arrow list therefore
    comes too late: hundreds of discarded face samples have already been
    evaluated.  Every model attribute passed here has a geometry ``ref``.
    Using the largest sample count in a mixed list keeps placement work within
    the same budget while ``_limited`` retains the first, last and evenly
    distributed assignments.
    """

    if not items:
        return []
    samples = max(_entity_symbol_samples(item.ref) for item in items)
    if samples <= 0:
        return []
    return _limited(items, max(1, OVERLAY_SYMBOL_LIMIT // samples))


def _geometry_bounds(
    geometry: GeometryModel,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not geometry.vertices:
        return None
    positions = np.array(
        [
            geometry.vertex_position(vertex_id)
            for vertex_id in sorted(geometry.vertices)
        ],
        dtype=float,
    )
    return positions.min(axis=0), positions.max(axis=0)


def geometry_characteristic_size(geometry: GeometryModel) -> float:
    """Model span for annotation sizing, without constructing a scene."""

    bounds = _geometry_bounds(geometry)
    if bounds is None:
        return 1.0
    span = float(np.linalg.norm(bounds[1] - bounds[0]))
    return span if span > 0.0 else 1.0


def _geometry_centre(geometry: GeometryModel) -> np.ndarray:
    bounds = _geometry_bounds(geometry)
    if bounds is None:
        return np.zeros(3, dtype=float)
    return 0.5 * (bounds[0] + bounds[1])


def entity_sample_points(
    geometry: GeometryModel, ref: EntityRef, mesh: Optional[Mesh] = None
) -> List[np.ndarray]:
    """A few representative points on an entity, for placing symbols."""

    if isinstance(geometry,_MeshOverlayGeometry):return geometry.sample_points(ref)
    if ref.kind == "vertex":
        return [geometry.vertex_position(ref.id)]
    if ref.kind == "edge":
        return list(
            geometry.sample_edge(ref.id, np.linspace(0.15, 0.85, _EDGE_SAMPLES))
        )
    if ref.kind == "face":
        steps = np.linspace(0.25, 0.75, _FACE_SAMPLES)
        return list(_face_grid(geometry, ref.id, steps).reshape((-1, 3)))
    return []


def face_normal(geometry: GeometryModel, face_id: int) -> np.ndarray:
    """The authoritative surface normal at the centre of a plate."""

    return geometry.face_normal(face_id, 0.5, 0.5)


def _imperfection_amplitude(item, coordinates: np.ndarray) -> float:
    if item.amplitude is not None:
        return float(item.amplitude)
    if len(coordinates) < 2:
        return 0.0
    if item.resolved_kind == "plate_mode":
        axes = [int(item.axes[0]), int(item.axes[1])]
        spans = np.ptp(coordinates[:, axes], axis=0)
        return float(np.min(spans) / 200.0)
    return float(np.linalg.norm(coordinates[-1] - coordinates[0]) / 300.0)


def _preview_factor(amplitude: float, span: float) -> float:
    """Make small fabrication-scale offsets visible without changing the model."""

    if amplitude <= 0.0:
        return 1.0
    return min(50.0, max(1.0, 0.03 * span / amplitude))


def _plate_preview_grid(geometry, item, mesh: Optional[Mesh]) -> np.ndarray:
    if mesh is not None and item.ref.id in mesh.grid_of_face:
        identifiers = np.asarray(mesh.grid_of_face[item.ref.id])
        if identifiers.ndim == 2 and identifiers.size:
            return np.asarray(
                [[mesh.nodes[int(node)] for node in row] for row in identifiers],
                dtype=float,
            )
    parameters = np.linspace(0.0, 1.0, 13)
    return _face_grid(geometry, item.ref.id, parameters)


def _plate_imperfection_preview(
    geometry, item, mesh: Optional[Mesh], span: float
) -> tuple[list[np.ndarray], Optional[Arrow]]:
    grid = _plate_preview_grid(geometry, item, mesh)
    coordinates = grid.reshape(-1, 3)
    amplitude = _imperfection_amplitude(item, coordinates)
    direction = np.asarray(item.direction, dtype=float)
    direction /= float(np.linalg.norm(direction))
    axes = [int(item.axes[0]), int(item.axes[1])]
    selected = coordinates[:, axes]
    lower = selected.min(axis=0)
    spans = np.maximum(selected.max(axis=0) - lower, 1.0e-14)
    sx = (selected[:, 0] - lower[0]) / spans[0]
    sy = (selected[:, 1] - lower[1]) / spans[1]
    shape = (
        np.sin(int(item.waves[0]) * np.pi * sx)
        * np.sin(int(item.waves[1]) * np.pi * sy)
    )
    offsets = (
        _preview_factor(amplitude, span)
        * amplitude
        * shape[:, None]
        * direction[None, :]
    )
    displaced = (coordinates + offsets).reshape(grid.shape)
    lines = [np.asarray(row) for row in displaced]
    lines.extend(np.asarray(column) for column in np.swapaxes(displaced, 0, 1))
    peak = int(np.argmax(np.linalg.norm(offsets, axis=1)))
    arrow = None
    if float(np.linalg.norm(offsets[peak])) > 0.0:
        arrow = Arrow(
            start=coordinates[peak],
            end=coordinates[peak] + offsets[peak],
            color=COLOR_IMPERFECTION,
        )
    return lines, arrow


def _member_imperfection_preview(
    geometry, item, mesh: Optional[Mesh], span: float
) -> tuple[np.ndarray, Optional[Arrow]]:
    if mesh is not None and item.ref.id in mesh.nodes_of_edge:
        coordinates = np.asarray(
            [mesh.nodes[int(node)] for node in mesh.nodes_of_edge[item.ref.id]],
            dtype=float,
        )
    else:
        coordinates = np.asarray(
            geometry.sample_edge(item.ref.id, np.linspace(0.0, 1.0, 25)),
            dtype=float,
        )
    amplitude = _imperfection_amplitude(item, coordinates)
    axis = coordinates[-1] - coordinates[0]
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        return coordinates, None
    axis /= length
    direction = np.asarray(item.direction, dtype=float)
    direction -= float(direction @ axis) * axis
    if float(np.linalg.norm(direction)) <= 1.0e-14:
        basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
        direction = basis - float(basis @ axis) * axis
    direction /= float(np.linalg.norm(direction))
    distances = np.asarray(
        [float((coordinate - coordinates[0]) @ axis) / length for coordinate in coordinates]
    )
    offsets = (
        _preview_factor(amplitude, span)
        * amplitude
        * np.sin(np.pi * np.clip(distances, 0.0, 1.0))[:, None]
        * direction[None, :]
    )
    peak = int(np.argmax(np.linalg.norm(offsets, axis=1)))
    arrow = None
    if float(np.linalg.norm(offsets[peak])) > 0.0:
        arrow = Arrow(
            start=coordinates[peak],
            end=coordinates[peak] + offsets[peak],
            color=COLOR_IMPERFECTION,
        )
    return coordinates + offsets, arrow


def _display_attributes(project, items):
    """Materialize canonical live scopes without rewriting authored records.

    Unresolved intent keeps its project validation diagnostic and contributes
    no geometry symbols. Imported mesh references use their existing adapter.
    """
    if project.mesh_only:
        return list(items)
    from ..model.regions import RegionDomain
    geometry = project.geometry
    result = []
    stores = {"vertex": geometry.vertices, "edge": geometry.edges, "face": geometry.faces}
    for item in items:
        region_ref = getattr(item, "region", None) or project.raw_geometry_attachment_region(item)
        try:
            if region_ref is None:
                targets = geometry.resolve_ref(item.ref)
            else:
                region = project.regions[region_ref.id]
                if region.domain is not RegionDomain.GEOMETRY:
                    continue
                targets = project.regions.resolve(region.id, geometry=geometry,
                    candidates=tuple(EntityRef(region.entity_kind, identifier)
                                     for identifier in stores.get(region.entity_kind, {})),
                    feature_resolver=lambda anchor: geometry.features.resolve(anchor, geometry))
        except (KeyError, ValueError):
            continue
        for target in dict.fromkeys(targets):
            if isinstance(target, EntityRef) and target.kind == item.ref.kind:
                result.append(item if target == item.ref else replace(item, ref=target))
    return result


def build_imperfection_overlay(
    project, *, mesh: Optional[Mesh] = None, scale: Optional[float] = None
) -> Scene:
    """Draw the stress-free imperfect reference shape as purple wirework."""

    scene = Scene()
    geometry = project.geometry
    span = float(scale or geometry_characteristic_size(geometry))
    for item in _display_attributes(project, project.imperfections):
        if item.ref.kind == "face" and item.ref.id in geometry.faces:
            lines, arrow = _plate_imperfection_preview(geometry, item, mesh, span)
            scene.lines.extend(
                Polyline(
                    ref=item.ref,
                    points=points,
                    color=COLOR_IMPERFECTION,
                    width=3,
                    annotation=True,
                )
                for points in lines
            )
        elif item.ref.kind == "edge" and item.ref.id in geometry.edges:
            points, arrow = _member_imperfection_preview(geometry, item, mesh, span)
            scene.lines.append(
                Polyline(
                    ref=item.ref,
                    points=points,
                    color=COLOR_IMPERFECTION,
                    width=4,
                    annotation=True,
                )
            )
        else:
            continue
        if arrow is not None:
            scene.arrows.append(arrow)
    return scene


class _MeshOverlayGeometry:
    """Annotation sampling from imported mesh associations, without CAD."""
    def __init__(self,mesh):
        self.mesh=mesh;self.vertices=mesh.nodes

    def vertex_position(self,identifier):return self.mesh.nodes[identifier]

    def sample_points(self,ref):
        if ref.kind=="node":return [self.mesh.nodes[ref.id]]
        identifiers=self.mesh.nodes_on(ref)
        return [np.asarray(self.mesh.nodes[node]) for node in _limited(identifiers,9)]

    def pressure_samples(self,ref):
        samples=[]
        for identifier in _limited(self.mesh.elements_on(ref),9):
            nodes=self.mesh.quads.get(identifier,self.mesh.tris.get(identifier))
            if nodes is None:continue
            points=np.asarray([self.mesh.nodes[node] for node in nodes[:4]])
            normal=_unit(np.cross(points[1]-points[0],points[2]-points[0]))
            if normal is not None:samples.append((points.mean(axis=0),normal))
        return samples


def build_attribute_overlay(
    project,
    *,
    case_name: Optional[str] = "default",
    scale: Optional[float] = None,
    show_supports: bool = True,
    show_loads: bool = True,
    show_masses: bool = True,
    show_imperfections: bool = True,
    mesh: Optional[Mesh] = None,
) -> Scene:
    """Draw supports, loads and masses over the model.

    Nothing here carries an entity tag, so an arrow drawn over a plate never
    steals the click: picking walks past untagged items to the geometry behind.
    """

    geometry = _MeshOverlayGeometry(mesh) if project.mesh_only and mesh is not None else project.geometry
    scene = Scene()
    span = (
        scale
        if scale is not None
        else geometry_characteristic_size(geometry)
    )
    arrow_length = ARROW_FRACTION * span
    symbol = SYMBOL_FRACTION * span

    if show_imperfections:
        scene.merge(build_imperfection_overlay(project, mesh=mesh, scale=span))

    if show_supports:
        supports = Scene()
        for support in _budgeted_assignments(_display_attributes(project, project.supports)):
            _draw_support(supports, geometry, support, symbol)
        scene.points.extend(_limited(supports.points))
        scene.lines.extend(_limited(supports.lines))
        scene.arrows.extend(_limited(supports.arrows))

    if show_masses:
        markers: List[PointMarker] = []
        for mass in _budgeted_assignments(_display_attributes(project, project.masses)):
            for point in entity_sample_points(geometry, mass.ref):
                markers.append(
                    PointMarker(
                        ref=None, position=point, color=COLOR_MASS,
                        size=1.6 * symbol,
                    )
                )
        scene.points.extend(_limited(markers))

    if show_supports and project.connections and not isinstance(geometry, _MeshOverlayGeometry):
        _draw_connections(scene, project, geometry, symbol)

    if show_loads:
        _draw_wet_surfaces(scene, project, geometry)
    if show_loads and case_name is not None and project.connections and not isinstance(geometry, _MeshOverlayGeometry):
        case = _display_case(project, case_name)
        if case is not None and case.connection_loads:
            _draw_connection_loads(scene, project, case, arrow_length)
    if show_loads and case_name is not None:
        case = _display_case(project, case_name)
        if case is not None and case.wet_pressures:
            _draw_wet_pressures(scene, project, geometry, case, mesh, arrow_length)
    if show_loads and case_name is not None:
        case = _display_case(project, case_name)
        if case is not None:
            display_case = SimpleNamespace(**{name: _display_attributes(project, getattr(case, name))
                for name in ("point_loads", "pressures", "line_loads", "surface_tractions")}, gravity=case.gravity)
            _draw_loads(scene, geometry, display_case, arrow_length)

    return scene


COLOR_CONNECTION = "#00695c"


def _display_case(project, case_name):
    """The case to draw: a step shows its definition's loads, a definition its first step.

    Loads whose envelope has them switched off at that step are left out, so the
    picture shows what acts in the step being looked at.
    """

    from copy import copy

    case = project.load_cases.get(case_name)
    if case is None:
        return None
    try:
        if case.times is not None:
            steps = project.series_children(case.name)
            if not steps:
                return None
            case = steps[0]
        case = project.resolve_case(case)
    except Exception:  # an orphaned step: validation reports it, nothing to draw
        return None
    if case is project.load_cases.get(case_name) and case.parent is None and case.times is None:
        view = copy(case)
    else:
        view = case
    for name in ("point_loads", "pressures", "line_loads", "surface_tractions",
                 "wet_pressures", "connection_loads"):
        kept = []
        for load in getattr(case, name):
            try:
                acting = case.envelope_factor(getattr(load, "envelope", None)) != 0.0
            except Exception:
                acting = True
            if acting:
                kept.append(load)
        setattr(view, name, kept)
    return view


def _connection_reference(project, connection):
    try:
        return project.connection_reference_point(connection)
    except Exception:  # unresolved scope: validation reports it, nothing to draw
        return None


def _draw_connections(scene: Scene, project, geometry, symbol: float) -> None:
    """A marker at each reference point with a spoke to every tied entity.

    The spokes are untagged annotations, so they never steal a click from the
    plate or line underneath.
    """

    markers: List[PointMarker] = []
    spokes: List[Polyline] = []
    for connection in _budgeted_assignments(list(project.connections)):
        reference = _connection_reference(project, connection)
        if reference is None:
            continue
        markers.append(
            PointMarker(
                ref=None, position=reference, color=COLOR_CONNECTION,
                size=1.8 * symbol,
            )
        )
        try:
            targets = project.connection_targets(connection)
        except Exception:
            continue
        for target in _limited(list(targets), 24):
            samples = entity_sample_points(geometry, target)
            if not samples:
                continue
            centre = np.mean(np.asarray(samples, dtype=float), axis=0)
            spokes.append(
                Polyline(
                    ref=None,
                    points=np.vstack([reference, centre]),
                    color=COLOR_CONNECTION,
                    width=1 if connection.kind == "distributing" else 2,
                    annotation=True,
                )
            )
    scene.points.extend(_limited(markers))
    scene.lines.extend(_limited(spokes))


def _draw_connection_loads(scene: Scene, project, case, arrow_length: float) -> None:
    for load in _limited(list(case.connection_loads)):
        connection = next((item for item in project.connections if item.id == load.connection), None)
        if connection is None:
            continue
        reference = _connection_reference(project, connection)
        if reference is None:
            continue
        force = _unit(load.force)
        if force is not None:
            scene.arrows.append(
                Arrow(start=reference - arrow_length * force, end=reference)
            )
        moment = _unit(load.moment)
        if moment is not None:
            scene.arrows.append(
                Arrow(
                    start=reference - 0.5 * arrow_length * moment,
                    end=reference + 0.5 * arrow_length * moment,
                    color=COLOR_MOMENT,
                )
            )
            scene.points.append(
                PointMarker(
                    ref=None, position=reference, color=COLOR_MOMENT,
                    size=0.25 * arrow_length,
                )
            )


def _draw_wet_surfaces(scene: Scene, project, geometry) -> None:
    """Outline every wet plate, with a cross through it, in the wet colour.

    The lines are untagged annotations, so they never steal a click from the
    plate underneath.
    """

    if isinstance(geometry, _MeshOverlayGeometry) or not project.wet_surfaces:
        return
    parameters = np.linspace(0.0, 1.0, 9)
    seen: set[int] = set()
    outlines: List[Polyline] = []
    for surface in _budgeted_assignments(_display_attributes(project, project.wet_surfaces)):
        face_id = surface.ref.id
        if surface.ref.kind != "face" or face_id in seen or face_id not in geometry.faces:
            continue
        seen.add(face_id)
        try:
            grid = _face_grid(geometry, face_id, parameters)
        except Exception:  # an undisplayable plate is reported by validation
            continue
        middle = len(parameters) // 2
        for line, width in (
            (np.vstack([grid[0, :], grid[1:, -1], grid[-1, -2::-1], grid[-2::-1, 0]]), 3),
            (grid[middle, :], 1),
            (grid[:, middle], 1),
        ):
            outlines.append(
                Polyline(
                    ref=None,
                    points=np.asarray(line, dtype=float),
                    color=COLOR_WET,
                    width=width,
                    annotation=True,
                )
            )
    scene.lines.extend(outlines)


def _wet_pressure_samples(project, geometry, case, mesh):
    """``(point, normal, signed pressure)`` for the wet pressures of a case.

    With a mesh the solver's own evaluation is used, element by element. Before
    meshing, plates are sampled at a few interior points, which is enough to
    show the side the fluid is on and how the pressure varies.
    """

    from ..model.wetfield import WetFieldError, field_pressure, fluid_sign

    if mesh is not None:
        from ..model.project import ProjectError
        from ..solve.build import wet_element_pressures

        try:
            items = wet_element_pressures(
                project, mesh, case, limit=OVERLAY_SYMBOL_LIMIT
            )
        except ProjectError:
            return []
        return [(item.centroid, item.normal, item.pressure) for item in items]

    if isinstance(geometry, _MeshOverlayGeometry):
        return []
    # A pressure field needs a finer sampling than a symbol: a hydrostatic
    # gradient is only readable with several points across the depth.
    surfaces = _limited(
        _display_attributes(project, project.wet_surfaces),
        max(1, OVERLAY_SYMBOL_LIMIT // _WET_GRID**2),
    )
    by_id = {surface.id: surface for surface in project.wet_surfaces}
    samples = []
    for load in case.wet_pressures:
        wanted = set(load.surfaces) if load.surfaces else set(by_id)
        for surface in surfaces:
            if not (wanted & {surface.id}) or surface.ref.id not in geometry.faces:
                continue
            try:
                grid = _face_grid(
                    geometry, surface.ref.id, np.linspace(0.1, 0.9, _WET_GRID)
                )
                points = list(grid.reshape((-1, 3)))
                normal = face_normal(geometry, surface.ref.id)
            except Exception:  # an undisplayable plate is reported by validation
                continue
            try:
                if load.kind != "mapping":
                    magnitude = field_pressure(case, load, np.array(points))
                elif load.mapping_key == "face":
                    magnitude = load.scale * np.full(
                        len(points), load.mapping.get(surface.ref.id, 0.0)
                    )
                else:
                    continue  # element ids exist only on a mesh
                for point, value in zip(points, magnitude):
                    samples.append(
                        (
                            point,
                            normal,
                            fluid_sign(surface, point, normal) * float(value),
                        )
                    )
            except WetFieldError:
                continue
    return samples


def _draw_wet_pressures(scene: Scene, project, geometry, case, mesh, arrow_length: float) -> None:
    """Arrows into each wet plate, longer where the pressure is higher."""

    samples = [
        item
        for item in _wet_pressure_samples(project, geometry, case, mesh)
        if item[2] != 0.0
    ]
    if not samples:
        return
    peak = max(abs(item[2]) for item in samples)
    arrows: List[Arrow] = []
    for point, normal, pressure in _limited(samples):
        # The signed pressure is positive along the plate normal: the arrow
        # shows the direction the fluid pushes, ending on the plate.
        direction = normal if pressure > 0.0 else -normal
        length = arrow_length * (0.25 + 0.75 * abs(pressure) / peak)
        arrows.append(
            Arrow(
                start=np.asarray(point) - length * direction,
                end=np.asarray(point),
                color=COLOR_WET,
            )
        )
    scene.arrows.extend(arrows)


def _draw_support(scene: Scene, geometry, support, symbol: float) -> None:
    """Draw a restraint marker plus axes for the constrained directions."""

    held = len(support.constraints)
    prescribed_motion = any(
        abs(float(value)) > 0.0 for value in support.constraints.values()
    )
    colour = COLOR_LOAD if prescribed_motion else COLOR_SUPPORT
    size = symbol * (0.6 + 0.4 * held / 6.0)
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    for point in entity_sample_points(geometry, support.ref):
        scene.points.append(
            PointMarker(ref=None, position=point, color=colour, size=size)
        )
        for dof, value in support.constraints.items():
            axis = axes.get(dof[-1:])
            if axis is None:
                continue
            if dof.startswith("u"):
                scene.lines.append(
                    Polyline(
                        ref=None,
                        points=np.vstack([point, point + 1.8 * symbol * axis]),
                        color=COLOR_SUPPORT,
                        width=3,
                    )
                )
                if abs(float(value)) > 0.0:
                    direction = axis if float(value) > 0.0 else -axis
                    scene.arrows.append(
                        Arrow(
                            start=point,
                            end=point + 2.6 * symbol * direction,
                            color=COLOR_LOAD,
                        )
                    )
            elif dof.startswith("r"):
                scene.lines.append(
                    Polyline(
                        ref=None,
                        points=np.vstack(
                            [
                                point - 0.9 * symbol * axis,
                                point + 0.9 * symbol * axis,
                            ]
                        ),
                        color=COLOR_ROTATION,
                        width=2,
                    )
                )


def _draw_loads(scene: Scene, geometry, case, arrow_length: float) -> None:
    point_arrows: List[Arrow] = []
    moment_arrows: List[Arrow] = []
    moment_markers: List[PointMarker] = []
    force_loads = [load for load in case.point_loads if _unit(load.force) is not None]
    for load in _limited(force_loads):
        direction = _unit(load.force)
        point = geometry.vertex_position(load.ref.id)
        point_arrows.append(
            Arrow(start=point - arrow_length * direction, end=point)
        )

    moment_loads = [
        load for load in case.point_loads if _unit(load.moment) is not None
    ]
    for load in _limited(moment_loads):
        moment_direction = _unit(load.moment)
        point = geometry.vertex_position(load.ref.id)
        moment_arrows.append(
            Arrow(
                start=point - 0.5 * arrow_length * moment_direction,
                end=point + 0.5 * arrow_length * moment_direction,
                color=COLOR_MOMENT,
            )
        )
        moment_markers.append(
            PointMarker(
                ref=None,
                position=point,
                color=COLOR_MOMENT,
                size=0.25 * arrow_length,
            )
        )
    scene.arrows.extend(_limited(point_arrows))
    scene.arrows.extend(_limited(moment_arrows))
    scene.points.extend(_limited(moment_markers))

    line_arrows: List[Arrow] = []
    line_loads = [
        load for load in case.line_loads
        if _unit(load.force_per_length) is not None
    ]
    for load in _budgeted_assignments(line_loads):
        direction = _unit(load.force_per_length)
        for point in entity_sample_points(geometry, load.ref):
            line_arrows.append(
                Arrow(start=point - arrow_length * direction, end=point)
            )
    scene.arrows.extend(_limited(line_arrows))

    traction_arrows: List[Arrow] = []
    traction_loads = [
        load for load in case.surface_tractions if _unit(load.traction) is not None
    ]
    for load in _budgeted_assignments(traction_loads):
        direction = _unit(load.traction)
        for point in entity_sample_points(geometry, load.ref):
            traction_arrows.append(
                Arrow(start=point - arrow_length * direction, end=point)
            )
    scene.arrows.extend(_limited(traction_arrows))

    pressure_arrows: List[Arrow] = []
    pressure_loads = [load for load in case.pressures if load.value != 0.0]
    for load in _budgeted_assignments(pressure_loads):
        # Positive pressure pushes along the plate normal; the arrows show
        # which face it acts on, which is the thing that is easy to get wrong.
        if isinstance(geometry,_MeshOverlayGeometry):samples=geometry.pressure_samples(load.ref)
        else:
            normal = face_normal(geometry, load.ref.id)
            samples = [(point, normal) for point in entity_sample_points(geometry, load.ref)]
        for point,normal in samples:
            direction = normal if load.value > 0.0 else -normal
            pressure_arrows.append(
                Arrow(
                    start=point - 0.6 * arrow_length * direction,
                    end=point,
                    color=COLOR_PRESSURE,
                )
            )
    scene.arrows.extend(_limited(pressure_arrows))

    gravity_direction = _unit(case.gravity) if case.gravity is not None else None
    if gravity_direction is not None:
        point = _geometry_centre(geometry)
        scene.arrows.append(
            Arrow(
                start=point - arrow_length * gravity_direction,
                end=point,
                color=COLOR_MASS,
            )
        )


def _unit(vector) -> Optional[np.ndarray]:
    array = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(array))
    if length <= 0.0:
        return None
    return array / length


COLOR_SPHERE = "#455a64"
COLOR_PATH = "#78909c"


def build_collision_overlay(solution, index: int = 0) -> Scene:
    """Draw the sphere where it is at one step, and the path it took.

    The sphere is drawn at its actual radius, so an obviously wrong radius or
    aim is visible rather than only showing up as a strange contact force.
    """

    scene = Scene()
    positions = np.asarray(
        getattr(solution, "sphere_positions", np.zeros((0, 3))), dtype=float
    )
    if positions.size == 0:
        return scene

    collision = getattr(solution, "collision", None)
    radius = 0.1 if collision is None else float(collision.radius)

    step = min(max(int(index), 0), len(positions) - 1)
    scene.spheres.append(
        Sphere(centre=positions[step], radius=radius, color=COLOR_SPHERE)
    )
    if len(positions) > 1:
        scene.lines.append(
            Polyline(ref=None, points=positions, color=COLOR_PATH, width=2)
        )
    return scene
