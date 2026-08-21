"""Semantic selection values and projected, tile-indexed query geometry.

This module deliberately has no GUI dependency.  Backends remain responsible
for building projected primitives from their retained or compiled scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple


Point2 = Tuple[float, float]
Rect = Tuple[float, float, float, float]


class SelectionDepth(str, Enum):
    """Whether a region query stops at the visible surface."""

    VISIBLE = "visible"
    THROUGH = "through"


class SelectionTool(str, Enum):
    """Shape made by a primary-button drag."""

    SINGLE = "single"
    BOX = "box"
    LASSO = "lasso"


class SelectionOperation(str, Enum):
    """How an application combines a gesture with its selection."""

    REPLACE = "replace"
    ADD = "add"
    REMOVE = "remove"
    TOGGLE = "toggle"


class SelectionGesture(str, Enum):
    CLICK = "click"
    WINDOW = "window"
    CROSSING = "crossing"
    LASSO = "lasso"


@dataclass(frozen=True)
class PickOwner:
    """One semantic object represented by a rendered primitive."""

    key: str
    kind: str = ""
    priority: int = 0
    identity: object | None = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.key):
            raise ValueError("a pick owner needs a non-empty key")
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "priority", int(self.priority))


@dataclass(frozen=True)
class PickBinding:
    """Semantic owners attached to one face, line, or marker."""

    owners: Tuple[PickOwner, ...]

    def __post_init__(self) -> None:
        cleaned = tuple(
            owner if isinstance(owner, PickOwner) else PickOwner(*owner)
            for owner in self.owners
        )
        if not cleaned:
            raise ValueError("a pick binding needs at least one owner")
        if len({owner.key for owner in cleaned}) != len(cleaned):
            raise ValueError("owner keys in one pick binding must be unique")
        object.__setattr__(self, "owners", cleaned)

    @classmethod
    def one(cls, key: str, kind: str = "", priority: int = 0) -> "PickBinding":
        return cls((PickOwner(key, kind, priority),))


@dataclass(frozen=True)
class SelectionFilter:
    """Restrict queries by semantic kind and/or stable-key prefix."""

    kinds: FrozenSet[str] = frozenset()
    key_prefixes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kinds", frozenset(str(value) for value in self.kinds))
        object.__setattr__(
            self, "key_prefixes", tuple(str(value) for value in self.key_prefixes)
        )

    def accepts(self, owner: PickOwner) -> bool:
        if self.kinds and owner.kind not in self.kinds:
            return False
        if self.key_prefixes and not any(
            owner.key.startswith(prefix) for prefix in self.key_prefixes
        ):
            return False
        return True


@dataclass(frozen=True)
class SelectionConfig:
    """Interaction and query policy for a commercial selection profile."""

    filter: SelectionFilter = field(default_factory=SelectionFilter)
    depth: SelectionDepth = SelectionDepth.VISIBLE
    tool: SelectionTool = SelectionTool.BOX
    directional: bool = True
    drag_threshold_px: int = 4
    click_radius_px: int = 4
    cycle_radius_px: int = 5
    cycle_timeout_ms: int = 1500
    click_on_press: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.filter, SelectionFilter):
            raise TypeError("filter must be a SelectionFilter")
        object.__setattr__(self, "depth", SelectionDepth(self.depth))
        object.__setattr__(self, "tool", SelectionTool(self.tool))
        object.__setattr__(self, "click_on_press", bool(self.click_on_press))
        for name, minimum in (
            ("drag_threshold_px", 1),
            ("click_radius_px", 0),
            ("cycle_radius_px", 0),
            ("cycle_timeout_ms", 0),
        ):
            object.__setattr__(self, name, max(minimum, int(getattr(self, name))))


@dataclass(frozen=True)
class SelectionHit:
    """One semantic owner hit by a point or region query."""

    owner: PickOwner
    primitive: int
    depth: float
    screen_distance: float = 0.0
    visible: bool = True
    item: int = -1
    identity: object | None = None

    def __post_init__(self) -> None:
        if self.identity is None:
            object.__setattr__(
                self,
                "identity",
                self.owner.identity if self.owner.identity is not None else self.owner,
            )

    @property
    def key(self) -> str:
        return self.owner.key

    @property
    def kind(self) -> str:
        return self.owner.kind


@dataclass(frozen=True)
class SelectionEvent:
    """Completed click, box, or lasso gesture."""

    gesture: SelectionGesture
    operation: SelectionOperation
    hits: Tuple[SelectionHit, ...] = ()
    candidates: Tuple[SelectionHit, ...] = ()
    start: Tuple[int, int] = (0, 0)
    end: Tuple[int, int] = (0, 0)
    points: Tuple[Tuple[int, int], ...] = ()
    cycle_index: int = 0
    cycle_total: int = 0


@dataclass(frozen=True)
class ProjectedPrimitive:
    """One projected polygon, segment, or fixed-size marker."""

    index: int
    shape: str
    points: Tuple[Point2, ...]
    depths: Tuple[float, ...]
    binding: Optional[PickBinding] = None
    layer: float = 0.0
    radius: float = 0.0
    item: int = -1

    @property
    def depth(self) -> float:
        if len(self.depths) == 1:
            return self.depths[0]
        return sum(self.depths) / max(1, len(self.depths))

    @property
    def bbox(self) -> Rect:
        pad = max(0.0, float(self.radius))
        if len(self.points) == 1:
            x, y = self.points[0]
            return x - pad, y - pad, x + pad, y + pad
        if len(self.points) == 2:
            first, second = self.points
            return (
                min(first[0], second[0]) - pad,
                min(first[1], second[1]) - pad,
                max(first[0], second[0]) + pad,
                max(first[1], second[1]) + pad,
            )
        min_x = max_x = self.points[0][0]
        min_y = max_y = self.points[0][1]
        for x, y in self.points[1:]:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return min_x - pad, min_y - pad, max_x + pad, max_y + pad


def _rect_normalized(rect: Sequence[float]) -> Rect:
    x0, y0, x1, y1 = (float(value) for value in rect)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _bbox_intersects(first: Rect, second: Rect) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _point_in_rect(point: Point2, rect: Rect) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def _point_distance_to_segment(point: Point2, start: Point2, end: Point2) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 1.0e-18:
        return math.hypot(px - ax, py - ay)
    parameter = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    return math.hypot(px - (ax + parameter * dx), py - (ay + parameter * dy))


def _point_in_polygon(point: Point2, polygon: Sequence[Point2]) -> bool:
    """Even/odd containment with boundary points treated as inside."""

    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_distance_to_segment(point, previous, current) <= 1.0e-7:
            return True
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x >= x:
                inside = not inside
        previous = current
    return inside


def _orientation(a: Point2, b: Point2, c: Point2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point2, b: Point2, c: Point2, d: Point2) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    epsilon = 1.0e-9
    if ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and (
        (o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon)
    ):
        return True
    if abs(o1) <= epsilon and _point_distance_to_segment(c, a, b) <= epsilon:
        return True
    if abs(o2) <= epsilon and _point_distance_to_segment(d, a, b) <= epsilon:
        return True
    if abs(o3) <= epsilon and _point_distance_to_segment(a, c, d) <= epsilon:
        return True
    if abs(o4) <= epsilon and _point_distance_to_segment(b, c, d) <= epsilon:
        return True
    return False


def _segment_intersects_rect(start: Point2, end: Point2, rect: Rect) -> bool:
    if _point_in_rect(start, rect) or _point_in_rect(end, rect):
        return True
    x0, y0, x1, y1 = rect
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return any(
        _segments_intersect(start, end, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    )


def _primitive_distance(primitive: ProjectedPrimitive, point: Point2) -> float:
    if primitive.shape == "point":
        distance = math.hypot(
            point[0] - primitive.points[0][0],
            point[1] - primitive.points[0][1],
        )
        return max(0.0, distance - primitive.radius)
    if primitive.shape == "segment":
        return max(
            0.0,
            _point_distance_to_segment(point, primitive.points[0], primitive.points[1])
            - primitive.radius,
        )
    if _point_in_polygon(point, primitive.points):
        return 0.0
    return min(
        _point_distance_to_segment(
            point, primitive.points[index - 1], primitive.points[index]
        )
        for index in range(len(primitive.points))
    )


def _primitive_crosses_rect(primitive: ProjectedPrimitive, rect: Rect) -> bool:
    if primitive.shape == "point":
        return _point_in_rect(primitive.points[0], (
            rect[0] - primitive.radius,
            rect[1] - primitive.radius,
            rect[2] + primitive.radius,
            rect[3] + primitive.radius,
        ))
    if not _bbox_intersects(primitive.bbox, rect):
        return False
    if primitive.shape == "segment":
        return _segment_intersects_rect(primitive.points[0], primitive.points[1], rect)
    if any(_point_in_rect(point, rect) for point in primitive.points):
        return True
    x0, y0, x1, y1 = rect
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    if any(_point_in_polygon(corner, primitive.points) for corner in corners):
        return True
    return any(
        _segment_intersects_rect(
            primitive.points[index - 1], primitive.points[index], rect
        )
        for index in range(len(primitive.points))
    )


def _primitive_inside_rect(primitive: ProjectedPrimitive, rect: Rect) -> bool:
    pad = primitive.radius if primitive.shape == "point" else 0.0
    inner = (rect[0] + pad, rect[1] + pad, rect[2] - pad, rect[3] - pad)
    if inner[0] > inner[2] or inner[1] > inner[3]:
        return False
    return all(_point_in_rect(point, inner) for point in primitive.points)


def _primitive_crosses_polygon(
    primitive: ProjectedPrimitive, polygon: Sequence[Point2]
) -> bool:
    if len(polygon) < 3:
        return False
    if primitive.shape == "point":
        return _point_in_polygon(primitive.points[0], polygon)

    def segment_crosses(start: Point2, end: Point2) -> bool:
        return (
            _point_in_polygon(start, polygon)
            or _point_in_polygon(end, polygon)
            or any(
                _segments_intersect(
                    start, end, polygon[index - 1], polygon[index]
                )
                for index in range(len(polygon))
            )
        )

    if primitive.shape == "segment":
        return segment_crosses(primitive.points[0], primitive.points[1])
    if any(_point_in_polygon(point, polygon) for point in primitive.points):
        return True
    if any(_point_in_polygon(point, primitive.points) for point in polygon):
        return True
    return any(
        segment_crosses(primitive.points[index - 1], primitive.points[index])
        for index in range(len(primitive.points))
    )


class ProjectedSelectionIndex:
    """Spatial index and deterministic screen-space selection queries."""

    def __init__(
        self,
        primitives: Iterable[ProjectedPrimitive],
        width: int,
        height: int,
        *,
        tile_size: int = 16,
    ) -> None:
        self.primitives = tuple(primitives)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.tile_size = max(8, int(tile_size))
        self._tiles: Dict[Tuple[int, int], List[int]] = {}
        self._global: List[int] = []
        self._owners: Dict[str, List[int]] = {}
        self._owner_values: Dict[str, PickOwner] = {}
        self._build()

    def _build(self) -> None:
        max_x = max(0, (self.width - 1) // self.tile_size)
        max_y = max(0, (self.height - 1) // self.tile_size)
        for index, primitive in enumerate(self.primitives):
            if primitive.binding is not None:
                for owner in primitive.binding.owners:
                    self._owners.setdefault(owner.key, []).append(index)
                    current = self._owner_values.get(owner.key)
                    if current is None or owner.priority > current.priority:
                        self._owner_values[owner.key] = owner
            x0, y0, x1, y1 = primitive.bbox
            if x1 < 0 or y1 < 0 or x0 > self.width or y0 > self.height:
                continue
            low_x = max(0, min(max_x, math.floor(x0 / self.tile_size)))
            high_x = max(0, min(max_x, math.floor(x1 / self.tile_size)))
            low_y = max(0, min(max_y, math.floor(y0 / self.tile_size)))
            high_y = max(0, min(max_y, math.floor(y1 / self.tile_size)))
            cell_total = (high_x - low_x + 1) * (high_y - low_y + 1)
            if cell_total > 256:
                self._global.append(index)
                continue
            for cell_y in range(low_y, high_y + 1):
                for cell_x in range(low_x, high_x + 1):
                    self._tiles.setdefault((cell_x, cell_y), []).append(index)

    def _indices_for_rect(self, rect: Rect) -> List[int]:
        x0, y0, x1, y1 = _rect_normalized(rect)
        if x1 < 0 or y1 < 0 or x0 > self.width or y0 > self.height:
            return []
        max_x = max(0, (self.width - 1) // self.tile_size)
        max_y = max(0, (self.height - 1) // self.tile_size)
        low_x = max(0, min(max_x, math.floor(x0 / self.tile_size)))
        high_x = max(0, min(max_x, math.floor(x1 / self.tile_size)))
        low_y = max(0, min(max_y, math.floor(y0 / self.tile_size)))
        high_y = max(0, min(max_y, math.floor(y1 / self.tile_size)))
        found = set(self._global)
        for cell_y in range(low_y, high_y + 1):
            for cell_x in range(low_x, high_x + 1):
                found.update(self._tiles.get((cell_x, cell_y), ()))
        return sorted(found)

    def _front_primitive_at(
        self,
        point: Point2,
        radius: float = 0.75,
        selection_filter: Optional[SelectionFilter] = None,
    ) -> Optional[int]:
        rect = (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)
        candidates = []
        for index in self._indices_for_rect(rect):
            primitive = self.primitives[index]
            if _primitive_distance(primitive, point) > radius:
                continue
            # Surfaces are real occluders even when their owner is filtered
            # out.  Filtered point/line annotations are selection-transparent,
            # matching Tk's legacy behaviour where untagged overlays never
            # steal a click from geometry behind them.
            if selection_filter is not None and primitive.shape != "polygon":
                binding = primitive.binding
                if binding is None or not any(
                    selection_filter.accepts(owner) for owner in binding.owners
                ):
                    continue
            candidates.append(index)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda index: (
                self.primitives[index].depth,
                -self.primitives[index].layer,
                -index,
            ),
        )

    def point_hits(
        self,
        x: float,
        y: float,
        selection_filter: SelectionFilter,
        *,
        radius: float = 4.0,
    ) -> Tuple[SelectionHit, ...]:
        point = (float(x), float(y))
        query = (x - radius, y - radius, x + radius, y + radius)
        primitive_hits: List[Tuple[int, float]] = []
        for index in self._indices_for_rect(query):
            primitive = self.primitives[index]
            distance = _primitive_distance(primitive, point)
            if distance <= radius:
                primitive_hits.append((index, distance))
        if not primitive_hits:
            return ()

        selectable_front = [
            index
            for index, _distance in primitive_hits
            if self.primitives[index].shape == "polygon"
            or (
                self.primitives[index].binding is not None
                and any(
                    selection_filter.accepts(owner)
                    for owner in self.primitives[index].binding.owners
                )
            )
        ]
        if not selectable_front:
            return ()
        front = min(
            selectable_front,
            key=lambda index: (
                self.primitives[index].depth,
                -self.primitives[index].layer,
                -index,
            ),
        )
        best: Dict[str, SelectionHit] = {}
        for index, distance in primitive_hits:
            primitive = self.primitives[index]
            if primitive.binding is None:
                continue
            visible = index == front
            for owner in primitive.binding.owners:
                if not selection_filter.accepts(owner):
                    continue
                hit = SelectionHit(
                    owner=owner,
                    primitive=primitive.index,
                    depth=primitive.depth,
                    screen_distance=distance,
                    visible=visible,
                    item=primitive.item,
                )
                current = best.get(owner.key)
                if current is None or self._hit_order(hit) < self._hit_order(current):
                    best[owner.key] = hit
        return tuple(sorted(best.values(), key=self._hit_order))

    @staticmethod
    def _hit_order(hit: SelectionHit) -> Tuple[object, ...]:
        return (
            not hit.visible,
            -hit.owner.priority,
            hit.screen_distance,
            hit.depth,
            hit.primitive,
            hit.owner.key,
        )

    @staticmethod
    def _samples(primitive: ProjectedPrimitive) -> Tuple[Point2, ...]:
        points = primitive.points
        if primitive.shape == "point":
            return points
        if primitive.shape == "segment":
            start, end = points
            return (start, ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5), end)
        centroid = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        middles = tuple(
            (
                (points[index - 1][0] + points[index][0]) * 0.5,
                (points[index - 1][1] + points[index][1]) * 0.5,
            )
            for index in range(len(points))
        )
        return (centroid,) + points + middles

    def _owner_visible(
        self,
        owner_key: str,
        indices: Sequence[int],
        rect: Rect,
        selection_filter: SelectionFilter,
    ) -> bool:
        for index in indices:
            primitive = self.primitives[index]
            for sample in self._samples(primitive):
                if not _point_in_rect(sample, rect):
                    continue
                front = self._front_primitive_at(
                    sample, selection_filter=selection_filter
                )
                if front is None:
                    continue
                front_binding = self.primitives[front].binding
                if front == index or (
                    front_binding is not None
                    and any(owner.key == owner_key for owner in front_binding.owners)
                ):
                    return True
        return False

    def rectangle_hits(
        self,
        rect: Sequence[float],
        selection_filter: SelectionFilter,
        *,
        crossing: bool,
        depth: SelectionDepth,
    ) -> Tuple[SelectionHit, ...]:
        area = _rect_normalized(rect)
        touched = self._indices_for_rect(area)
        # Crossing-through is the high-volume FE workflow: it needs one best
        # primitive per semantic owner, but neither the complete per-owner
        # primitive lists required by window containment nor visible-depth
        # samples.  Avoiding 50k one-item lists keeps a full-model drag within
        # the interaction budget and materially lowers transient memory.
        if crossing and depth == SelectionDepth.THROUGH:
            best_indices: Dict[str, int] = {}
            for index in touched:
                primitive = self.primitives[index]
                if primitive.binding is None or not _primitive_crosses_rect(
                    primitive, area
                ):
                    continue
                for owner in primitive.binding.owners:
                    if not selection_filter.accepts(owner):
                        continue
                    current_index = best_indices.get(owner.key)
                    if current_index is None:
                        best_indices[owner.key] = index
                        continue
                    current = self.primitives[current_index]
                    if (primitive.depth, -primitive.layer, primitive.index) < (
                        current.depth,
                        -current.layer,
                        current.index,
                    ):
                        best_indices[owner.key] = index
            hits = tuple(
                SelectionHit(
                    owner=self._owner_values[owner_key],
                    primitive=self.primitives[index].index,
                    depth=self.primitives[index].depth,
                    visible=True,
                    item=self.primitives[index].item,
                )
                for owner_key, index in best_indices.items()
            )
            return tuple(sorted(hits, key=self._hit_order))

        owner_candidates: Dict[str, List[int]] = {}
        for index in touched:
            primitive = self.primitives[index]
            if primitive.binding is None:
                continue
            qualifies = (
                _primitive_crosses_rect(primitive, area)
                if crossing
                else _primitive_inside_rect(primitive, area)
            )
            if not qualifies:
                continue
            for owner in primitive.binding.owners:
                if selection_filter.accepts(owner):
                    owner_candidates.setdefault(owner.key, []).append(index)

        hits: List[SelectionHit] = []
        for owner_key, qualified in owner_candidates.items():
            if not crossing:
                all_indices = self._owners.get(owner_key, ())
                if not all_indices or not all(
                    _primitive_inside_rect(self.primitives[index], area)
                    for index in all_indices
                ):
                    continue
                qualified = list(all_indices)
            visible = True
            if depth == SelectionDepth.VISIBLE:
                visible = self._owner_visible(
                    owner_key, qualified, area, selection_filter
                )
                if not visible:
                    continue
            primitive = min(
                (self.primitives[index] for index in qualified),
                key=lambda value: (value.depth, -value.layer, value.index),
            )
            hits.append(
                SelectionHit(
                    owner=self._owner_values[owner_key],
                    primitive=primitive.index,
                    depth=primitive.depth,
                    visible=visible,
                    item=primitive.item,
                )
            )
        return tuple(sorted(hits, key=self._hit_order))

    def polygon_hits(
        self,
        points: Sequence[Point2],
        selection_filter: SelectionFilter,
        *,
        depth: SelectionDepth,
    ) -> Tuple[SelectionHit, ...]:
        """Crossing query for a lasso polygon."""

        polygon = tuple((float(point[0]), float(point[1])) for point in points)
        if len(polygon) < 3:
            return ()
        area = (
            min(point[0] for point in polygon),
            min(point[1] for point in polygon),
            max(point[0] for point in polygon),
            max(point[1] for point in polygon),
        )
        owner_candidates: Dict[str, List[int]] = {}
        for index in self._indices_for_rect(area):
            primitive = self.primitives[index]
            if primitive.binding is None or not _primitive_crosses_polygon(
                primitive, polygon
            ):
                continue
            for owner in primitive.binding.owners:
                if selection_filter.accepts(owner):
                    owner_candidates.setdefault(owner.key, []).append(index)

        hits: List[SelectionHit] = []
        for owner_key, qualified in owner_candidates.items():
            visible = True
            if depth == SelectionDepth.VISIBLE:
                visible = False
                for index in qualified:
                    for sample in self._samples(self.primitives[index]):
                        if not _point_in_polygon(sample, polygon):
                            continue
                        front = self._front_primitive_at(
                            sample, selection_filter=selection_filter
                        )
                        if front is None:
                            continue
                        front_binding = self.primitives[front].binding
                        if front == index or (
                            front_binding is not None
                            and any(
                                owner.key == owner_key
                                for owner in front_binding.owners
                            )
                        ):
                            visible = True
                            break
                    if visible:
                        break
                if not visible:
                    continue
            primitive = min(
                (self.primitives[index] for index in qualified),
                key=lambda value: (value.depth, -value.layer, value.index),
            )
            hits.append(
                SelectionHit(
                    owner=self._owner_values[owner_key],
                    primitive=primitive.index,
                    depth=primitive.depth,
                    visible=visible,
                    item=primitive.item,
                )
            )
        return tuple(sorted(hits, key=self._hit_order))


__all__ = [
    "PickBinding",
    "PickOwner",
    "ProjectedPrimitive",
    "ProjectedSelectionIndex",
    "SelectionConfig",
    "SelectionDepth",
    "SelectionEvent",
    "SelectionFilter",
    "SelectionGesture",
    "SelectionHit",
    "SelectionOperation",
    "SelectionTool",
]
