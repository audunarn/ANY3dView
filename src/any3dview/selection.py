"""Semantic selection values and projected, tile-indexed query geometry.

This module deliberately has no GUI dependency.  Backends remain responsible
for building projected primitives from their retained or compiled scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import merge
import math
from itertools import islice
from typing import Callable, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


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


def _float_exact(value: object) -> bool:
    """True when the scalar predicates on ``value`` compute in float64.

    Python floats (and ``numpy.float64``, a float subclass) do; Python
    integers do up to 2**53, where float64 conversion is exact.  Lower or
    higher precision numbers (float16/float32/longdouble, numpy integers that
    can wrap) keep the scalar path, whose arithmetic the screen cannot mirror.
    """
    kind = type(value)
    if kind is float or kind is np.float64:
        return True
    if kind is int or kind is bool:
        return -(2 ** 53) <= value <= 2 ** 53
    return False


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
        self._bboxes: List[Rect] = []
        self._build()
        # Projected primitives are immutable for this index. Reuse their
        # deterministic occlusion order across all visibility sample pixels.
        self._front_keys = tuple((p.depth, -p.layer, -index)
                                 for index, p in enumerate(self.primitives))
        # Visibility samples are answered by screened, batched front queries
        # (see ``_front_many``); their caches live and die with this index.
        self._bbox_array = np.asarray(self._bboxes, dtype=np.float64).reshape((-1, 4))
        self._front_exact: Optional[bool] = None
        self._front_rank: Optional[np.ndarray] = None
        self._front_triangles: Optional[Tuple[np.ndarray, ...]] = None
        self._front_opaque: Dict[object, np.ndarray] = {}
        self._front_fine: Dict[object, np.ndarray] = {}
        self._front_retained = 0

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
            x0, y0, x1, y1 = bbox = primitive.bbox
            self._bboxes.append(bbox)
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

    def _tile_range(self, rect: Rect):
        x0, y0, x1, y1 = _rect_normalized(rect)
        if x1 < 0 or y1 < 0 or x0 > self.width or y0 > self.height:
            return None
        max_x = max(0, (self.width - 1) // self.tile_size)
        max_y = max(0, (self.height - 1) // self.tile_size)
        low_x = max(0, min(max_x, math.floor(x0 / self.tile_size)))
        high_x = max(0, min(max_x, math.floor(x1 / self.tile_size)))
        low_y = max(0, min(max_y, math.floor(y0 / self.tile_size)))
        high_y = max(0, min(max_y, math.floor(y1 / self.tile_size)))
        return low_x, high_x, low_y, high_y

    def _indices_for_rect(self, rect: Rect) -> List[int]:
        cells = self._tile_range(rect)
        if cells is None:
            return []
        low_x, high_x, low_y, high_y = cells
        found = set(self._global)
        for cell_y in range(low_y, high_y + 1):
            for cell_x in range(low_x, high_x + 1):
                found.update(self._tiles.get((cell_x, cell_y), ()))
        return sorted(found)

    _FINE_CELL = 4
    # Screening only ever drops a primitive that the exact scalar test would
    # also reject.  The half-plane screen is applied to pixels and triangles
    # within this coordinate range, where the rounding of both the screen
    # and the scalar test is orders of magnitude below the margins used.
    _SCREEN_LIMIT = 1.0e6
    # Retained per-cell candidate ids (8 bytes each) before caches restart.
    _FRONT_CACHE_IDS = 4_000_000
    # Distinct selection filters whose opacity masks are retained.
    _FRONT_FILTER_MASKS = 8

    def _front_exact_only(self) -> bool:
        """True when screened queries cannot reproduce the scalar order.

        A NaN depth or layer makes the front-to-back keys unordered, so a
        sort of a tile's candidates depends on their input order; and
        integer coordinates beyond 2**53 compare exactly in Python but not
        after conversion to float64.  Such indexes keep the scalar query.
        """
        exact = self._front_exact
        if exact is None:
            exact = any(
                key[0] != key[0] or key[1] != key[1] for key in self._front_keys
            ) or any(
                not _float_exact(value) for box in self._bboxes for value in box
            )
            self._front_exact = exact
        return exact

    def _rank(self) -> np.ndarray:
        """Position of every primitive in the deterministic front-to-back order."""
        rank = self._front_rank
        if rank is None:
            order = sorted(range(len(self.primitives)), key=self._front_keys.__getitem__)
            rank = np.empty(len(order), dtype=np.int64)
            rank[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)
            self._front_rank = rank
        return rank

    def _front_geometry(self) -> Tuple[np.ndarray, ...]:
        """Screening data of every primitive.

        Returns ``(screenable, origin, edge, length, scale)``.  ``screenable``
        marks counter-clockwise-normalised, clearly non-degenerate triangles
        inside the screening range; their edge origins, edge vectors, edge
        lengths and coordinate scale feed the conservative half-plane test.
        """
        geometry = self._front_triangles
        if geometry is None:
            count = len(self.primitives)
            corners = np.zeros((count, 3, 2), dtype=np.float64)
            screenable = np.zeros(count, dtype=bool)
            for index, primitive in enumerate(self.primitives):
                points = primitive.points
                if primitive.shape == "polygon" and len(points) == 3 and all(
                    _float_exact(value) for point in points for value in point
                ):
                    corners[index] = points
                    screenable[index] = True
            with np.errstate(all="ignore"):
                scale = np.max(np.abs(corners), axis=(1, 2))
                edge = np.roll(corners, -1, axis=1) - corners
                length = np.sqrt(np.einsum("tij,tij->ti", edge, edge))
                area = edge[:, 0, 0] * edge[:, 1, 1] - edge[:, 0, 1] * edge[:, 1, 0]
                # The area estimate's rounding is far below this bound, so a
                # screened triangle's orientation is certain.
                bound = 1.0e-9 * (length[:, 0] * length[:, 1] + scale * (length[:, 0] + length[:, 1])) + 1.0e-9
                screenable &= (
                    np.isfinite(area)
                    & (np.abs(area) > bound)
                    & (scale <= self._SCREEN_LIMIT)
                )
                flip = area < 0.0
                corners[flip] = corners[flip][:, ::-1]
                edge = np.roll(corners, -1, axis=1) - corners
                length = np.sqrt(np.einsum("tij,tij->ti", edge, edge))
            geometry = (screenable, corners, edge, length, scale)
            self._front_triangles = geometry
        return geometry

    def _opaque(self, selection_filter: Optional[SelectionFilter]) -> np.ndarray:
        """Primitives that can answer a front query under ``selection_filter``.

        Under a filter, point and line primitives whose owners the filter
        rejects are transparent: the scalar scan skips them unconditionally,
        so leaving them out cannot change its answer.
        """
        mask = self._front_opaque.get(selection_filter)
        if mask is None:
            if len(self._front_opaque) >= self._FRONT_FILTER_MASKS:
                # One boolean per primitive and filter: keep only a few.
                self._front_opaque.clear()
            if selection_filter is None:
                mask = np.ones(len(self.primitives), dtype=bool)
            else:
                mask = np.fromiter(
                    (
                        primitive.shape == "polygon"
                        or (
                            primitive.binding is not None
                            and any(selection_filter.accepts(owner) for owner in primitive.binding.owners)
                        )
                        for primitive in self.primitives
                    ),
                    dtype=bool,
                    count=len(self.primitives),
                )
            self._front_opaque[selection_filter] = mask
        return mask

    def _cache_ids(self, key: object, ids: np.ndarray) -> np.ndarray:
        """Retain candidate ids under a bounded total; restart when exceeded."""
        if self._front_retained + len(ids) > self._FRONT_CACHE_IDS:
            self._front_fine.clear()
            self._front_retained = 0
        self._front_fine[key] = ids
        self._front_retained += len(ids)
        return ids

    def _ordered_ids(self, ids: np.ndarray) -> np.ndarray:
        """Unique ids in the deterministic front-to-back order."""
        ids = np.unique(ids)
        return ids[np.argsort(self._rank()[ids], kind="stable")]

    def _fine_ids(
        self, cell: Tuple[int, int], selection_filter: Optional[SelectionFilter]
    ) -> np.ndarray:
        """Front-to-back candidates of one fine cell under one filter.

        Every primitive whose box reaches the cell is a member of the global
        list or of a tile overlapping the cell.  Fine cells use the same
        clamped floor mapping as tiles, so a query visits a superset of the
        primitives whose boxes reach its pixel.
        """
        key = (cell, selection_filter)
        ids = self._front_fine.get(key)
        if ids is not None:
            return ids
        fine = self._FINE_CELL
        fine_x, fine_y = cell
        tile = self.tile_size
        max_tile_x = max(0, (self.width - 1) // tile)
        max_tile_y = max(0, (self.height - 1) // tile)
        members = list(self._global)
        for tile_y in range(fine_y * fine // tile, min(max_tile_y, ((fine_y + 1) * fine - 1) // tile) + 1):
            for tile_x in range(fine_x * fine // tile, min(max_tile_x, ((fine_x + 1) * fine - 1) // tile) + 1):
                members.extend(self._tiles.get((tile_x, tile_y), ()))
        candidates = np.asarray(members, dtype=np.int64)
        if len(candidates):
            boxes = self._bbox_array[candidates]
            max_x = max(0, (self.width - 1) // fine)
            max_y = max(0, (self.height - 1) // fine)
            low_x = np.clip(np.floor(boxes[:, 0] / fine), 0, max_x)
            high_x = np.clip(np.floor(boxes[:, 2] / fine), 0, max_x)
            low_y = np.clip(np.floor(boxes[:, 1] / fine), 0, max_y)
            high_y = np.clip(np.floor(boxes[:, 3] / fine), 0, max_y)
            keep = (
                self._opaque(selection_filter)[candidates]
                & (low_x <= fine_x) & (fine_x <= high_x)
                & (low_y <= fine_y) & (fine_y <= high_y)
            )
            candidates = self._ordered_ids(candidates[keep])
        return self._cache_ids(key, candidates)

    def _cells_ids(
        self, cells: Tuple[Tuple[int, int], ...], selection_filter: Optional[SelectionFilter]
    ) -> np.ndarray:
        """One duplicate-free front-to-back candidate list over several fine cells."""
        if len(cells) == 1:
            return self._fine_ids(cells[0], selection_filter)
        key = (cells, selection_filter)
        ids = self._front_fine.get(key)
        if ids is None:
            parts = [self._fine_ids(cell, selection_filter) for cell in cells]
            ids = self._cache_ids(key, self._ordered_ids(np.concatenate(parts)))
        return ids

    def _fine_cells(
        self, point: Point2, radius: float
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        """Fine cells a front query at ``point`` visits (``None``: off screen)."""
        rect = (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)
        if self._tile_range(rect) is None:
            return None
        x0, y0, x1, y1 = _rect_normalized(rect)
        fine = self._FINE_CELL
        max_x = max(0, (self.width - 1) // fine)
        max_y = max(0, (self.height - 1) // fine)
        low_x = max(0, min(max_x, math.floor(x0 / fine)))
        high_x = max(0, min(max_x, math.floor(x1 / fine)))
        low_y = max(0, min(max_y, math.floor(y0 / fine)))
        high_y = max(0, min(max_y, math.floor(y1 / fine)))
        return tuple(
            (cell_x, cell_y)
            for cell_y in range(low_y, high_y + 1)
            for cell_x in range(low_x, high_x + 1)
        )

    def _front_reference(
        self,
        point: Point2,
        radius: float = 0.75,
        selection_filter: Optional[SelectionFilter] = None,
    ) -> Optional[int]:
        """The unscreened scalar front query: all tile candidates, sorted."""
        rect = (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)
        candidates = sorted(self._indices_for_rect(rect), key=self._front_keys.__getitem__)
        for index in candidates:
            primitive = self.primitives[index]
            x0, y0, x1, y1 = self._bboxes[index]
            if point[0] < x0-radius or point[0] > x1+radius or point[1] < y0-radius or point[1] > y1+radius:
                continue
            if selection_filter is not None and primitive.shape != "polygon":
                binding = primitive.binding
                if binding is None or not any(selection_filter.accepts(owner) for owner in binding.owners):
                    continue
            if _primitive_distance(primitive, point) <= radius:
                return index
        return None

    def _front_many(
        self,
        points: Sequence[Point2],
        radius: float = 0.75,
        selection_filter: Optional[SelectionFilter] = None,
    ) -> List[Optional[int]]:
        """Front primitive at each point, as :meth:`_front_reference` defines it.

        Points that visit the same fine cells share one front-to-back
        candidate list, walked in growing chunks.  In each chunk a vectorised
        bounding-box test (the scalar test's exact complement) and a
        conservative triangle half-plane test screen out primitives; the
        survivors are visited in order by the exact scalar test.  A point is
        answered by the first survivor that passes, and every candidate before
        it was either screened out or failed the exact test, so the answer is
        the one the scalar query gives.
        """
        results: List[Optional[int]] = [None] * len(points)
        if self._front_exact_only():
            return [self._front_reference(point, radius, selection_filter) for point in points]
        groups: Dict[Tuple[Tuple[int, int], ...], List[int]] = {}
        for position, point in enumerate(points):
            if not (_float_exact(point[0]) and _float_exact(point[1])):
                results[position] = self._front_reference(point, radius, selection_filter)
                continue
            cells = self._fine_cells(point, radius)
            if cells is not None:
                groups.setdefault(cells, []).append(position)
        for cells, members in groups.items():
            ids = self._cells_ids(cells, selection_filter)
            total = len(ids)
            unresolved = members
            start, size = 0, 32
            while unresolved and start < total:
                stop = min(total, start + size)
                unresolved = self._front_chunk(
                    points, unresolved, results, radius, selection_filter, ids[start:stop]
                )
                start, size = stop, size * 4
        return results

    def _front_chunk(
        self,
        points: Sequence[Point2],
        positions: List[int],
        results: List[Optional[int]],
        radius: float,
        selection_filter: Optional[SelectionFilter],
        indices: np.ndarray,
    ) -> List[int]:
        """Resolve ``positions`` against one front-to-back chunk; return the rest."""
        boxes = self._bbox_array[indices]
        screenable, origin, edge, length, scale = self._front_geometry()
        triangle = screenable[indices] if radius >= 0.0 else np.zeros(len(indices), dtype=bool)
        has_triangles = bool(triangle.any())
        unresolved: List[int] = []
        # Keep each screening matrix bounded (rows x members).
        step = max(1, 262_144 // max(1, len(indices)))
        for start in range(0, len(positions), step):
            rows = positions[start:start + step]
            coordinates = np.asarray(
                [points[position] for position in rows], dtype=np.float64
            ).reshape((-1, 2))
            x = coordinates[:, 0:1]
            y = coordinates[:, 1:2]
            # Scalar rejection was ``x < x0-r or x > x1+r or y < y0-r or
            # y > y1+r``; keep exactly its complement (NaN-safe).
            rejected = (
                (x < boxes[:, 0] - radius)
                | (x > boxes[:, 2] + radius)
                | (y < boxes[:, 1] - radius)
                | (y > boxes[:, 3] + radius)
            )
            row_index, column = np.nonzero(~rejected)
            if has_triangles and len(column):
                # A pixel farther than ``radius`` outside an edge line of a
                # triangle is farther than ``radius`` from the triangle.  The
                # margins (relative to the radius and to the coordinate
                # scale) are orders of magnitude above the rounding of this
                # estimate and of the exact scalar test inside the screening
                # range, so only primitives the exact test rejects are dropped.
                ids = indices[column]
                px = coordinates[row_index, 0]
                py = coordinates[row_index, 1]
                pixel_scale = np.maximum(np.abs(px), np.abs(py))
                active = triangle[column] & (pixel_scale <= self._SCREEN_LIMIT)
                pair_edge = edge[ids]
                pair_origin = origin[ids]
                with np.errstate(all="ignore"):
                    cross = (
                        pair_edge[:, :, 0] * (py[:, None] - pair_origin[:, :, 1])
                        - pair_edge[:, :, 1] * (px[:, None] - pair_origin[:, :, 0])
                    )
                    margin = radius * (1.0 + 1.0e-6) + 1.0e-6 + 1.0e-9 * (scale[ids] + pixel_scale)
                    outside = np.any(cross < -margin[:, None] * length[ids], axis=1)
                keep = ~(active & outside)
                row_index = row_index[keep]
                column = column[keep]
            counts = np.bincount(row_index, minlength=len(rows)).tolist()
            # Row-major order keeps the chunk's front-to-back order per row.
            flat = indices[column].tolist()
            offset = 0
            for position, count in zip(rows, counts):
                front = None
                if count:
                    front = self._front_scan(
                        points[position], flat[offset:offset + count],
                        radius, selection_filter,
                    )
                    offset += count
                if front is None:
                    unresolved.append(position)
                else:
                    results[position] = front
        return unresolved

    def _front_scan(
        self,
        point: Point2,
        candidates: Iterable[int],
        radius: float,
        selection_filter: Optional[SelectionFilter],
    ) -> Optional[int]:
        """First candidate, in the given order, within ``radius`` of ``point``."""
        for index in candidates:
            primitive = self.primitives[index]
            # Filtered point/line annotations are transparent; surfaces remain
            # occluders even when their owners do not match the selection filter.
            if selection_filter is not None and primitive.shape != "polygon":
                binding = primitive.binding
                if binding is None or not any(selection_filter.accepts(owner) for owner in binding.owners):
                    continue
            if _primitive_distance(primitive, point) <= radius:
                return index
        return None

    def _front_primitive_at(
        self,
        point: Point2,
        radius: float = 0.75,
        selection_filter: Optional[SelectionFilter] = None,
    ) -> Optional[int]:
        # The answer is the first primitive, in the deterministic
        # front-to-back order, whose exact distance to the pixel is within
        # ``radius``; farther primitives cannot change it.
        return self._front_many((point,), radius, selection_filter)[0]

    def _fronts(
        self, points: Sequence[Point2], selection_filter: SelectionFilter
    ) -> List[Optional[int]]:
        """Front primitives for visibility samples.

        A subclass that redefines :meth:`_front_primitive_at` (a reference
        implementation, for instance) keeps its own definition per point.
        """
        if type(self)._front_primitive_at is not ProjectedSelectionIndex._front_primitive_at:
            return [
                self._front_primitive_at(point, selection_filter=selection_filter)
                for point in points
            ]
        return self._front_many(points, selection_filter=selection_filter)

    def _visible_owners(
        self,
        owners: Dict[str, Sequence[int]],
        samples_of: Callable[[ProjectedPrimitive], Iterable[Point2]],
        selection_filter: SelectionFilter,
    ) -> set:
        """Owners with at least one sample whose front primitive is theirs.

        Samples are resolved in growing rounds over all still-undecided
        owners, so owners that are plainly visible stop after a few samples
        while hidden owners get their remaining samples batched.  The result
        is an existence test per owner and does not depend on the order.
        """

        def pairs(indices: Sequence[int]) -> Iterator[Tuple[int, Point2]]:
            for index in indices:
                for sample in samples_of(self.primitives[index]):
                    yield index, sample

        pending = {key: pairs(indices) for key, indices in owners.items()}
        visible = set()
        fronts: Dict[Point2, Optional[int]] = {}
        chunk = 4
        while pending:
            batch = {}
            for key, iterator in pending.items():
                items = list(islice(iterator, chunk))
                if items:
                    batch[key] = items
            pending = {key: pending[key] for key in batch}
            needed = list(dict.fromkeys(
                sample for items in batch.values() for _index, sample in items
                if sample not in fronts
            ))
            fronts.update(zip(needed, self._fronts(needed, selection_filter)))
            for key, items in batch.items():
                for index, sample in items:
                    if sample in fronts:
                        front = fronts[sample]
                    else:  # unhashable-equal values such as NaN
                        front = self._fronts((sample,), selection_filter)[0]
                    if front is None:
                        continue
                    binding = self.primitives[front].binding
                    if front == index or (
                        binding is not None
                        and any(owner.key == key for owner in binding.owners)
                    ):
                        visible.add(key)
                        del pending[key]
                        break
            chunk *= 4
        return visible

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

    @staticmethod
    def _rectangle_samples(primitive: ProjectedPrimitive, rect: Rect) -> Tuple[Point2, ...]:
        """Sample the portion of a line actually inside the gesture rectangle."""
        if primitive.shape != "segment":
            return ProjectedSelectionIndex._samples(primitive)
        start, end = primitive.points
        dx, dy = end[0]-start[0], end[1]-start[1]
        low, high = 0.0, 1.0
        for direction, distance in ((-dx,start[0]-rect[0]), (dx,rect[2]-start[0]),
                                    (-dy,start[1]-rect[1]), (dy,rect[3]-start[1])):
            if direction == 0:
                if distance < 0:
                    return ()
                continue
            fraction = distance/direction
            if direction < 0:
                low = max(low, fraction)
            else:
                high = min(high, fraction)
            if low > high:
                return ()
        return tuple((start[0]+dx*t, start[1]+dy*t) for t in ((low+high)*0.5,low,high))

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

        qualified_owners: Dict[str, List[int]] = {}
        for owner_key, qualified in owner_candidates.items():
            if not crossing:
                all_indices = self._owners.get(owner_key, ())
                if not all_indices or not all(
                    _primitive_inside_rect(self.primitives[index], area)
                    for index in all_indices
                ):
                    continue
                qualified = list(all_indices)
            qualified_owners[owner_key] = qualified
        visible_owners = None
        if depth == SelectionDepth.VISIBLE:
            visible_owners = self._visible_owners(
                qualified_owners,
                lambda primitive: [
                    sample for sample in self._rectangle_samples(primitive, area)
                    if _point_in_rect(sample, area)
                ],
                selection_filter,
            )

        hits: List[SelectionHit] = []
        for owner_key, qualified in qualified_owners.items():
            visible = True
            if visible_owners is not None and owner_key not in visible_owners:
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

        visible_owners = None
        if depth == SelectionDepth.VISIBLE:
            visible_owners = self._visible_owners(
                owner_candidates,
                lambda primitive: [
                    sample for sample in self._samples(primitive)
                    if _point_in_polygon(sample, polygon)
                ],
                selection_filter,
            )
        hits: List[SelectionHit] = []
        for owner_key, qualified in owner_candidates.items():
            visible = True
            if visible_owners is not None and owner_key not in visible_owners:
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
