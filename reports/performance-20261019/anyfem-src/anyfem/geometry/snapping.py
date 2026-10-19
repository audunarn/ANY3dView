"""Deterministic workplane snapping for interactive construction.

The engine is Tk-free.  The viewport supplies a world point obtained from its
camera ray, while this module ranks grid and model candidates in workplane
coordinates.  Object-snap precedence is explicit and stable so a click gives
the same answer independent of dictionary insertion or rendering order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Iterable, Sequence

import numpy as np

from ..model.workplanes import Workplane, WorkplaneFrame

__all__ = [
    "GeometrySnapData",
    "SnapCandidate",
    "SnapEngine",
    "SnapKind",
    "SnapPoint",
    "SnapResult",
    "SnapSegment",
    "geometry_snap_data",
]


class SnapKind(str, Enum):
    ENDPOINT = "endpoint"
    INTERSECTION = "intersection"
    MIDPOINT = "midpoint"
    AXIS = "axis"
    GRID = "grid"


# Established CAD object snaps take precedence over construction aids.  Within
# one kind, nearest wins, followed by a stable source key.
_PRIORITY = {
    SnapKind.ENDPOINT: 0,
    SnapKind.INTERSECTION: 1,
    SnapKind.MIDPOINT: 2,
    SnapKind.AXIS: 3,
    SnapKind.GRID: 4,
}


def _position(value: Sequence[float], label: str = "snap position") -> tuple[float, float, float]:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label} needs three finite components")
    return tuple(float(item) for item in point)


@dataclass(frozen=True)
class SnapPoint:
    key: str
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        key = str(self.key)
        if not key:
            raise ValueError("a snap point needs a stable key")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "position", _position(self.position))


@dataclass(frozen=True)
class SnapSegment:
    key: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    owner_key: str = ""

    def __post_init__(self) -> None:
        key = str(self.key)
        if not key:
            raise ValueError("a snap segment needs a stable key")
        start = _position(self.start, "segment start")
        end = _position(self.end, "segment end")
        if np.linalg.norm(np.asarray(end) - np.asarray(start)) <= 1.0e-14:
            raise ValueError("a snap segment cannot have zero length")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "owner_key", str(self.owner_key or key))


@dataclass(frozen=True)
class GeometrySnapData:
    endpoints: tuple[SnapPoint, ...] = ()
    midpoints: tuple[SnapPoint, ...] = ()
    segments: tuple[SnapSegment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoints", tuple(self.endpoints))
        object.__setattr__(self, "midpoints", tuple(self.midpoints))
        object.__setattr__(self, "segments", tuple(self.segments))


@dataclass(frozen=True)
class SnapCandidate:
    kind: SnapKind
    position: tuple[float, float, float]
    distance: float
    source_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", SnapKind(self.kind))
        object.__setattr__(self, "position", _position(self.position))
        distance = float(self.distance)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("snap distance must be finite and non-negative")
        object.__setattr__(self, "distance", distance)
        object.__setattr__(
            self, "source_keys", tuple(sorted(str(item) for item in self.source_keys))
        )

    @property
    def rank(self) -> tuple[int, float, tuple[str, ...], tuple[float, float, float]]:
        return (
            _PRIORITY[self.kind],
            self.distance,
            self.source_keys,
            self.position,
        )


@dataclass(frozen=True)
class SnapResult:
    raw_position: tuple[float, float, float]
    position: tuple[float, float, float]
    candidate: SnapCandidate | None
    candidates: tuple[SnapCandidate, ...] = field(default_factory=tuple)

    @property
    def snapped(self) -> bool:
        return self.candidate is not None

    @property
    def kind(self) -> SnapKind | None:
        return None if self.candidate is None else self.candidate.kind


def _nearest_multiple(value: float, spacing: float) -> float:
    quotient = value / spacing
    lower = math.floor(quotient)
    upper = lower + 1
    # A half-grid tie goes to the lower integer for platform-independent
    # behavior (Python/NumPy rounding modes otherwise differ at half values).
    index = lower if quotient - lower <= upper - quotient else upper
    return float(index * spacing)


def _intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    first = first_end - first_start
    second = second_end - second_start
    denominator = float(first[0] * second[1] - first[1] * second[0])
    scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)), 1.0)
    if abs(denominator) <= 1.0e-12 * scale * scale:
        return None
    delta = second_start - first_start
    first_parameter = float(
        (delta[0] * second[1] - delta[1] * second[0]) / denominator
    )
    second_parameter = float(
        (delta[0] * first[1] - delta[1] * first[0]) / denominator
    )
    epsilon = 1.0e-12
    if not (
        -epsilon <= first_parameter <= 1.0 + epsilon
        and -epsilon <= second_parameter <= 1.0 + epsilon
    ):
        return None
    return first_start + np.clip(first_parameter, 0.0, 1.0) * first


class SnapEngine:
    """Rank construction candidates in one resolved workplane.

    Intersection work is bounded: only segments whose projected bounding boxes
    enter the cursor tolerance are considered, and at most
    ``max_intersection_segments`` participate.  This keeps a dense model from
    turning one mouse move into an unbounded all-pairs query.
    """

    def __init__(self, *, max_intersection_segments: int = 64) -> None:
        maximum = int(max_intersection_segments)
        if maximum < 2:
            raise ValueError("max_intersection_segments must be at least two")
        self.max_intersection_segments = maximum

    def snap(
        self,
        position: Sequence[float],
        workplane: Workplane,
        frame: WorkplaneFrame,
        data: GeometrySnapData | None = None,
    ) -> SnapResult:
        raw_world = frame.project(position)
        raw_local = frame.plane_coordinates(raw_world)
        tolerance = float(workplane.snap_tolerance)
        geometry = data or GeometrySnapData()
        candidates: list[SnapCandidate] = []

        def add(kind: SnapKind, local: Sequence[float], *sources: str) -> None:
            coordinates = np.asarray(local, dtype=float)
            distance = float(np.linalg.norm(coordinates - raw_local))
            if distance <= tolerance + max(1.0e-14, tolerance * 1.0e-12):
                candidates.append(
                    SnapCandidate(
                        kind,
                        _position(frame.world_position(coordinates)),
                        distance,
                        tuple(sources),
                    )
                )

        def add_points(kind: SnapKind, points: tuple[SnapPoint, ...]) -> None:
            if not points:
                return
            world = np.asarray([item.position for item in points], dtype=float)
            delta = world - frame.origin
            local = np.column_stack((delta @ frame.x_axis, delta @ frame.y_axis))
            distances = np.linalg.norm(local - raw_local, axis=1)
            accepted = np.flatnonzero(
                distances
                <= tolerance + max(1.0e-14, tolerance * 1.0e-12)
            )
            for index in accepted:
                add(kind, local[int(index)], points[int(index)].key)

        if workplane.snap_endpoints:
            add_points(SnapKind.ENDPOINT, geometry.endpoints)

        if workplane.snap_intersections and len(geometry.segments) >= 2:
            projected: list[tuple[float, str, str, np.ndarray, np.ndarray]] = []
            world_start = np.asarray(
                [segment.start for segment in geometry.segments], dtype=float
            )
            world_end = np.asarray(
                [segment.end for segment in geometry.segments], dtype=float
            )
            start_delta = world_start - frame.origin
            end_delta = world_end - frame.origin
            local_start = np.column_stack(
                (start_delta @ frame.x_axis, start_delta @ frame.y_axis)
            )
            local_end = np.column_stack(
                (end_delta @ frame.x_axis, end_delta @ frame.y_axis)
            )
            lower = np.minimum(local_start, local_end)
            upper = np.maximum(local_start, local_end)
            offsets = np.maximum(
                np.maximum(lower - raw_local, raw_local - upper), 0.0
            )
            bbox_distances = np.linalg.norm(offsets, axis=1)
            for index in np.flatnonzero(bbox_distances <= tolerance):
                item = int(index)
                segment = geometry.segments[item]
                projected.append(
                    (
                        float(bbox_distances[item]),
                        segment.key,
                        segment.owner_key,
                        local_start[item],
                        local_end[item],
                    )
                )
            projected.sort(key=lambda item: (item[0], item[1]))
            projected = projected[: self.max_intersection_segments]
            for index, first in enumerate(projected):
                for second in projected[index + 1 :]:
                    if first[2] == second[2]:
                        continue
                    crossing = _intersection(first[3], first[4], second[3], second[4])
                    if crossing is not None:
                        add(SnapKind.INTERSECTION, crossing, first[1], second[1])

        if workplane.snap_midpoints:
            add_points(SnapKind.MIDPOINT, geometry.midpoints)

        if workplane.snap_axes:
            add(SnapKind.AXIS, (raw_local[0], 0.0), "axis:x")
            add(SnapKind.AXIS, (0.0, raw_local[1]), "axis:y")

        if workplane.snap_grid:
            add(
                SnapKind.GRID,
                (
                    _nearest_multiple(float(raw_local[0]), workplane.grid_spacing),
                    _nearest_multiple(float(raw_local[1]), workplane.grid_spacing),
                ),
                "grid",
            )

        # Remove coincident duplicates of the same semantic kind.  Intersected
        # tessellation pieces can otherwise return a dozen equivalent points.
        unique: dict[tuple[SnapKind, tuple[float, float, float]], SnapCandidate] = {}
        for candidate in candidates:
            key = (
                candidate.kind,
                tuple(round(value, 12) for value in candidate.position),
            )
            previous = unique.get(key)
            if previous is None or candidate.rank < previous.rank:
                unique[key] = candidate
        ranked = tuple(sorted(unique.values(), key=lambda item: item.rank))
        chosen = ranked[0] if ranked else None
        raw_tuple = _position(raw_world)
        return SnapResult(
            raw_position=raw_tuple,
            position=raw_tuple if chosen is None else chosen.position,
            candidate=chosen,
            candidates=ranked,
        )


def geometry_snap_data(geometry, *, curve_segments: int = 16) -> GeometrySnapData:
    """Extract stable endpoint/midpoint/projected-segment candidates.

    Curves are represented by deterministic chords for intersection snapping;
    their midpoint comes from the exact public curve sampler.  This is a
    retained-data boundary suitable for rebuilding off the Tk event thread.
    """

    subdivisions = int(curve_segments)
    if subdivisions < 2:
        raise ValueError("curve_segments must be at least two")
    endpoints = tuple(
        SnapPoint(f"vertex:{identifier}", tuple(vertex.position))
        for identifier, vertex in sorted(geometry.vertices.items())
    )
    midpoints: list[SnapPoint] = []
    segments: list[SnapSegment] = []
    for identifier, edge in sorted(geometry.edges.items()):
        owner = f"edge:{identifier}"
        midpoint = np.asarray(
            geometry.sample_edge(identifier, np.asarray((0.5,), dtype=float))[0],
            dtype=float,
        )
        midpoints.append(SnapPoint(owner, tuple(midpoint)))
        straight = edge.curve.__class__.__name__.lower() == "straight"
        parameters = np.linspace(0.0, 1.0, 2 if straight else subdivisions + 1)
        samples = np.asarray(geometry.sample_edge(identifier, parameters), dtype=float)
        for index, (start, end) in enumerate(zip(samples, samples[1:])):
            if float(np.linalg.norm(end - start)) <= 1.0e-14:
                continue
            segments.append(
                SnapSegment(
                    f"{owner}/segment:{index}",
                    tuple(start),
                    tuple(end),
                    owner_key=owner,
                )
            )
    return GeometrySnapData(endpoints, tuple(midpoints), tuple(segments))
