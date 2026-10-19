"""Non-mutating point, line and polyline construction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np
from anygeometry.entities import EntityRef

from ..commands import AddLine, AddPoint, AddPolyline, GeometryCommand
from ..model.project import Project
from ..model.workplanes import WorkplaneFrame
from .snapping import SnapResult

__all__ = [
    "ConstructionMode",
    "ConstructionResult",
    "ConstructionTask",
    "CoordinateConstruction",
]


class ConstructionMode(str, Enum):
    POINT = "point"
    LINE = "line"
    POLYLINE = "polyline"


def _coordinate(value: Sequence[float]) -> tuple[float, float, float]:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("construction point needs three finite coordinates")
    return tuple(float(item) for item in point)


@dataclass(frozen=True)
class ConstructionResult:
    vertices: tuple[EntityRef, ...]
    edges: tuple[EntityRef, ...] = ()


@dataclass(eq=False)
class CoordinateConstruction(GeometryCommand):
    """Commit coordinate-driven geometry as one atomic undo item.

    Internally this composes the established feature-recording ``AddPoint``,
    ``AddLine`` and ``AddPolyline`` commands.  Construction therefore does not
    create an opaque special topology path, while the user still receives one
    Apply and one undo operation.
    """

    coordinates: Sequence[Sequence[float]]
    mode: ConstructionMode | str = ConstructionMode.POLYLINE
    close: bool = False
    label: str = "construct geometry"

    def __post_init__(self) -> None:
        GeometryCommand.__init__(self)
        self.mode = ConstructionMode(self.mode)
        self.coordinates = tuple(_coordinate(item) for item in self.coordinates)
        required = {
            ConstructionMode.POINT: 1,
            ConstructionMode.LINE: 2,
            ConstructionMode.POLYLINE: 2,
        }[self.mode]
        if self.mode in (ConstructionMode.POINT, ConstructionMode.LINE):
            if len(self.coordinates) != required:
                raise ValueError(
                    f"{self.mode.value} construction needs exactly {required} point(s)"
                )
        elif len(self.coordinates) < required:
            raise ValueError("polyline construction needs at least two points")
        if self.close and (
            self.mode is not ConstructionMode.POLYLINE
            or len(self.coordinates) < 3
        ):
            raise ValueError("only a polyline with at least three points can close")
        self._commands: list[GeometryCommand] = []
        self._result: ConstructionResult | None = None

    def operate(self, project: Project) -> ConstructionResult:  # pragma: no cover
        # ``do`` intentionally sequences established commands instead of
        # using GeometryCommand's one-operation implementation.
        raise NotImplementedError

    def do(self, project: Project) -> ConstructionResult:
        commands: list[GeometryCommand] = []
        vertex_ids: list[int] = []
        edge_ids: list[int] = []
        try:
            for coordinate in self.coordinates:
                command = AddPoint(*coordinate)
                vertex_ids.append(int(command.do(project)))
                commands.append(command)
            if self.mode is ConstructionMode.LINE:
                line = AddLine(vertex_ids[0], vertex_ids[1])
                edge_ids.append(int(line.do(project)))
                commands.append(line)
            elif self.mode is ConstructionMode.POLYLINE:
                polyline = AddPolyline(tuple(vertex_ids), close=bool(self.close))
                edge_ids.extend(int(item) for item in polyline.do(project))
                commands.append(polyline)
        except BaseException:
            for command in reversed(commands):
                command.undo(project)
            raise
        self._commands = commands
        self._result = ConstructionResult(
            tuple(EntityRef("vertex", item) for item in vertex_ids),
            tuple(EntityRef("edge", item) for item in edge_ids),
        )
        return self._result

    def undo(self, project: Project) -> None:
        for command in reversed(self._commands):
            command.undo(project)

    def redo(self, project: Project) -> ConstructionResult:
        vertex_ids: list[int] = []
        edge_ids: list[int] = []
        for command in self._commands:
            result = command.redo(project)
            if isinstance(command, AddPoint):
                vertex_ids.append(int(result))
            elif isinstance(command, (AddLine, AddPolyline)):
                if isinstance(result, (tuple, list)):
                    edge_ids.extend(int(item) for item in result)
                else:
                    edge_ids.append(int(result))
        self._result = ConstructionResult(
            tuple(EntityRef("vertex", item) for item in vertex_ids),
            tuple(EntityRef("edge", item) for item in edge_ids),
        )
        return self._result


class ConstructionTask:
    """Working-copy clicks and previews for one construction command."""

    def __init__(
        self,
        mode: ConstructionMode | str = ConstructionMode.POLYLINE,
        *,
        close: bool = False,
    ) -> None:
        self.mode = ConstructionMode(mode)
        self.close = bool(close)
        if self.close and self.mode is not ConstructionMode.POLYLINE:
            raise ValueError("only polyline construction can close")
        self._points: list[tuple[float, float, float]] = []
        self._snaps: list[SnapResult | None] = []
        self.cancelled = False

    @property
    def points(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self._points)

    @property
    def snap_results(self) -> tuple[SnapResult | None, ...]:
        return tuple(self._snaps)

    @property
    def ready(self) -> bool:
        if self.mode is ConstructionMode.POINT:
            return len(self._points) == 1
        if self.mode is ConstructionMode.LINE:
            return len(self._points) == 2
        return len(self._points) >= (3 if self.close else 2)

    @property
    def accepts_more(self) -> bool:
        return self.mode is ConstructionMode.POLYLINE or not self.ready

    @property
    def preview_segments(
        self,
    ) -> tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ]:
        pairs = list(zip(self._points, self._points[1:]))
        if self.close and len(self._points) >= 3:
            pairs.append((self._points[-1], self._points[0]))
        return tuple(pairs)

    def add(self, value: Sequence[float] | SnapResult) -> tuple[float, float, float]:
        if not self.accepts_more:
            raise ValueError(
                f"{self.mode.value} construction already has all required points"
            )
        if isinstance(value, SnapResult):
            point = _coordinate(value.position)
            snap = value
        else:
            point = _coordinate(value)
            snap = None
        self._points.append(point)
        self._snaps.append(snap)
        self.cancelled = False
        return point

    def add_plane_coordinates(
        self,
        u: float,
        v: float,
        frame: WorkplaneFrame,
    ) -> tuple[float, float, float]:
        """Add an exact numeric point in active-workplane coordinates."""

        values = np.asarray((u, v), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("numeric workplane coordinates must be finite")
        return self.add(frame.world_position(values))

    def backspace(self) -> bool:
        if not self._points:
            return False
        self._points.pop()
        self._snaps.pop()
        return True

    def reset(self) -> None:
        self._points.clear()
        self._snaps.clear()
        self.cancelled = False

    def cancel(self) -> None:
        self._points.clear()
        self._snaps.clear()
        self.cancelled = True

    def command(self) -> CoordinateConstruction:
        if not self.ready:
            minimum = {
                ConstructionMode.POINT: 1,
                ConstructionMode.LINE: 2,
                ConstructionMode.POLYLINE: 3 if self.close else 2,
            }[self.mode]
            raise ValueError(
                f"{self.mode.value} construction needs at least {minimum} point(s)"
            )
        return CoordinateConstruction(self.points, self.mode, self.close)

    def apply(self, execute: Callable[[CoordinateConstruction], Any]) -> Any:
        """Execute one atomic command and clear only after successful Apply."""

        command = self.command()
        result = execute(command)
        self.reset()
        return result
