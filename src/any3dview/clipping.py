"""World-space clipping primitives shared by rendering backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

from .core import Point3D, as_point


@dataclass(frozen=True, eq=False)
class SectionPlane:
    """A plane retaining points for which ``normal · point >= offset``.

    ``normal`` is interpreted as a direction and normalized without rescaling
    ``offset``.  The offset is therefore always a world-space distance along
    the unit normal.
    """

    normal: Point3D = Point3D(1.0, 0.0, 0.0)
    offset: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        normal = as_point(self.normal)
        components = normal.to_tuple()
        if not all(math.isfinite(value) for value in components):
            raise ValueError("section-plane normal must contain three finite values")
        length = normal.length()
        if length <= 0.0:
            raise ValueError("section-plane normal cannot be zero")
        offset = float(self.offset)
        if not math.isfinite(offset):
            raise ValueError("section-plane offset must be finite")
        object.__setattr__(self, "normal", normal / length)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "enabled", bool(self.enabled))

    @property
    def key(self) -> Tuple[float, float, float, float, bool]:
        return (*self.normal.to_tuple(), self.offset, self.enabled)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SectionPlane) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def signed_distance(self, point: Point3D | Sequence[float]) -> float:
        return self.normal.dot(as_point(point)) - self.offset

    def contains(self, point: Point3D | Sequence[float], *, tolerance: float = 1.0e-9) -> bool:
        return not self.enabled or self.signed_distance(point) >= -abs(float(tolerance))

    def clip_segment(
        self,
        start: Point3D | Sequence[float],
        end: Point3D | Sequence[float],
    ) -> Optional[Tuple[Point3D, Point3D]]:
        """Return the retained part of a segment, or ``None`` when discarded."""

        first = as_point(start)
        second = as_point(end)
        if not self.enabled:
            return first, second
        first_distance = self.signed_distance(first)
        second_distance = self.signed_distance(second)
        first_inside = first_distance >= 0.0
        second_inside = second_distance >= 0.0
        if first_inside and second_inside:
            return first, second
        if not first_inside and not second_inside:
            return None
        denominator = first_distance - second_distance
        if abs(denominator) <= 1.0e-15:
            return None
        parameter = first_distance / denominator
        intersection = first + (second - first) * parameter
        return (first, intersection) if first_inside else (intersection, second)

    def clip_polygon(
        self, points: Sequence[Point3D | Sequence[float]]
    ) -> Tuple[Point3D, ...]:
        """Clip a polygon with Sutherland-Hodgman half-space clipping."""

        polygon = tuple(as_point(point) for point in points)
        if not self.enabled or not polygon:
            return polygon
        result = []
        for index, current in enumerate(polygon):
            following = polygon[(index + 1) % len(polygon)]
            current_distance = self.signed_distance(current)
            following_distance = self.signed_distance(following)
            current_inside = current_distance >= 0.0
            following_inside = following_distance >= 0.0
            if current_inside:
                result.append(current)
            if current_inside != following_inside:
                denominator = current_distance - following_distance
                if abs(denominator) > 1.0e-15:
                    parameter = current_distance / denominator
                    result.append(current + (following - current) * parameter)
        return tuple(result)




@dataclass(frozen=True)
class ClipIntersection:
    """Intersection of normalized world half spaces (render and query contract)."""
    planes: tuple[SectionPlane, ...]

    @property
    def enabled(self): return any(p.enabled for p in self.planes)
    @property
    def key(self): return tuple(p.key for p in self.planes if p.enabled)
    def contains(self, point, *, tolerance=1e-9):
        return all(p.contains(point, tolerance=tolerance) for p in self.planes)
    def clip_polygon(self, points):
        for plane in self.planes:
            points=plane.clip_polygon(points)
            if not len(points): break
        return tuple(points)
    def clip_segment(self, start, end):
        segment=(start,end)
        for plane in self.planes:
            segment=plane.clip_segment(*segment)
            if segment is None: return None
        return segment

@dataclass(frozen=True)
class SectionBox:
    """Oriented visual section box; quaternion order is x,y,z,w."""
    center: Point3D
    half_extents: Point3D
    orientation: tuple = (0.,0.,0.,1.)
    enabled: bool = True

    def __post_init__(self):
        center,extents=as_point(self.center),as_point(self.half_extents)
        q=tuple(float(v) for v in self.orientation)
        if len(q)!=4 or not all(math.isfinite(v) for v in (*center.to_tuple(),*extents.to_tuple(),*q)):
            raise ValueError('section box requires finite center, extents and xyzw quaternion')
        if min(extents.to_tuple())<=0: raise ValueError('section box half extents must be positive')
        length=math.hypot(*q)
        if length<=1e-15: raise ValueError('section box quaternion cannot be zero')
        object.__setattr__(self,'center',center);object.__setattr__(self,'half_extents',extents)
        object.__setattr__(self,'orientation',tuple(v/length for v in q))
        object.__setattr__(self,'enabled',bool(self.enabled))

    @property
    def planes(self):
        x,y,z,w=self.orientation
        axes=((1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)),
              (2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)),
              (2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)))
        return tuple(SectionPlane(as_point(axis)*sign,as_point(axis).dot(self.center)*sign-extent,self.enabled)
                     for axis,extent in zip(axes,self.half_extents.to_tuple()) for sign in (1.,-1.))
    @property
    def key(self): return (*self.center.to_tuple(),*self.half_extents.to_tuple(),*self.orientation,self.enabled)
    def contains(self, point, **kwargs): return ClipIntersection(self.planes).contains(point,**kwargs)
    def clip_polygon(self, points): return ClipIntersection(self.planes).clip_polygon(points)
    def clip_segment(self, start,end): return ClipIntersection(self.planes).clip_segment(start,end)

__all__=['SectionPlane','SectionBox','ClipIntersection','CutCap']

@dataclass(frozen=True)
class CutCap:
    """Owner-qualified neutral display cap, never inferred solid/scalar truth.

    An installed provider certifies that its source is a supported closed volume.
    ``owners`` uses the existing MeshArrays semantic-owner registration contract.
    """
    mesh: object
    source_handle: object
    owners: object
