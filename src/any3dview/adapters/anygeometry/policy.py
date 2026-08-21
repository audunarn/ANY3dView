"""Renderer-neutral ANYgeometry display policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DisplayMode(str, Enum):
    STRUCTURAL = "structural"
    GEOMETRY = "geometry"
    TOPOLOGY_DEBUG = "topology_debug"
    RELATIONSHIPS = "relationships"
    COMBINED = "combined"


@dataclass(frozen=True, slots=True)
class TessellationPolicy:
    chord_tolerance: float = 1.0e-3
    relative_chord_tolerance: float = 1.0e-3
    angular_tolerance: float = 0.15
    max_curve_segments: int = 128
    lod_levels: int = 3
    preserve_boundaries: bool = True

    def __post_init__(self) -> None:
        if self.chord_tolerance <= 0 or self.relative_chord_tolerance <= 0:
            raise ValueError("tessellation tolerances must be positive")
        if self.angular_tolerance <= 0:
            raise ValueError("angular_tolerance must be positive")
        if self.max_curve_segments < 2:
            raise ValueError("max_curve_segments must be at least two")
        if self.lod_levels < 1:
            raise ValueError("lod_levels must be positive")


@dataclass(frozen=True, slots=True)
class DisplayPolicy:
    mode: DisplayMode = DisplayMode.STRUCTURAL
    tessellation: TessellationPolicy = TessellationPolicy()
    lod: int = 1
    chunk_span: int = 256
    external_coordinates: bool = False
    threaded_updates: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", DisplayMode(self.mode))
        if not isinstance(self.tessellation, TessellationPolicy):
            raise TypeError("tessellation must be TessellationPolicy")
        if self.lod < 0 or self.lod >= self.tessellation.lod_levels:
            raise ValueError("lod is outside configured tessellation levels")
        if self.chunk_span < 1:
            raise ValueError("chunk_span must be positive")
