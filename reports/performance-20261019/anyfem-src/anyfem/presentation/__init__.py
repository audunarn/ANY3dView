"""Renderer- and toolkit-neutral presentation API.

The existing Tk package retains compatibility imports during the frontend
migration.  New frontends should import presentation data from this namespace
rather than from :mod:`anyfem.ui`.
"""

from .live_progress import GRAPH_CHOICES, LiveProgressData
from .result_display import (
    DISPLAY_UNIT_SYSTEMS,
    ENGINEERING_DISPLAY,
    SI_DISPLAY,
    converted_series,
    unit_transform,
)
from .result_export import lazy_field_to_csv, pillow_available, save_gif, save_png
from .result_summary import NonlinearPathSummary, nonlinear_path_summary
from .scene import *
from .scene import __all__ as _scene_all
from .visualization import GEOMETRY_DETAILS, RENDER_MODES, VisualizationStyle

__all__ = [
    *_scene_all,
    "DISPLAY_UNIT_SYSTEMS",
    "ENGINEERING_DISPLAY",
    "GEOMETRY_DETAILS",
    "GRAPH_CHOICES",
    "LiveProgressData",
    "NonlinearPathSummary",
    "RENDER_MODES",
    "SI_DISPLAY",
    "VisualizationStyle",
    "converted_series",
    "lazy_field_to_csv",
    "nonlinear_path_summary",
    "pillow_available",
    "save_gif",
    "save_png",
    "unit_transform",
]
