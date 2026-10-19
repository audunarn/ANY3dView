"""An XY plot on a Tk canvas.

Hand-written rather than matplotlib, for the same reason ANYtk3D draws its 3D
viewport on a bare ``Canvas``: the GUI's whole dependency set stays Tk, which
is in the standard library, and a plot of two arrays with axes and a readout is
not enough work to justify a plotting stack as an install requirement.  What it
does not do -- log axes, multiple y scales, export to vector formats -- is what
would justify one, and none of it is needed to read a load-displacement path.

The arithmetic is deliberately separate from the drawing.  ``nice_ticks`` and
``map_to_canvas`` are module-level functions with no widget in sight, so the
part that can be wrong in a way a screenshot would not reveal is testable
without a display.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..post.history import Series

__all__ = ["HistoryPlot", "map_to_canvas", "nice_ticks", "padded_range"]

_MARGIN_LEFT = 64
_MARGIN_RIGHT = 16
_MARGIN_TOP = 22
_MARGIN_BOTTOM = 40

COLOR_AXIS = "#555555"
COLOR_GRID = "#e6e6e6"
COLOR_CURVE = "#1f5fa9"
COLOR_MARKER = "#c1440e"
COLOR_START = "#2e7d32"
COLOR_END = "#6a1b9a"
COLOR_ACTIVE = "#00838f"
COLOR_TEXT = "#333333"


def nice_ticks(low: float, high: float, count: int = 5) -> List[float]:
    """Round tick values spanning a range.

    Steps are 1, 2 or 5 times a power of ten, which is what makes axis labels
    readable: a tick at 0.0237 is arithmetically fine and useless to look at.
    A degenerate range still yields a single tick rather than an empty axis or
    a division by zero.
    """

    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("tick range must be finite")
    if count < 2:
        raise ValueError("a tick count below two describes no axis")
    if high < low:
        low, high = high, low
    span = high - low
    if span <= 0.0:
        return [low]

    raw = span / (count - 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))

    # Choose the step whose tick count lands nearest the request, rather than
    # the first step at least as large as the ideal one.  That rule sounds
    # equivalent and is not: a span of 10 asking for 5 ticks has an ideal step
    # of 2.5, rounds up to 5, and draws three ticks on an axis that had room
    # for six.  Ties go to the smaller step, since a slightly busy axis reads
    # better than a bare one.
    best_step = magnitude
    best_score: Optional[Tuple[int, float]] = None
    for multiple in (1.0, 2.0, 5.0, 10.0):
        step = multiple * magnitude
        produced = math.floor(high / step) - math.ceil(low / step) + 1
        score = (abs(produced - count), step)
        if best_score is None or score < best_score:
            best_score, best_step = score, step
    step = best_step

    # Stay inside the range: a tick past the end would draw its grid line
    # outside the plot box.  The tolerance is only there to keep an endpoint
    # that float division lands a hair beyond.
    tolerance = 1.0e-9 * max(1.0, abs(low), abs(high))
    ticks: List[float] = []
    value = math.ceil(low / step) * step
    # Guard the loop on a count as well as the float comparison, so a
    # pathological step cannot spin.
    while value <= high + tolerance and len(ticks) <= 4 * count:
        # Snap away the float noise that turns 0.30000000000000004 into a label.
        ticks.append(round(value / step) * step)
        value += step
    return ticks or [low]


def padded_range(values: Sequence[float], pad: float = 0.05) -> Tuple[float, float]:
    """A drawing range for a set of values, with a margin.

    A flat series gets a range around its value rather than a zero-width one,
    so a constant curve draws as a line through the middle instead of vanishing
    onto an axis.  Non-finite values are ignored: an imported result can carry
    NaN where a component is absent, and one of those must not blank the plot.
    """

    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0
    low, high = float(data.min()), float(data.max())
    if high <= low:
        margin = abs(low) * pad or 1.0
        return low - margin, low + margin
    margin = (high - low) * pad
    return low - margin, high + margin


def map_to_canvas(
    values: Sequence[float],
    low: float,
    high: float,
    start: float,
    end: float,
    *,
    invert: bool = False,
) -> List[float]:
    """Map data values onto canvas pixels along one axis.

    ``invert`` for the vertical axis, where canvas y grows downward and data y
    grows upward.
    """

    span = high - low
    if span == 0.0:
        span = 1.0
    scaled = [(float(value) - low) / span for value in values]
    if invert:
        return [end - fraction * (end - start) for fraction in scaled]
    return [start + fraction * (end - start) for fraction in scaled]


def _format_tick(value: float) -> str:
    """A short label for a tick value."""

    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1.0e5 or magnitude < 1.0e-3:
        return f"{value:.1e}"
    return f"{value:.4g}"


class HistoryPlot(ttk.Frame):
    """A single-series XY plot with axes, a grid and a readout.

    One series at a time, chosen from a list, rather than several overlaid: the
    curves a result offers have unlike units -- metres against newtons against
    a dimensionless load factor -- and putting them on one axis would only look
    like a comparison.
    """

    def __init__(self, master: tk.Misc, width: int = 460, height: int = 260) -> None:
        super().__init__(master)
        self._series: List[Series] = []
        self._current: Optional[Series] = None
        self._active_index: Optional[int] = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="series", width=8).pack(side="left")
        self._choice = tk.StringVar()
        self._box = ttk.Combobox(
            top, textvariable=self._choice, state="readonly", width=28
        )
        self._box.pack(side="left", fill="x", expand=True)
        self._box.bind("<<ComboboxSelected>>", lambda _event: self._draw())

        self.canvas = tk.Canvas(
            self, width=width, height=height, background="white",
            highlightthickness=1, highlightbackground="#cccccc",
        )
        self.canvas.pack(fill="both", expand=True, pady=(4, 2))
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda _event: self._readout.set(""))

        self._readout = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._readout, foreground="#666666").pack(
            anchor="w"
        )

    # ------------------------------------------------------------------
    def show(self, series: Sequence[Series]) -> None:
        """Display a set of series, selecting the first."""

        self._active_index = None
        self._series = [item for item in series if not item.is_empty]
        names = [item.name for item in self._series]
        self._box.configure(values=names)
        if names:
            self._choice.set(names[0])
        else:
            self._choice.set("")
        self._draw()

    def clear(self) -> None:
        self._active_index = None
        self.show([])

    def set_active_index(self, index: Optional[int]) -> None:
        """Mark the displayed saved frame on the selected history curve."""

        self._active_index = None if index is None else int(index)
        self._draw()

    @property
    def series_names(self) -> List[str]:
        return [item.name for item in self._series]

    def select(self, name: str) -> None:
        if name not in self.series_names:
            raise KeyError(f"no series named {name!r}")
        self._choice.set(name)
        self._draw()

    # ------------------------------------------------------------------
    def _selected(self) -> Optional[Series]:
        wanted = self._choice.get()
        for item in self._series:
            if item.name == wanted:
                return item
        return self._series[0] if self._series else None

    def _plot_box(self) -> Tuple[float, float, float, float]:
        width = max(int(self.canvas.winfo_width()), 120)
        height = max(int(self.canvas.winfo_height()), 100)
        return (
            _MARGIN_LEFT,
            _MARGIN_TOP,
            width - _MARGIN_RIGHT,
            height - _MARGIN_BOTTOM,
        )

    def _draw(self) -> None:
        self.canvas.delete("all")
        series = self._selected()
        self._current = series
        left, top, right, bottom = self._plot_box()
        if right <= left or bottom <= top:
            return
        if series is None:
            self.canvas.create_text(
                0.5 * (left + right), 0.5 * (top + bottom),
                text="nothing to plot", fill="#999999",
            )
            return

        x_low, x_high = padded_range(series.x)
        y_low, y_high = padded_range(series.y)

        for value in nice_ticks(y_low, y_high):
            y = map_to_canvas([value], y_low, y_high, top, bottom, invert=True)[0]
            self.canvas.create_line(left, y, right, y, fill=COLOR_GRID)
            self.canvas.create_text(
                left - 6, y, text=_format_tick(value), anchor="e",
                fill=COLOR_TEXT, font=("TkDefaultFont", 7),
            )
        for value in nice_ticks(x_low, x_high):
            x = map_to_canvas([value], x_low, x_high, left, right)[0]
            self.canvas.create_line(x, top, x, bottom, fill=COLOR_GRID)
            self.canvas.create_text(
                x, bottom + 6, text=_format_tick(value), anchor="n",
                fill=COLOR_TEXT, font=("TkDefaultFont", 7),
            )

        self.canvas.create_rectangle(
            left, top, right, bottom, outline=COLOR_AXIS
        )
        self.canvas.create_text(
            0.5 * (left + right), bottom + 24,
            text=self._axis_label(series.x_label, series.x_unit),
            fill=COLOR_TEXT, font=("TkDefaultFont", 8),
        )
        self.canvas.create_text(
            left - 52, 0.5 * (top + bottom),
            text=self._axis_label(series.y_label, series.y_unit),
            fill=COLOR_TEXT, font=("TkDefaultFont", 8), angle=90,
        )

        xs = map_to_canvas(series.x, x_low, x_high, left, right)
        ys = map_to_canvas(series.y, y_low, y_high, top, bottom, invert=True)
        points: List[float] = []
        for x, y in zip(xs, ys):
            if math.isfinite(x) and math.isfinite(y):
                points.extend((x, y))
        if len(points) >= 4:
            self.canvas.create_line(*points, fill=COLOR_CURVE, width=2)
        elif len(points) == 2:
            self.canvas.create_oval(
                points[0] - 2, points[1] - 2, points[0] + 2, points[1] + 2,
                outline=COLOR_CURVE, fill=COLOR_CURVE,
            )

        peak_x, peak_y = series.peak()
        marker_x = map_to_canvas([peak_x], x_low, x_high, left, right)[0]
        marker_y = map_to_canvas(
            [peak_y], y_low, y_high, top, bottom, invert=True
        )[0]
        self.canvas.create_oval(
            marker_x - 3, marker_y - 3, marker_x + 3, marker_y + 3,
            outline=COLOR_MARKER, width=2,
        )
        start_x, start_y = xs[0], ys[0]
        end_x, end_y = xs[-1], ys[-1]
        self.canvas.create_oval(
            start_x - 3, start_y - 3, start_x + 3, start_y + 3,
            outline=COLOR_START, fill=COLOR_START,
        )
        self.canvas.create_oval(
            end_x - 4, end_y - 4, end_x + 4, end_y + 4,
            outline=COLOR_END, width=2,
        )
        active_text = ""
        if self._active_index is not None and len(series):
            active = min(max(self._active_index, 0), len(series) - 1)
            active_x, active_y = xs[active], ys[active]
            self.canvas.create_oval(
                active_x - 5, active_y - 5, active_x + 5, active_y + 5,
                outline=COLOR_ACTIVE, width=3,
            )
            active_text = f"   |   CURRENT {series.y[active]:.4g}"
        self.canvas.create_text(
            0.5 * (left + right), top - 10,
            text=(
                f"START {series.y[0]:.4g}   |   "
                f"PEAK {peak_y:.4g} at {peak_x:.4g}   |   "
                f"LAST {series.y[-1]:.4g}{active_text}"
            ),
            fill=COLOR_TEXT, font=("TkDefaultFont", 8, "bold"),
        )

    @staticmethod
    def _axis_label(label: str, unit: str) -> str:
        return f"{label} [{unit}]" if unit else label

    def _hover(self, event: tk.Event) -> None:
        """Report the data point nearest the pointer."""

        series = self._current
        if series is None or series.is_empty:
            return
        left, top, right, bottom = self._plot_box()
        if not (left <= event.x <= right and top <= event.y <= bottom):
            self._readout.set("")
            return
        x_low, x_high = padded_range(series.x)
        fraction = (event.x - left) / max(right - left, 1)
        target = x_low + fraction * (x_high - x_low)
        index = int(np.argmin(np.abs(np.asarray(series.x, dtype=float) - target)))
        self._readout.set(
            f"{series.x_label} {series.x[index]:.5g}"
            f"{' ' + series.x_unit if series.x_unit else ''}   "
            f"{series.y_label} {series.y[index]:.5g}"
            f"{' ' + series.y_unit if series.y_unit else ''}"
        )
