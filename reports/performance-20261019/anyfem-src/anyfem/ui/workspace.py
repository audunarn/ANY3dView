"""Persistent Details workspace, selection strip and job summary widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from ..selection import SELECTION_KINDS, mode_label


_SELECTION_KIND_BY_LABEL = {
    "Point": "vertex",
    "Line": "edge",
    "Plate": "face",
    "Node": "node",
    "Element": "element",
    "Element face": "element_face",
}
_QUICK_FILTERS_BY_DOMAIN = {
    "Geometry": ("Point", "Line", "Plate"),
    "Mesh": ("Node", "Element", "Element face"),
}


def quick_filter_labels(domain: str) -> tuple[str, ...]:
    """Return the compact selection choices shown for one domain."""

    return _QUICK_FILTERS_BY_DOMAIN.get(str(domain), ())

__all__ = [
    "DetailsWorkspace",
    "JobStatusView",
    "SelectionStrip",
    "quick_filter_labels",
]


class DetailsWorkspace(ttk.Frame):
    """A commercial-style task/details host replacing stage notebook tabs."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._panels: dict[str, ttk.Frame] = {}
        self._buttons: dict[str, ttk.Button] = {}
        self._current: str | None = None
        self._on_select: Callable[[str], None] | None = None

        header = ttk.Frame(self, padding=(8, 6))
        header.pack(fill="x")
        self._title = ttk.Label(header, text="Details", font=("TkDefaultFont", 11, "bold"))
        self._title.pack(side="left")
        self._hint = ttk.Label(header, text="", foreground="#666666")
        self._hint.pack(side="right")

        self._navigation = ttk.Frame(self, padding=(6, 0, 6, 6))
        self._navigation.pack(fill="x")
        # Details pages contain progressively disclosed engineering forms and
        # can legitimately be taller than the window.  A plain Frame clips
        # everything below its allocated height.  Keep one scrollable host for
        # every page and Geometry sub-tab rather than teaching each panel its
        # own scrolling implementation.
        content_host = ttk.Frame(self)
        content_host.pack(fill="both", expand=True)
        background = ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0"
        self._canvas = tk.Canvas(
            content_host,
            borderwidth=0,
            highlightthickness=0,
            background=background,
        )
        self._scrollbar = ttk.Scrollbar(
            content_host, orient="vertical", command=self._canvas.yview
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.pack(side="right", fill="y")
        self._content = ttk.Frame(self._canvas)
        self._content_window = self._canvas.create_window(
            (0, 0), window=self._content, anchor="nw"
        )
        self._content.bind("<Configure>", self._content_configured)
        self._canvas.bind("<Configure>", self._canvas_configured)
        self._canvas.bind("<MouseWheel>", self._mousewheel, add="+")
        self._content.bind("<MouseWheel>", self._mousewheel, add="+")

    def add(self, panel: ttk.Frame, *, text: str) -> None:
        if text in self._panels:
            raise ValueError(f"details page {text!r} already exists")
        self._panels[text] = panel
        button = ttk.Button(
            self._navigation,
            text=text,
            command=lambda name=text: self.select(name),
        )
        button.pack(side="left", padx=1, pady=1)
        self._buttons[text] = button
        if self._current is None:
            self.select(text)

    def select(self, page: str | ttk.Frame) -> None:
        if not isinstance(page, str):
            name = next(
                (key for key, candidate in self._panels.items() if candidate is page),
                None,
            )
            if name is None:
                raise KeyError("panel is not registered in the Details workspace")
        else:
            name = page
        if name not in self._panels:
            raise KeyError(f"unknown Details page {name!r}")
        for key, panel in self._panels.items():
            if key == name:
                panel.pack(fill="both", expand=True)
                self._buttons[key].state(["disabled"])
            else:
                panel.pack_forget()
                self._buttons[key].state(["!disabled"])
        self._current = name
        self._title.configure(text=name)
        self._canvas.yview_moveto(0.0)
        if self._on_select is not None:
            self._on_select(name)

    def _content_configured(self, _event=None) -> None:
        requested = max(
            self._content.winfo_reqheight(), self._canvas.winfo_height()
        )
        self._canvas.itemconfigure(self._content_window, height=requested)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _canvas_configured(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._content_window, width=max(int(event.width), 1))
        self._content_configured()

    def _mousewheel(self, event: tk.Event) -> None:
        """Scroll only when the pointer is over this Details workspace."""

        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (AttributeError, tk.TclError):
            return
        while widget is not None and widget is not self:
            widget = getattr(widget, "master", None)
        if widget is self and event.delta:
            self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def scroll_to(self, widget: tk.Misc) -> None:
        """Bring a disclosed task section to the top of the visible area."""

        self.update_idletasks()
        y = 0
        current: tk.Misc | None = widget
        while current is not None and current is not self._content:
            y += int(current.winfo_y())
            current = getattr(current, "master", None)
        if current is not self._content:
            return
        total = max(self._content.winfo_height(), self._content.winfo_reqheight(), 1)
        self._canvas.yview_moveto(min(max(y / total, 0.0), 1.0))

    def set_select_handler(self, handler: Callable[[str], None] | None) -> None:
        """Run application view/context policy after a Details page is chosen."""

        self._on_select = handler

    def current(self) -> str | None:
        return self._current

    def set_hint(self, text: str) -> None:
        self._hint.configure(text=text)


class SelectionStrip(ttk.Frame):
    """Always-visible selection domain/filter/tool/depth controls."""

    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=(6, 3))
        self.app = app
        ttk.Label(self, text="Selection:").pack(side="left")

        self.domain = tk.StringVar(value="Geometry")
        domain_box = ttk.Combobox(
            self,
            textvariable=self.domain,
            values=("Geometry", "Mesh"),
            state="readonly",
            width=9,
        )
        domain_box.pack(side="left", padx=(4, 2))
        domain_box.bind("<<ComboboxSelected>>", self._set_domain)

        self.filter = tk.StringVar(value=mode_label(app.selection.mode))
        self._filter_box = ttk.Combobox(
            self,
            textvariable=self.filter,
            values=("Point", "Line", "Plate", "Node", "Element", "Element face"),
            state="readonly",
            width=12,
        )
        self._filter_box.pack(side="left", padx=2)
        self._filter_box.bind("<<ComboboxSelected>>", self._set_filter)

        self.tool = tk.StringVar(value="Box")
        tool_box = ttk.Combobox(
            self,
            textvariable=self.tool,
            values=("Single", "Box", "Lasso"),
            state="readonly",
            width=7,
        )
        tool_box.pack(side="left", padx=2)
        tool_box.bind("<<ComboboxSelected>>", self._selection_control_changed)

        self.depth = tk.StringVar(value="Visible")
        depth_box = ttk.Combobox(
            self,
            textvariable=self.depth,
            values=("Visible", "Through"),
            state="readonly",
            width=8,
        )
        depth_box.pack(side="left", padx=2)
        depth_box.bind("<<ComboboxSelected>>", self._selection_control_changed)

        self.operation = tk.StringVar(value="Replace")
        operation_box = ttk.Combobox(
            self,
            textvariable=self.operation,
            values=("Replace", "Add", "Remove", "Toggle"),
            state="readonly",
            width=8,
        )
        operation_box.pack(side="left", padx=2)
        operation_box.bind(
            "<<ComboboxSelected>>", self._selection_control_changed
        )

        self._count = ttk.Label(self, text="0 selected")
        self._count.pack(side="right")
        self._hint = ttk.Label(
            self,
            text="LMB select/drag  |  MMB pan  |  RMB orbit",
            foreground="#666666",
        )
        self._hint.pack(side="right", padx=12)

        ttk.Separator(self, orient="vertical").pack(
            side="left", fill="y", padx=(8, 6)
        )
        self._quick_frame = ttk.Frame(self)
        self._quick_frame.pack(side="left", fill="x", expand=True)
        self._quick_buttons: dict[str, ttk.Radiobutton] = {}
        for label in _SELECTION_KIND_BY_LABEL:
            button = ttk.Radiobutton(
                self._quick_frame,
                text=label,
                value=label,
                variable=self.filter,
                command=self._quick_filter_selected,
                style="Toolbutton",
                takefocus=True,
            )
            self._quick_buttons[label] = button
        self._show_quick_filters(self.domain.get())
        app.selection.add_listener(self.refresh)

    def refresh(self) -> None:
        domain = getattr(self.app.selection.domain, "value", "geometry")
        domain_label = "Mesh" if domain == "mesh" else "Geometry"
        self.domain.set(domain_label)
        self.filter.set(mode_label(self.app.selection.mode))
        self._show_quick_filters(domain_label)
        count = len(self.app.selection)
        self._count.configure(text=f"{count} selected")

    def set_context(self, kind: str, hint: str = "") -> None:
        self._activate_selection_mode()
        if kind in SELECTION_KINDS:
            self.app.selection.set_mode(kind)
        if hint:
            self._hint.configure(text=hint)
        self.refresh()
        self._apply_canvas()

    def _set_filter(self, _event=None) -> None:
        self._activate_selection_mode()
        kind = _SELECTION_KIND_BY_LABEL.get(self.filter.get())
        if kind is not None:
            self.app.selection.set_mode(kind)
        self.refresh()
        self._apply_canvas()

    def _set_domain(self, _event=None) -> None:
        self._activate_selection_mode()
        kind = "face" if self.domain.get() == "Geometry" else "element"
        self.app.selection.set_mode(kind)
        self.refresh()
        self._apply_canvas()

    def _quick_filter_selected(self) -> None:
        """Apply a toolbar radio choice through the canonical filter path."""

        self._set_filter()

    def _show_quick_filters(self, domain: str) -> None:
        """Show only choices valid for the selected geometry/mesh domain."""

        labels = quick_filter_labels(domain)
        self._filter_box.configure(values=labels)
        for button in self._quick_buttons.values():
            button.pack_forget()
        for label in labels:
            self._quick_buttons[label].pack(side="left", padx=1)

    def _selection_control_changed(self, _event=None) -> None:
        self._activate_selection_mode()
        self._apply_canvas()

    def _activate_selection_mode(self) -> None:
        """An explicit selection control leaves click-construction mode.

        Construction and selection both use LMB.  Keeping construction active
        after an engineer chooses Point/Line/Plate makes a valid hover appear
        clickable while every click is silently captured by the old task.
        """

        viewport = getattr(self.app, "viewport", None)
        cancel = getattr(viewport, "cancel_construction", None)
        if callable(cancel) and cancel():
            self.app.set_status(
                "click construction cancelled; selection controls are active"
            )

    def _apply_canvas(self) -> None:
        viewport = getattr(self.app, "viewport", None)
        if viewport is not None and hasattr(viewport, "configure_selection"):
            viewport.configure_selection(
                tool=self.tool.get().lower(),
                depth=self.depth.get().lower(),
                operation=self.operation.get().lower(),
            )


class JobStatusView(ttk.Frame):
    """Compact queued/running/completed job area below the viewport."""

    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=(4, 2))
        self.app = app
        ttk.Label(self, text="Jobs", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self._text = ttk.Label(self, text="No jobs", foreground="#555555")
        self._text.pack(side="left", padx=8)
        ttk.Button(self, text="Open", width=6, command=lambda: app.details.select("Solve")).pack(side="right")

    def refresh(self) -> None:
        manager = getattr(self.app, "job_manager", None)
        if manager is None:
            return
        tree = getattr(self.app, "tree", None)
        if tree is not None and hasattr(tree, "refresh_job_states"):
            tree.refresh_job_states()
        active = manager.active_job_id
        queued = len(manager.queued)
        if active:
            text = f"Running {active[:8]}  |  {queued} queued"
        elif queued:
            text = f"{queued} queued"
        elif getattr(self.app.project, "jobs", None):
            records = list(self.app.project.jobs.values())
            text = f"{len(records)} retained  |  {records[-1].status.value}"
        else:
            text = "No jobs"
        self._text.configure(text=text)
