"""Deterministic exports for retained result fields and viewport frames.

This module deliberately has no Tk dependency.  The Results panel can use it
for large, lazily stored fields without first materialising every frame, and
the formatting rules are straightforward to qualify in headless tests.
"""

from __future__ import annotations

import csv
import io
from numbers import Integral
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["lazy_field_to_csv", "pillow_available", "save_gif", "save_png"]


def write_client_export(path, text, cancel_event=None, *, source=''):
    """Atomic client transport; committed output survives later supersession."""
    from ..application.result_requests import CommittedResult
    from anypost.errors import RequestCancelled
    def check():
        if cancel_event is not None and cancel_event.is_set():raise RequestCancelled('export cancelled before commit')
    destination=Path(path);check();destination.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+destination.name+'.',suffix='.tmp',dir=destination.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8',newline='\n') as stream:
            for start in range(0,len(text),65536):check();stream.write(text[start:start+65536])
            stream.flush();os.fsync(stream.fileno())
        check();os.replace(temporary,destination)
        return CommittedResult(destination,source)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def pillow_available() -> bool:
    """Whether bitmap export support is installed."""

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def _format_number(value: Any) -> str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return "nan"
        if np.isposinf(number):
            return "inf"
        if np.isneginf(number):
            return "-inf"
        return format(number, ".17g")
    return str(value)


def _association(dataset, key: str, location: str, row_count: int):
    table_kind = "node" if location == "node" else "element"
    table_key = f"{key}_{table_kind}_ids"
    if table_key not in getattr(dataset, "table_keys", ()):
        return ("row_index",), np.arange(row_count, dtype=np.int64).reshape(-1, 1)

    identifiers = np.asarray(dataset.table(table_key))
    if identifiers.ndim == 0:
        identifiers = identifiers.reshape(1, 1)
    elif identifiers.ndim == 1:
        identifiers = identifiers.reshape(-1, 1)
    else:
        identifiers = identifiers.reshape(identifiers.shape[0], -1)
    if identifiers.shape[0] != row_count:
        raise ValueError(
            f"{key!r} has {row_count} value rows but {table_key!r} has "
            f"{identifiers.shape[0]} associations"
        )

    if identifiers.shape[1] == 1:
        names = (f"{table_kind}_id",)
    elif location == "element_face" and identifiers.shape[1] == 2:
        names = ("element_id", "face_id")
    else:
        names = tuple(
            f"{table_kind}_id_{index + 1}"
            for index in range(identifiers.shape[1])
        )
    return names, identifiers


def _value_layout(values: np.ndarray, components: Sequence[str]):
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    component_names = tuple(str(value) for value in components)
    if (
        component_names
        and array.ndim == 1
        and array.shape[0] == len(component_names)
    ):
        # A global vector is one row with components, whereas a spatial
        # vector has one row per node/element and therefore arrives as 2-D.
        array = array.reshape(1, -1)
    row_count = int(array.shape[0])
    if component_names:
        if array.ndim < 2 or array.shape[-1] != len(component_names):
            raise ValueError(
                "persisted field component metadata does not match its values"
            )
        subshape = array.shape[1:-1]
    else:
        subshape = array.shape[1:]
    return array, row_count, component_names, subshape


def lazy_field_to_csv(dataset, key: str, *, frame: int | None = None) -> str:
    """Return one retained HDF5 field as stable, lossless-enough CSV.

    If the artifact carries frame metadata, ``frame`` selects one frame and
    only that HDF5 chunk is read.  Pass ``None`` to export all stored frames.
    Entity association tables are used when present.  When they are absent we
    explicitly write ``row_index`` rather than pretending array positions are
    node or element IDs.
    """

    stored = dataset.field(key)
    descriptor = stored.descriptor
    shape = tuple(int(value) for value in stored.shape)
    frames = np.asarray(getattr(descriptor, "frames", ()), dtype=float).reshape(-1)
    if not len(frames):
        frames = np.asarray(getattr(dataset, "frames", ()), dtype=float).reshape(-1)
    has_frame_axis = bool(frames.size and shape and shape[0] == frames.size)
    provenance = getattr(descriptor, "provenance", {})
    frame_labels = provenance.get("frame_labels", provenance.get("load_cases", ()))
    if frame_labels and len(frame_labels) != len(frames):
        raise ValueError("persisted frame labels do not match frame coordinates")
    source_frame_indices = provenance.get("source_frame_indices")
    if source_frame_indices is not None and (not has_frame_axis or len(source_frame_indices) != len(frames)
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                   for value in source_frame_indices)):
        raise ValueError("persisted source frame indices do not match field frames")
    recovery_status = provenance.get("node_recovery_status")
    if recovery_status is not None and (not has_frame_axis or len(recovery_status) != len(frames)
                                        or descriptor.location != "node"):
        raise ValueError("persisted node recovery status does not match field frames/location")

    if has_frame_axis:
        if frame is None:
            frame_indices: Iterable[int] = range(len(frames))
        else:
            selected = int(frame)
            if selected < 0 or selected >= len(frames):
                raise IndexError(
                    f"frame {selected} is outside 0..{len(frames) - 1}"
                )
            frame_indices = (selected,)
    else:
        frame_indices = (0,)

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    header_written = False
    expected_header: tuple[str, ...] | None = None

    for frame_index in frame_indices:
        values = np.asarray(
            stored.read(frame_index) if has_frame_axis else stored.read(None)
        )
        sample_axes = getattr(descriptor, "provenance", {}).get("scalar_sample_axes")
        if sample_axes is not None:
            expected = list(range(2, len(shape)))
            if (descriptor.location != "integration_point" or len(descriptor.components) != 1
                    or list(sample_axes) != expected or len(shape) < 3):
                raise ValueError("persisted scalar sample layout is invalid")
            values = values[..., np.newaxis]
        values, row_count, components, subshape = _value_layout(
            values, descriptor.components
        )
        association_names, identifiers = _association(
            dataset, key, descriptor.location, row_count
        )
        frame_columns = ("frame_index", "frame_value") if has_frame_axis else ()
        if has_frame_axis and frame_labels:
            frame_columns += ("frame_label",)
        if source_frame_indices is not None:
            frame_columns += ("source_frame_index",)
        sub_columns = tuple(
            f"subindex_{index + 1}" for index in range(len(subshape))
        )
        if components:
            value_columns = tuple(
                f"{component} [{descriptor.unit}]" if descriptor.unit else component
                for component in components
            )
        else:
            label = descriptor.label or key
            value_columns = (
                f"{label} [{descriptor.unit}]" if descriptor.unit else label,
            )
        status_columns = ("recovery_status",) if recovery_status is not None else ()
        if status_columns and association_names != ("node_id",):
            raise ValueError("node recovery status requires explicit node associations")
        header = frame_columns + association_names + status_columns + sub_columns + value_columns
        if expected_header is None:
            expected_header = header
        elif expected_header != header:
            raise ValueError("persisted frames do not share one CSV layout")
        if not header_written:
            writer.writerow(header)
            header_written = True

        subindices = tuple(np.ndindex(subshape)) if subshape else ((),)
        for row_index in range(row_count):
            association = tuple(
                _format_number(value) for value in identifiers[row_index]
            )
            status_values = (str(recovery_status[frame_index].get(association[0], "unclassified")),) if status_columns else ()
            frame_values = (
                (str(frame_index), _format_number(frames[frame_index]))
                if has_frame_axis
                else ()
            )
            if has_frame_axis and frame_labels:
                frame_values += (str(frame_labels[frame_index]),)
            if source_frame_indices is not None:
                frame_values += (str(source_frame_indices[frame_index]),)
            for subindex in subindices:
                if components:
                    selected_values = np.asarray(
                        values[(row_index,) + subindex]
                    ).reshape(-1)
                else:
                    selected_values = np.asarray(
                        values[(row_index,) + subindex]
                    ).reshape(1)
                writer.writerow(
                    frame_values
                    + association
                    + status_values
                    + tuple(str(value) for value in subindex)
                    + tuple(_format_number(value) for value in selected_values)
                )

    return output.getvalue().rstrip("\n")


def save_png(image, path: str | Path) -> Path:
    """Write a captured Pillow image as RGB/RGBA PNG."""

    if not pillow_available():
        raise RuntimeError("PNG export needs Pillow; install ANYfem[gui]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted = image if getattr(image, "mode", "") in ("RGB", "RGBA") else image.convert("RGB")
    converted.save(destination, format="PNG")
    return destination


def save_gif(
    images: Sequence[Any],
    path: str | Path,
    *,
    duration_ms: int = 80,
    loop: int = 0,
) -> Path:
    """Write captured Pillow frames as a reproducible looping GIF."""

    if not pillow_available():
        raise RuntimeError("GIF export needs Pillow; install ANYfem[gui]")
    if not images:
        raise ValueError("cannot export an animation without captured frames")
    if int(duration_ms) <= 0:
        raise ValueError("GIF frame duration must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = [image.convert("RGBA") for image in images]
    descriptor,temporary=tempfile.mkstemp(prefix=".anyfem-gif-",suffix=".gif",dir=destination.parent)
    os.close(descriptor)
    try:
        frames[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=int(duration_ms),
            loop=int(loop),
            disposal=2,
            optimize=False,
        )
        os.replace(temporary,destination)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return destination
