"""Named, immutable views of requested native result quantities.

Native fields stay available for the workbench. Each requested view is a real
subset with independent association/frame metadata, not a filter on live state.
No basis transformation, recovery, or unavailable value is invented here.
"""

from __future__ import annotations

from dataclasses import replace
from urllib.parse import quote
import numpy as np

from ..model.records import OutputRequest

_BASIS_ALIASES = {"local": "element_local", "element": "element_local"}


def _matches(key, descriptor, quantity):
    if quantity == key:
        return True, None
    quantity = quantity.replace(":", ".")
    for separator in (".", "_"):
        if quantity.startswith(key + separator):
            component = quantity[len(key) + 1:]
            if component in descriptor.components:
                return True, component
    family, separator, component = quantity.partition(".")
    if not separator:
        return key.startswith(family + "_"), None
    if key == family and component in descriptor.components:
        return True, component
    if descriptor.recovery == "patch" and key == family + "_patch_" + component:
        return True, None
    return key == family + "_" + component, None


def _view(descriptor, values, source_key, scope, request, component, tables):
    aliases = {"average": "mean", "abs_max": "max_abs"}
    reduction = aliases.get(request.reduction, request.reduction)
    stored_reduction = aliases.get(descriptor.reduction, descriptor.reduction)
    reduce_samples = (
        descriptor.location == "integration_point" and request.location == "element"
        and reduction in {"mean", "min", "max", "max_abs"}
        and stored_reduction == "none"
    )
    if descriptor.location != request.location and not reduce_samples:
        raise ValueError(f"location {descriptor.location!r} does not match {request.location!r}")
    requested_basis = _BASIS_ALIASES.get(request.basis, request.basis)
    stored_basis = _BASIS_ALIASES.get(descriptor.basis, descriptor.basis)
    if requested_basis != stored_basis:
        raise ValueError(f"basis {request.basis!r} is unavailable (stored {descriptor.basis!r})")
    if request.recovery not in {"native", descriptor.recovery}:
        raise ValueError(f"recovery {request.recovery!r} is unavailable (stored {descriptor.recovery!r})")
    if reduction != "none" and reduction != stored_reduction and not reduce_samples:
        raise ValueError(f"reduction {request.reduction!r} requires an explicit qualified quantity")
    if request.frame_policy == "selected" and not request.frame_indices:
        raise ValueError("selected-frame requests require explicit frame indices; none are recorded")

    data = np.asarray(values)
    sample_axes = descriptor.provenance.get("scalar_sample_axes")
    if sample_axes is not None:
        if (descriptor.location != "integration_point" or len(descriptor.components) != 1
                or list(sample_axes) != list(range(2, data.ndim)) or data.ndim < 3):
            raise ValueError("stored scalar sample layout is invalid")
        data = data[..., np.newaxis]
    frames = tuple(descriptor.frames)
    if not frames or data.shape[0] != len(frames):
        raise ValueError("stored quantity has no unambiguous frame association")
    association = None
    identifiers = None
    missing = []
    if request.location in {"node", "element", "integration_point", "element_face"}:
        association = "node_ids" if request.location == "node" else "element_ids"
        table = tables.get(f"{source_key}_{association}")
        if table is None:
            raise ValueError("stored quantity has no explicit entity association")
        source_association = np.asarray(table, dtype=np.int64)
        if source_association.ndim == 1:
            source_ids = source_association
        elif request.location == "element_face" and source_association.ndim == 2 and source_association.shape[1] == 2:
            source_ids = source_association[:, 0]
        else:
            raise ValueError("stored entity association has an unsupported layout")
        if data.ndim < 2 or data.shape[1] != len(source_ids):
            raise ValueError("stored entity association does not match the field")
        wanted = scope[association]
        missing = sorted(set(wanted).difference(map(int, source_ids)))
        indices = np.flatnonzero(np.isin(source_ids, wanted))
        if request.location == "element_face" and scope.get("element_faces"):
            if source_association.ndim != 2:
                raise ValueError("stored quantity has no explicit element-face association")
            faces = {tuple(item) for item in scope["element_faces"]}
            missing = sorted(faces.difference(map(tuple, source_association)))
            indices = np.asarray([index for index in indices
                                  if tuple(source_association[index]) in faces], dtype=np.intp)
        if not len(indices):
            raise ValueError("requested region has no entities carrying this quantity")
        data = np.take(data, indices, axis=1)
        identifiers = source_association[indices]

    components = descriptor.components
    unit = descriptor.unit
    if component is not None:
        if data.shape[-1] != len(components):
            raise ValueError("stored component association does not match the field")
        index = components.index(component)
        data = data[..., index:index + 1]
        if unit.startswith("mixed:"):
            units = unit.removeprefix("mixed:").split(",")
            if len(units) == len(components):
                unit = units[index]
            elif len(units) == 2:
                unit = units[1 if component in {"rx", "ry", "rz", "mx", "my", "mz"} else 0]
        components = (component,)
    if reduce_samples:
        if sample_axes is None or not all(data.shape[axis] for axis in sample_axes):
            raise ValueError("integration-point reduction requires explicit nonempty sample axes")
        # Keep frames, entities and components separate. Match post.fields._reduce,
        # including the sign of the first maximum-absolute sample on a tie.
        samples = data.reshape(data.shape[:2] + (-1, data.shape[-1]))
        # NaN marks absent samples, including padding. Keep entirely missing
        # rows as NaN so spatial readers leave them unavailable and selected
        # frames with valid data remain accessible.
        if reduction == "mean":
            data = np.nanmean(samples, axis=2)
        elif reduction == "min":
            data = np.nanmin(samples, axis=2)
        elif reduction == "max":
            data = np.nanmax(samples, axis=2)
        else:
            magnitudes = np.where(np.isnan(samples), -np.inf, np.abs(samples))
            indices = np.argmax(magnitudes, axis=2, keepdims=True)
            data = np.take_along_axis(samples, indices, axis=2).squeeze(axis=2)
    source_frame_indices = tuple(range(len(frames)))
    if request.frame_policy in {"first", "last"}:
        index = 0 if request.frame_policy == "first" else len(frames) - 1
        source_frame_indices = (index,)
        data = data[index:index + 1]
        frames = (frames[index],)
    elif request.frame_policy == "selected":
        if any(index >= len(frames) for index in request.frame_indices):
            raise ValueError(f"requested frame indices {request.frame_indices} exceed stored range 0..{len(frames)-1}")
        data = np.take(data, request.frame_indices, axis=0)
        source_frame_indices = request.frame_indices
        frames = tuple(frames[index] for index in request.frame_indices)
    elif request.frame_policy == "envelope":
        # Match ANYfem's existing signed maximum-absolute envelope convention.
        indices = np.argmax(np.abs(data), axis=0)
        data = np.take_along_axis(data, indices[np.newaxis, ...], axis=0)
        frames = (0.0,)
        source_frame_indices = ()

    provenance = {
        **descriptor.provenance,
        "output_request": request.to_dict(),
        "source_quantity": source_key,
        "scope": dict(scope),
        "frame_policy": request.frame_policy,
        "missing_requested_entities": missing,
    }
    provenance.pop("source_frame_indices", None)
    if request.frame_policy != "envelope":
        provenance["source_frame_indices"] = list(source_frame_indices)
    for name in ("step_indices", "frame_recovery", "owner_recovery", "case_recovery"):
        items = descriptor.provenance.get(name)
        if items is None:
            continue
        if len(items) != len(descriptor.frames):
            raise ValueError(f"stored {name} has no unambiguous frame association")
        if request.frame_policy == "envelope":
            provenance.pop(name, None)
            provenance[f"envelope_source_{name}"] = list(items)
        else:
            provenance[name] = [items[index] for index in source_frame_indices]
    if sample_axes is not None:
        # Requested views have an explicit component axis; the native scalar
        # layout marker must not make readers append another component axis.
        provenance.pop("scalar_sample_axes", None)
        provenance["source_scalar_sample_axes"] = list(sample_axes)
    case_labels = descriptor.provenance.get("load_cases", ())
    if len(case_labels) == len(descriptor.frames):
        provenance["frame_labels"] = (
            [] if request.frame_policy == "envelope" else
            [case_labels[index] for index in source_frame_indices]
        )
    statuses = descriptor.provenance.get("node_recovery_status")
    if statuses is not None:
        if len(statuses) != len(descriptor.frames) or request.location != "node":
            raise ValueError("stored node recovery status has no unambiguous frame association")
        if request.frame_policy == "envelope":
            selected_statuses = [{str(node): "qualified" if all(
                item.get(str(node)) == "qualified" for item in statuses) else "fallback"
                for node in identifiers}]
        else:
            selected_statuses = [statuses[index] for index in source_frame_indices]
        provenance["node_recovery_status"] = [
            {str(node): item.get(str(node), "unclassified") for node in identifiers}
            for item in selected_statuses]
        provenance["unqualified_requested_nodes"] = sorted({int(node)
            for item in provenance["node_recovery_status"] for node, status in item.items()
            if status != "qualified"})
        provenance["unqualified_node_ids"] = provenance["unqualified_requested_nodes"]
    if reduce_samples:
        provenance["sample_reduction"] = reduction
    if request.frame_policy == "envelope":
        provenance["envelope_source_frames"] = list(descriptor.frames)
        provenance["envelope_source_frame_indices"] = list(range(len(descriptor.frames)))
        provenance["envelope_convention"] = "signed maximum absolute value per entry"
    return replace(
        descriptor, label=f"{request.label}: {descriptor.label}", components=components,
        location=request.location, reduction=reduction if reduce_samples else descriptor.reduction,
        unit=unit, frames=frames, deformation_required=False, provenance=provenance,
    ), data, association, identifiers


def add_output_request_views(payload, scopes):
    """Materialize only available requested subsets and disclose refusals."""

    fields = dict(payload.fields)
    tables = dict(payload.tables)
    outcomes = []
    for scope in scopes:
        request = OutputRequest.from_dict(scope["request"])
        outcome = {"request_id": request.id, "label": request.label, "fields": [], "diagnostics": []}
        for quantity in request.quantity_keys:
            matched = False
            for key, (descriptor, values) in payload.fields.items():
                if descriptor.recovery == "patch" and request.recovery != "patch":
                    continue
                matches, component = _matches(key, descriptor, quantity)
                if not matches:
                    continue
                if request.recovery == "patch" and descriptor.recovery != "patch":
                    continue
                matched = True
                try:
                    view, data, association, identifiers = _view(
                        descriptor, values, key, scope, request, component, payload.tables,
                    )
                except (KeyError, ValueError, IndexError) as error:
                    outcome["diagnostics"].append(f"{quantity} ({key}): {error}")
                    continue
                view_key = "request_" + ".".join(quote(value, safe="") for value in (request.id, quantity, key))
                fields[view_key] = (replace(view, key=view_key), data)
                if association is not None:
                    tables[f"{view_key}_{association}"] = identifiers
                outcome["fields"].append(view_key)
                if view.provenance["missing_requested_entities"]:
                    outcome["diagnostics"].append(
                        f"{quantity} ({key}): missing requested entities "
                        f"{view.provenance['missing_requested_entities']}"
                    )
                if view.provenance.get("unqualified_requested_nodes"):
                    outcome["diagnostics"].append(
                        f"{quantity} ({key}): owner patch recovery has fallback/unclassified values at nodes "
                        f"{view.provenance['unqualified_requested_nodes']}")
            if not matched:
                outcome["diagnostics"].append(f"{quantity}: unavailable in the retained solver output (recovery {request.recovery})")
        outcome["status"] = (
            "partial" if outcome["fields"] and outcome["diagnostics"] else
            "available" if outcome["fields"] else "unavailable"
        )
        outcomes.append(outcome)
    tables["output_request_outcomes"] = outcomes
    return replace(payload, fields=fields, tables=tables,
                   summary={**payload.summary, "field_keys": sorted(fields),
                            "output_request_outcomes": outcomes},
                   provenance={**payload.provenance, "output_request_outcomes": outcomes})
