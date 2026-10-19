"""Opt-in fixed-source responses with conservative original-time fallback."""
from __future__ import annotations

import ast
from copy import copy
from dataclasses import dataclass, replace
from decimal import Decimal
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..model.attributes import LoadCase
from ..model.expressions import (BUILTIN_CONSTANTS, COORDINATE_NAMES,
    check_expression, evaluate_expression, expression_names)
from ..model.project import ProjectError
from . import build as builder
from .prepared_static import (PreparedStaticEvaluator, _CAPTURED_LOAD_LISTS,
    _capture_external_case, _capture_request, _input_storage_bytes, _prepare_linear_static_evaluator)
from .static_batch import StaticLoadRequest, _resolve_record

__all__ = ["PeriodicLoadSource", "PeriodicSample", "PatternLoadRequest",
           "PreparedStaticResponseEvaluator", "prepare_linear_static_response_evaluator"]


@dataclass(frozen=True)
class PeriodicLoadSource:
    """Declared periodic wet-pressure source on an original, uniform time grid.

    Envelopes stay at original time. All period phases are prepared, including
    phases absent from a subsequently requested subset. Unsupported source
    expressions/families use direct original-time cases with a diagnostic.
    """
    load_case: str
    period: float
    time_step: float
    times: Sequence[float] | None = None
    origin: float = 0.0


@dataclass(frozen=True)
class PeriodicSample:
    case_id: str
    time: float


@dataclass(frozen=True)
class PatternLoadRequest:
    case_id: str
    coefficients: Any
    time: float | None = None
    canonical_request: Any = None


def _periodic_expression(expression, parameters, period):
    """Conservative structural admission using the existing safe evaluator.

    Every time occurrence must be inside sin/cos of an affine time argument
    classified as a declared-period harmonic. Clipping and time-free spatial
    expressions remain verbatim. The harmonic tolerance is classification,
    not a proof of recurrence or source equivalence. The solver compares the
    actual original-time full load vector for every periodic sample.
    """
    check_expression(expression)
    if expression_names(expression) - set(parameters) - set(BUILTIN_CONSTANTS) - set(COORDINATE_NAMES) - {"t"}:
        raise ValueError("periodic source uses unsupported sample-dependent names")
    tree = ast.parse(expression, mode="eval")

    def has_time(node):
        return any(isinstance(item, ast.Name) and item.id == "t" for item in ast.walk(node))

    def constant(node):
        text = ast.unparse(node)
        if expression_names(text) - set(parameters) - set(BUILTIN_CONSTANTS):
            raise ValueError("time slope depends on spatial or unknown data")
        return float(evaluate_expression(text, parameters))

    def slope(node):
        if not has_time(node):
            return 0.0
        if isinstance(node, ast.Name) and node.id == "t":
            return 1.0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return slope(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return slope(node.left) + (-1 if isinstance(node.op, ast.Sub) else 1) * slope(node.right)
            if isinstance(node.op, ast.Mult):
                if not has_time(node.left):
                    return constant(node.left) * slope(node.right)
                if not has_time(node.right):
                    return slope(node.left) * constant(node.right)
            if isinstance(node.op, ast.Div) and not has_time(node.right):
                return slope(node.left) / constant(node.right)
        raise ValueError("time argument is not affine")

    def visit(node):
        if not has_time(node):
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("sin", "cos"):
            if len(node.args) != 1:
                raise ValueError("periodic trigonometric source must have one argument")
            harmonic = slope(node.args[0]) * period / (2 * math.pi)
            if not math.isfinite(harmonic) or not math.isclose(harmonic, round(harmonic), rel_tol=0., abs_tol=1e-12):
                raise ValueError("expression frequency does not match declared period")
            return
        if isinstance(node, ast.Name) and node.id == "t":
            raise ValueError("time occurs outside an admitted periodic function")
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)


def _sample_case(source, time, index, times):
    case = copy(source)
    case.times = None
    case.parent = None
    case._window_times = times
    case.time = float(time)
    case.step = int(index)
    return case


def _unit_case(source, family, load, time, index, times):
    case = _sample_case(source, time, index, times)
    for name in _CAPTURED_LOAD_LISTS:
        setattr(case, name, [])
    case.gravity = None
    setattr(case, family, [replace(load, envelope=None)])
    return case


def _phase(time, origin, step, count):
    if not math.isfinite(float(time)):
        return None
    position = (Decimal(str(float(time))) - Decimal(str(origin))) / Decimal(str(step))
    if position != position.to_integral_value():
        return None
    return int(position) % count


def _prepare_sources(owned, context, token, patterns, baseline, periodic, memory_limit=None):
    from anysolver.control import cancellation_safe_point
    native = {}
    retained = _input_storage_bytes((owned.project, owned.mesh, owned.connections))

    def build(request):
        nonlocal retained
        cancellation_safe_point(token, "prepared_response.pattern_load")
        context.clear_case_caches()
        try:
            result = _resolve_record(owned.project, owned, context, _capture_request(request)).load_case
            retained += _input_storage_bytes(result)
            if memory_limit is not None and retained > memory_limit:
                raise ProjectError("prepared response input library exceeds memory_limit_bytes")
            return result
        finally:
            context.clear_case_caches()

    for name, request in patterns.items():
        if type(name) is not str or not name or name.startswith("__periodic__"):
            raise ValueError("pattern names must be nonempty strings outside reserved __periodic__ prefix")
        native[name] = build(request)
    native_baseline = None if baseline is None else build(baseline)
    state = {"periodic": None, "pattern_names": tuple(native), "pattern_build_count": len(native)}
    if periodic is not None:
        period, step, origin = float(periodic.period), float(periodic.time_step), float(periodic.origin)
        if not all(math.isfinite(value) for value in (period, step, origin)) or period <= 0 or step <= 0:
            raise ValueError("period, time_step and origin must be finite, with positive period and step")
        ratio = Decimal(str(period)) / Decimal(str(step))
        if ratio < 1 or ratio != ratio.to_integral_value():
            raise ValueError("period must contain an integral number of declared grid steps")
        count = int(ratio)
        if memory_limit is not None and count * 64 > memory_limit - retained:
            raise ProjectError("declared phase grid exceeds memory_limit_bytes")
        # This is a source declaration, intentionally allowed to name a series
        # definition. The ordinary case resolver correctly refuses definitions.
        source = owned.project.load_cases.get(periodic.load_case)
        if source is None:
            raise ProjectError("periodic source case is missing")
        source = _capture_external_case(owned.project.resolve_case(source))
        declared = periodic.times if periodic.times is not None else source.times or ()
        original = source.times if source.times is not None else declared
        if memory_limit is not None and (len(declared) + len(original)) * 64 > memory_limit - retained:
            raise ProjectError("declared original sample catalog exceeds memory_limit_bytes")
        times = tuple(float(value) for value in declared)
        original_times = tuple(float(value) for value in original)
        if not times or not all(math.isfinite(value) for value in original_times + times) or len(set(original_times)) != len(original_times):
            raise ValueError("periodic source requires a finite original grid with unique times")
        indices = {time: i for i, time in enumerate(original_times)}
        if any(time not in indices for time in times):
            raise ValueError("declared sample is absent from the source's original times")
        reason = None
        try:
            if any(_phase(value, origin, step, count) is None for value in original_times):
                raise ValueError("declared source times do not lie exactly on the original decimal grid")
            if source.gravity is not None or any(getattr(source, key) for key in
                    ("point_loads", "pressures", "line_loads", "surface_tractions")):
                raise ValueError("periodic v1 admits wet expressions and fixed connection loads only")
            if not source.wet_pressures:
                raise ValueError("periodic source has no wet expression")
            for load in source.wet_pressures:
                if load.expression is None:
                    raise ValueError("periodic wet source must use a safe expression")
                _periodic_expression(load.expression, source.parameters, period)
        except (ValueError, TypeError) as error:
            reason = str(error)
        # No nearest-phase snapping. Decimal construction matches declared
        # decimal grids; actual original samples override equal phase slots.
        phase_times = [] if reason is not None else [float(Decimal(str(origin)) + index * Decimal(str(step))) for index in range(count)]
        seen = set()
        for time in original_times:
            phase = _phase(time, origin, step, count)
            if reason is None and phase not in seen:
                phase_times[phase] = time
                seen.add(phase)
        names = []
        if reason is None:
            for load_index, load in enumerate(source.wet_pressures):
                load_names = []
                for phase, time in enumerate(phase_times):
                    name = f"__periodic__wet:{load_index}:phase:{phase}"
                    native[name] = build(_unit_case(source, "wet_pressures", load, time, indices.get(time, phase), original_times))
                    load_names.append(name)
                names.append(tuple(load_names))
            for index, load in enumerate(source.connection_loads):
                name = f"__periodic__connection:{index}"
                native[name] = build(_unit_case(source, "connection_loads", load, original_times[0], 0, original_times))
        state["periodic"] = {"source": source, "times": times, "original_times": original_times, "time_indices": indices,
            "period": period, "step": step, "origin": origin, "phase_count": count,
            "phase_times": tuple(phase_times), "wet_names": tuple(names), "direct_reason": reason}
    state["pattern_names"] = tuple(native)
    state["pattern_build_count"] = len(native)
    return {"patterns": native, "baseline_load_case": native_baseline}, state


class PreparedStaticResponseEvaluator(PreparedStaticEvaluator):
    def _configure_prepared(self, state):
        self._response_state = state
        self._response_counts = {"periodic_pattern_samples": 0, "periodic_direct_samples": 0}
        periodic = state["periodic"]
        self._periodic_diagnostics = None if periodic is None else {key: periodic[key] for key in
            ("period", "step", "origin", "phase_count", "phase_times", "direct_reason")}

    def close(self):
        try:
            super().close()
        finally:
            if hasattr(self, "_response_state"):
                self._response_state = {**self._response_state, "periodic": None}

    def diagnostics(self):
        result = super().diagnostics()
        result["route"] = "anyfem.prepared_static_response"
        result["response"] = {"pattern_names": self._response_state["pattern_names"],
            "pattern_build_count": self._response_state["pattern_build_count"],
            "source_equivalence": "periodic recurrence unproved; periodic samples supply original-time canonical loads; fixed patterns define their own source",
            **self._response_counts,
            "periodic": None if self._periodic_diagnostics is None else dict(self._periodic_diagnostics)}
        return result

    def iter_batches(self, load_cases=None, **kwargs):
        periodic = self._response_state["periodic"]
        if load_cases is None and periodic is not None:
            load_cases = (PeriodicSample(f"{periodic['source'].id}:{periodic['time_indices'][time]}", time)
                          for time in periodic["times"])
        return super().iter_batches(load_cases, **kwargs)

    def _resolve_prepared_request(self, request):
        from anysolver import PatternCase, StaticCase
        if type(request) in (PatternCase, StaticCase):
            return request
        if isinstance(request, PatternLoadRequest):
            canonical = None if request.canonical_request is None else super()._resolve_prepared_request(request.canonical_request).load_case
            return PatternCase(request.case_id, request.coefficients, request.time, canonical_load_case=canonical)
        if not isinstance(request, PeriodicSample):
            return super()._resolve_prepared_request(request)
        periodic = self._response_state["periodic"]
        if periodic is None:
            raise ValueError("PeriodicSample requires a declared periodic source")
        time = float(request.time)
        if type(request.case_id) is not str or not request.case_id or not math.isfinite(time):
            raise ValueError("periodic sample requires a nonempty ID and finite original time")
        index = periodic["time_indices"].get(time)
        # Off-grid samples have no declared case-number identity. Require callers
        # to supply an ordinary StaticLoadRequest for these arbitrary times.
        if index is None:
            raise ValueError("periodic sample time is outside the declared original grid; use StaticLoadRequest")
        case = _sample_case(periodic["source"], time, index, periodic["original_times"])
        record = _resolve_record(self._owned.project, self._owned, self._context,
                                 StaticLoadRequest(request.case_id, load_case=case, time=time))
        if periodic["direct_reason"] is not None:
            self._response_counts["periodic_direct_samples"] += 1
            return record
        phase = _phase(time, periodic["origin"], periodic["step"], periodic["phase_count"])
        coefficients = {}
        for names, load in zip(periodic["wet_names"], case.wet_pressures):
            factor = builder._envelope_factor(case, load)
            if factor != 0:
                coefficients[names[phase]] = factor
        for load_index, load in enumerate(case.connection_loads):
            factor = builder._envelope_factor(case, load)
            if factor != 0:
                coefficients[f"__periodic__connection:{load_index}"] = factor
        self._response_counts["periodic_pattern_samples"] += 1
        return PatternCase(request.case_id, coefficients, time, canonical_load_case=record.load_case)


def prepare_linear_static_response_evaluator(project=None, *, patterns: Mapping[str, Any] | None = None,
        baseline_load_case: Any = None, periodic_source: PeriodicLoadSource | None = None, **kwargs):
    """Prepare opt-in source responses; default direct APIs are unchanged.

    Patterns are named ANYfem cases/StaticLoadRequests. Periodic v1 retains the
    original-time canonical load for every sample; no recurrence equivalence or
    cheaper governing-envelope certificate is claimed.
    """
    patterns = {} if patterns is None else dict(patterns)
    if periodic_source is not None and not isinstance(periodic_source, PeriodicLoadSource):
        raise TypeError("periodic_source must be PeriodicLoadSource")
    if periodic_source is not None and baseline_load_case is not None:
        raise ValueError("periodic source already defines its full canonical load; supply its baseline within the source")
    resources = kwargs.get("resource_config")
    memory_limit = None if resources is None else resources.memory_limit_bytes
    def prepare(owned, context, token):
        return _prepare_sources(owned, context, token, patterns, baseline_load_case, periodic_source, memory_limit)
    return _prepare_linear_static_evaluator(project, _factory_name="prepare_static_response_evaluator",
        _prepare_application=prepare, _adapter_type=PreparedStaticResponseEvaluator, **kwargs)
