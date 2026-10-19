"""Collect owner measurements without changing prepared or materialized state."""
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Mapping


@contextmanager
def collect_geometry_runtime():
    """Use optional owner instrumentation while supporting older geometry wheels."""
    try:
        from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
    except ModuleNotFoundError as error:
        if error.name != "anygeometry.runtime_diagnostics":
            raise
        yield {}
    else:
        with collect_runtime_diagnostics() as counts:
            yield counts


def merge_geometry_runtime(mesh, counts):
    """Add outer preparation counts to the existing execution-only mesh block."""
    if not counts:
        return
    diagnostics = getattr(mesh, "hybrid_diagnostics", {})
    diagnostics = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
    runtime = diagnostics.get("runtime", {})
    runtime = deepcopy(dict(runtime)) if isinstance(runtime, Mapping) else {}
    operations = dict(runtime.get("operation_counts", {}))
    for name, value in counts.items():
        key = f"geometry.{name}"
        operations[key] = int(operations.get(key, 0)) + int(value)
    runtime["operation_counts"] = operations
    diagnostics["runtime"] = runtime
    mesh.hybrid_diagnostics = diagnostics
