"""Physical beam axis ordering, independent of GUI and FE subdivision."""
from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError


def ordered_beam_chain(geometry, edge_ids):
    ids = tuple(dict.fromkeys(edge_ids))
    if not ids:raise GeometryError("Select beam axis lines")
    incident = {}
    for identifier in ids:
        if identifier not in geometry.edges:raise GeometryError(f"No beam axis edge {identifier}")
        edge = geometry.edges[identifier]
        for vertex in (edge.start, edge.end):incident.setdefault(vertex, []).append(identifier)
    ends = [vertex for vertex, edges in incident.items() if len(edges) == 1]
    if len(ends) != 2 or any(len(edges) > 2 for edges in incident.values()):
        raise GeometryError("A beam axis must be one open, unbranched connected chain")
    # Stable direction independent of box-selection ordering.
    vertex = min(ends)
    remaining = set(ids)
    result = []
    while remaining:
        candidates = [identifier for identifier in incident[vertex] if identifier in remaining]
        if len(candidates) != 1:raise GeometryError("Beam axis lines are disconnected")
        identifier = candidates[0]
        edge = geometry.edges[identifier]
        forward = edge.start == vertex
        result.append(OrientedEdge(identifier, forward))
        vertex = edge.end if forward else edge.start
        remaining.remove(identifier)
    return tuple(result)
