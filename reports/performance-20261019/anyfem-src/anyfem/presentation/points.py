"""Persistent presentation of deliberate points in models with generated topology."""
USER_POINT_GROUP = "anyfem:user_points"


def displayed_geometry_point_ids(geometry):
    """Use an explicit point group when present; preserve ordinary model display.

    Individual point commands remain visible even in a model whose generated
    vertices are omitted. Entity IDs always come from the geometry owner.
    """
    if USER_POINT_GROUP not in geometry.groups:
        return geometry.vertices
    identifiers = {ref.id for ref in geometry.group(USER_POINT_GROUP) if ref.kind == "vertex"}
    for feature in geometry.features.records:
        if feature.kind == "geometry.point" and not feature.suppressed and feature.state == "ok":
            identifiers.update(ref.id for ref in feature.outputs.values() if ref.kind == "vertex")
    return tuple(sorted(identifier for identifier in identifiers if identifier in geometry.vertices))
