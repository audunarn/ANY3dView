"""Transient display visibility; no project or solver inputs are modified."""
from dataclasses import replace


class SelectionVisibility:
    def __init__(self):
        self.hidden = set()

    @staticmethod
    def key(ref):
        return ref.kind, ref.id

    def is_hidden(self, ref):
        return ref is not None and self.key(ref) in self.hidden

    def set_hidden(self, refs, hidden=True):
        keys = {self.key(ref) for ref in refs}
        if hidden:
            self.hidden.update(keys)
        else:
            self.hidden.difference_update(keys)

    def filter_scene(self, scene):
        """Filter draw/pick payloads together, retaining cached source arrays."""
        result = scene.copy()
        if not self.hidden:
            return result
        result.faces = []
        for patch in scene.faces:
            keep = [i for i in range(len(patch.polygons))
                    if not any(self.is_hidden(ref) for ref in
                               (patch.polygon_visibility_owners[i]
                                if patch.polygon_visibility_owners is not None
                                else patch.owners_for_polygon(i)))]
            if not keep:
                continue
            owners = tuple(ref for ref in patch.owners if not self.is_hidden(ref))
            polygon_owners = None if patch.polygon_owners is None else [
                tuple(ref for ref in patch.polygon_owners[i] if not self.is_hidden(ref)) for i in keep]
            visibility_owners = None if patch.polygon_visibility_owners is None else [
                patch.polygon_visibility_owners[i] for i in keep]
            ref = patch.ref
            if self.is_hidden(ref):
                ref = next(iter(owners), None)
            result.faces.append(replace(patch, ref=ref, owners=owners,
                polygons=[patch.polygons[i] for i in keep],
                colors=[patch.colors[i] for i in keep], polygon_owners=polygon_owners,
                polygon_visibility_owners=visibility_owners))
        for name in ('lines', 'points'):
            setattr(result, name, [item for item in getattr(scene, name)
                if not any(self.is_hidden(ref) for ref in
                           (*item.owners, *(() if item.ref is None else (item.ref,))))])
        return result
