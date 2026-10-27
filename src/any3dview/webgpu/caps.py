"""Adapter for explicitly owner-qualified ANYgeometry closed box operands."""
from ..clipping import CutCap
from ..arrays import MeshArrays
from ..ownership import PackedOwnerTable
from ..selection import PickBinding

def box_volume_cap_provider(volume,source_handle,owner):
    """Bind a canonical world-space BoxVolume to its displayed source and owner.

    The application supplies this relation; no shell closure is inferred here.
    The volume must match the currently displayed source coordinates. Scalars
    are intentionally absent because a visual cap has no owner-supplied field.
    """
    if not isinstance(owner,PickBinding):raise TypeError('cap owner must be a source PickBinding')
    def provide(viewer,planes):
        import numpy as np
        if source_handle.deformation_scale or not np.array_equal(source_handle.transform,np.eye(4)):
            raise ValueError('BoxVolume cap adapter requires source coordinates in the supplied world-volume frame; transformed/deformed sources need an owner-updated provider')
        from anygeometry.display_caps import box_volume_cut_caps
        result=[]
        for cap in box_volume_cut_caps(volume,[(p.normal.to_tuple(),p.offset) for p in planes if p.enabled]):
            triangles=[(0,i,i+1) for i in range(1,len(cap.vertices)-1)]
            mesh=MeshArrays(cap.vertices,triangles)
            owners=PackedOwnerTable.from_owners(triangles=[owner]*len(triangles))
            result.append(CutCap(mesh,source_handle,owners))
        return tuple(result)
    return provide
