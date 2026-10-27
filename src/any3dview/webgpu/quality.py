"""Display-only crease classification; scientific connectivity is untouched."""
import numpy as np


def feature_line_mask(positions,triangles,lines,angle=30.):
    """Keep boundaries, creases, degeneracies and nonmanifold/unmatched lines.

    Exact coincident display vertices account for split flat-shading normals.
    This classification never changes selection owners or source arrays.
    """
    positions=np.asarray(positions)
    _,vertices=np.unique(positions,axis=0,return_inverse=True)
    points=positions[triangles]
    normals=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
    lengths=np.linalg.norm(normals,axis=1)
    normals=normals/np.maximum(lengths[:,None],np.finfo(float).tiny)
    adjacency={}
    for row,triangle in enumerate(vertices[triangles]):
        for a,b in zip(triangle,np.roll(triangle,-1)):
            adjacency.setdefault(tuple(sorted((int(a),int(b)))),[]).append(row)
    threshold=np.cos(np.deg2rad(angle));result=[]
    for a,b in vertices[lines]:
        rows=adjacency.get(tuple(sorted((int(a),int(b)))),())
        result.append(len(rows)!=2 or any(lengths[row]==0 for row in rows)
            or abs(float(normals[rows[0]]@normals[rows[1]]))<threshold)
    return np.asarray(result,dtype=bool)
