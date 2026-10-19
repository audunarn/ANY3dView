"""Compatibility exports for mapped meshing now owned by ANYmesher."""

from anymesher.errors import MeshError
from anygeometry.chains import chain_breaks, chain_point, sample_chain
from anymesher.mapped import ELEMENT_ORDERS, coons_grid, generate_mesh, nodal_normals
from anymesher.mesh import Coupling, Mesh

__all__ = [
    "Coupling",
    "ELEMENT_ORDERS",
    "Mesh",
    "MeshError",
    "chain_breaks",
    "chain_point",
    "coons_grid",
    "generate_mesh",
    "nodal_normals",
    "sample_chain",
]
