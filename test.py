#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import ctypes as ct
import os
import sys
import numpy as np
import scipy.sparse as sparse

from cfem import *
sys.path.append(os.path.expanduser("~/Documents/Code/ArgyrisPack"))
sys.path.append(os.path.expanduser("~/Documents/Code/PODSUPG"))

import ap.mesh.meshes as m
import supg.core.gmsh as gmsh
import supg.core.supg as spg
import supg.cdr.Operators2DCDR as operators

# Create example data.
mesh_file = util.unique_file(suffix=".msh")
geo_file = util.unique_file(suffix=".geo")
gmsh.write_geo_file(gmsh.unit_square(0.05), geo_file=geo_file)
gmsh.call_gmsh(geo_file=geo_file, mesh_file=mesh_file)
mesh = m.mesh_factory(mesh_file)

# compute the PODSUPG operators too.
fe_facade = spg.SUPGFacade(order=2)
ops = operators.Operators2DCDR(mesh, fe_facade, stabilization_coeff=3.0)

if mesh.interior_nodes.min() == 0:
    last_boundary_node = max(mesh.boundary_nodes['land'])
else:
    last_boundary_node = max(mesh.boundary_nodes['land']) - 1
first_interior_node = last_boundary_node + 1

mass = get_matrix('mass', mesh, stabilization_coeff=3.0).todense()
convection = get_matrix('convection', mesh, stabilization_coeff=3.0).todense()
stiffness = get_matrix('stiffness', mesh, stabilization_coeff=3.0).todense()
hessian = get_matrix('hessian', mesh, stabilization_coeff=3.0).todense()

for cfem_matrix, podsupg_matrix in [[mass, ops.mass], [convection, ops.convection],
                                    [stiffness, ops.stiffness]]:
    print(np.linalg.norm(cfem_matrix[first_interior_node:, first_interior_node:]
                         - podsupg_matrix.todense()))

load_vector = get_load_vector(mesh, 1.0, stabilization_coeff=3.0)
podsupg_load_vector = ops.get_load_vector(1.0)
print(np.linalg.norm(load_vector[first_interior_node:] - podsupg_load_vector))
