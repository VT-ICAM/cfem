#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ctypes as ct
import os
import sys
import numpy as np
import scipy.sparse as sparse

sys.path.append(os.path.expanduser("~/Documents/Code/ArgyrisPack"))
sys.path.append(os.path.expanduser("~/Documents/Code/PODSUPG"))
import ap.mesh.meshes as m
import supg.core.util as util
import supg.core.gmsh as gmsh
import supg.core.ElementFunctions as ef
import supg.core.supg as sg
import supg.cdr.Operators2DCDR as operators

PDOUBLE = ct.POINTER(ct.c_double)
PINT = ct.POINTER(ct.c_int)

# Create example data.
mesh_file = util.unique_file(suffix=".msh")
geo_file = util.unique_file(suffix=".geo")
gmsh.write_geo_file(gmsh.unit_square(1.0), geo_file=geo_file)
gmsh.call_gmsh(geo_file=geo_file, mesh_file=mesh_file)
mesh = m.mesh_factory(mesh_file)

elements = mesh.elements.astype(np.int32) - 1
if elements.min() > 0:
    raise ValueError("the minimal node number should be zero.")
nodes = mesh.nodes

quad_data = ef._QuadratureValues(2)

# Define classes for the interface.
class CTripletMatrix(ct.Structure):
    _fields_ = [("length", ct.c_int), ("rows", PINT), ("columns", PINT),
                ("values", PDOUBLE)]

class CLocalElement(ct.Structure):
    _fields_ = [("xs", 3*ct.c_double),
                ("ys", 3*ct.c_double),
                ("B", 4*ct.c_double),
                ("b", 2*ct.c_double),
                ("jacobian", ct.c_double),
                ("supg_stabilization_constant", ct.c_double)]

class CMesh(ct.Structure):
    _fields_ = [("num_nodes", ct.c_int),
                ("nodes", PDOUBLE),
                ("num_elements", ct.c_int),
                ("num_basis_functions", ct.c_int),
                ("elements", PINT)]

class CRefArrays(ct.Structure):
    _fields_ = [("num_points", ct.c_int),
                ("num_basis_functions", ct.c_int),
                ("xs", PDOUBLE),
                ("ys", PDOUBLE),
                ("weights", PDOUBLE),
                ("values", PDOUBLE),
                ("dx", PDOUBLE),
                ("dy", PDOUBLE),
                ("dxx", PDOUBLE),
                ("dxy", PDOUBLE),
                ("dyy", PDOUBLE),
                ("global_supg_constant", ct.c_double)]

class CConvection(ct.Structure):
    _fields_ = [("value", 2*ct.c_double),
                ("dx", 2*ct.c_double),
                ("dy", 2*ct.c_double)]

class CVector(ct.Structure):
    _fields_ = [("length", ct.c_int),
                ("values", PDOUBLE)]

# define function pointers.
CForcingFunction = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double, ct.c_double)
CConvectionFunction = ct.CFUNCTYPE(CConvection, ct.c_double, ct.c_double)

cfem = np.ctypeslib.load_library("libcfem.so", "./")
cfem.cf_build_mass.restype = ct.c_int
cfem.cf_build_mass.argtypes = [CMesh, CRefArrays, ct.c_void_p, CTripletMatrix]

cfem.cf_build_convection.restype = ct.c_int
cfem.cf_build_convection.argtypes = [CMesh, CRefArrays, ct.c_void_p, CTripletMatrix]

cfem.cf_build_stiffness.restype = ct.c_int
cfem.cf_build_stiffness.argtypes = [CMesh, CRefArrays, ct.c_void_p, CTripletMatrix]

cfem.cf_build_hessian.restype = ct.c_int
cfem.cf_build_hessian.argtypes = [CMesh, CRefArrays, ct.c_void_p, CTripletMatrix]

cfem.cf_build_load.restype = ct.c_int
cfem.cf_build_load.argtypes = [ct.POINTER(CMesh), ct.POINTER(CRefArrays), ct.c_void_p,
                               ct.c_void_p, ct.c_double, ct.POINTER(CVector)]

cmesh = CMesh(num_nodes=nodes.shape[0], nodes=nodes.ctypes.data_as(PDOUBLE),
              num_elements=elements.shape[0],
              num_basis_functions=elements.shape[1],
              elements=elements.ctypes.data_as(PINT))

ref_data = CRefArrays(num_points=len(quad_data.weights),
                      num_basis_functions=elements.shape[1],
                      xs=quad_data.x.ctypes.data_as(PDOUBLE),
                      ys=quad_data.y.ctypes.data_as(PDOUBLE),
                      weights=quad_data.weights.ctypes.data_as(PDOUBLE),
                      values=quad_data.values.ctypes.data_as(PDOUBLE),
                      dx=quad_data.dx.ctypes.data_as(PDOUBLE),
                      dy=quad_data.dy.ctypes.data_as(PDOUBLE),
                      dxx=quad_data.dxx.ctypes.data_as(PDOUBLE),
                      dxy=quad_data.dxy.ctypes.data_as(PDOUBLE),
                      dyy=quad_data.dyy.ctypes.data_as(PDOUBLE),
                      global_supg_constant=3.0)

length = elements.shape[0]*elements.shape[1]**2
rows = np.zeros(length, dtype=np.int32) + -999
columns = np.zeros(length, dtype=np.int32) + -999
values = np.zeros(length, dtype=np.double) + -999
matrix = CTripletMatrix(length=length,
                        rows=rows.ctypes.data_as(PINT),
                        columns=columns.ctypes.data_as(PINT),
                        values=values.ctypes.data_as(PDOUBLE))
load_vector = np.zeros(nodes.shape[0])
vector = CVector(length=nodes.shape[0],
                 values=load_vector.ctypes.data_as(PDOUBLE))

# compute the PODSUPG operators too.
fe_facade = sg.SUPGFacade(order=2)
ops = operators.Operators2DCDR(mesh, fe_facade, stabilization_coeff=3.0)

convection_function = ct.cast(cfem.cf_standard_cosinesine_convection, ct.c_void_p)
forcing_function = ct.cast(cfem.cf_standard_forcing, ct.c_void_p)

cfem.cf_build_mass(cmesh, ref_data, convection_function, matrix)
mass = sparse.coo_matrix((values, (rows, columns))).todense()[8:, 8:]
cfem.cf_build_convection(cmesh, ref_data, convection_function, matrix)
convection = sparse.coo_matrix((values, (rows, columns))).todense()[8:, 8:]
cfem.cf_build_stiffness(cmesh, ref_data, convection_function, matrix)
stiffness = sparse.coo_matrix((values, (rows, columns))).todense()[8:, 8:]
cfem.cf_build_hessian(cmesh, ref_data, convection_function, matrix)
hessian = sparse.coo_matrix((values, (rows, columns))).todense()[8:, 8:]

cfem.cf_build_load(ct.byref(cmesh), ct.byref(ref_data), convection_function,
                   forcing_function, 1.0, ct.byref(vector))
