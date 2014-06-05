#!/usr/bin/env python
# -*- coding: utf-8 -*-
import atexit
import ctypes as ct
import os
import sys
import tempfile as tf
import shutil
import numpy as np
import scipy.sparse as sparse
import sage.all as sg
import sympy
import sympy.printing as sp

sg.var('t, x, y')

sys.path.append(os.path.expanduser("~/Documents/Code/ArgyrisPack"))
sys.path.append(os.path.expanduser("~/Documents/Code/PODSUPG"))
import supg.core.util as util
import supg.core.ElementFunctions as ef

_module_path = os.path.dirname(unicode(os.path.abspath(__file__),
        sys.getfilesystemencoding())) + os.sep

PDOUBLE = ct.POINTER(ct.c_double)
PINT = ct.POINTER(ct.c_int)

# define classes for the interface.
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

cfem = np.ctypeslib.load_library("libcfem.so", _module_path)

for builder in [cfem.cf_build_mass, cfem.cf_build_convection, cfem.cf_build_stiffness,
                cfem.cf_build_hessian]:
    builder.restype = ct.c_int
    builder.argtypes = [CMesh, CRefArrays, ct.c_void_p, CTripletMatrix]

cfem.cf_build_load.restype = ct.c_int
cfem.cf_build_load.argtypes = [ct.POINTER(CMesh), ct.POINTER(CRefArrays), ct.c_void_p,
                               ct.c_void_p, ct.c_double, ct.POINTER(CVector)]

# define caches.
_compiled_lookup = dict()
_quadrature_lookup = dict()

def get_matrix(matrix, mesh, convection_field=(sg.cos(sg.pi/3), sg.sin(sg.pi/3)),
               forcing_function=1, stabilization_coeff=0.0):
    """
    Build a finite element matrix.
    """
    builder = {'mass': cfem.cf_build_mass,
               'convection': cfem.cf_build_convection,
               'stiffness': cfem.cf_build_stiffness,
               'hessian': cfem.cf_build_hessian}[matrix]
    convection, _, ref_data = _get_lookups(convection_field, forcing_function,
                                           stabilization_coeff, mesh.order)
    cmesh, elements = _get_cmesh(mesh)

    length = elements.shape[0]*elements.shape[1]**2
    rows = np.zeros(length, dtype=np.int32) + -99999
    columns = np.zeros(length, dtype=np.int32) + -99999
    values = np.zeros(length, dtype=np.double) + -99999
    triplet_matrix = CTripletMatrix(length=length,
                                    rows=rows.ctypes.data_as(PINT),
                                    columns=columns.ctypes.data_as(PINT),
                                    values=values.ctypes.data_as(PDOUBLE))

    builder(cmesh, ref_data, convection, triplet_matrix)
    return sparse.coo_matrix((values, (rows, columns))).tocsc()

def get_load_vector(mesh, time,
                    convection_field=(sg.cos(sg.pi/3), sg.sin(sg.pi/3)),
                    forcing_function=1, stabilization_coeff=0.0):
    """
    Form a load vector from symbolic expressions for the convection
    field and forcing function. The symbolic expressions are cached.
    """
    convection, forcing, ref_data = _get_lookups(
        convection_field, forcing_function, stabilization_coeff, mesh.order)
    cmesh, elements = _get_cmesh(mesh)

    load_vector = np.zeros(mesh.nodes.shape[0])
    cvector = CVector(length=mesh.nodes.shape[0],
                      values=load_vector.ctypes.data_as(PDOUBLE))

    cfem.cf_build_load(ct.byref(cmesh), ct.byref(ref_data), convection, forcing,
                       time, cvector)

    return load_vector

def _get_cmesh(mesh):
    # TODO is this cast a significant bottleneck?
    elements = mesh.elements.astype(np.int32)
    if elements.min() == 1:
        elements -= 1
    if elements.min() != 0:
        raise ValueError("elements must have a minimal node number of 1 or 0.")

    # note that to keep elements alive (i.e., prevent it from being garbage
    # collected) we must return it too.
    return CMesh(num_nodes=mesh.nodes.shape[0],
                 nodes=mesh.nodes.ctypes.data_as(PDOUBLE),
                 num_elements=elements.shape[0],
                 num_basis_functions=elements.shape[1],
                 elements=elements.ctypes.data_as(PINT)), elements

_forcing_template = \
"""
double cf_forcing(double t, double x, double y)
{{
        return {forcing_expr};
}}
"""

_convection_template = \
"""
cf_convection_s cf_convection(double x, double y)
{{
        cf_convection_s convection = {{.value = {{{value0_expr}, {value1_expr}}},
                                      .dx = {{{dx0_expr}, {dx1_expr}}},
                                      .dy = {{{dy0_expr}, {dy1_expr}}},
        }};
        return convection;
}}
"""

class _CompiledFunctions(object):
    def __init__(self, convection_field, forcing_function):
        self._forcing_function = forcing_function
        self._convection_field = convection_field

        dx0 = sg.diff(convection_field[0], x)
        dx1 = sg.diff(convection_field[1], x)
        dy0 = sg.diff(convection_field[0], y)
        dy1 = sg.diff(convection_field[1], y)

        to_ccode = lambda u: sp.ccode(
            sympy.sympify(sg.symbolic_expression(u)._sympy_()))

        self._forcing_code = _forcing_template.format(
            **{'forcing_expr': to_ccode(forcing_function)})

        self._convection_code = _convection_template.format(**
            {'value0_expr': to_ccode(convection_field[0]),
             'value1_expr': to_ccode(convection_field[1]),
             'dx0_expr': to_ccode(dx0),
             'dx1_expr': to_ccode(dx1),
             'dy0_expr': to_ccode(dy0),
             'dy1_expr': to_ccode(dy1)})

        self.so_folder = tf.mkdtemp(prefix="cfem_") + os.sep
        # even if the object is not instantiated, we can still clean it up at
        # exit.
        atexit.register(lambda folder=self.so_folder: shutil.rmtree(folder))
        shutil.copy(_module_path + "makefile", self.so_folder)
        shutil.copy(_module_path + "cfem.h", self.so_folder)
        with open(self.so_folder + "funcs.c", 'w') as fhandle:
            fhandle.write("#include <math.h>\n")
            fhandle.write("#include \"cfem.h\"\n")
            fhandle.write(self._forcing_code)
            fhandle.write(self._convection_code)

        current_directory = os.getcwd()
        try:
            os.chdir(self.so_folder)
            util.run_make(command="autogenerated_functions")
            self.so = np.ctypeslib.load_library("libfuncs.so", "./")
            self.so.cf_forcing.restype = ct.c_double
            self.so.cf_forcing.argtypes = [ct.c_double, ct.c_double,
                                           ct.c_double]
            self.so.cf_convection.restype = CConvection
            self.so.cf_convection.argtypes = [ct.c_double, ct.c_double]
        finally:
            os.chdir(current_directory)

    def functions(self):
        return (ct.cast(self.so.cf_convection, ct.c_void_p),
                ct.cast(self.so.cf_forcing, ct.c_void_p))

    def close(self):
        shutil.rmtree(self.so_folder)

def _get_lookups(convection_field, forcing_function, stabilization_coeff, order):
    try:
        ref_data, quad_data = _quadrature_lookup[(order, stabilization_coeff)]
    except KeyError:
        quad_data = ef._QuadratureValues(order)
        num_basis_functions = {1: 3, 2: 6}[order]
        ref_data = CRefArrays(num_points=len(quad_data.weights),
                              num_basis_functions=num_basis_functions,
                              xs=quad_data.x.ctypes.data_as(PDOUBLE),
                              ys=quad_data.y.ctypes.data_as(PDOUBLE),
                              weights=quad_data.weights.ctypes.data_as(PDOUBLE),
                              values=quad_data.values.ctypes.data_as(PDOUBLE),
                              dx=quad_data.dx.ctypes.data_as(PDOUBLE),
                              dy=quad_data.dy.ctypes.data_as(PDOUBLE),
                              dxx=quad_data.dxx.ctypes.data_as(PDOUBLE),
                              dxy=quad_data.dxy.ctypes.data_as(PDOUBLE),
                              dyy=quad_data.dyy.ctypes.data_as(PDOUBLE),
                              global_supg_constant=stabilization_coeff)
        # note that ref_data is just a struct of (not reference counted)
        # pointers to arrays controlled by quad_data. Hence we must prevent
        # quad_data from being garbage collected somehow.
        _quadrature_lookup[(order, stabilization_coeff)] = (ref_data, quad_data)

    try:
        convection, forcing = _compiled_lookup[
            (convection_field, forcing_function)].functions()
    except KeyError:
        _compiled_lookup[(convection_field, forcing_function)] = (
            _CompiledFunctions(convection_field, forcing_function))
        convection, forcing = _compiled_lookup[
            (convection_field, forcing_function)].functions()

    return convection, forcing, ref_data
