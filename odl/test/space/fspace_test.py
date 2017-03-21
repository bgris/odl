# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

import inspect
import numpy as np
import pytest
import sys

import odl
from odl import FunctionSpace
from odl.discr.grid import sparse_meshgrid
from odl.util.testutils import (all_almost_equal, all_equal, almost_equal,
                                simple_fixture)


# --- Helper functions --- #


PY2 = sys.version_info.major < 3
getargspec = inspect.getargspec if PY2 else inspect.getfullargspec


def _points(domain, num):
    """Helper to generate ``num`` points in ``domain``."""
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    points = np.random.uniform(low=0, high=1, size=(ndim, num))
    for i in range(ndim):
        points[i, :] = min_pt[i] + (max_pt[i] - min_pt[i]) * points[i]
    return points


def _meshgrid(domain, shape):
    """Helper to generate a ``shape`` meshgrid of points in ``domain``."""
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    coord_vecs = []
    for i in range(ndim):
        vec = np.random.uniform(low=min_pt[i], high=max_pt[i], size=shape[i])
        vec.sort()
        coord_vecs.append(vec)
    return sparse_meshgrid(*coord_vecs)


def _standard_setup_2d():
    rect = odl.IntervalProd([0, 0], [1, 2])
    points = _points(rect, num=5)
    mg = _meshgrid(rect, shape=(2, 3))
    return rect, points, mg


class FuncList(list):  # So we can set __name__
    pass


# --- pytest fixtures --- #


out_dtype_params = ['float32', 'float64', 'complex64']
out_dtype = simple_fixture('out_dtype', out_dtype_params,
                           fmt=' {name} = {value!r} ')

out_shape = simple_fixture('out_shape', [(1,), (2,), (2, 3)])
domain_ndim = simple_fixture('domain_ndim', [1, 2])
vectorized = simple_fixture('vectorized', [True, False])


@pytest.fixture(scope='module')
def fspace_scal(domain_ndim, out_dtype):
    """Fixture returning a function space with given properties."""
    domain = odl.IntervalProd([0] * domain_ndim, [1] * domain_ndim)
    return FunctionSpace(domain, out_dtype=out_dtype)


# --- fixtures with test functions --- #


def func_nd_oop(x):
    return sum(x)


def func_nd_ip(x, out):
    out[:] = sum(x)


def func_nd_dual(x, out=None):
    if out is None:
        return sum(x)
    else:
        out[:] = sum(x)


def func_nd_bcast_oop(x):
    return x[0]


def func_nd_bcast_ip(x, out):
    out[:] = x[0]


func_nd_params = [func_nd_oop, func_nd_ip, func_nd_dual, func_nd_bcast_oop,
                  func_nd_bcast_ip]
func_nd = simple_fixture('func_nd', func_nd_params,
                         fmt=' {name} = {value.__name__} ')


def func_param_nd_oop(x, c):
    return sum(x) + c


def func_param_nd_ip(x, out, c):
    out[:] = sum(x) + c


def func_param_switched_nd_ip(x, c, out):
    out[:] = sum(x) + c


def func_param_bcast_nd_oop(x, c):
    return x[0] + c


def func_param_bcast_nd_ip(x, out, c):
    out[:] = x[0]


func_nd_with_param_params = [func_param_nd_oop, func_param_nd_ip,
                             func_param_switched_nd_ip,
                             func_param_bcast_nd_oop, func_param_bcast_nd_ip]
func_nd_with_param = simple_fixture('func_with_param',
                                    func_nd_with_param_params,
                                    fmt=' {name} = {value.__name__} ')


def func_vec_nd_oop(x):
    return (sum(x) + 1, sum(x) - 1)


func_nd_oop_seq = FuncList([lambda x: sum(x) + 1, lambda x: sum(x) - 1])
func_nd_oop_seq.__name__ = 'func_nd_oop_seq'


def func_vec_nd_ip(x, out):
    out[0] = sum(x) + 1
    out[1] = sum(x) - 1


def comp0_nd(x, out):
    out[:] = sum(x) + 1


def comp1_nd(x, out):
    out[:] = sum(x) - 1


func_nd_ip_seq = FuncList([comp0_nd, comp1_nd])
func_nd_ip_seq.__name__ = 'func_nd_ip_seq'

func_nd_vec_params = [func_vec_nd_oop, func_nd_oop_seq, func_vec_nd_ip,
                      func_nd_ip_seq]
func_nd_vec = simple_fixture('func_nd_vec', func_nd_vec_params,
                             fmt=' {name} = {value.__name__} ')


def func_1d_oop(x):
    return x * 2


def func_1d_ip(x, out):
    out[:] = x * 2


func_1d_params = [func_1d_oop, func_1d_ip]
func_1d = simple_fixture('func_1d', func_1d_params,
                         fmt=' {name} = {value.__name__} ')


def func_vec_1d_oop(x):
    return (x * 2, x + 1)


func_1d_oop_seq = FuncList([lambda x: x * 2, lambda x: x + 1])
func_1d_oop_seq.__name__ = 'func_1d_oop_seq'


def func_vec_1d_ip(x, out):
    out[0] = x * 2
    out[1] = x + 1


def comp0_1d(x, out):
    out[:] = x * 2


def comp1_1d(x, out):
    out[:] = x + 1


func_1d_ip_seq = FuncList([comp0_1d, comp1_1d])
func_1d_ip_seq.__name__ = 'func_1d_ip_seq'


func_1d_vec_params = [func_vec_1d_oop, func_1d_oop_seq, func_vec_1d_ip,
                      func_1d_ip_seq]
func_1d_vec = simple_fixture('func_1d_vec', func_1d_vec_params,
                             fmt=' {name} = {value.__name__} ')


# --- FunctionSpace tests --- #


def test_fspace_init():
    """Check if all initialization patterns work."""
    intv = odl.IntervalProd(0, 1)
    FunctionSpace(intv)
    FunctionSpace(intv, out_dtype=float)
    FunctionSpace(intv, out_dtype=complex)
    FunctionSpace(intv, out_dtype=(float, (2, 3)))

    str3 = odl.Strings(3)
    FunctionSpace(str3, out_dtype=int)

    # Make sure this doesn't raise an exception due to a bug
    repr(FunctionSpace(intv, out_dtype=(float, (2, 3))))


def test_fspace_attributes():
    """Check attribute access and correct values."""
    intv = odl.IntervalProd(0, 1)

    # Scalar-valued function spaces
    fspace = FunctionSpace(intv)
    fspace_r = FunctionSpace(intv, out_dtype=float)
    fspace_c = FunctionSpace(intv, out_dtype=complex)
    fspace_s = FunctionSpace(intv, out_dtype='U1')
    scalar_spaces = (fspace, fspace_r, fspace_c, fspace_s)

    assert fspace.domain == intv
    assert fspace.field == odl.RealNumbers()
    assert fspace_r.field == odl.RealNumbers()
    assert fspace_c.field == odl.ComplexNumbers()
    assert fspace_s.field is None

    assert fspace.out_dtype == float
    assert fspace_r.out_dtype == float
    assert fspace_r.real_out_dtype == float
    assert fspace_r.complex_out_dtype == complex
    assert fspace_c.out_dtype == complex
    assert fspace_c.real_out_dtype == float
    assert fspace_c.complex_out_dtype == complex
    assert fspace_s.out_dtype == np.dtype('U1')
    assert fspace_s.real_out_dtype is None

    assert all(spc.scalar_out_dtype == spc.out_dtype for spc in scalar_spaces)
    assert all(spc.out_shape == () for spc in scalar_spaces)
    assert all(not spc.tensor_valued for spc in scalar_spaces)

    # Vector-valued function space
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    assert fspace_vec.field == odl.RealNumbers()
    assert fspace_vec.out_dtype == np.dtype((float, (2,)))
    assert fspace_vec.scalar_out_dtype == float
    assert fspace_vec.out_shape == (2,)
    assert fspace_vec.tensor_valued


def _test_eq(x, y):
    """Test equality of x and y."""
    assert x == y
    assert not x != y
    assert hash(x) == hash(y)


def _test_neq(x, y):
    """Test non-equality of x and y."""
    assert x != y
    assert not x == y
    assert hash(x) != hash(y)


def test_equals():
    """Test equality check and hash."""
    intv = odl.IntervalProd(0, 1)
    intv2 = odl.IntervalProd(-1, 1)
    fspace = FunctionSpace(intv)
    fspace_r = FunctionSpace(intv, out_dtype=float)
    fspace_c = FunctionSpace(intv, out_dtype=complex)
    fspace_intv2 = FunctionSpace(intv2)
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))

    _test_eq(fspace, fspace)
    _test_eq(fspace, fspace_r)
    _test_eq(fspace_c, fspace_c)

    _test_neq(fspace, fspace_c)
    _test_neq(fspace, fspace_intv2)
    _test_neq(fspace_r, fspace_vec)


def test_fspace_astype():
    """Check that converting function spaces to new out_dtype works."""
    rspace = FunctionSpace(odl.IntervalProd(0, 1))
    cspace = FunctionSpace(odl.IntervalProd(0, 1), out_dtype=complex)
    rspace_s = FunctionSpace(odl.IntervalProd(0, 1), out_dtype='float32')
    cspace_s = FunctionSpace(odl.IntervalProd(0, 1), out_dtype='complex64')

    assert rspace.astype('complex64') == cspace_s
    assert rspace.astype('complex128') == cspace
    assert rspace.astype('complex128') is rspace.complex_space
    assert rspace.astype('float32') == rspace_s
    assert rspace.astype('float64') is rspace.real_space

    assert cspace.astype('float32') == rspace_s
    assert cspace.astype('float64') == rspace
    assert cspace.astype('float64') is cspace.real_space
    assert cspace.astype('complex64') == cspace_s
    assert cspace.astype('complex128') is cspace.complex_space


# --- FunctionSpaceElement tests --- #


def test_fspace_elem_vectorized_init(vectorized):
    """Check init of fspace elements with(out) vectorization."""
    intv = odl.IntervalProd(0, 1)

    fspace_scal = FunctionSpace(intv)
    fspace_scal.element(func_nd_oop, vectorized=vectorized)

    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    fspace_vec.element(func_vec_nd_oop, vectorized=vectorized)
    fspace_vec.element(func_nd_oop_seq, vectorized=vectorized)


def test_fspace_scal_elem_eval(fspace_scal, func_nd):
    """Check evaluation of scalar-valued function elements."""
    points = _points(fspace_scal.domain, 3)
    mesh_shape = tuple(range(2, 2 + fspace_scal.domain.ndim))
    mesh = _meshgrid(fspace_scal.domain, mesh_shape)

    func_argspec = getargspec(func_nd)
    if 'out' in func_argspec.args:
        true_values_points = np.empty(3, dtype=fspace_scal.scalar_out_dtype)
        true_values_mesh = np.empty(mesh_shape,
                                    dtype=fspace_scal.scalar_out_dtype)
        func_nd(points, out=true_values_points)
        func_nd(mesh, out=true_values_mesh)
    else:
        true_values_points = func_nd(points)
        true_values_mesh = func_nd(mesh)

    func_elem = fspace_scal.element(func_nd)

    # Out of place
    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.dtype == fspace_scal.scalar_out_dtype
    assert result_mesh.dtype == fspace_scal.scalar_out_dtype

    # In place
    out_points = np.empty(3, dtype=fspace_scal.scalar_out_dtype)
    out_mesh = np.empty(mesh_shape, dtype=fspace_scal.scalar_out_dtype)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)


def test_fspace_scal_elem_with_param_eval(func_nd_with_param):
    """Check evaluation of scalar-valued function elements with parameters."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_scal = FunctionSpace(intv)
    points = _points(fspace_scal.domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(fspace_scal.domain, mesh_shape)

    func_argspec = getargspec(func_nd_with_param)
    if 'out' in func_argspec.args:
        true_values_points = np.empty(3, dtype=fspace_scal.scalar_out_dtype)
        true_values_mesh = np.empty(mesh_shape,
                                    dtype=fspace_scal.scalar_out_dtype)
        func_nd_with_param(points, out=true_values_points, c=2.5)
        func_nd_with_param(mesh, out=true_values_mesh, c=2.5)
    else:
        true_values_points = func_nd_with_param(points, c=2.5)
        true_values_mesh = func_nd_with_param(mesh, c=2.5)

    func_elem = fspace_scal.element(func_nd_with_param)

    # Out of place
    result_points = func_elem(points, c=2.5)
    result_mesh = func_elem(mesh, c=2.5)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)

    # In place
    out_points = np.empty(3, dtype=fspace_scal.scalar_out_dtype)
    out_mesh = np.empty(mesh_shape, dtype=fspace_scal.scalar_out_dtype)
    func_elem(points, out=out_points, c=2.5)
    func_elem(mesh, out=out_mesh, c=2.5)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Complex output
    fspace_complex = FunctionSpace(intv, out_dtype=complex)
    if 'out' in func_argspec.args:
        true_values_points = np.empty(3, dtype=fspace_complex.scalar_out_dtype)
        true_values_mesh = np.empty(mesh_shape,
                                    dtype=fspace_complex.scalar_out_dtype)
        func_nd_with_param(points, out=true_values_points, c=2j)
        func_nd_with_param(mesh, out=true_values_mesh, c=2j)
    else:
        true_values_points = func_nd_with_param(points, c=2j)
        true_values_mesh = func_nd_with_param(mesh, c=2j)

    func_elem = fspace_complex.element(func_nd_with_param)

    result_points = func_elem(points, c=2j)
    result_mesh = func_elem(mesh, c=2j)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)


def test_fspace_vec_elem_eval(func_nd_vec, out_dtype):
    """Check evaluation of scalar-valued function elements."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    points = _points(fspace_vec.domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(fspace_vec.domain, mesh_shape)
    values_points_shape = (2, 3)
    values_mesh_shape = (2, 2, 3)

    print(points)
    # Get true results, a bit more complicated for lists of functions
    if isinstance(func_nd_vec, list):
        true_values_points = np.empty(values_points_shape,
                                      dtype=fspace_vec.scalar_out_dtype)
        true_values_mesh = np.empty(values_mesh_shape,
                                    dtype=fspace_vec.scalar_out_dtype)
        for i, f in enumerate(func_nd_vec):
            if 'out' in getargspec(f).args:
                f(points[i][None, :], out=true_values_points[i])
                f(mesh[i], out=true_values_mesh[i])
            else:
                true_values_points[i] = f(points[i][None, :])
                true_values_mesh[i] = f(mesh[i])
    else:
        func_argspec = getargspec(func_nd)
        if 'out' in func_argspec.args:
            true_values_points = np.empty(values_points_shape,
                                          dtype=fspace_vec.scalar_out_dtype)
            true_values_mesh = np.empty(values_mesh_shape,
                                        dtype=fspace_vec.scalar_out_dtype)
            func_nd_vec(points, out=true_values_points)
            func_nd_vec(mesh, out=true_values_mesh)
        else:
            true_values_points = func_nd_vec(points)
            true_values_mesh = func_nd_vec(mesh)

    func_elem = fspace_vec.element(func_nd_vec)

    # Out of place
    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.dtype == fspace_vec.scalar_out_dtype
    assert result_mesh.dtype == fspace_vec.scalar_out_dtype

    # In place
    out_points = np.empty(values_points_shape,
                          dtype=fspace_vec.scalar_out_dtype)
    out_mesh = np.empty(values_mesh_shape,
                        dtype=fspace_vec.scalar_out_dtype)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)


def test_fspace_eval_unusual_dtypes():
    """Check evaluation with unusual data types."""
    str3 = odl.Strings(3)
    fspace = FunctionSpace(str3, out_dtype=int)
    strings = np.array(['aa', 'b', 'cab', 'aba'])
    out_vec = np.empty((4,), dtype=int)

    # Vectorized for arrays only
    func_elem = fspace.element(
        lambda s: np.array([str(si).count('a') for si in s]))
    true_values = [2, 0, 1, 2]

    assert func_elem('abc') == 1
    assert all_equal(func_elem(strings), true_values)
    func_elem(strings, out=out_vec)
    assert all_equal(out_vec, true_values)


def test_fspace_vector_eval_real():
    rect, points, mg = _standard_setup_2d()

    fspace = FunctionSpace(rect)
    f_novec = fspace.element(func_2d_novec, vectorized=False)
    f_vec_oop = fspace.element(func_2d_vec_oop, vectorized=True)
    f_vec_ip = fspace.element(func_2d_vec_ip, vectorized=True)
    f_vec_dual = fspace.element(func_2d_vec_dual, vectorized=True)

    true_arr = func_2d_vec_oop(points)
    true_mg = func_2d_vec_oop(mg)

    # Out-of-place
    assert f_novec([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert f_vec_oop([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_oop(points), true_arr)
    assert all_equal(f_vec_oop(mg), true_mg)

    # In-place standard implementation
    out_arr = np.empty((5,), dtype='float64')
    out_mg = np.empty((2, 3), dtype='float64')

    f_vec_oop(points, out=out_arr)
    f_vec_oop(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    with pytest.raises(TypeError):  # ValueError: invalid vectorized input
        f_vec_oop(points[0])

    # In-place-only
    out_arr = np.empty((5,), dtype='float64')
    out_mg = np.empty((2, 3), dtype='float64')

    f_vec_ip(points, out=out_arr)
    f_vec_ip(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Standard out-of-place evaluation
    assert f_vec_ip([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_ip(points), true_arr)
    assert all_equal(f_vec_ip(mg), true_mg)

    # Dual use
    assert f_vec_dual([0.5, 1.5]) == func_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_dual(points), true_arr)
    assert all_equal(f_vec_dual(mg), true_mg)

    out_arr = np.empty((5,), dtype='float64')
    out_mg = np.empty((2, 3), dtype='float64')

    f_vec_dual(points, out=out_arr)
    f_vec_dual(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)


def test_fspace_vector_eval_complex():
    rect, points, mg = _standard_setup_2d()

    fspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    f_novec = fspace.element(cfunc_2d_novec, vectorized=False)
    f_vec_oop = fspace.element(cfunc_2d_vec_oop, vectorized=True)
    f_vec_ip = fspace.element(cfunc_2d_vec_ip, vectorized=True)
    f_vec_dual = fspace.element(cfunc_2d_vec_dual, vectorized=True)

    true_arr = cfunc_2d_vec_oop(points)
    true_mg = cfunc_2d_vec_oop(mg)

    # Out-of-place
    assert f_novec([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert f_vec_oop([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_oop(points), true_arr)
    assert all_equal(f_vec_oop(mg), true_mg)

    # In-place standard implementation
    out_arr = np.empty((5,), dtype='complex128')
    out_mg = np.empty((2, 3), dtype='complex128')

    f_vec_oop(points, out=out_arr)
    f_vec_oop(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    with pytest.raises(TypeError):  # ValueError: invalid vectorized input
        f_vec_oop(points[0])

    # In-place-only
    out_arr = np.empty((5,), dtype='complex128')
    out_mg = np.empty((2, 3), dtype='complex128')

    f_vec_ip(points, out=out_arr)
    f_vec_ip(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Standard out-of-place evaluation
    assert f_vec_ip([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_ip(points), true_arr)
    assert all_equal(f_vec_ip(mg), true_mg)

    # Dual use
    assert f_vec_dual([0.5, 1.5]) == cfunc_2d_novec([0.5, 1.5])
    assert all_equal(f_vec_dual(points), true_arr)
    assert all_equal(f_vec_dual(mg), true_mg)

    out_arr = np.empty((5,), dtype='complex128')
    out_mg = np.empty((2, 3), dtype='complex128')

    f_vec_dual(points, out=out_arr)
    f_vec_dual(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)


def test_fspace_vector_with_params():
    rect, points, mg = _standard_setup_2d()

    def f(x, c):
        return sum(x) + c

    def f_out1(x, out, c):
        out[:] = sum(x) + c

    def f_out2(x, c, out):
        out[:] = sum(x) + c

    fspace = FunctionSpace(rect)
    true_result_arr = f(points, c=2)
    true_result_mg = f(mg, c=2)

    f_elem = fspace.element(f)
    assert all_equal(f_elem(points, c=2), true_result_arr)
    out_arr = np.empty((5,))
    f_elem(points, c=2, out=out_arr)
    assert all_equal(out_arr, true_result_arr)
    assert all_equal(f_elem(mg, c=2), true_result_mg)
    out_mg = np.empty((2, 3))
    f_elem(mg, c=2, out=out_mg)
    assert all_equal(out_mg, true_result_mg)

    f_out1_elem = fspace.element(f_out1)
    assert all_equal(f_out1_elem(points, c=2), true_result_arr)
    out_arr = np.empty((5,))
    f_out1_elem(points, c=2, out=out_arr)
    assert all_equal(out_arr, true_result_arr)
    assert all_equal(f_out1_elem(mg, c=2), true_result_mg)
    out_mg = np.empty((2, 3))
    f_out1_elem(mg, c=2, out=out_mg)
    assert all_equal(out_mg, true_result_mg)

    f_out2_elem = fspace.element(f_out2)
    assert all_equal(f_out2_elem(points, c=2), true_result_arr)
    out_arr = np.empty((5,))
    f_out2_elem(points, c=2, out=out_arr)
    assert all_equal(out_arr, true_result_arr)
    assert all_equal(f_out2_elem(mg, c=2), true_result_mg)
    out_mg = np.empty((2, 3))
    f_out2_elem(mg, c=2, out=out_mg)
    assert all_equal(out_mg, true_result_mg)


def test_fspace_vector_ufunc():
    intv = odl.IntervalProd(0, 1)
    points = _points(intv, num=5)
    mg = _meshgrid(intv, shape=(5,))

    fspace = FunctionSpace(intv)
    f_vec = fspace.element(np.sin)

    assert f_vec(0.5) == np.sin(0.5)
    assert all_equal(f_vec(points), np.sin(points.squeeze()))
    assert all_equal(f_vec(mg), np.sin(mg[0]))


def test_fspace_vector_equality():
    rect = odl.IntervalProd([0, 0], [1, 2])
    fspace = FunctionSpace(rect)

    f_novec = fspace.element(func_2d_novec, vectorized=False)

    f_vec_oop = fspace.element(func_2d_vec_oop, vectorized=True)
    f_vec_oop_2 = fspace.element(func_2d_vec_oop, vectorized=True)

    f_vec_ip = fspace.element(func_2d_vec_ip, vectorized=True)
    f_vec_ip_2 = fspace.element(func_2d_vec_ip, vectorized=True)

    f_vec_dual = fspace.element(func_2d_vec_dual, vectorized=True)
    f_vec_dual_2 = fspace.element(func_2d_vec_dual, vectorized=True)

    assert f_novec == f_novec
    assert f_novec != f_vec_oop
    assert f_novec != f_vec_ip
    assert f_novec != f_vec_dual

    assert f_vec_oop == f_vec_oop
    assert f_vec_oop == f_vec_oop_2
    assert f_vec_oop != f_vec_ip
    assert f_vec_oop != f_vec_dual

    assert f_vec_ip == f_vec_ip
    assert f_vec_ip == f_vec_ip_2
    assert f_vec_ip != f_vec_dual

    assert f_vec_dual == f_vec_dual
    assert f_vec_dual == f_vec_dual_2


def test_fspace_vector_assign():
    fspace = FunctionSpace(odl.IntervalProd(0, 1))

    f_novec = fspace.element(func_1d_oop, vectorized=False)
    f_vec_ip = fspace.element(func_1d_ip, vectorized=True)
    f_vec_dual = fspace.element(func_1d_dual, vectorized=True)

    f_out = fspace.element()
    f_out.assign(f_novec)
    assert f_out == f_novec

    f_out = fspace.element()
    f_out.assign(f_vec_ip)
    assert f_out == f_vec_ip

    f_out = fspace.element()
    f_out.assign(f_vec_dual)
    assert f_out == f_vec_dual


def test_fspace_vector_copy():
    fspace = FunctionSpace(odl.IntervalProd(0, 1))

    f_novec = fspace.element(func_1d_oop, vectorized=False)
    f_vec_ip = fspace.element(func_1d_ip, vectorized=True)
    f_vec_dual = fspace.element(func_1d_dual, vectorized=True)

    f_out = f_novec.copy()
    assert f_out == f_novec

    f_out = f_vec_ip.copy()
    assert f_out == f_vec_ip

    f_out = f_vec_dual.copy()
    assert f_out == f_vec_dual


def test_fspace_vector_real_imag():
    rect, _, mg = _standard_setup_2d()
    cspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    f = cspace.element(cfunc_2d_vec_oop)

    # real / imag on complex functions
    assert all_equal(f.real(mg), cfunc_2d_vec_oop(mg).real)
    assert all_equal(f.imag(mg), cfunc_2d_vec_oop(mg).imag)
    out_mg = np.empty((2, 3))
    f.real(mg, out=out_mg)
    assert all_equal(out_mg, cfunc_2d_vec_oop(mg).real)
    f.imag(mg, out=out_mg)
    assert all_equal(out_mg, cfunc_2d_vec_oop(mg).imag)

    # real / imag on real functions, should be the function itself / zero
    rspace = FunctionSpace(rect)
    f = rspace.element(func_2d_vec_oop)
    assert all_equal(f.real(mg), f(mg))
    assert all_equal(f.imag(mg), rspace.zero()(mg))

    # Complex conjugate
    f = cspace.element(cfunc_2d_vec_oop)
    fbar = f.conj()
    assert all_equal(fbar(mg), cfunc_2d_vec_oop(mg).conj())
    out_mg = np.empty((2, 3), dtype='complex128')
    fbar(mg, out=out_mg)
    assert all_equal(out_mg, cfunc_2d_vec_oop(mg).conj())


def test_fspace_zero():
    rect, points, mg = _standard_setup_2d()

    # real
    fspace = FunctionSpace(rect)
    zero_vec = fspace.zero()

    assert zero_vec([0.5, 1.5]) == 0.0
    assert all_equal(zero_vec(points), np.zeros(5, dtype=float))
    assert all_equal(zero_vec(mg), np.zeros((2, 3), dtype=float))

    # complex
    fspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    zero_vec = fspace.zero()

    assert zero_vec([0.5, 1.5]) == 0.0 + 1j * 0.0
    assert all_equal(zero_vec(points), np.zeros(5, dtype=complex))
    assert all_equal(zero_vec(mg), np.zeros((2, 3), dtype=complex))


def test_fspace_one():
    rect, points, mg = _standard_setup_2d()

    # real
    fspace = FunctionSpace(rect)
    one_vec = fspace.one()

    assert one_vec([0.5, 1.5]) == 1.0
    assert all_equal(one_vec(points), np.ones(5, dtype=float))
    assert all_equal(one_vec(mg), np.ones((2, 3), dtype=float))

    # complex
    fspace = FunctionSpace(rect, range=odl.ComplexNumbers())
    one_vec = fspace.one()

    assert one_vec([0.5, 1.5]) == 1.0 + 1j * 0.0
    assert all_equal(one_vec(points), np.ones(5, dtype=complex))
    assert all_equal(one_vec(mg), np.ones((2, 3), dtype=complex))


a = simple_fixture('a', [2.0, 0.0, -1.0])
b = simple_fixture('b', [2.0, 0.0, -1.0])


def test_fspace_lincomb(a, b):
    rect, points, mg = _standard_setup_2d()
    point = points.T[0]

    fspace = FunctionSpace(rect)

    # Note: Special cases and alignment are tested later in the magic methods

    # Not vectorized
    true_novec = a * func_2d_novec(point) + b * other_func_2d_novec(point)
    f_novec = fspace.element(func_2d_novec, vectorized=False)
    g_novec = fspace.element(other_func_2d_novec, vectorized=False)
    out_novec = fspace.element(vectorized=False)
    fspace.lincomb(a, f_novec, b, g_novec, out_novec)
    assert almost_equal(out_novec(point), true_novec)

    # Vectorized
    true_arr = (a * func_2d_vec_oop(points) +
                b * other_func_2d_vec_oop(points))
    true_mg = (a * func_2d_vec_oop(mg) + b * other_func_2d_vec_oop(mg))

    # Out-of-place
    f_vec_oop = fspace.element(func_2d_vec_oop, vectorized=True)
    g_vec_oop = fspace.element(other_func_2d_vec_oop, vectorized=True)
    out_vec = fspace.element()
    fspace.lincomb(a, f_vec_oop, b, g_vec_oop, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)
    assert almost_equal(out_vec(point), true_novec)
    out_arr = np.empty((5,), dtype=float)
    out_mg = np.empty((2, 3), dtype=float)
    out_vec(points, out=out_arr)
    out_vec(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # In-place
    f_vec_ip = fspace.element(func_2d_vec_ip, vectorized=True)
    g_vec_ip = fspace.element(other_func_2d_vec_ip, vectorized=True)
    out_vec = fspace.element()
    fspace.lincomb(a, f_vec_ip, b, g_vec_ip, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)
    assert almost_equal(out_vec(point), true_novec)
    out_arr = np.empty((5,), dtype=float)
    out_mg = np.empty((2, 3), dtype=float)
    out_vec(points, out=out_arr)
    out_vec(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Dual use
    f_vec_dual = fspace.element(func_2d_vec_dual, vectorized=True)
    g_vec_dual = fspace.element(other_func_2d_vec_dual, vectorized=True)
    out_vec = fspace.element()
    fspace.lincomb(a, f_vec_dual, b, g_vec_dual, out_vec)

    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)
    assert almost_equal(out_vec(point), true_novec)
    out_arr = np.empty((5,), dtype=float)
    out_mg = np.empty((2, 3), dtype=float)
    out_vec(points, out=out_arr)
    out_vec(mg, out=out_mg)
    assert all_equal(out_arr, true_arr)
    assert all_equal(out_mg, true_mg)

    # Mix of vectorized and non-vectorized -> manual vectorization
    fspace.lincomb(a, f_vec_dual, b, g_novec, out_vec)
    assert all_equal(out_vec(points), true_arr)
    assert all_equal(out_vec(mg), true_mg)


# NOTE: multiply and divide are tested via magic methods

power = simple_fixture('power', [3, 1.0, 0.5, 6.0])


def test_fspace_power(power):
    rect, points, mg = _standard_setup_2d()
    point = points.T[0]
    out_arr = np.empty(5)
    out_mg = np.empty((2, 3))

    fspace = FunctionSpace(rect)

    # Not vectorized
    true_novec = func_2d_novec(point) ** power

    f_novec = fspace.element(func_2d_novec, vectorized=False)
    pow_novec = f_novec ** power
    assert almost_equal(pow_novec(point), true_novec)

    pow_novec = f_novec.copy()
    pow_novec **= power

    assert almost_equal(pow_novec(point), true_novec)

    # Vectorized
    true_arr = func_2d_vec_oop(points) ** power
    true_mg = func_2d_vec_oop(mg) ** power

    f_vec = fspace.element(func_2d_vec_dual, vectorized=True)
    pow_vec = f_vec ** power

    assert all_almost_equal(pow_vec(points), true_arr)
    assert all_almost_equal(pow_vec(mg), true_mg)

    pow_vec = f_vec.copy()
    pow_vec **= power

    assert all_almost_equal(pow_vec(points), true_arr)
    assert all_almost_equal(pow_vec(mg), true_mg)

    pow_vec(points, out=out_arr)
    pow_vec(mg, out=out_mg)

    assert all_almost_equal(out_arr, true_arr)
    assert all_almost_equal(out_mg, true_mg)


op = simple_fixture('op', ['+', '+=', '-', '-=', '*', '*=', '/', '/='])


var_params = ['vv', 'vs', 'sv']
var_ids = [' vec <op> vec ', ' vec <op> scal ', ' scal <op> vec ']


@pytest.fixture(scope="module", ids=var_ids, params=var_params)
def variant(request):
    return request.param


def _op(a, op, b):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b
    if op == '+=':
        a += b
        return a
    elif op == '-=':
        a -= b
        return a
    elif op == '*=':
        a *= b
        return a
    elif op == '/=':
        a /= b
        return a
    else:
        raise ValueError("bad operator '{}'.".format(op))


def test_fspace_vector_arithmetic(variant, op):
    if variant == 'sv' and '=' in op:  # makes no sense, quit
        return

    # Setup
    rect, points, mg = _standard_setup_2d()
    point = points.T[0]

    fspace = FunctionSpace(rect)
    a = -1.5
    b = 2.0
    array_out = np.empty((5,), dtype=float)
    mg_out = np.empty((2, 3), dtype=float)

    # Initialize a bunch of elements
    f_novec = fspace.element(func_2d_novec, vectorized=False)
    f_vec = fspace.element(func_2d_vec_dual, vectorized=True)
    g_novec = fspace.element(other_func_2d_novec, vectorized=False)
    g_vec = fspace.element(other_func_2d_vec_dual, vectorized=True)

    out_novec = fspace.element(vectorized=False)
    out_vec = fspace.element(vectorized=True)

    if variant[0] == 'v':
        true_l_novec = func_2d_novec(point)
        true_l_arr = func_2d_vec_oop(points)
        true_l_mg = func_2d_vec_oop(mg)

        test_l_novec = f_novec
        test_l_vec = f_vec
    else:  # 's'
        true_l_novec = true_l_arr = true_l_mg = a
        test_l_novec = test_l_vec = a

    if variant[1] == 'v':
        true_r_novec = other_func_2d_novec(point)
        true_r_arr = other_func_2d_vec_oop(points)
        true_r_mg = other_func_2d_vec_oop(mg)

        test_r_novec = g_novec
        test_r_vec = g_vec
    else:  # 's'
        true_r_novec = true_r_arr = true_r_mg = b
        test_r_novec = test_r_vec = b

    true_novec = _op(true_l_novec, op, true_r_novec)
    true_arr = _op(true_l_arr, op, true_r_arr)
    true_mg = _op(true_l_mg, op, true_r_mg)

    out_novec = _op(test_l_novec, op, test_r_novec)
    out_vec = _op(test_l_vec, op, test_r_vec)

    assert almost_equal(out_novec(point), true_novec)
    assert all_equal(out_vec(points), true_arr)
    out_vec(points, out=array_out)
    assert all_equal(array_out, true_arr)
    assert all_equal(out_vec(mg), true_mg)
    out_vec(mg, out=mg_out)
    assert all_equal(mg_out, true_mg)


# ---- Test function definitions ----

# 'ip' = in-place, 'oop' = out-of-place, 'dual' = dual-use


def cfunc_nd_oop(x):
    return sum(x) + 1j


def other_func_2d_novec(x):
    return x[0] + abs(x[1])


def other_func_2d_vec_oop(x):
    return x[0] + abs(x[1])


def other_func_2d_vec_ip(x, out):
    out[:] = x[0] + abs(x[1])


def other_func_2d_vec_dual(x, out=None):
    if out is None:
        return x[0] + abs(x[1])
    else:
        out[:] = x[0] + abs(x[1])


def other_cfunc_2d_novec(x):
    return 1j * x[0] + abs(x[1])


def other_cfunc_2d_vec_oop(x):
    return 1j * x[0] + abs(x[1])


def other_cfunc_2d_vec_ip(x, out):
    out[:] = 1j * x[0] + abs(x[1])


def other_cfunc_2d_vec_dual(x, out=None):
    if out is None:
        return 1j * x[0] + abs(x[1])
    else:
        out[:] = 1j * x[0] + abs(x[1])


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
