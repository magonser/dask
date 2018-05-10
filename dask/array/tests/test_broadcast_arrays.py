from __future__ import absolute_import, division, print_function

from distutils.version import LooseVersion
import pytest
from numpy.testing import assert_array_equal, assert_equal
import dask.array as da
import numpy as np
from dask.array.utils import assert_eq


from dask.array.broadcast_arrays import _where, _inverse
from dask.array.broadcast_arrays import _parse_signature, broadcast_arrays


def test__where():
    assert _where([], 1) is None
    assert _where([0, 2], 1) is None
    assert _where([0, 2], 2) == 1
    assert _where([0, 2, None], None) == 2


def test__inverse():
    assert _inverse([0]) == [0]
    assert _inverse([0, 1]) == [0, 1]
    assert _inverse([2, 1]) == [None, 1, 0]
    assert _inverse([None, 1, 0]) == [2, 1]


def test__parse_signature():
    assert_equal(_parse_signature(''), ([], []))
    assert_equal(_parse_signature('()'), ([tuple()], []))
    assert_equal(_parse_signature('()->()'), ([tuple()], [tuple()]))
    assert_equal(_parse_signature('->()'), ([], [tuple()]))
    assert_equal(_parse_signature('(i,k)->(j)'), ([('i', 'k')], [('j',)]))
    assert_equal(_parse_signature('(i,k),(j)->(k),()'), ([('i', 'k'),('j',)], [('k',), tuple()]))


# Test from `dask.array.tests.test_array_core.py`
def test_broadcast_arrays():
    # Calling `broadcast_arrays` with no arguments only works in NumPy 1.13.0+.
    if LooseVersion(np.__version__) >= LooseVersion("1.13.0"):
        assert np.broadcast_arrays() == broadcast_arrays()

    a = np.arange(4)
    d_a = da.from_array(a, chunks=tuple(s // 2 for s in a.shape))

    a_0 = np.arange(4)[None, :]
    a_1 = np.arange(4)[:, None]

    d_a_0 = d_a[None, :]
    d_a_1 = d_a[:, None]

    a_r = np.broadcast_arrays(a_0, a_1)
    d_r = broadcast_arrays(d_a_0, d_a_1)

    assert isinstance(d_r, list)
    assert len(a_r) == len(d_r)

    for e_a_r, e_d_r in zip(a_r, d_r):
        assert_eq(e_a_r, e_d_r)


@pytest.mark.parametrize("sparse", [True, False])
def test_broadcast_arrays_00(sparse):
    x = np.array([1, 2, 3])
    y = np.array([10, 20])
    X, Y = np.meshgrid(x, y, indexing="ij", sparse=sparse)
    Z = X + Y
    DX, DY = broadcast_arrays(x, y, signature="(a),(b)", sparse=sparse)
    DZ = DX + DY
    assert_array_equal(Z, DZ.compute())


def test_broadcast_arrays_01():
    x = da.random.normal(size=(3, 1), chunks=(2, 3))
    y = da.random.normal(size=(1, 4), chunks=(2, 3))
    X, Y = broadcast_arrays(x, y, signature="(i,j),(i,j)", sparse=False)
    Z = X + Y
    assert Z.shape == (3, 4)


def test_broadcast_arrays_02():
    a = np.random.randn(3, 1)
    b = np.random.randn(4,)
    A, B = broadcast_arrays(a, b, sparse=False)
    assert A.shape == (3, 4)
    assert B.shape == (3, 4)


def test_broadcast_arrays_03():
    x = da.random.normal(size=(2, 4, 5), chunks=(1, 3, 4))
    y = da.random.normal(size=(6, 2, 8, 7), chunks=(2, 1, 5, 6))
    X, Y = broadcast_arrays(x, y, signature="(i,V,U),(W,i,j,X)->(U,V),(W,X)", sparse=False)
    assert X.shape == (2, 8, 5, 4)
    assert Y.shape == (2, 8, 6, 7)


def test_broadcast_arrays_04():
    x = da.random.normal(size=(2, 8, 4, 5), chunks=(1, 5, 3, 4))
    y = da.random.normal(size=(6, 8, 2, 7), chunks=(2, 5, 1, 6))
    X, Y = broadcast_arrays(x, y, signature="(i,j,V,U),(W,j,i,X)->(U,V),(W,X)", sparse=False)
    assert X.shape == (2, 8, 5, 4)
    assert Y.shape == (2, 8, 6, 7)


def test_broadcast_arrays_05():
    x = da.random.normal(size=(20, 30, 3), chunks=(5, 6, 3))
    y = da.random.normal(size=(3, 30, 20, 2), chunks=(3, 6, 5, 2))
    X, Y = broadcast_arrays(x, y, signature="(i,j,U),(U,j,i,V)->(U),(V,U)", sparse=False)
    assert X.shape == (20, 30, 3)
    assert Y.shape == (20, 30, 2, 3)


# 1) Broadcast two scalars each with distinct loop dims against each other
@pytest.mark.parametrize("signature", ["(i),(j)",
                                       "(i),(j)->(),()",
                                       "(i),(j)->(i,j),(i,j)"])
def test_broadcast_arrays_scalar_distinct_loop_dims(signature):
    a = np.random.randn(3)
    b = np.random.randn(4,)
    A, B = broadcast_arrays(a, b, signature=signature, sparse=False)
    assert A.shape == (3, 4)
    assert B.shape == (3, 4)


# 2) Broadcast two vectors each with distinct loop dims against each other
@pytest.mark.parametrize("signature", ["(i),(j)",
                                       "(i),(j)->(U),(V)",  # Advanced usage, maybe not recommended
                                       "(i,U),(j,V)->(U),(V)",
                                       "(i,U),(j,V)->(i,j,U),(i,j,V)"])
def test_broadcast_arrays_vectors_distinct_loop_dims(signature):
    a = np.random.randn(3, 5)
    b = np.random.randn(4, 6)
    A, B = broadcast_arrays(a, b, signature=signature, sparse=False)
    assert A.shape == (3, 4, 5)
    assert B.shape == (3, 4, 6)


# 3) Broadcast two scalars with same loop dim against each other
@pytest.mark.parametrize("signature", [None,
                                       "(i),(i)",
                                       "->(),()",
                                       "(i),(i)->(),()",
                                       "(i),(i)->(i),(i)"])
def test_broadcast_arrays_scaVars_same_loop_dims(signature):
    a = np.random.randn(3)
    b = np.random.randn(3)
    A, B = broadcast_arrays(a, b, signature=signature, sparse=False)
    assert A.shape == (3,)
    assert B.shape == (3,)


# 4) Broadcast two vectors each with same loop dim against each other
@pytest.mark.parametrize("signature", ["(i),(i)",
                                       "->(U),(V)",
                                       "(i),(i)->(U),(V)",  # Advanced usage, maybe not recommended
                                       "(i,U),(i,V)->(U),(V)",
                                       "(i,U),(i,V)->(i,U),(i,V)"])
def test_broadcast_arrays_vectors_same_loop_dims(signature):
    a = np.random.randn(3, 5)
    b = np.random.randn(3, 6)
    A, B = broadcast_arrays(a, b, signature=signature, sparse=False)
    assert A.shape == (3, 5)
    assert B.shape == (3, 6)


# 5a) Broadcast two vectors each with partially same loop dim against each other no proposed loop dim order
@pytest.mark.parametrize("signature", ["(j),(i,j)",
                                       "->(U),(V)",
                                       "(j),(i,j)->(U),(V)",  # Advanced usage, maybe not recommended
                                       "(j,U),(i,j,V)->(U),(V)",
                                       "(j,U),(i,j,V)->(i,j,U),(i,j,V)"])
def test_broadcast_arrays_vectors_mixed_loop_dims(signature):
    a = np.random.randn(4, 5)
    b = np.random.randn(3, 4, 6)
    A, B = broadcast_arrays(a, b, signature=signature, sparse=False)
    assert A.shape == (3, 4, 5)
    assert B.shape == (3, 4, 6)


# 5b) Broadcast two vectors each with partially same loop dim against each other no proposed loop dim order
@pytest.mark.parametrize("signature", ["(i),(i,j)",
                                       "(i),(i,j)->(U),(V)",  # Advanced usage, maybe not recommended
                                       "(i,U),(i,j,V)->(U),(V)",
                                       "(i,U),(i,j,V)->(i,j,U),(i,j,V)"])
def test_broadcast_arrays_vectors_mixed_loop_dims_02(signature):
    a = np.random.randn(3, 5)
    b = np.random.randn(3, 4, 6)
    A, B = broadcast_arrays(a, b, signature=signature, sparse=False)
    assert A.shape == (3, 4, 5)
    assert B.shape == (3, 4, 6)
