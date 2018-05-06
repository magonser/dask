from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
import operator
from pytest import raises

assert_raises = raises
assert_raises_regex = lambda ex, regex: raises(ex, match=regex)

from dask.array.vectorize import _parse_gufunc_signature, vectorize


def test_parse_gufunc_signature():
    assert_equal(_parse_gufunc_signature('(x)->()'), ([('x',)], ()))
    assert_equal(_parse_gufunc_signature('(x,y)->()'),
                 ([('x', 'y')], ()))
    assert_equal(_parse_gufunc_signature('(x),(y)->()'),
                 ([('x',), ('y',)], ()))
    assert_equal(_parse_gufunc_signature('(x)->(y)'),
                 ([('x',)], ('y',)))
    assert_equal(_parse_gufunc_signature('(x)->(y),()'),
                 ([('x',)], [('y',), ()]))
    assert_equal(_parse_gufunc_signature('(),(a,b,c),(d)->(d,e)'),
                 ([(), ('a', 'b', 'c'), ('d',)], ('d', 'e')))
    with assert_raises(ValueError):
        _parse_gufunc_signature('(x)(y)->()')
    with assert_raises(ValueError):
        _parse_gufunc_signature('(x),(y)->')
    with assert_raises(ValueError):
        _parse_gufunc_signature('((x))->(x)')
    with assert_raises(ValueError):
        _parse_gufunc_signature(',(y)->')

    # Extension to length 1 output tuples
    assert_equal(_parse_gufunc_signature('(x)->(y),'),
                 ([('x',)], [('y',)]))
    assert_equal(_parse_gufunc_signature('(x,)->(y),'),
                 ([('x',)], [('y',)]))

    # Allow zero inputs (vectorize already accepts this behaviour, if no signature is given):
    assert_equal(_parse_gufunc_signature('->(y)'),
                 ([], ('y',)))


def test_signature_simple():
    def addsubtract(a, b):
        if a > b:
            return a - b
        else:
            return a + b

    f = vectorize(addsubtract, signature='(),()->()')
    r = f([0, 3, 6, 9], [1, 3, 5, 7])
    assert_array_equal(r, [1, 6, 1, 2])


def test_signature_mean_last():
    def mean(a):
        return a.mean()

    f = vectorize(mean, signature='(n)->()')
    r = f([[1, 3], [2, 4]])
    assert_array_equal(r, [2, 3])


def test_signature_center():
    def center(a):
        return a - a.mean()

    f = vectorize(center, signature='(n)->(n)')
    r = f([[1, 3], [2, 4]])
    assert_array_equal(r, [[-1, 1], [-1, 1]])


def test_signature_one_outputs():
    f = vectorize(lambda x: (x,), signature='()->(),')
    r = f([1, 2, 3])
    assert_(isinstance(r, tuple) and len(r) == 1)
    assert_array_equal(r[0], [1, 2, 3])


def test_signature_two_outputs():
    f = vectorize(lambda x: (x, x), signature='()->(),()')
    r = f([1, 2, 3])
    assert_(isinstance(r, tuple) and len(r) == 2)
    assert_array_equal(r[0], [1, 2, 3])
    assert_array_equal(r[1], [1, 2, 3])


def test_signature_outer():
    f = vectorize(np.outer, signature='(a),(b)->(a,b)')
    r = f([1, 2], [1, 2, 3])
    assert_array_equal(r, [[1, 2, 3], [2, 4, 6]])

    r = f([[[1, 2]]], [1, 2, 3])
    assert_array_equal(r, [[[[1, 2, 3], [2, 4, 6]]]])

    r = f([[1, 0], [2, 0]], [1, 2, 3])
    assert_array_equal(r, [[[1, 2, 3], [0, 0, 0]],
                           [[2, 4, 6], [0, 0, 0]]])

    r = f([1, 2], [[1, 2, 3], [0, 0, 0]])
    assert_array_equal(r, [[[1, 2, 3], [2, 4, 6]],
                           [[0, 0, 0], [0, 0, 0]]])


def test_signature_computed_size():
    f = vectorize(lambda x: x[:-1], signature='(n)->(m)')
    r = f([1, 2, 3])
    assert_array_equal(r, [1, 2])

    r = f([[1, 2, 3], [2, 3, 4]])
    assert_array_equal(r, [[1, 2], [2, 3]])


def test_signature_excluded():

    def foo(a, b=1):
        return a + b

    f = vectorize(foo, signature='()->()', excluded={'b'})
    assert_array_equal(f([1, 2, 3]), [2, 3, 4])
    assert_array_equal(f([1, 2, 3], b=0), [1, 2, 3])


def test_signature_otypes():
    f = vectorize(lambda x: x, signature='(n)->(n)', otypes=['float64'])
    r = f([1, 2, 3])
    assert_equal(r.dtype, np.dtype('float64'))
    assert_array_equal(r, [1, 2, 3])


def test_signature_invalid_inputs():
    f = vectorize(operator.add, signature='(n),(n)->(n)')
    with assert_raises_regex(TypeError, 'wrong number of positional'):
        f([1, 2])
    with assert_raises_regex(
            ValueError, 'does not have enough dimensions'):
        f(1, 2)
    with assert_raises_regex(
            ValueError, 'inconsistent size for core dimension'):
        f([1, 2], [1, 2, 3])

    f = vectorize(operator.add, signature='()->()')
    with assert_raises_regex(TypeError, 'wrong number of positional'):
        f(1, 2)


def test_signature_invalid_outputs():

    f = vectorize(lambda x: x[:-1], signature='(n)->(n)')
    with assert_raises_regex(
            ValueError, 'inconsistent size for core dimension'):
        f([1, 2, 3])

    f = vectorize(lambda x: x, signature='()->(),()')
    with assert_raises_regex(ValueError, 'wrong number of outputs'):
        f(1)

    f = vectorize(lambda x: (x, x), signature='()->()')
    with assert_raises_regex(ValueError, 'wrong number of outputs'):
        f([1, 2])


def test_size_zero_output():
    # see issue 5868
    f = vectorize(lambda x: x)
    x = np.zeros([0, 5], dtype=int)
    with assert_raises_regex(ValueError, 'otypes'):
        f(x)

    f.otypes = 'i'
    assert_array_equal(f(x), x)

    f = vectorize(lambda x: x, signature='()->()')
    with assert_raises_regex(ValueError, 'otypes'):
        f(x)

    f = vectorize(lambda x: x, signature='()->()', otypes='i')
    assert_array_equal(f(x), x)

    f = vectorize(lambda x: x, signature='(n)->(n)', otypes='i')
    assert_array_equal(f(x), x)

    f = vectorize(lambda x: x, signature='(n)->(n)')
    assert_array_equal(f(x.T), x.T)

    f = vectorize(lambda x: [x], signature='()->(n)', otypes='i')
    with assert_raises_regex(ValueError, 'new output dimensions'):
        f(x)