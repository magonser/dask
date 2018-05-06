from __future__ import absolute_import, division, print_function

import re

import numpy as np
from numpy.core.numerictypes import typecodes
from numpy import broadcast_to, asanyarray, asarray
import numpy.core.numeric as _nx
import sys

if sys.version_info[0] < 3:
    # Force range to be a generator, for np.delete's usage.
    range = xrange
    import __builtin__ as builtins
else:
    import builtins


def iterable(y):
    """
    Check whether or not an object can be iterated over.

    Parameters
    ----------
    y : object
      Input object.

    Returns
    -------
    b : bool
      Return ``True`` if the object has an iterator method or is a
      sequence and ``False`` otherwise.


    Examples
    --------
    >>> np.iterable([1, 2, 3])
    True
    >>> np.iterable(2)
    False

    """
    try:
        iter(y)
    except TypeError:
        return False
    return True


# See http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*,?)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_INPUT_ARGUMENTS = '(?:{0:}(?:,{0:})*,?)?'.format(_ARGUMENT)
_OUTPUT_ARGUMENTS = '{0:}(?:,{0:})*,?'.format(_ARGUMENT)
#_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{1:}$'.format(_INPUT_ARGUMENTS, _OUTPUT_ARGUMENTS)


def _parse_gufunc_signature(signature):
    """
    Parse string signatures for a generalized universal function.

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
        for ``np.matmul``.

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]], except for one output. Then output core
    dimension is not a list, but of the form Tuple[str, ...]
    """
    signature = signature.replace(' ', '')
    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid gufunc signature: {}'.format(signature))
    in_txt, out_txt = signature.split('->')
    ins = [tuple(re.findall(_DIMENSION_NAME, arg))
           for arg in re.findall(_ARGUMENT, in_txt)]
    outs = [tuple(re.findall(_DIMENSION_NAME, arg))
            for arg in re.findall(_ARGUMENT, out_txt)]
    outs = outs[0] if ((len(outs) == 1) and (out_txt[-1] != ',')) else outs
    return ins, outs


def _update_dim_sizes(dim_sizes, arg, core_dims):
    """
    Incrementally check and update core dimension sizes for a single argument.

    Arguments
    ---------
    dim_sizes : Dict[str, int]
        Sizes of existing core dimensions. Will be updated in-place.
    arg : ndarray
        Argument to examine.
    core_dims : Tuple[str, ...]
        Core dimensions for this argument.
    """
    if not core_dims:
        return

    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            '%d-dimensional argument does not have enough '
            'dimensions for all core dimensions %r'
            % (arg.ndim, core_dims))

    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError(
                    'inconsistent size for core dimension %r: %r vs %r'
                    % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size


def _parse_input_dimensions(args, input_core_dims):
    """
    Parse broadcast and core dimensions for vectorize with a signature.

    Arguments
    ---------
    args : Tuple[ndarray, ...]
        Tuple of input arguments to examine.
    input_core_dims : List[Tuple[str, ...]]
        List of core dimensions corresponding to each input.

    Returns
    -------
    broadcast_shape : Tuple[int, ...]
        Common shape to broadcast all non-core dimensions to.
    dim_sizes : Dict[str, int]
        Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib.stride_tricks._broadcast_shape(*broadcast_args)
    return broadcast_shape, dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    """Helper for calculating broadcast shapes with core dimensions."""
    return [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
            for core_dims in list_of_core_dims]


def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes):
    """Helper for creating output arrays in vectorize."""
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    arrays = tuple(np.empty(shape, dtype=dtype)
                   for shape, dtype in zip(shapes, dtypes))
    return arrays


class vectorize(object):
    """
    vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False,
              signature=None)

    Generalized function class.

    Define a vectorized function which takes a nested sequence of objects or
    numpy arrays as inputs and returns an single or tuple of numpy array as
    output. The vectorized function evaluates `pyfunc` over successive tuples
    of the input arrays like the python map function, except it uses the
    broadcasting rules of numpy.

    The data type of the output of `vectorized` is determined by calling
    the function with the first element of the input.  This can be avoided
    by specifying the `otypes` argument.

    Parameters
    ----------
    pyfunc : callable
        A python function or method.
    otypes : str or list of dtypes, optional
        The output data type. It must be specified as either a string of
        typecode characters or a list of data type specifiers. There should
        be one data type specifier for each output.
    doc : str, optional
        The docstring for the function. If `None`, the docstring will be the
        ``pyfunc.__doc__``.
    excluded : set, optional
        Set of strings or integers representing the positional or keyword
        arguments for which the function will not be vectorized.  These will be
        passed directly to `pyfunc` unmodified.

    cache : bool, optional
       If `True`, then cache the first function call that determines the number
       of outputs if `otypes` is not provided.

    signature : string, optional
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
        be called with (and expected to return) arrays with shapes given by the
        size of corresponding core dimensions. By default, ``pyfunc`` is
        assumed to take scalars as input and output.

        .. versionadded:: 1.12.0

    Returns
    -------
    vectorized : callable
        Vectorized function.

    Examples
    --------
    >>> def myfunc(a, b):
    ...     "Return a-b if a>b, otherwise return a+b"
    ...     if a > b:
    ...         return a - b
    ...     else:
    ...         return a + b

    >>> vfunc = np.vectorize(myfunc)
    >>> vfunc([1, 2, 3, 4], 2)
    array([3, 4, 1, 2])

    The docstring is taken from the input function to `vectorize` unless it
    is specified:

    >>> vfunc.__doc__
    'Return a-b if a>b, otherwise return a+b'
    >>> vfunc = np.vectorize(myfunc, doc='Vectorized `myfunc`')
    >>> vfunc.__doc__
    'Vectorized `myfunc`'

    The output type is determined by evaluating the first element of the input,
    unless it is specified:

    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <type 'numpy.int32'>
    >>> vfunc = np.vectorize(myfunc, otypes=[float])
    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <type 'numpy.float64'>

    The `excluded` argument can be used to prevent vectorizing over certain
    arguments.  This can be useful for array-like arguments of a fixed length
    such as the coefficients for a polynomial as in `polyval`:

    >>> def mypolyval(p, x):
    ...     _p = list(p)
    ...     res = _p.pop(0)
    ...     while _p:
    ...         res = res*x + _p.pop(0)
    ...     return res
    >>> vpolyval = np.vectorize(mypolyval, excluded=['p'])
    >>> vpolyval(p=[1, 2, 3], x=[0, 1])
    array([3, 6])

    Positional arguments may also be excluded by specifying their position:

    >>> vpolyval.excluded.add(0)
    >>> vpolyval([1, 2, 3], x=[0, 1])
    array([3, 6])

    The `signature` argument allows for vectorizing functions that act on
    non-scalar arrays of fixed length. For example, you can use it for a
    vectorized calculation of Pearson correlation coefficient and its p-value:

    >>> import scipy.stats
    >>> pearsonr = np.vectorize(scipy.stats.pearsonr,
    ...                         signature='(n),(n)->(),()')
    >>> pearsonr([[0, 1, 2, 3]], [[1, 2, 3, 4], [4, 3, 2, 1]])
    (array([ 1., -1.]), array([ 0.,  0.]))

    Or for a vectorized convolution:

    >>> convolve = np.vectorize(np.convolve, signature='(n),(m)->(k)')
    >>> convolve(np.eye(4), [1, 2, 1])
    array([[ 1.,  2.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  2.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  2.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  2.,  1.]])

    See Also
    --------
    frompyfunc : Takes an arbitrary Python function and returns a ufunc

    Notes
    -----
    The `vectorize` function is provided primarily for convenience, not for
    performance. The implementation is essentially a for loop.

    If `otypes` is not specified, then a call to the function with the
    first argument will be used to determine the number of outputs.  The
    results of this call will be cached if `cache` is `True` to prevent
    calling the function twice.  However, to implement the cache, the
    original function must be wrapped which will slow down subsequent
    calls, so only do this if your function is expensive.

    The new keyword argument interface and `excluded` argument support
    further degrades performance.

    References
    ----------
    .. [1] NumPy Reference, section `Generalized Universal Function API
           <http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html>`_.
    """

    def __init__(self, pyfunc, otypes=None, doc=None, excluded=None,
                 cache=False, signature=None):
        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature
        self._ufunc = None    # Caching to improve default performance

        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc

        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes['All']:
                    raise ValueError("Invalid otype specified: %s" % (char,))
        elif iterable(otypes):
            otypes = ''.join([_nx.dtype(x).char for x in otypes])
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes

        # Excluded variable support
        if excluded is None:
            excluded = set()
        self.excluded = set(excluded)

        if signature is not None:
            self._in_and_out_core_dims = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dims = None


    def __call__(self, *args, **kwargs):
        """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
        excluded = self.excluded
        if not kwargs and not excluded:
            func = self.pyfunc
            vargs = args
        else:
            # The wrapper accepts only positional arguments: we use `names` and
            # `inds` to mutate `the_args` and `kwargs` to pass to the original
            # function.
            nargs = len(args)

            names = [_n for _n in kwargs if _n not in excluded]
            inds = [_i for _i in range(nargs) if _i not in excluded]
            the_args = list(args)

            def func(*vargs):
                for _n, _i in enumerate(inds):
                    the_args[_i] = vargs[_n]
                kwargs.update(zip(names, vargs[len(inds):]))
                return self.pyfunc(*the_args, **kwargs)

            vargs = [args[_i] for _i in inds]
            vargs.extend([kwargs[_n] for _n in names])

        return self._vectorize_call(func=func, args=vargs)

    def _vectorize_call(self, func, args):
        """Vectorized call to `func` over positional `args`."""
        if self.signature is not None:
            res = self._vectorize_call_with_signature(func, args)
        elif not args:
            res = func()
        else:
            raise NotImplementedError()
        return res

    def _vectorize_call_with_signature(self, func, args):
        """Vectorized call over positional arguments with a signature."""
        input_core_dims, output_core_dims = self._in_and_out_core_dims

        if len(args) != len(input_core_dims):
            raise TypeError('wrong number of positional arguments: '
                            'expected %r, got %r'
                            % (len(input_core_dims), len(args)))
        args = tuple(asanyarray(arg) for arg in args)

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims)
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes,
                                         input_core_dims)
        args = [broadcast_to(arg, shape, subok=True)
                for arg, shape in zip(args, input_shapes)]

        outputs = None
        scalaroutput = not isinstance(output_core_dims, list)
        nout = 1 if scalaroutput else len(output_core_dims)
        output_core_dims = output_core_dims if not scalaroutput else [output_core_dims]
        #####onetupleoutput = not scalaroutput and (nout == 1)
        otypes = self.otypes if self.otypes is None or not scalaroutput else (self.otypes,)

        for index in np.ndindex(*broadcast_shape):
            results = func(*(arg[index] for arg in args))

            n_results = len(results) if isinstance(results, tuple) else 1

            if nout != n_results:
                raise ValueError(
                    'wrong number of outputs from pyfunc: expected %r, got %r'
                    % (nout, n_results))

            if scalaroutput:
                results = (results,)

            if outputs is None:
                for result, core_dims in zip(results, output_core_dims):
                    _update_dim_sizes(dim_sizes, result, core_dims)

                if otypes is None:
                    otypes = [asarray(result).dtype for result in results]

                outputs = _create_arrays(broadcast_shape, dim_sizes,
                                         output_core_dims, otypes)

            for output, result in zip(outputs, results):
                output[index] = result

        if outputs is None:
            # did not call the function even once
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            if builtins.any(dim not in dim_sizes
                            for dims in output_core_dims
                            for dim in dims):
                raise ValueError('cannot call `vectorize` with a signature '
                                 'including new output dimensions on size 0 '
                                 'inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     output_core_dims, otypes)
        print(outputs)
        return outputs[0] if scalaroutput else outputs