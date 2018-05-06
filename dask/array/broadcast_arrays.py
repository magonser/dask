from __future__ import absolute_import, division, print_function

import re
from functools import partial
from itertools import count
from operator import itemgetter
import numpy as np
try:
    from cytoolz import concat, groupby, valmap, unique, compose
except ImportError:
    from toolz import concat, groupby, valmap, unique, compose

from .core import asarray, asanyarray, broadcast_to


_DIMENSION_NAME = r'\w+'
_DIMENSION_LIST = '(?:{0:}(?:,{0:})*,?)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_DIMENSION_LIST)
_ARGUMENTS = '(?:{0:}(?:,{0:})*,?)?'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{1:}$'.format(_ARGUMENTS, _ARGUMENTS)


def _parse_signature(signature):
    """
    Parse string signatures for broadcast arrays.

    Arguments
    ---------
    signature : String

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]]. 
    """
    signature = signature.replace(' ', '')
    signature = signature if '->' in signature else signature + '->'
    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid signature: {}'.format(signature))
    in_txt, out_txt = signature.split('->')
    ins = [tuple(re.findall(_DIMENSION_NAME, arg))
           for arg in re.findall(_ARGUMENT, in_txt)]
    outs = [tuple(re.findall(_DIMENSION_NAME, arg))
            for arg in re.findall(_ARGUMENT, out_txt)]
    return ins, outs


def _where(seq, elem):
    """
    Returns first occurrence of ``elem`` in ``seq``
    or ``None``, if ``elem`` is not found
    """
    try:
        return [i for i, e in enumerate(seq) if e == elem][0]
    except IndexError:
        return None


def _inverse(seq):
    """
    Returns the inverse of a sequence of int. 
    ``None`` is ignored.

    Examples
    --------
    >>> _inverse([0, 1])
    [0, 1]
    >>> _inverse([2, 1])
    [None, 1, 0]
    >>> _inverse([None, 1, 0])
    [2, 1] 
    """
    n = max(filter(None, seq))
    return [_where(seq, i) for i in range(n+1)]


def broadcast_arrays(*args, **kwargs):
    """
    Broadcasts arrays against each other.


    Parameters
    ----------
    *args: Arrays
        Arrays to be broadcast against each other.

    signature: Optional; String, Iterable of Iterable, or None
        Specifies loop and core dimensions within each array, where only loop
        dimensions are broadcast. E.g. ``"(i,K),(j)->(K),()"`` is the signature
        for two arrays, where the first array has ``"K"`` as core dimension and
        the two loop dimensions are ``"i"`` and ``"j"``. Specification of core
        dimensions could also be omitted, e.g. ``"(i),(j)"``. Then only loop
        dimensions could be specified and the signature is left aligned,
        meaning trailing array dimensions are considered core dimensions.

        Dimension sizes for same loop dimensions must match or be of length
        ``1``. Dimension sizes of same core dimensions must match.

        Chunk sizes for same loop or core dimensions must be same,
        except where a loop dimension is sparse and of length ``1``.

        Order of loop dimensions is in order of their appearance.
        Core dimensions are put to the end in the specified order.

        Defaults to ``None``.

        Note: while the syntax is similar to numpy generalized 
        ufuncs [2]_, its meaning here is different. 
       
    sparse: Optional; Bool
        Specifies if a broadcast should be sparse, i.e. new broadcast
        dimensions are of length 1. If not sparse existing sparse
        dimensions are broadcast to their full size (as stated by
        the other passed arguments). Defaults to ``False``.
    
    subok: Bool


    Returns
    -------
     : Arrays
        Broadcast arrays


    Examples
    --------
    >>> a = np.random.randn(3, 1)
    >>> b = np.random.randn(1, 4)
    >>> A, B = broadcast_arrays(a, b)
    >>> A.shape, B.shape
    (3, 4), (3, 4)


    >>> a = np.random.randn(3)
    >>> b = np.random.randn(4)
    >>> A, B = broadcast_arrays(a, b, signature="(i),(j)")
    >>> A.shape, B.shape
    (3, 4), (3, 4)


    >>> a = np.random.randn(3)
    >>> b = np.random.randn(4)
    >>> A, B = broadcast_arrays(a, b, signature="(i),(j)", sparse=True)
    >>> A.shape, B.shape
    (3, 1), (1, 4)


    >>> x = da.random.normal(size=(2, 4, 5), chunks=(1, 3, 4))
    >>> y = da.random.normal(size=(6, 2, 8, 7), chunks=(2, 1, 5, 6))
    >>> X, Y = broadcast_arrays(x, y, signature="(L1,K2,K1),(K3,L1,L2,K4)->(K1,K2),(K3,K4)", sparse=True)
    >>> X.shape, Y.shape
    (2, 1, 5, 4), (2, 8, 6, 7)


    References
    ----------
    .. [1] http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html


    """
    signature = kwargs.pop('signature', None)
    sparse = kwargs.pop('sparse', False)
    subok = bool(kwargs.pop('subok', False))
    if kwargs:
        raise TypeError("Unsupported keyword argument(s) provided")
    
    # Input processing
    to_array = asanyarray if subok else asarray
    args = tuple(to_array(e) for e in args)
    shapes = [e.shape for e in args]
    chunkss = [[i[0] for i in e.chunks] for e in args]
    ndimss = [len(s) for s in shapes]

    # Inject default behavior for signature
    if signature is None:
        signature = [tuple(reversed(range(n))) for n in ndimss]

    # Parse signature
    if isinstance(signature, str):
        array_dimss, core_dimss = _parse_signature(signature)
    elif isinstance(signature, tuple):
        array_dimss, core_dimss = signature
    elif isinstance(signature, list):
        array_dimss = signature
        core_dimss = []
    elif signature is None:
        array_dimss = [reversed(range(n) for n in ndimss)]
        core_dimss = []
    else:
        raise ValueError("``signature`` is invalid")
    core_dimss = core_dimss if core_dimss else [tuple()]*len(array_dimss)

    # Check consistency of passed arguments
    if len(array_dimss) != len(args):
        raise ValueError("``signature`` does not match number of input arrays")
    if len(core_dimss) != len(args):
        raise ValueError("``signature`` does not match number of input arrays on right hand side")

    for idx, ndims, array_dims, core_dims in zip(count(1), ndimss, array_dimss, core_dimss):
        if len(set(array_dims)) != len(array_dims):
            raise ValueError("Repeated dimension name for array #{} in signature".format(idx))
        if len(array_dims) > ndims:
            raise ValueError("Too many dimension(s) for input array #{} in signature given".format(idx))
        if not set(core_dims).issubset(array_dims):
            raise ValueError("Core dimension(s) ``{}`` are not given on left hand side ``{}`` for array #{} in signature".format(core_dims, array_dims, idx))
    
    # Extend missing core dims
    new_core_dimss = [tuple('__broadcast_arrays_coredim_{}_{}'.format(i, j) for j in range(n - len(ad))) \
                      for i, n, ad in zip(count(), ndimss, array_dimss)]
    core_dimss = [cd + ncd for cd, ncd in zip(core_dimss, new_core_dimss)]
    array_dimss = [ad + ncd for ad, ncd in zip(array_dimss, new_core_dimss)]

    # Check that the arrays have same length for same dimensions
    _temp = groupby(0, concat(zip(ad, s) for ad, s in zip(array_dimss, shapes)))
    dimsizess = valmap(compose(set, partial(map, itemgetter(1))), _temp)
    for dim, sizes in dimsizess.items():
        if sizes.union({1}) != {1, max(sizes)}:
            raise ValueError("Dimension ``{}`` with different lengths in arrays".format(dim))
    dimsizes = valmap(max, dimsizess)

    # Check if arrays have same chunk size for the same dimension
    _temp = groupby(0, concat(zip(ad, s, c) for ad, s, c in zip(array_dimss, shapes, chunkss)))
    dimchunksizess = valmap(compose(set,
                                    partial(map, itemgetter(1)),
                                    partial(filter, lambda e: e != (1, 1)),
                                    partial(map, lambda tpl: tpl[1:])),
                            _temp)
    for dim, dimchunksizes in dimchunksizess.items():
        if len(dimchunksizes) > 1:
            raise ValueError('Dimension ``{}`` with different chunksize present'.format(dim))
    dimchunksizes = valmap(max, dimchunksizess)

    # Find loop dims and union of all loop dims in order of appearance
    _temp = ((d for d in ad if d not in cd) for ad, cd in zip(array_dimss, core_dimss))
    total_loop_dims = tuple(unique(concat(_temp)))

    total_loop_dims_id = dict(zip(total_loop_dims, count(-1, -1)))
    total_loop_dims_id_inv = {v: k for k, v in total_loop_dims_id.items()}

    # Find order of transposition for each array and perform transformations
    new_args = []
    for arg, array_dims, core_dims, shape, chunks in \
        zip(args, array_dimss, core_dimss, shapes, chunkss):

        # Find new position of given dimension and maybe indicate dimensions which have to be created
        old2new_poss = []
        for loop_dim in total_loop_dims:
            oidx = _where(array_dims, loop_dim)
            old2new_poss.append(oidx if oidx is not None else total_loop_dims_id[loop_dim])  # Insert new dim if not present
        for core_dim in core_dims:
            old2new_poss.append(_where(array_dims, core_dim))

        # Determine the new shape size by pre-pending newly created dimensions
        if sparse is True:
            new_shape = tuple(1 for i in old2new_poss if i < 0) + shape
        else:
            # If we had size `1` amongst existing loop dims, we also need to have a new shape for them
            new_shape = tuple(dimsizes[total_loop_dims_id_inv[i]] for i in old2new_poss if i < 0) \
                      + tuple(s if d not in total_loop_dims else dimsizes[d] for d, s in zip(array_dims, shape))

        # Chunks can be original size, in case of `sparse=True` it will be cut back to `1` later
        new_chunks = tuple(dimchunksizes[total_loop_dims_id_inv[i]] for i in old2new_poss if i < 0) + tuple(chunks)

        # Apply `dask.array.broadcast_to`
        new_arg = broadcast_to(arg, shape=new_shape, chunks=new_chunks)

        # Determine order for transpose and do so
        idcs = _inverse(np.argsort(old2new_poss).tolist())
        new_arg = new_arg.transpose(idcs)

        new_args.append(new_arg)
    
    return new_args
