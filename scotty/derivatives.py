import numpy as np

from typing import Callable, Dict, Optional, Tuple, Union
from scotty.typing import ArrayLike, FloatArray

import functools


def _maybe_bytes(arg):
    """Return either ``arg`` as bytes if possible, or just ``arg``"""
    try:
        return arg.tobytes()
    except AttributeError:
        return arg


def cache(func):
    """Simple caching of a function compatible with numpy array arguments"""
    hits = misses = 0

    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        nonlocal hits, misses
        cache_key = hash(
            (
                tuple(map(_maybe_bytes, args)),
                tuple((k, _maybe_bytes(v)) for k, v in kwargs.items()),
            )
        )

        try:
            result = wrapper_cache.cache[cache_key]
            hits += 1
        except KeyError:
            result = func(*args, **kwargs)
            wrapper_cache.cache[cache_key] = result
            misses += 1

        return result

    def cache_info():
        """Return some information on cache usage by this function"""
        nonlocal hits, misses
        return {"hits": hits, "misses": misses}

    wrapper_cache.cache = {}
    wrapper_cache.cache_info = cache_info
    return wrapper_cache


Stencil = Dict[Tuple, float]

STENCILS: Dict[str, Stencil] = {
    "d1_FFD2": {(0,): -3 / 2, (1,): 2, (2,): -0.5},
    "d2_FFD2": {(0,): 2, (1,): -5, (2,): 4, (3,): -1},
    "d1_CFD2": {(-1,): -0.5, (1,): 0.5},
    "d1_CFD4": {(-2,): 1 / 12, (-1,): -2 / 3, (1,): 2 / 3, (2,): -1 / 12},
    "d2_CFD2": {(-1,): 1, (0,): -2, (1,): 1},
    "d1d1_FFD_FFD2": {
        (0, 0): 2.25,
        (0, 1): -3,
        (0, 2): 0.75,
        (1, 0): -3,
        (1, 1): 4,
        (1, 2): -1,
        (2, 0): 0.75,
        (2, 1): -1,
        (2, 2): 0.25,
    },
    "d1d1_FFD_CFD2": {
        (0, -1): 0.25,
        (0, 0): 1,
        (0, 1): -1.25,
        (1, 0): -2,
        (1, 1): 2,
        (2, -1): -0.25,
        (2, 0): 1,
        (2, 1): -0.75,
    },
    "d1d1_CFD_CFD2": {
        (1, 1): 0.25,
        (1, -1): -0.25,
        (-1, 1): -0.25,
        (-1, -1): 0.25,
    },
}
"""Finite difference stencils for `derivative`.

The second-derivative stencils are optimised for minimal function evaluations.

The naming scheme here is ``<order>_<kind><error>`` with:

- ``<order>`` is ``d<n>`` for the ``n``th derivative, and this is repeated for
  multiple dimensions
- ``<kind>`` is either ``FFD`` for forward finite difference or ``CFD`` for
  central finite difference, again repeated for each dimension
- ``<error>`` is the order of the error scaling

So ``d1d1_FFD_CFD2`` is the first mixed-derivative of two dimensions, using
forward differences for the first dimension and central for the second, with
second-order error overall
"""

_derivative_function_cache: Dict[Callable, Callable] = {}


def derivative(
    func: Callable,
    dims: Union[str, Tuple[str, ...]],
    args: Dict[str, ArrayLike],
    spacings: Union[float, Dict[str, float]] = 1e-8,
    stencil: Optional[str] = None,
    use_cache: bool = True,
) -> FloatArray:
    """Partial derivative of a function along one or more of its arguments.

    Currently this can take partial derivatives in one or two arguments, given
    by ``dims`` and evaluated at ``args``. The arguments to ``func`` must be
    exactly the keys of ``args``, and ``func`` must be able to take them as
    keyword arguments.

    The function will be evaluated at points around ``args`` given by the
    integer offsets in the stencil multiplied by the step size in each dimension
    given in ``spacings``.

    By default, calls to ``func`` with a given set of arguments will be cached,
    which enables re-use of evaluation points across multiple derivatives. This
    is particularly useful when needing to take partial derivatives in many
    directions. However, if ``func`` has side effects, such as printing
    variables, these will not work, and you should set ``use_cache = False``.

    Parameters
    ----------
    func:
        Function to take the derivative of
    dims:
        The name(s) of the dimension(s) to take the derivative along. ``dims``
        must appear in the keys of ``args``. For second derivatives in one
        dimension, pass a tuple with the argument name repeated, for example
        ``("x", "x")`` for the second derivative in ``x``
    args:
        Arguments to ``func`` at the point to evaluate the derivative
    spacings:
        Step size for derivative. If a single float, used for all dimensions,
        otherwise if a mapping of dimension names to step size then the keys
        must be identical to those of ``dims``
    stencil:
        Stencil name (see `STENCILS` for supported stencils). Defaults to
        central differences
    use_cache:
        If true, then wraps ``func`` with the decorator `cache`

    Returns
    -------
    ArrayLike
        Derivative of ``func`` evaluated at ``location``

    Examples
    --------
    >>> derivative(lambda x: np.sin(x), "x", {"x": 0.0})
    1.0

    >>> derivative(lambda x, y: np.sin(x) * y**2, ("x", "y"), {"x": 0.0, "y": 2.0})
    4.0

    """

    # Ensure that dims is a tuple
    if isinstance(dims, str):
        dims = (dims,)

    for dim in dims:
        if dim not in args:
            raise ValueError(f"Dimension '{dim}' not in 'args' ({list(args.keys())})")

    if not isinstance(spacings, dict):
        default_spacing = spacings
        spacings = {dim: default_spacing for dim in dims}

    # Collect the relative spacings for the derivative dimensions
    dim_spacings = [spacings[dim] for dim in dims]
    # For second order derivatives, remove repeated dimensions. We need to do
    # this after getting the list of step sizes so that this is correct for
    # second order derivatives
    if len(dims) == 2 and dims[1] == dims[0]:
        dims = (dims[0],)

    if stencil is None:
        if len(dims) == 2:
            stencil = "d1d1_CFD_CFD2"
        elif len(dim_spacings) == 2:
            stencil = "d2_CFD2"
        elif len(dim_spacings) == 1:
            stencil = "d1_CFD2"
        else:
            raise ValueError(
                f"No stencil given and unsupported derivative order (dims={dims})"
            )

    try:
        stencil_ = STENCILS[stencil]
    except KeyError:
        raise ValueError(
            f"Unknown stencil name '{stencil}', expected one of {list(STENCILS.keys())}"
        )

    # Retrieve or create a version of the user function that caches results
    global _derivative_function_cache
    if use_cache:
        try:
            cached_func = _derivative_function_cache[func]
        except KeyError:
            cached_func = cache(func)
            _derivative_function_cache[func] = cached_func
    else:
        cached_func = func

    # We need an array with a shape compatible with input arrays in `args`
    result = np.zeros(np.broadcast(*args.values()).shape)

    # Now we can actually compute the derivative
    for stencil_offsets, weight in stencil_.items():
        offsets = dict(zip(dims, stencil_offsets))
        # Convert integer offsets to absolute positions
        coords = {
            dim: start + offsets.get(dim, 0) * spacings.get(dim, 0.0)
            for dim, start in args.items()
        }
        result += weight * cached_func(**coords)
    return result / np.prod(dim_spacings)
