from dataclasses import dataclass, asdict
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
    "d2d2_FFD_FFD2": {
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
    "d2d2_FFD_CFD2": {
        (0, -1): 0.25,
        (0, 0): 1,
        (0, 1): -1.25,
        (1, 0): -2,
        (1, 1): 2,
        (2, -1): -0.25,
        (2, 0): 1,
        (2, 1): -0.75,
    },
    "d2d2_CFD_CFD2": {
        (1, 1): 0.25,
        (1, -1): -0.25,
        (-1, 1): -0.25,
        (-1, -1): 0.25,
    },
}


@dataclass(frozen=True)
class CoordOffset:
    """Helper class to store relative offsets in a dict"""

    q_R: int = 0
    q_Z: int = 0
    K_R: int = 0
    K_zeta: int = 0
    K_Z: int = 0


def derivative(
    func: Callable,
    dims: Union[str, Tuple[str, ...]],
    location: Dict[str, ArrayLike],
    spacings: Dict[str, float],
    stencil: str,
    cache: Optional[Dict[CoordOffset, ArrayLike]] = None,
) -> FloatArray:
    """Take the derivative of a function in arbitrary dimensions.

    The arguments to ``func`` must be exactly the keys of ``location``. The
    function will be evaluated at positions around ``location`` given by the
    integer offsets in the stencil multiplied by the step size in each dimension
    given in ``spacings``.

    An optional cache can be given in order to reuse function evaluations
    between derivatives of the function. The keys of the cache mapping will be
    `CoordOffset` instances, representing the integer offsets of the
    stencil. Practically, this means that the cache should only be used for
    derivatives at the same location, and should be cleared to evaluate
    derivatives at other locations.

    Parameters
    ----------
    func : Callable
        Function to take the derivative of
    dims : Union[str, Tuple[str, ...]]
        The name(s) of the dimension(s) to take the derivative along. ``dims``
        must appear in the keys of ``location``
    location : Dict[str, ArrayLike]
        Mapping of dimension names to evaluation point values. The evaluation
        points may be scalars or arrays. If arrays, they should all be the same
        size and ``func`` must be capable of vectorised arguments.
    spacings : Dict[str, float]
        Mapping of dimension names to step size. The keys must be identical to
        those of ``location``
    stencil : Optional[str]
        Stencil name (see `STENCILS` for supported stencils)
    cache : Optional[Dict[CoordOffset, ArrayLike]]
        Mapping of integer offsets to previous evaluations of ``func``

    Returns
    -------
    ArrayLike
        Derivative of ``func`` evaluated at ``location``

    Examples
    --------
    >>> derivative(lambda q_R: np.sin(q_R), "q_R", {"q_R": 0.0}, spacings={"q_R": 1e-8})
    1.0

    >>>

    """

    # Ensure that dims is a tuple
    if isinstance(dims, str):
        dims = (dims,)

    # Collect the relative spacings for the derivative dimensions
    dim_spacings = [spacings[dim] for dim in dims]
    # For second order derivatives, remove repeated dimensions
    if len(dims) == 2 and dims[1] == dims[0]:
        dims = (dims[0],)

    # We need an array the same shape as the input arrays in `location`
    result = np.zeros(np.broadcast(*location.values()).shape)

    if cache is None:
        cache = {}

    try:
        stencil_ = STENCILS[stencil]
    except KeyError:
        raise ValueError(
            f"Unknown stencil name '{stencil}', expected one of {list(STENCILS.keys())}"
        )

    for stencil_offsets, weight in stencil_.items():
        coord_offsets = CoordOffset(**dict(zip(dims, stencil_offsets)))
        try:
            # See if we've already evaluated func here...
            func_at_offset = cache[coord_offsets]
        except KeyError:
            # ...if not, let's do so now
            offsets = asdict(coord_offsets)
            # Convert integer offsets to absolute positions
            coords = {
                dim: start + offsets[dim] * spacings.get(dim, 0.0)
                for dim, start in location.items()
            }
            # Evaluate the derivative at this offset and cache it
            func_at_offset = func(**coords)
            cache[coord_offsets] = func_at_offset

        result += weight * func_at_offset
    return result / np.prod(dim_spacings)
