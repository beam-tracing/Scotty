# -*- coding: utf-8 -*-
# Copyright 2017 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Functions for finding derivatives of H using central finite difference.
"""

import itertools
from typing import Callable, Dict, Union, Tuple

import numpy as np


WEIGHTS_AND_STEPS: Dict[int, Dict[str, list]] = {
    1: {"weights": [-0.5, 0.5], "steps": [-1, 1]},
    2: {"weights": [1, -2, 1], "steps": [-1, 0, 1]},
}


def cfd_gradient(
    f: Callable,
    directions: Union[str, Tuple[str]],
    dx: Union[float, Tuple[float]] = 1e-4,
    d_order=1,
    **kwargs,
):
    """Compute the derivative of a function using central finite differences

    Arguments
    ---------
    f:
        Function to compute derivative of
    directions:
        Tuple of directions to take derivative in. These must match
        the name of the corresponding arguments to ``f``
    dx:
        Tuple of step sizes for each direction
    d_order:
        Order of the derivative (NOT the error scaling order of the
        stencil)
    **kwargs:
        Arguments passed to ``f``

    """

    if isinstance(directions, str):
        directions = (directions,)
    if isinstance(dx, float):
        dx = (dx,)

    num_directions = len(directions)

    if num_directions != len(dx):
        raise ValueError(
            f"Length of 'dx' ({len(dx)}) must match length of directions ({num_directions})"
        )
    if d_order < num_directions:
        raise ValueError(
            f"Order of derivative (d_order={d_order}) must be greater or equal to the number "
            f"of directions ({num_directions})"
        )

    d_order = d_order - (num_directions - 1)

    try:
        weights = WEIGHTS_AND_STEPS[d_order]["weights"]
        steps = WEIGHTS_AND_STEPS[d_order]["steps"]
    except KeyError:
        raise ValueError(f"No weights for derivative order {d_order}")

    all_weights = itertools.product(*(weights,) * num_directions)
    all_steps = itertools.product(*(steps,) * num_directions)

    original_directions = {direction: kwargs[direction] for direction in directions}

    result = 0.0
    for weight, step in zip(all_weights, all_steps):
        for direction, dx_, step_ in zip(directions, dx, step):
            kwargs[direction] = original_directions[direction] + step_ * dx_
        result += np.prod(weight) * f(**kwargs)

    return result / np.prod(dx * d_order)


def find_dpolflux_dR(q_R, q_Z, delta_R, interp_poloidal_flux):
    return cfd_gradient(
        interp_poloidal_flux, q_R=q_R, q_Z=q_Z, directions="q_R", dx=delta_R
    )


def find_dpolflux_dZ(q_R, q_Z, delta_Z, interp_poloidal_flux):
    return cfd_gradient(
        interp_poloidal_flux, q_R=q_R, q_Z=q_Z, directions="q_Z", dx=delta_Z
    )
