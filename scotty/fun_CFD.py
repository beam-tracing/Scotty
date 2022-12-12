# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

Functions for finding derivatives of H using central finite difference.

@author: chenv
Valerian Hongjie Hall-Chen
valerian_hall-chen@ihpc.a-star.edu.sg

Run in Python 3,  does not work in Python 2
"""

# import numpy as np
# from scipy import constants as constants
from scotty.fun_general import find_H

from copy import deepcopy
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


def find_dH_dR(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_R,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):
    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="q_R",
        dx=delta_R,
    )


def find_dH_dZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="q_Z",
        dx=delta_Z,
    )


def find_dH_dKR(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_R,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="K_R",
        dx=delta_K_R,
    )


def find_dH_dKzeta(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_zeta,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):
    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="K_zeta",
        dx=delta_K_zeta,
    )


def find_dH_dKZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):
    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="K_Z",
        dx=delta_K_Z,
    )


def find_d2H_dR2(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_R,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="q_R",
        dx=delta_R,
        d_order=2,
    )


def find_d2H_dZ2(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="q_Z",
        dx=delta_Z,
        d_order=2,
    )


def find_d2H_dR_dZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_R,
    delta_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("q_R", "q_Z"),
        dx=(delta_R, delta_Z),
        d_order=2,
    )


def find_d2H_dKR2(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_R,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="K_R",
        dx=delta_K_R,
        d_order=2,
    )


def find_d2H_dKR_dKzeta(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_R,
    delta_K_zeta,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("K_R", "K_zeta"),
        dx=(delta_K_R, delta_K_zeta),
        d_order=2,
    )


def find_d2H_dKR_dKZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_R,
    delta_K_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("K_R", "K_Z"),
        dx=(delta_K_R, delta_K_Z),
        d_order=2,
    )


def find_d2H_dKzeta2(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_zeta,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="K_zeta",
        dx=delta_K_zeta,
        d_order=2,
    )


def find_d2H_dKzeta_dKZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_zeta,
    delta_K_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("K_zeta", "K_Z"),
        dx=(delta_K_zeta, delta_K_Z),
        d_order=2,
    )


def find_d2H_dKZ2(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions="K_Z",
        dx=delta_K_Z,
        d_order=2,
    )


def find_d2H_dKR_dR(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_R,
    delta_R,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("K_R", "q_R"),
        dx=(delta_K_R, delta_R),
        d_order=2,
    )


def find_d2H_dKR_dZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_R,
    delta_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("K_R", "q_Z"),
        dx=(delta_K_R, delta_Z),
        d_order=2,
    )


def find_d2H_dKzeta_dR(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_zeta,
    delta_R,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("q_R", "K_zeta"),
        dx=(delta_R, delta_K_zeta),
        d_order=2,
    )


def find_d2H_dKzeta_dZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_zeta,
    delta_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("q_Z", "K_zeta"),
        dx=(delta_Z, delta_K_zeta),
        d_order=2,
    )


def find_d2H_dKZ_dR(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_Z,
    delta_R,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("K_Z", "q_R"),
        dx=(delta_K_Z, delta_R),
        d_order=2,
    )


def find_d2H_dKZ_dZ(
    q_R,
    q_Z,
    K_R,
    K_zeta,
    K_Z,
    launch_angular_frequency,
    mode_flag,
    delta_K_Z,
    delta_Z,
    interp_poloidal_flux,
    find_density_1D,
    find_B_R,
    find_B_T,
    find_B_Z,
):

    return cfd_gradient(
        find_H,
        q_R=q_R,
        q_Z=q_Z,
        K_R=K_R,
        K_zeta=K_zeta,
        K_Z=K_Z,
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=mode_flag,
        interp_poloidal_flux=interp_poloidal_flux,
        find_density_1D=find_density_1D,
        find_B_R=find_B_R,
        find_B_T=find_B_T,
        find_B_Z=find_B_Z,
        directions=("K_Z", "q_Z"),
        dx=(delta_K_Z, delta_Z),
        d_order=2,
    )


def find_dpolflux_dR(q_R, q_Z, delta_R, interp_poloidal_flux):
    return cfd_gradient(
        interp_poloidal_flux, q_R=q_R, q_Z=q_Z, directions="q_R", dx=delta_R
    )


def find_dpolflux_dZ(q_R, q_Z, delta_Z, interp_poloidal_flux):
    return cfd_gradient(
        interp_poloidal_flux, q_R=q_R, q_Z=q_Z, directions="q_Z", dx=delta_Z
    )
