# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Functions for Scotty to evolve the beam or ray.
I've separated this from scotty.fun_general to prevent circular importing

"""

from typing import Tuple

import numpy as np

from scotty.cart_hamiltonian import cart_Hamiltonian, hessians
from scotty.typing import FloatArray, ArrayLike


def cart_pack_beam_parameters(
    q_X: ArrayLike,
    q_Y: ArrayLike,
    q_Z: ArrayLike,
    K_X: ArrayLike,
    K_Y: ArrayLike,
    K_Z: ArrayLike,
    Psi: FloatArray,
) -> FloatArray:
    """Pack coordinates and Psi matrix into single flat array for"""

    # This used to be complex, with a length of 11, but the solver
    # throws a warning saying that something is casted to real It
    # seems to be fine, bu
    beam_parameters = np.zeros(18)

    beam_parameters[0] = q_X
    beam_parameters[1] = q_Y
    beam_parameters[2] = q_Z

    beam_parameters[3] = K_X
    beam_parameters[4] = K_Y
    beam_parameters[5] = K_Z

    beam_parameters[6] = np.real(Psi[0, 0])
    beam_parameters[7] = np.real(Psi[1, 1])
    beam_parameters[8] = np.real(Psi[2, 2])
    beam_parameters[9] = np.real(Psi[0, 1])
    beam_parameters[10] = np.real(Psi[0, 2])
    beam_parameters[11] = np.real(Psi[1, 2])

    beam_parameters[12] = np.imag(Psi[0, 0])
    beam_parameters[13] = np.imag(Psi[1, 1])
    beam_parameters[14] = np.imag(Psi[2, 2])
    beam_parameters[15] = np.imag(Psi[0, 1])
    beam_parameters[16] = np.imag(Psi[0, 2])
    beam_parameters[17] = np.imag(Psi[1, 2])
    return beam_parameters


def cart_unpack_beam_parameters(
    beam_parameters: FloatArray,
) -> Tuple[
    ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, FloatArray
]:
    """Unpack the flat solver state vector into separate coordinate
    variables and Psi matrix

    """
    q_X = beam_parameters[0, ...]
    q_Y = beam_parameters[1, ...]
    q_Z = beam_parameters[2, ...]
    K_X = beam_parameters[3, ...]
    K_Y = beam_parameters[4, ...]
    K_Z = beam_parameters[5, ...]

    n_points = beam_parameters.shape[1] if beam_parameters.ndim == 2 else 1

    Psi_3D = np.zeros([n_points, 3, 3], dtype="complex128")
    # Psi_RR
    Psi_3D[:, 0, 0] = beam_parameters[6, ...] + 1j * beam_parameters[12, ...]
    # Psi_zetazeta
    Psi_3D[:, 1, 1] = beam_parameters[7, ...] + 1j * beam_parameters[13, ...]
    # Psi_ZZ
    Psi_3D[:, 2, 2] = beam_parameters[8, ...] + 1j * beam_parameters[14, ...]
    # Psi_Rzeta
    Psi_3D[:, 0, 1] = beam_parameters[9, ...] + 1j * beam_parameters[15, ...]
    # Psi_RZ
    Psi_3D[:, 0, 2] = beam_parameters[10, ...] + 1j * beam_parameters[16, ...]
    # Psi_zetaZ
    Psi_3D[:, 1, 2] = beam_parameters[11, ...] + 1j * beam_parameters[17, ...]
    # Psi_3D is symmetric
    Psi_3D[:, 1, 0] = Psi_3D[:, 0, 1]
    Psi_3D[:, 2, 0] = Psi_3D[:, 0, 2]
    Psi_3D[:, 2, 1] = Psi_3D[:, 1, 2]

    return q_X, q_Y, q_Z, K_X, K_Y, K_Z, np.squeeze(Psi_3D)


def cart_beam_evolution_fun(tau, beam_parameters, hamiltonian: cart_Hamiltonian):
    """
    Parameters
    ----------
    tau : float
        Parameter along the ray.
    beam_parameters : complex128
        q_R, q_zeta, q_Z, K_R, K_Z, Psi_RR, Psi_zetazeta, Psi_ZZ, Psi_Rzeta, Psi_RZ, Psi_zetaZ.

    Returns
    -------
    d_beam_parameters_d_tau
        d (beam_parameters) / d tau
    """

    q_X, q_Y, q_Z, K_X, K_Y, K_Z, Psi_3D = cart_unpack_beam_parameters(beam_parameters)

    # Find derivatives of H
    dH = hamiltonian.derivatives(q_X, q_Y, q_Z, K_X, K_Y, K_Z, second_order=True)

    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH)
    grad_gradK_H = np.transpose(gradK_grad_H)

    dH_dX = dH["dH_dX"]
    dH_dY = dH["dH_dY"]
    dH_dZ = dH["dH_dZ"]
    dH_dKX = dH["dH_dKX"]
    dH_dKY = dH["dH_dKY"]
    dH_dKZ = dH["dH_dKZ"]

    d_Psi_d_tau = (
        -grad_grad_H
        - np.matmul(Psi_3D, gradK_grad_H)
        - np.matmul(grad_gradK_H, Psi_3D)
        - np.matmul(np.matmul(Psi_3D, gradK_gradK_H), Psi_3D)
    )

    return cart_pack_beam_parameters(
        dH_dKX, dH_dKY, dH_dKZ, -dH_dX, -dH_dY, -dH_dZ, d_Psi_d_tau
    )