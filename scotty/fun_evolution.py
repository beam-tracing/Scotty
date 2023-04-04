# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Functions for Scotty to evolve the beam or ray.
I've separated this from scotty.fun_general to prevent circular importing

"""

from typing import Tuple

import numpy as np

from scotty.hamiltonian import Hamiltonian, hessians
from scotty.typing import FloatArray, ArrayLike


def pack_beam_parameters(
    q_R: ArrayLike,
    q_zeta: ArrayLike,
    q_Z: ArrayLike,
    K_R: ArrayLike,
    K_Z: ArrayLike,
    Psi: FloatArray,
) -> FloatArray:
    """Pack coordinates and Psi matrix into single flat array for"""

    # This used to be complex, with a length of 11, but the solver
    # throws a warning saying that something is casted to real It
    # seems to be fine, bu
    beam_parameters = np.zeros(17)

    beam_parameters[0] = q_R
    beam_parameters[1] = q_zeta
    beam_parameters[2] = q_Z

    beam_parameters[3] = K_R
    beam_parameters[4] = K_Z

    beam_parameters[5] = np.real(Psi[0, 0])
    beam_parameters[6] = np.real(Psi[1, 1])
    beam_parameters[7] = np.real(Psi[2, 2])
    beam_parameters[8] = np.real(Psi[0, 1])
    beam_parameters[9] = np.real(Psi[0, 2])
    beam_parameters[10] = np.real(Psi[1, 2])

    beam_parameters[11] = np.imag(Psi[0, 0])
    beam_parameters[12] = np.imag(Psi[1, 1])
    beam_parameters[13] = np.imag(Psi[2, 2])
    beam_parameters[14] = np.imag(Psi[0, 1])
    beam_parameters[15] = np.imag(Psi[0, 2])
    beam_parameters[16] = np.imag(Psi[1, 2])
    return beam_parameters


def unpack_beam_parameters(
    beam_parameters: FloatArray,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, FloatArray]:
    """Unpack the flat solver state vector into separate coordinate
    variables and Psi matrix

    """
    q_R = beam_parameters[0, ...]
    q_zeta = beam_parameters[1, ...]
    q_Z = beam_parameters[2, ...]
    K_R = beam_parameters[3, ...]
    K_Z = beam_parameters[4, ...]

    n_points = beam_parameters.shape[1] if beam_parameters.ndim == 2 else 1

    Psi_3D = np.zeros([n_points, 3, 3], dtype="complex128")
    # Psi_RR
    Psi_3D[:, 0, 0] = beam_parameters[5, ...] + 1j * beam_parameters[11, ...]
    # Psi_zetazeta
    Psi_3D[:, 1, 1] = beam_parameters[6, ...] + 1j * beam_parameters[12, ...]
    # Psi_ZZ
    Psi_3D[:, 2, 2] = beam_parameters[7, ...] + 1j * beam_parameters[13, ...]
    # Psi_Rzeta
    Psi_3D[:, 0, 1] = beam_parameters[8, ...] + 1j * beam_parameters[14, ...]
    # Psi_RZ
    Psi_3D[:, 0, 2] = beam_parameters[9, ...] + 1j * beam_parameters[15, ...]
    # Psi_zetaZ
    Psi_3D[:, 1, 2] = beam_parameters[10, ...] + 1j * beam_parameters[16, ...]
    # Psi_3D is symmetric
    Psi_3D[:, 1, 0] = Psi_3D[:, 0, 1]
    Psi_3D[:, 2, 0] = Psi_3D[:, 0, 2]
    Psi_3D[:, 2, 1] = Psi_3D[:, 1, 2]

    return q_R, q_zeta, q_Z, K_R, K_Z, np.squeeze(Psi_3D)


def beam_evolution_fun(tau, beam_parameters, K_zeta, hamiltonian: Hamiltonian):
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

    q_R, q_zeta, q_Z, K_R, K_Z, Psi_3D = unpack_beam_parameters(beam_parameters)

    # Find derivatives of H
    dH = hamiltonian.derivatives(q_R, q_Z, K_R, K_zeta, K_Z, second_order=True)

    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH)
    grad_gradK_H = np.transpose(gradK_grad_H)

    dH_dR = dH["dH_dR"]
    dH_dZ = dH["dH_dZ"]
    dH_dKR = dH["dH_dKR"]
    dH_dKzeta = dH["dH_dKzeta"]
    dH_dKZ = dH["dH_dKZ"]

    d_Psi_d_tau = (
        -grad_grad_H
        - np.matmul(Psi_3D, gradK_grad_H)
        - np.matmul(grad_gradK_H, Psi_3D)
        - np.matmul(np.matmul(Psi_3D, gradK_gradK_H), Psi_3D)
    )

    return pack_beam_parameters(dH_dKR, dH_dKzeta, dH_dKZ, -dH_dR, -dH_dZ, d_Psi_d_tau)
