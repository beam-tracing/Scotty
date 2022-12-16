# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

Functions for Scotty to evolve the beam or ray.
I've separated this from scotty.fun_general to prevent circular importing

@author: chenv
Valerian Hongjie Hall-Chen
valerian_hall-chen@ihpc.a-star.edu.sg

Run in Python 3,  does not work in Python 2
"""

import numpy as np

from scotty.hamiltonian import Hamiltonian, laplacians


# Functions (solver)
# Defines the ray evolution function
# Not necessary for what I want to do, but it does make life easier
def ray_evolution_2D_fun(tau, ray_parameters_2D, K_zeta, hamiltonian: Hamiltonian):
    """
    Parameters
    ----------
    tau : float
        Parameter along the ray.
    ray_parameters_2D : complex128
        q_R, q_Z, K_R, K_Z
    hamiltonian:
        Hamiltonian object

    Returns
    -------
    d_beam_parameters_d_tau
        d (beam_parameters) / d tau

    Notes
    -------

    """

    # Clean input up. Not necessary, but aids readability
    q_R = ray_parameters_2D[0]
    q_Z = ray_parameters_2D[1]
    K_R = ray_parameters_2D[2]
    K_Z = ray_parameters_2D[3]

    # Find derivatives of H
    dH = hamiltonian.derivatives(q_R, q_Z, K_R, K_zeta, K_Z, order=1)

    d_ray_parameters_2D_d_tau = np.zeros_like(ray_parameters_2D)

    # d (q_R) / d tau
    d_ray_parameters_2D_d_tau[0] = dH["dH_dKR"]
    # d (q_Z) / d tau
    d_ray_parameters_2D_d_tau[1] = dH["dH_dKZ"]
    # d (K_R) / d tau
    d_ray_parameters_2D_d_tau[2] = -dH["dH_dR"]
    # d (K_Z) / d tau
    d_ray_parameters_2D_d_tau[3] = -dH["dH_dZ"]

    return d_ray_parameters_2D_d_tau


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

    # Clean input up. Not necessary, but aids readability
    q_R = beam_parameters[0]
    # q_zeta = beam_parameters[1]
    q_Z = beam_parameters[2]
    K_R = beam_parameters[3]
    K_Z = beam_parameters[4]

    Psi_3D = np.zeros([3, 3], dtype="complex128")
    Psi_3D[0, 0] = beam_parameters[5] + 1j * beam_parameters[11]  # Psi_RR
    Psi_3D[1, 1] = beam_parameters[6] + 1j * beam_parameters[12]  # Psi_zetazeta
    Psi_3D[2, 2] = beam_parameters[7] + 1j * beam_parameters[13]  # Psi_ZZ
    Psi_3D[0, 1] = beam_parameters[8] + 1j * beam_parameters[14]  # Psi_Rzeta
    Psi_3D[0, 2] = beam_parameters[9] + 1j * beam_parameters[15]  # Psi_RZ
    Psi_3D[1, 2] = beam_parameters[10] + 1j * beam_parameters[16]  # Psi_zetaZ
    Psi_3D[1, 0] = Psi_3D[0, 1]  # Psi_3D is symmetric
    Psi_3D[2, 0] = Psi_3D[0, 2]
    Psi_3D[2, 1] = Psi_3D[1, 2]

    # Find derivatives of H
    dH = hamiltonian.derivatives(q_R, q_Z, K_R, K_zeta, K_Z, order=2)

    grad_grad_H, gradK_grad_H, gradK_gradK_H = laplacians(dH)
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

    d_beam_parameters_d_tau = np.zeros_like(beam_parameters)

    d_beam_parameters_d_tau[0] = dH_dKR  # d (q_R) / d tau
    d_beam_parameters_d_tau[1] = dH_dKzeta  # d (q_zeta) / d tau

    d_beam_parameters_d_tau[2] = dH_dKZ  # d (q_Z) / d tau
    d_beam_parameters_d_tau[3] = -dH_dR  # d (K_R) / d tau
    d_beam_parameters_d_tau[4] = -dH_dZ  # d (K_Z) / d tau

    d_beam_parameters_d_tau[5] = np.real(d_Psi_d_tau[0, 0])  # d (Psi_RR) / d tau
    d_beam_parameters_d_tau[6] = np.real(d_Psi_d_tau[1, 1])  # d (Psi_zetazeta) / d tau
    d_beam_parameters_d_tau[7] = np.real(d_Psi_d_tau[2, 2])  # d (Psi_ZZ) / d tau
    d_beam_parameters_d_tau[8] = np.real(d_Psi_d_tau[0, 1])  # d (Psi_Rzeta) / d tau
    d_beam_parameters_d_tau[9] = np.real(d_Psi_d_tau[0, 2])  # d (Psi_RZ) / d tau
    d_beam_parameters_d_tau[10] = np.real(d_Psi_d_tau[1, 2])  # d (Psi_zetaZ) / d tau

    d_beam_parameters_d_tau[11] = np.imag(d_Psi_d_tau[0, 0])  # d (Psi_RR) / d tau
    d_beam_parameters_d_tau[12] = np.imag(d_Psi_d_tau[1, 1])  # d (Psi_zetazeta) / d tau
    d_beam_parameters_d_tau[13] = np.imag(d_Psi_d_tau[2, 2])  # d (Psi_ZZ) / d tau
    d_beam_parameters_d_tau[14] = np.imag(d_Psi_d_tau[0, 1])  # d (Psi_Rzeta) / d tau
    d_beam_parameters_d_tau[15] = np.imag(d_Psi_d_tau[0, 2])  # d (Psi_RZ) / d tau
    d_beam_parameters_d_tau[16] = np.imag(d_Psi_d_tau[1, 2])  # d (Psi_zetaZ) / d tau

    return d_beam_parameters_d_tau
