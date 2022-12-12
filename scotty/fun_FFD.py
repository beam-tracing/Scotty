# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

Functions for finding derivatives of H using forward finite difference.

@author: chenv
Valerian Hongjie Hall-Chen
valerian_hall-chen@ihpc.a-star.edu.sg

Run in Python 3,  does not work in Python 2
"""

# import numpy as np
# from scipy import constants as constants
from scotty.fun_general import find_H


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R + delta_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R + 2 * delta_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dR = ((-3 / 2) * H_0 + (2) * H_1 + (-1 / 2) * H_2) / (delta_R)

    return dH_dR


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z + delta_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z + 2 * delta_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dZ = ((-3 / 2) * H_0 + (2) * H_1 + (-1 / 2) * H_2) / (delta_Z)

    return dH_dZ


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z,
        K_R + delta_K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z,
        K_R + 2 * delta_K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dKR = ((-3 / 2) * H_0 + (2) * H_1 + (-1 / 2) * H_2) / (delta_K_R)

    return dH_dKR


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta + delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta + 2 * delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dKzeta = ((-3 / 2) * H_0 + (2) * H_1 + (-1 / 2) * H_2) / (delta_K_zeta)

    return dH_dKzeta


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + 2 * delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dKZ = ((-3 / 2) * H_0 + (2) * H_1 + (-1 / 2) * H_2) / (delta_K_Z)

    return dH_dKZ


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R + delta_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R + 2 * delta_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_3 = find_H(
        q_R + 3 * delta_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    #    H_4 = find_H(q_R+4*delta_R,
    #                 q_Z, K_R, K_zeta, K_Z, launch_angular_frequency, mode_flag,
    #                 interp_poloidal_flux, find_density_1D, find_B_R, find_B_T, find_B_Z)
    d2H_dR2 = ((2) * H_0 + (-5) * H_1 + (4) * H_2 + (-1) * H_3) / (delta_R**2)
    #    d2H_dR2 = ( (1)*H_0 + (-2)*H_1 + (1)*H_2 ) / (delta_R**2)

    return d2H_dR2


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z + delta_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z + 2 * delta_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_3 = find_H(
        q_R,
        q_Z + 3 * delta_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dZ2 = ((2) * H_0 + (-5) * H_1 + (4) * H_2 + (-1) * H_3) / (delta_Z**2)

    return d2H_dZ2


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

    dH_dZ_0 = find_dH_dZ(
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
    )
    dH_dZ_1 = find_dH_dZ(
        q_R + delta_R,
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
    )
    dH_dZ_2 = find_dH_dZ(
        q_R + 2 * delta_R,
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
    )
    d2H_dR_dZ = ((-3 / 2) * dH_dZ_0 + (2) * dH_dZ_1 + (-1 / 2) * dH_dZ_2) / (delta_R)

    return d2H_dR_dZ


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z,
        K_R + delta_K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z,
        K_R + 2 * delta_K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_3 = find_H(
        q_R,
        q_Z,
        K_R + 3 * delta_K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKR2 = ((2) * H_0 + (-5) * H_1 + (4) * H_2 + (-1) * H_3) / (delta_K_R**2)

    return d2H_dKR2


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

    dH_dKzeta_0 = find_dH_dKzeta(
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
    )
    dH_dKzeta_1 = find_dH_dKzeta(
        q_R,
        q_Z,
        K_R + delta_K_R,
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
    )
    dH_dKzeta_2 = find_dH_dKzeta(
        q_R,
        q_Z,
        K_R + 2 * delta_K_R,
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
    )
    d2H_dKR_dKzeta = (
        (-3 / 2) * dH_dKzeta_0 + (2) * dH_dKzeta_1 + (-1 / 2) * dH_dKzeta_2
    ) / (delta_K_R)

    return d2H_dKR_dKzeta


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

    dH_dKZ_0 = find_dH_dKZ(
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
    )
    dH_dKZ_1 = find_dH_dKZ(
        q_R,
        q_Z,
        K_R + delta_K_R,
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
    )
    dH_dKZ_2 = find_dH_dKZ(
        q_R,
        q_Z,
        K_R + 2 * delta_K_R,
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
    )
    d2H_dKR_dKz = ((-3 / 2) * dH_dKZ_0 + (2) * dH_dKZ_1 + (-1 / 2) * dH_dKZ_2) / (
        delta_K_R
    )

    return d2H_dKR_dKz


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta + delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta + 2 * delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_3 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta + 3 * delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKzeta2 = ((2) * H_0 + (-5) * H_1 + (4) * H_2 + (-1) * H_3) / (
        delta_K_zeta**2
    )

    return d2H_dKzeta2


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

    dH_dKZ_0 = find_dH_dKZ(
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
    )
    dH_dKZ_1 = find_dH_dKZ(
        q_R,
        q_Z,
        K_R,
        K_zeta + delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_K_Z,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dKZ_2 = find_dH_dKZ(
        q_R,
        q_Z,
        K_R,
        K_zeta + 2 * delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_K_Z,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKzeta_dKZ = ((-3 / 2) * dH_dKZ_0 + (2) * dH_dKZ_1 + (-1 / 2) * dH_dKZ_2) / (
        delta_K_zeta
    )

    return d2H_dKzeta_dKZ


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

    H_0 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_1 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_2 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + 2 * delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    H_3 = find_H(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + 3 * delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKZ2 = ((2) * H_0 + (-5) * H_1 + (4) * H_2 + (-1) * H_3) / (delta_K_Z**2)

    return d2H_dKZ2


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

    dH_dR_0 = find_dH_dR(
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
    )
    dH_dR_1 = find_dH_dR(
        q_R,
        q_Z,
        K_R + delta_K_R,
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
    )
    dH_dR_2 = find_dH_dR(
        q_R,
        q_Z,
        K_R + 2 * delta_K_R,
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
    )
    d2H_dKR_dR = ((-3 / 2) * dH_dR_0 + (2) * dH_dR_1 + (-1 / 2) * dH_dR_2) / (delta_K_R)

    return d2H_dKR_dR


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

    dH_dZ_0 = find_dH_dZ(
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
    )
    dH_dZ_1 = find_dH_dZ(
        q_R,
        q_Z,
        K_R + delta_K_R,
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
    )
    dH_dZ_2 = find_dH_dZ(
        q_R,
        q_Z,
        K_R + 2 * delta_K_R,
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
    )
    d2H_dKR_dZ = ((-3 / 2) * dH_dZ_0 + (2) * dH_dZ_1 + (-1 / 2) * dH_dZ_2) / (delta_K_R)

    return d2H_dKR_dZ


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

    dH_dR_0 = find_dH_dR(
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
    )
    dH_dR_1 = find_dH_dR(
        q_R,
        q_Z,
        K_R,
        K_zeta + delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_R,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dR_2 = find_dH_dR(
        q_R,
        q_Z,
        K_R,
        K_zeta + 2 * delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_R,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKzeta_dR = ((-3 / 2) * dH_dR_0 + (2) * dH_dR_1 + (-1 / 2) * dH_dR_2) / (
        delta_K_zeta
    )

    return d2H_dKzeta_dR


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

    dH_dZ_0 = find_dH_dZ(
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
    )
    dH_dZ_1 = find_dH_dZ(
        q_R,
        q_Z,
        K_R,
        K_zeta + delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_Z,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dZ_2 = find_dH_dZ(
        q_R,
        q_Z,
        K_R,
        K_zeta + 2 * delta_K_zeta,
        K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_Z,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKzeta_dZ = ((-3 / 2) * dH_dZ_0 + (2) * dH_dZ_1 + (-1 / 2) * dH_dZ_2) / (
        delta_K_zeta
    )

    return d2H_dKzeta_dZ


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

    dH_dR_0 = find_dH_dR(
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
    )
    dH_dR_1 = find_dH_dR(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_R,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dR_2 = find_dH_dR(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + 2 * delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_R,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKZ_dR = ((-3 / 2) * dH_dR_0 + (2) * dH_dR_1 + (-1 / 2) * dH_dR_2) / (delta_K_Z)

    return d2H_dKZ_dR


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

    dH_dZ_0 = find_dH_dZ(
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
    )
    dH_dZ_1 = find_dH_dZ(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_Z,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    dH_dZ_2 = find_dH_dZ(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z + 2 * delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_Z,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKZ_dZ = ((-3 / 2) * dH_dZ_0 + (2) * dH_dZ_1 + (-1 / 2) * dH_dZ_2) / (delta_K_Z)

    return d2H_dKZ_dZ


def find_dpolflux_dR(q_R, q_Z, delta_R, interp_poloidal_flux):

    polflux_0 = interp_poloidal_flux(q_R, q_Z, grid=False)
    polflux_1 = interp_poloidal_flux(q_R + delta_R, q_Z, grid=False)
    polflux_2 = interp_poloidal_flux(q_R + 2 * delta_R, q_Z, grid=False)
    dpolflux_dR = ((-3 / 2) * polflux_0 + (2) * polflux_1 + (-1 / 2) * polflux_2) / (
        delta_R
    )

    return dpolflux_dR


def find_dpolflux_dZ(q_R, q_Z, delta_Z, interp_poloidal_flux):

    polflux_0 = interp_poloidal_flux(q_R, q_Z, grid=False)
    polflux_1 = interp_poloidal_flux(q_R, q_Z + delta_Z, grid=False)
    polflux_2 = interp_poloidal_flux(q_R, q_Z + 2 * delta_Z, grid=False)
    dpolflux_dZ = ((-3 / 2) * polflux_0 + (2) * polflux_1 + (-1 / 2) * polflux_2) / (
        delta_Z
    )

    return dpolflux_dZ
