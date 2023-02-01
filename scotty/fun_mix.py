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
from scotty.fun_FFD import find_dH_dR, find_dH_dZ  # \nabla H


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
    dH_dR_plus = find_dH_dR(
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
    dH_dR_minus = find_dH_dR(
        q_R,
        q_Z,
        K_R - delta_K_R,
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
    d2H_dKR_dR = (dH_dR_plus - dH_dR_minus) / (2 * delta_K_R)

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
    dH_dZ_plus = find_dH_dZ(
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
    dH_dZ_minus = find_dH_dZ(
        q_R,
        q_Z,
        K_R - delta_K_R,
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
    d2H_dKR_dZ = (dH_dZ_plus - dH_dZ_minus) / (2 * delta_K_R)

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
    dH_dR_plus = find_dH_dR(
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
    dH_dR_minus = find_dH_dR(
        q_R,
        q_Z,
        K_R,
        K_zeta - delta_K_zeta,
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
    d2H_dKzeta_dR = (dH_dR_plus - dH_dR_minus) / (2 * delta_K_zeta)

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
    dH_dZ_plus = find_dH_dZ(
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
    dH_dZ_minus = find_dH_dZ(
        q_R,
        q_Z,
        K_R,
        K_zeta - delta_K_zeta,
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
    d2H_dKzeta_dZ = (dH_dZ_plus - dH_dZ_minus) / (2 * delta_K_zeta)

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
    dH_dR_plus = find_dH_dR(
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
    dH_dR_minus = find_dH_dR(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z - delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_R,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKZ_dR = (dH_dR_plus - dH_dR_minus) / (2 * delta_K_Z)

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
    dH_dZ_plus = find_dH_dZ(
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
    dH_dZ_minus = find_dH_dZ(
        q_R,
        q_Z,
        K_R,
        K_zeta,
        K_Z - delta_K_Z,
        launch_angular_frequency,
        mode_flag,
        delta_Z,
        interp_poloidal_flux,
        find_density_1D,
        find_B_R,
        find_B_T,
        find_B_Z,
    )
    d2H_dKZ_dZ = (dH_dZ_plus - dH_dZ_minus) / (2 * delta_K_Z)

    return d2H_dKZ_dZ
