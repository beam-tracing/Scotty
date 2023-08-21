# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Functions for Scotty (excluding functions for finding derivatives of H).
"""

import numpy as np
from scipy import constants as constants
from scipy import interpolate as interpolate
from scipy import integrate as integrate
from typing import TextIO, List, Any, Tuple
from scotty.typing import ArrayLike, FloatArray

from .typing import ArrayLike
from typing import Optional, Union


def read_floats_into_list_until(terminator: str, lines: TextIO) -> FloatArray:
    """Reads the lines of a file until the string (terminator) is read.

    Currently used to read topfile.

    Written by NE Bricknell
    """
    lst: List[float] = []
    while True:
        try:
            line = lines.readline()
        except StopIteration:
            break
        if terminator in line:
            break
        elif not line:
            break
        lst.extend(map(float, line.split()))
    return np.asarray(lst)


def find_nearest(array: ArrayLike, value: Any) -> int:
    """Returns the index of the first element in ``array`` closest in
    absolute value to ``value``

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


def contract_special(arg_a: FloatArray, arg_b: FloatArray) -> FloatArray:
    """Dot product of arrays of vectors or matrices.

    Covers the case that matmul and dot don't do very elegantly, and
    avoids having to use a for loop to iterate over the array slices.

    Parameters
    ----------
    arg_a, arg_b:
        One of:

        - matrix with shape TxMxN and a vector with TxN or TxM
        - two vectors of shape TxN

        For each T, independently compute the dot product of the
        matrices/vectors

    """
    if (
        np.ndim(arg_a) == 3 and np.ndim(arg_b) == 2
    ):  # arg_a is the matrix and arg_b is the vector
        matrix = arg_a
        vector = arg_b
        intermediate_result = np.tensordot(matrix, vector, ((2), (1)))
        result = np.diagonal(
            intermediate_result, offset=0, axis1=0, axis2=2
        ).transpose()
    elif (
        np.ndim(arg_a) == 2 and np.ndim(arg_b) == 3
    ):  # arg_a is the vector and arg_b is the matrix
        vector = arg_a
        matrix = arg_b
        intermediate_result = np.tensordot(matrix, vector, ((1), (1)))
        result = np.diagonal(
            intermediate_result, offset=0, axis1=0, axis2=2
        ).transpose()
    elif (
        np.ndim(arg_a) == 2 and np.ndim(arg_b) == 2
    ):  # arg_a is the vector and arg_b is a vector
        vector1 = arg_a
        vector2 = arg_b
        intermediate_result = np.tensordot(vector1, vector2, ((1), (1)))
        result = np.diagonal(
            intermediate_result, offset=0, axis1=0, axis2=1
        ).transpose()
    else:
        print("Error: Invalid dimensions")
    return result


def make_unit_vector_from_cross_product(vector_a, vector_b):
    """
    Assume np.shape(vector_a) = np.shape(vector_b) = (n,3)
    or
    np.shape(vector_a) = (n,3) , np.shape(vector_b) = (3)
    """
    output_vector = np.cross(vector_a, vector_b)
    output_vector_magnitude = np.linalg.norm(output_vector, axis=-1)
    output_unit_vector = output_vector / np.tile(output_vector_magnitude, (3, 1)).T

    return output_unit_vector


def find_inverse_2D(matrix_2D):
    # Finds the inverse of a 2x2 matrix
    matrix_2D_inverse = np.zeros([2, 2], dtype="complex128")
    determinant = matrix_2D[0, 0] * matrix_2D[1, 1] - matrix_2D[0, 1] * matrix_2D[1, 0]
    matrix_2D_inverse[0, 0] = matrix_2D[1, 1] / determinant
    matrix_2D_inverse[1, 1] = matrix_2D[0, 0] / determinant
    matrix_2D_inverse[0, 1] = -matrix_2D[0, 1] / determinant
    matrix_2D_inverse[1, 0] = -matrix_2D[1, 0] / determinant
    return matrix_2D_inverse


def find_x0(xs, ys, y0):
    """
    xs,ys are the x and y coordinates of a line on a plane
    Finds the value of x corresponding to a certain y0
    This implementation is silly but I am impatient and want to move on to other things quickly
    """

    index_guess = find_nearest(ys, y0)

    interp_y = interpolate.interp1d(
        xs,
        ys,
        kind="linear",
        axis=-1,
        copy=True,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=False,
    )

    if index_guess == 0:
        xs_fine = np.linspace(xs[0], xs[index_guess + 1], 101)
    elif index_guess == len(xs) - 1:
        xs_fine = np.linspace(xs[index_guess - 1], xs[-1], 101)
    else:
        xs_fine = np.linspace(xs[index_guess - 1], xs[index_guess + 1], 101)

    ys_fine = interp_y(xs_fine)

    index = find_nearest(ys_fine, y0)
    x0 = xs_fine[index]

    return x0


def find_area_points(xs, ys, fraction_wanted):
    """
    ys = f(xs)

    Finds the maximum in ys, and finds an area around that
    Used to find localisation

    Assume xs sorted in ascending order
    """

    if fraction_wanted > 0.5 or fraction_wanted < 0.0:
        print("Invalid value given for fraction")
    x_vals = np.zeros(2)
    y_vals = np.zeros(2)

    cumulative_ys = integrate.cumulative_trapezoid(ys, xs, initial=0)
    cumulative_ys_at_y_max = cumulative_ys[ys.argmax()]
    total_ys = integrate.simps(ys, xs)
    fraction_at_y_max = cumulative_ys_at_y_max / total_ys

    if (fraction_at_y_max - fraction_wanted) < 0:
        lower_fraction = 0.0
        upper_fraction = 2 * fraction_wanted

    elif (fraction_at_y_max + fraction_wanted) > 1.0:
        lower_fraction = 1 - 2 * fraction_wanted
        upper_fraction = 1.0

    else:
        lower_fraction = fraction_at_y_max - fraction_wanted
        upper_fraction = fraction_at_y_max + fraction_wanted

    interp_x = interpolate.interp1d(
        cumulative_ys / total_ys, xs, fill_value="extrapolate"
    )
    x_vals[:] = interp_x([lower_fraction, upper_fraction])

    interp_y = interpolate.interp1d(xs, ys, fill_value="extrapolate")
    y_vals[:] = interp_y(x_vals)

    return x_vals, y_vals


# ----------------------------------


# Functions (Coordinate transformations)
def freq_GHz_to_angular_frequency(freq_GHz: float) -> float:
    """Convert frequency in GHz to angular frequency"""
    return 2 * np.pi * 1e9 * freq_GHz


def angular_frequency_to_wavenumber(angular_frequency: float) -> float:
    """Convert angular frequency to wavenumber"""
    return angular_frequency / constants.c


def freq_GHz_to_wavenumber(freq_GHz: float) -> float:
    """Converts frequency in GHz to wavenumber"""
    return angular_frequency_to_wavenumber(freq_GHz_to_angular_frequency(freq_GHz))


def find_vec_lab_Cartesian(vec_lab, q_zeta):
    # vec_lab to have shape (n,3) or (3) # I've not tested the second case

    vec_R = vec_lab.T[0]
    vec_zeta = vec_lab.T[1]
    vec_Z = vec_lab.T[2]

    vec_lab_Cartesian = np.zeros_like(vec_lab)
    vec_lab_Cartesian.T[0] = vec_R * np.cos(q_zeta) - vec_zeta * np.sin(q_zeta)
    vec_lab_Cartesian.T[1] = vec_R * np.sin(q_zeta) + vec_zeta * np.cos(q_zeta)
    vec_lab_Cartesian.T[2] = vec_Z
    return vec_lab_Cartesian


def find_q_lab_Cartesian(q_lab):
    q_R = q_lab[0]
    q_zeta = q_lab[1]
    q_Z = q_lab[2]

    q_lab_Cartesian = np.zeros_like(q_lab)
    q_lab_Cartesian[0] = q_R * np.cos(q_zeta)
    q_lab_Cartesian[1] = q_R * np.sin(q_zeta)
    q_lab_Cartesian[2] = q_Z
    return q_lab_Cartesian


def find_q_lab(q_lab_Cartesian):
    q_X = q_lab_Cartesian[0]
    q_Y = q_lab_Cartesian[1]
    q_Z = q_lab_Cartesian[2]

    q_lab = np.zeros_like(q_lab_Cartesian)
    q_lab[0] = np.sqrt(q_X**2 + q_Y**2)
    q_lab[1] = np.arctan2(q_Y, q_X)
    q_lab[2] = q_Z
    return q_lab


def find_K_lab_Cartesian(K_lab, q_lab):
    K_R = K_lab[0]
    K_zeta = K_lab[1]
    K_Z = K_lab[2]
    q_R = q_lab[0]
    q_zeta = q_lab[1]

    K_lab_Cartesian = np.zeros(3)
    K_lab_Cartesian[0] = K_R * np.cos(q_zeta) - K_zeta * np.sin(q_zeta) / q_R  # K_X
    K_lab_Cartesian[1] = K_R * np.sin(q_zeta) + K_zeta * np.cos(q_zeta) / q_R  # K_Y
    K_lab_Cartesian[2] = K_Z
    return K_lab_Cartesian


def find_K_lab(K_lab_Cartesian, q_lab_Cartesian):
    K_X = K_lab_Cartesian[0]
    K_Y = K_lab_Cartesian[1]
    K_Z = K_lab_Cartesian[2]

    [q_R, q_zeta, q_Z] = find_q_lab(q_lab_Cartesian)

    K_lab = np.zeros(3)
    K_lab[0] = K_X * np.cos(q_zeta) + K_Y * np.sin(q_zeta)  # K_R
    K_lab[1] = (-K_X * np.sin(q_zeta) + K_Y * np.cos(q_zeta)) * q_R  # K_zeta
    K_lab[2] = K_Z
    return K_lab


def find_Psi_3D_lab(Psi_3D_lab_Cartesian, q_R, q_zeta, K_R, K_zeta):
    """
    Converts Psi_3D from Cartesian to cylindrical coordinates, both in the lab frame (not the beam frame)
    """
    cos_zeta = np.cos(q_zeta)
    sin_zeta = np.sin(q_zeta)

    Psi_XX = Psi_3D_lab_Cartesian[0][0]
    Psi_YY = Psi_3D_lab_Cartesian[1][1]
    Psi_ZZ = Psi_3D_lab_Cartesian[2][2]
    Psi_XY = Psi_3D_lab_Cartesian[0][1]
    Psi_XZ = Psi_3D_lab_Cartesian[0][2]
    Psi_YZ = Psi_3D_lab_Cartesian[1][2]

    Psi_3D_lab = np.zeros([3, 3], dtype="complex128")

    Psi_3D_lab[0][0] = (
        Psi_XX * cos_zeta**2
        + 2 * Psi_XY * sin_zeta * cos_zeta
        + Psi_YY * sin_zeta**2
    )  # Psi_RR
    Psi_3D_lab[1][1] = (
        Psi_XX * sin_zeta**2
        - 2 * Psi_XY * sin_zeta * cos_zeta
        + Psi_YY * cos_zeta**2
    ) * q_R**2 - K_R * q_R  # Psi_zetazeta
    Psi_3D_lab[2][2] = Psi_ZZ  # Psi_ZZ

    Psi_3D_lab[0][1] = (
        -Psi_XX * sin_zeta * cos_zeta
        + Psi_XY * (cos_zeta**2 - sin_zeta**2)
        + Psi_YY * sin_zeta * cos_zeta
    ) * q_R + K_zeta / q_R  # Psi_Rzeta
    Psi_3D_lab[1][0] = Psi_3D_lab[0][1]

    Psi_3D_lab[0][2] = Psi_XZ * cos_zeta + Psi_YZ * sin_zeta  # Psi_RZ
    Psi_3D_lab[2][0] = Psi_3D_lab[0][2]
    Psi_3D_lab[1][2] = (-Psi_XZ * sin_zeta + Psi_YZ * cos_zeta) * q_R  # Psi_zetaZ
    Psi_3D_lab[2][1] = Psi_3D_lab[1][2]
    return Psi_3D_lab


def find_Psi_3D_lab_Cartesian(Psi_3D_lab, q_R, q_zeta, K_R, K_zeta):
    """
    Converts Psi_3D from cylindrical to Cartesian coordinates, both in the lab frame (not the beam frame)
    The shape of Psi_3D_lab must be either [3,3] or [numberOfDataPoints,3,3]
    """
    if Psi_3D_lab.ndim == 2:  # A single matrix of Psi
        Psi_RR = Psi_3D_lab[0, 0]
        Psi_zetazeta = Psi_3D_lab[1, 1]
        Psi_ZZ = Psi_3D_lab[2, 2]
        Psi_Rzeta = Psi_3D_lab[0, 1]
        Psi_RZ = Psi_3D_lab[0, 2]
        Psi_zetaZ = Psi_3D_lab[1, 2]

        temp_matrix_for_Psi = np.zeros(np.shape(Psi_3D_lab), dtype="complex128")

        temp_matrix_for_Psi[0, 0] = Psi_RR
        temp_matrix_for_Psi[0, 1] = Psi_Rzeta / q_R - K_zeta / q_R**2
        temp_matrix_for_Psi[0, 2] = Psi_RZ
        temp_matrix_for_Psi[1, 1] = Psi_zetazeta / q_R**2 + K_R / q_R
        temp_matrix_for_Psi[1, 2] = Psi_zetaZ / q_R
        temp_matrix_for_Psi[2, 2] = Psi_ZZ
        temp_matrix_for_Psi[1, 0] = temp_matrix_for_Psi[0, 1]
        temp_matrix_for_Psi[2, 0] = temp_matrix_for_Psi[0, 2]
        temp_matrix_for_Psi[2, 1] = temp_matrix_for_Psi[1, 2]
    elif Psi_3D_lab.ndim == 3:  # Matrices of Psi, residing in the first index
        Psi_RR = Psi_3D_lab[:, 0, 0]
        Psi_zetazeta = Psi_3D_lab[:, 1, 1]
        Psi_ZZ = Psi_3D_lab[:, 2, 2]
        Psi_Rzeta = Psi_3D_lab[:, 0, 1]
        Psi_RZ = Psi_3D_lab[:, 0, 2]
        Psi_zetaZ = Psi_3D_lab[:, 1, 2]

        temp_matrix_for_Psi = np.zeros(np.shape(Psi_3D_lab), dtype="complex128")

        temp_matrix_for_Psi[:, 0, 0] = Psi_RR
        temp_matrix_for_Psi[:, 0, 1] = Psi_Rzeta / q_R - K_zeta / q_R**2
        temp_matrix_for_Psi[:, 0, 2] = Psi_RZ
        temp_matrix_for_Psi[:, 1, 1] = Psi_zetazeta / q_R**2 + K_R / q_R
        temp_matrix_for_Psi[:, 1, 2] = Psi_zetaZ / q_R
        temp_matrix_for_Psi[:, 2, 2] = Psi_ZZ
        temp_matrix_for_Psi[:, 1, 0] = temp_matrix_for_Psi[:, 0, 1]
        temp_matrix_for_Psi[:, 2, 0] = temp_matrix_for_Psi[:, 0, 2]
        temp_matrix_for_Psi[:, 2, 1] = temp_matrix_for_Psi[:, 1, 2]
    else:
        print("Error: Psi_3D_lab has an invalid number of dimensions")

    rotation_matrix_xi = np.array(
        [
            [np.cos(q_zeta), -np.sin(q_zeta), np.zeros_like(q_zeta)],
            [np.sin(q_zeta), np.cos(q_zeta), np.zeros_like(q_zeta)],
            [np.zeros_like(q_zeta), np.zeros_like(q_zeta), np.ones_like(q_zeta)],
        ]
    )
    rotation_matrix_xi_inverse = np.swapaxes(rotation_matrix_xi, 0, 1)

    if Psi_3D_lab.ndim == 3:  # Matrices of Psi, residing in the last index
        # To change the rotation matrices from [3,3,numberOfDataPoints] to [numberOfDataPoints,3,3]
        # Ensures that matmul will broadcast correctly
        rotation_matrix_xi = np.moveaxis(rotation_matrix_xi, -1, 0)
        rotation_matrix_xi_inverse = np.moveaxis(rotation_matrix_xi_inverse, -1, 0)

    Psi_3D_lab_Cartesian = np.matmul(
        np.matmul(rotation_matrix_xi, temp_matrix_for_Psi), rotation_matrix_xi_inverse
    )
    return Psi_3D_lab_Cartesian


# ----------------------------------


# Functions (beam tracing 1)


def find_electron_mass(
    temperature=None,
):
    r"""Implements first-order relativistic corrections to electron mass.
    Tmperature is an optional argument. If no argument is passed, returns
    standard electron mass as a scalar. When passed an array of temperatures,
    returns an array of relativistically-corrected electron masses.

    Temperature needs to be in units of KeV.
    """

    if temperature is None:
        # print('electron_mass used is standard value.')
        return constants.m_e

    else:
        mazzu = 1 + temperature * 4.892 * (
            10 ** (-3)
        )  # Mazzucato's relativistic correction
        electron_mass = constants.m_e * mazzu
        # (5/2) / (constants.m_e * constants.c**2) * constants.e = 4.892 * (10 ** (-6))
        # But Te is in KeV not eV
        # print('electron_mass used is: ' + str(electron_mass))
        return electron_mass


def find_normalised_plasma_freq(
    electron_density, launch_angular_frequency, temperature=None
):
    # if electron_density < 0:
    #     print(electron_density)
    #     electron_density=0
    # Electron density in units of 10^19 m-3

    electron_mass = find_electron_mass(temperature)
    # print(electron_mass)
    normalised_plasma_freq = (
        constants.e
        * np.sqrt(electron_density * 10**19 / (constants.epsilon_0 * electron_mass))
    ) / launch_angular_frequency
    # normalised_plasma_freq = np.sqrt(electron_density*10**19 * 3187.042702) / launch_angular_frequency # Torbeam's implementation

    return normalised_plasma_freq


def find_normalised_gyro_freq(B_Total, launch_angular_frequency, temperature=None):
    electron_mass = find_electron_mass(temperature)
    normalised_gyro_freq = (
        constants.e * B_Total / (electron_mass * launch_angular_frequency)
    )

    return normalised_gyro_freq


def find_epsilon_para(electron_density, launch_angular_frequency, temperature=None):
    # also called epsilon_bb in my paper

    normalised_plasma_freq = find_normalised_plasma_freq(
        electron_density, launch_angular_frequency, temperature
    )
    epsilon_para = 1 - normalised_plasma_freq**2

    return epsilon_para


def find_epsilon_perp(
    electron_density, B_Total, launch_angular_frequency, temperature=None
):
    # also called epsilon_11 in my paper

    normalised_plasma_freq = find_normalised_plasma_freq(
        electron_density, launch_angular_frequency, temperature
    )
    normalised_gyro_freq = find_normalised_gyro_freq(
        B_Total, launch_angular_frequency, temperature
    )
    epsilon_perp = 1 - normalised_plasma_freq**2 / (1 - normalised_gyro_freq**2)

    return epsilon_perp


def find_epsilon_g(
    electron_density, B_Total, launch_angular_frequency, temperature=None
):
    # also called epsilon_12 in my paper

    normalised_plasma_freq = find_normalised_plasma_freq(
        electron_density, launch_angular_frequency, temperature
    )
    normalised_gyro_freq = find_normalised_gyro_freq(
        B_Total, launch_angular_frequency, temperature
    )
    epsilon_g = (
        (normalised_plasma_freq**2)
        * normalised_gyro_freq
        / (1 - normalised_gyro_freq**2)
    )

    return epsilon_g


def find_Booker_alpha(
    electron_density,
    B_Total,
    sin_theta_m_sq,
    launch_angular_frequency,
    temperature=None,
):
    epsilon_para = find_epsilon_para(
        electron_density, launch_angular_frequency, temperature
    )
    epsilon_perp = find_epsilon_perp(
        electron_density, B_Total, launch_angular_frequency, temperature
    )
    Booker_alpha = epsilon_para * sin_theta_m_sq + epsilon_perp * (1 - sin_theta_m_sq)

    return Booker_alpha


def find_Booker_beta(
    electron_density,
    B_Total,
    sin_theta_m_sq,
    launch_angular_frequency,
    temperature=None,
):
    epsilon_perp = find_epsilon_perp(
        electron_density, B_Total, launch_angular_frequency, temperature
    )
    epsilon_para = find_epsilon_para(
        electron_density, launch_angular_frequency, temperature
    )
    epsilon_g = find_epsilon_g(
        electron_density, B_Total, launch_angular_frequency, temperature
    )
    Booker_beta = -epsilon_perp * epsilon_para * (1 + sin_theta_m_sq) - (
        epsilon_perp**2 - epsilon_g**2
    ) * (1 - sin_theta_m_sq)

    return Booker_beta


def find_Booker_gamma(
    electron_density, B_Total, launch_angular_frequency, temperature=None
):
    epsilon_perp = find_epsilon_perp(
        electron_density, B_Total, launch_angular_frequency, temperature
    )
    epsilon_para = find_epsilon_para(
        electron_density, launch_angular_frequency, temperature
    )
    epsilon_g = find_epsilon_g(
        electron_density, B_Total, launch_angular_frequency, temperature
    )
    Booker_gamma = epsilon_para * (epsilon_perp**2 - epsilon_g**2)

    return Booker_gamma


# ----------------------------------


def find_H_numba(
    K_magnitude,
    electron_density,
    B_Total,
    sin_theta_m_sq,
    launch_angular_frequency,
    mode_flag,
    temperature,
):
    # For use with the numba package (parallelisation)
    # As such, doesn't take any functions as arguments
    # Still in development

    wavenumber_K0 = launch_angular_frequency / constants.c

    Booker_alpha = find_Booker_alpha(
        electron_density,
        B_Total,
        sin_theta_m_sq,
        launch_angular_frequency,
        temperature,
    )
    Booker_beta = find_Booker_beta(
        electron_density,
        B_Total,
        sin_theta_m_sq,
        launch_angular_frequency,
        temperature,
    )
    Booker_gamma = find_Booker_gamma(
        electron_density, B_Total, launch_angular_frequency, temperature
    )

    # Due to numerical errors, sometimes H_discriminant ends up being a very small negative number
    # That's why we take max(0, H_discriminant) in the sqrt

    H = (K_magnitude / wavenumber_K0) ** 2 + (
        Booker_beta
        - mode_flag
        * np.sqrt(
            np.maximum(
                np.zeros_like(Booker_beta),
                (Booker_beta**2 - 4 * Booker_alpha * Booker_gamma),
            )
        )
        # np.sqrt(Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
    ) / (2 * Booker_alpha)

    return H


# ----------------------------------


# Functions (interface)
# For going from vacuum to plasma (Will one day implement going from plasma to vacuum)


def find_Psi_3D_plasma_continuous(
    Psi_vacuum_3D,
    dH_dKR,
    dH_dKzeta,
    dH_dKZ,
    dH_dR,
    dH_dZ,
    d_poloidal_flux_d_R,
    d_poloidal_flux_d_Z,
):
    ## For continuous ne across the plasma-vacuum boundary
    ## Potential future improvement: write wrapper function to wrap find_Psi_3D_plasma_continuous and find_Psi_3D_plasma_discontinuous

    # Gradients are the plasma-vacuum boundary can be finnicky and it's good to check
    if np.isnan(dH_dKR):
        raise ValueError("Error, dH_dKR is NaN in find_Psi_3D_plasma_continuous")
    elif np.isnan(dH_dKzeta):
        raise ValueError("Error, dH_dKzeta is NaN in find_Psi_3D_plasma_continuous")
    elif np.isnan(dH_dKZ):
        raise ValueError("Error, dH_dKZ is NaN in find_Psi_3D_plasma_continuous")
    elif np.isnan(dH_dR):
        raise ValueError("Error, dH_dR is NaN in find_Psi_3D_plasma_continuous")
    elif np.isnan(dH_dZ):
        raise ValueError("Error, dH_dZ is NaN in find_Psi_3D_plasma_continuous")

    # When beam is entering plasma from vacuum
    Psi_v_R_R = Psi_vacuum_3D[0, 0]
    Psi_v_zeta_zeta = Psi_vacuum_3D[1, 1]
    Psi_v_Z_Z = Psi_vacuum_3D[2, 2]
    Psi_v_R_zeta = Psi_vacuum_3D[0, 1]
    Psi_v_R_Z = Psi_vacuum_3D[0, 2]
    Psi_v_zeta_Z = Psi_vacuum_3D[1, 2]

    interface_matrix = np.zeros([6, 6])
    interface_matrix[0][5] = 1
    interface_matrix[1][0] = d_poloidal_flux_d_Z**2
    interface_matrix[1][1] = -2 * d_poloidal_flux_d_R * d_poloidal_flux_d_Z
    interface_matrix[1][3] = d_poloidal_flux_d_R**2
    interface_matrix[2][2] = -d_poloidal_flux_d_Z
    interface_matrix[2][4] = d_poloidal_flux_d_R
    interface_matrix[3][0] = dH_dKR
    interface_matrix[3][1] = dH_dKZ
    interface_matrix[3][2] = dH_dKzeta
    interface_matrix[4][1] = dH_dKR
    interface_matrix[4][3] = dH_dKZ
    interface_matrix[4][4] = dH_dKzeta
    interface_matrix[5][2] = dH_dKR
    interface_matrix[5][4] = dH_dKZ
    interface_matrix[5][5] = dH_dKzeta

    # interface_matrix will be singular if one tries to transition while still in vacuum (and there's no plasma at all)
    # at least that's what happens, in my experience
    interface_matrix_inverse = np.linalg.inv(interface_matrix)

    [
        Psi_p_R_R,
        Psi_p_R_Z,
        Psi_p_R_zeta,
        Psi_p_Z_Z,
        Psi_p_Z_zeta,
        Psi_p_zeta_zeta,
    ] = np.matmul(
        interface_matrix_inverse,
        [
            Psi_v_zeta_zeta,
            Psi_v_R_R * d_poloidal_flux_d_Z**2
            - 2 * Psi_v_R_Z * d_poloidal_flux_d_R * d_poloidal_flux_d_Z
            + Psi_v_Z_Z * d_poloidal_flux_d_R**2,
            -Psi_v_R_zeta * d_poloidal_flux_d_Z + Psi_v_zeta_Z * d_poloidal_flux_d_R,
            -dH_dR,
            -dH_dZ,
            0,
        ],
    )

    Psi_3D_plasma = np.zeros([3, 3], dtype="complex128")
    Psi_3D_plasma[0, 0] = Psi_p_R_R
    Psi_3D_plasma[1, 1] = Psi_p_zeta_zeta
    Psi_3D_plasma[2, 2] = Psi_p_Z_Z
    Psi_3D_plasma[0, 1] = Psi_p_R_zeta
    Psi_3D_plasma[1, 0] = Psi_3D_plasma[0, 1]
    Psi_3D_plasma[0, 2] = Psi_p_R_Z
    Psi_3D_plasma[2, 0] = Psi_3D_plasma[0, 2]
    Psi_3D_plasma[1, 2] = Psi_p_Z_zeta
    Psi_3D_plasma[2, 1] = Psi_3D_plasma[1, 2]

    return Psi_3D_plasma


def find_K_plasma(
    q_R,
    K_v_R,
    K_v_zeta,
    K_v_Z,
    launch_angular_frequency,
    mode_flag,
    B_R,
    B_T,
    B_Z,
    electron_density_p,  # in the plasma
    dpolflux_dR,
    dpolflux_dZ,
    temperature=None,
):
    ## Finds

    ## Checks the plasma density
    plasma_freq = find_normalised_plasma_freq(
        electron_density_p, launch_angular_frequency, temperature
    )

    B_Total = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    gyro_freq = find_normalised_gyro_freq(B_Total, launch_angular_frequency)
    omega_R = 0.5 * (gyro_freq + np.sqrt(gyro_freq**2 + 4 * plasma_freq**2))
    omega_L = 0.5 * (-gyro_freq + np.sqrt(gyro_freq**2 + 4 * plasma_freq**2))
    omega_UH = np.sqrt(plasma_freq**2 + gyro_freq**2)

    if mode_flag == 1:
        ## O-mode
        if plasma_freq >= 1.0:
            ## That is, if the (normalised) cutoff frequency is higher than the launch beam frequency inside the plasma
            raise ValueError(
                "Error: cut-off freq higher than beam freq on plasma side of plasma-vac boundary"
            )

    elif mode_flag == -1:
        ## X-mode
        if (omega_R >= 1) and (omega_UH <= 1):
            raise ValueError(
                "Error: cut-off freq higher than beam freq on plasma side of plasma-vac boundary"
            )
        if omega_L >= 1:
            raise ValueError(
                "Error: cut-off freq higher than beam freq on plasma side of plasma-vac boundary"
            )
    ##

    ## Get components of the Booker quartic
    K_v_mag = np.sqrt(K_v_R**2 + (K_v_zeta / q_R) ** 2 + K_v_Z**2)
    sin_theta_m_vac = (B_R * K_v_R + B_T * K_v_zeta / q_R + B_Z * K_v_Z) / (
        B_Total * K_v_mag
    )  # B \cdot K / (abs (B) abs(K))
    sin_theta_m_sq = sin_theta_m_vac**2

    Booker_alpha = find_Booker_alpha(
        electron_density_p,
        B_Total,
        sin_theta_m_sq,
        launch_angular_frequency,
        temperature,
    )
    Booker_beta = find_Booker_beta(
        electron_density_p,
        B_Total,
        sin_theta_m_sq,
        launch_angular_frequency,
        temperature,
    )
    Booker_gamma = find_Booker_gamma(
        electron_density_p, B_Total, launch_angular_frequency, temperature
    )

    ## Calculate wavevector in plasma
    K_p_zeta = K_v_zeta

    K_v_pol = (-K_v_R * dpolflux_dZ + K_v_Z * dpolflux_dR) / np.sqrt(
        dpolflux_dR**2 + dpolflux_dZ**2
    )
    K_p_pol = K_v_pol

    ## TODO
    ## This only works if the mismatch angle is continuous. It is not
    ## I need to implement an iterative solver for this
    K_p_rad = -np.sqrt(
        abs(
            K_v_mag**2
            * (
                -Booker_beta
                + mode_flag
                * np.sqrt(Booker_beta**2 - 4 * Booker_alpha * Booker_gamma)
            )
            / (2 * Booker_alpha)
            - (K_p_zeta / q_R) ** 2
            - K_p_pol**2
        )
    )

    [
        K_p_R,
        K_p_Z,
    ] = np.matmul(
        np.linalg.inv([[-dpolflux_dZ, dpolflux_dR], [dpolflux_dR, dpolflux_dZ]])
        * np.sqrt(dpolflux_dR**2 + dpolflux_dZ**2),
        [
            K_p_pol,
            K_p_rad,
        ],
    )

    ## For checking
    H_bar = (K_p_R**2 + (K_p_zeta / q_R) ** 2 + K_p_Z**2) / (K_v_mag) ** 2 - (
        (
            -Booker_beta
            + mode_flag * np.sqrt(Booker_beta**2 - 4 * Booker_alpha * Booker_gamma)
        )
        / (2 * Booker_alpha)
    )
    tol = 1e-3

    if abs(H_bar) > tol:
        print("find_K_plasma not working properly")
        print("H_bar =", H_bar)

    return K_p_R, K_p_zeta, K_p_Z


def find_Psi_3D_plasma_discontinuous(
    Psi_vacuum_3D,
    K_v_R,
    K_v_zeta,
    K_v_Z,
    K_p_R,
    K_p_zeta,
    K_p_Z,
    dH_dKR,  # In the plasma
    dH_dKzeta,  # In the plasma
    dH_dKZ,  # In the plasma
    dH_dR,  # In the plasma
    dH_dZ,  # In the plasma
    dpolflux_dR,  # Continuous
    dpolflux_dZ,  # Continuous
    d2polflux_dR2,  # Continuous
    d2polflux_dZ2,  # Continuous
    d2polflux_dRdZ,  # Continuous
):
    ## For discontinuous ne

    eta = (
        -0.5
        * (
            d2polflux_dR2 * dpolflux_dZ**2
            + 2 * d2polflux_dRdZ * dpolflux_dR * dpolflux_dZ
            + d2polflux_dZ2 * dpolflux_dR**2
        )
        / (dpolflux_dR**2 + dpolflux_dZ**2)
    )

    # When beam is entering plasma from vacuum
    Psi_v_R_R = Psi_vacuum_3D[0, 0]
    Psi_v_zeta_zeta = Psi_vacuum_3D[1, 1]
    Psi_v_Z_Z = Psi_vacuum_3D[2, 2]
    Psi_v_R_zeta = Psi_vacuum_3D[0, 1]
    Psi_v_R_Z = Psi_vacuum_3D[0, 2]
    Psi_v_zeta_Z = Psi_vacuum_3D[1, 2]

    interface_matrix = np.zeros([6, 6])
    interface_matrix[0][5] = 1
    interface_matrix[1][0] = dpolflux_dZ**2
    interface_matrix[1][1] = -2 * dpolflux_dR * dpolflux_dZ
    interface_matrix[1][3] = dpolflux_dR**2
    interface_matrix[2][2] = -dpolflux_dZ
    interface_matrix[2][4] = dpolflux_dR
    interface_matrix[3][0] = dH_dKR
    interface_matrix[3][1] = dH_dKZ
    interface_matrix[3][2] = dH_dKzeta
    interface_matrix[4][1] = dH_dKR
    interface_matrix[4][3] = dH_dKZ
    interface_matrix[4][4] = dH_dKzeta
    interface_matrix[5][2] = dH_dKR
    interface_matrix[5][4] = dH_dKZ
    interface_matrix[5][5] = dH_dKzeta

    # interface_matrix will be singular if one tries to transition while still in vacuum (and there's no plasma at all)
    # at least that's what happens, in my experience
    interface_matrix_inverse = np.linalg.inv(interface_matrix)

    [
        Psi_p_R_R,
        Psi_p_R_Z,
        Psi_p_R_zeta,
        Psi_p_Z_Z,
        Psi_p_Z_zeta,
        Psi_p_zeta_zeta,
    ] = np.matmul(
        interface_matrix_inverse,
        [
            Psi_v_zeta_zeta,
            Psi_v_R_R * dpolflux_dZ**2
            - 2 * Psi_v_R_Z * dpolflux_dR * dpolflux_dZ
            + Psi_v_Z_Z * dpolflux_dR**2,
            -Psi_v_R_zeta * dpolflux_dZ
            + Psi_v_zeta_Z * dpolflux_dR
            + 2 * (K_v_R - K_p_R) * dpolflux_dR * eta
            + 2 * (K_v_Z - K_p_Z) * dpolflux_dZ * eta,
            -dH_dR,
            -dH_dZ,
            0,
        ],
    )

    Psi_3D_plasma = np.zeros([3, 3], dtype="complex128")
    Psi_3D_plasma[0, 0] = Psi_p_R_R
    Psi_3D_plasma[1, 1] = Psi_p_zeta_zeta
    Psi_3D_plasma[2, 2] = Psi_p_Z_Z
    Psi_3D_plasma[0, 1] = Psi_p_R_zeta
    Psi_3D_plasma[1, 0] = Psi_3D_plasma[0, 1]
    Psi_3D_plasma[0, 2] = Psi_p_R_Z
    Psi_3D_plasma[2, 0] = Psi_3D_plasma[0, 2]
    Psi_3D_plasma[1, 2] = Psi_p_Z_zeta
    Psi_3D_plasma[2, 1] = Psi_3D_plasma[1, 2]

    return Psi_3D_plasma


def apply_discontinuous_BC(
    q_R,
    q_Z,
    Psi_vacuum_3D,
    K_v_R,
    K_v_zeta,
    K_v_Z,
    launch_angular_frequency,
    mode_flag,
    delta_R,
    delta_Z,
    field,  # Field object
    hamiltonian,  # Hamiltonian object
    temperature=None,  # Currently an unused argument because downstream the H-booker functions
    # have an optional temperature argument. However apply_discontinuous_BC is only called by
    # launch_beam where the beam is propagating outside plasma currently, thus the temperature
    # argument is irrelevant. For cases where the beam is launched inside the plasma, the
    # temperature argument should be taken care of by the Hamiltonian (?).
):
    d_poloidal_flux_dR_boundary = field.d_poloidal_flux_dR(q_R, q_Z, delta_R)
    d_poloidal_flux_dZ_boundary = field.d_poloidal_flux_dZ(q_R, q_Z, delta_Z)
    d2_poloidal_flux_dR2_boundary = field.d2_poloidal_flux_dR2(q_R, q_Z, delta_R)
    d2_poloidal_flux_dZ2_boundary = field.d2_poloidal_flux_dZ2(q_R, q_Z, delta_Z)
    d2_poloidal_flux_dRdZ_boundary = field.d2_poloidal_flux_dRdZ(
        q_R, q_Z, delta_R, delta_Z
    )

    poloidal_flux_boundary = field.poloidal_flux(q_R, q_Z)

    K_plasma = find_K_plasma(
        q_R,
        K_v_R,
        K_v_zeta,
        K_v_Z,
        launch_angular_frequency,
        mode_flag,
        field.B_R(q_R, q_Z),
        field.B_T(q_R, q_Z),
        field.B_Z(q_R, q_Z),
        hamiltonian.density(poloidal_flux_boundary),  # in the plasma
        d_poloidal_flux_dR_boundary,
        d_poloidal_flux_dZ_boundary,
        temperature,
    )

    dH = hamiltonian.derivatives(
        q_R,
        q_Z,
        K_plasma[0],
        K_plasma[1],
        K_plasma[2],
    )

    Psi_3D_plasma = find_Psi_3D_plasma_discontinuous(
        Psi_vacuum_3D,
        K_v_R,
        K_v_zeta,
        K_v_Z,
        K_plasma[0],
        K_plasma[1],
        K_plasma[2],
        dH["dH_dKR"],  # In the plasma
        dH["dH_dKzeta"],  # In the plasma
        dH["dH_dKZ"],  # In the plasma
        dH["dH_dR"],  # In the plasma
        dH["dH_dZ"],  # In the plasma
        d_poloidal_flux_dR_boundary,  # Continuous
        d_poloidal_flux_dZ_boundary,  # Continuous
        d2_poloidal_flux_dR2_boundary,  # Continuous
        d2_poloidal_flux_dZ2_boundary,  # Continuous
        d2_poloidal_flux_dRdZ_boundary,  # Continuous
    )

    return K_plasma, Psi_3D_plasma


def apply_continuous_BC(
    q_R,
    q_Z,
    Psi_vacuum_3D,
    K_v_R,
    K_v_zeta,
    K_v_Z,
    delta_R,
    delta_Z,
    field,  # Field object
    hamiltonian,  # Hamiltonian object
):
    ## When the equilibrium is continuous, the wavevector is continuous
    K_plasma = [K_v_R, K_v_zeta, K_v_Z]

    dH = hamiltonian.derivatives(
        q_R,
        q_Z,
        K_plasma[0],
        K_plasma[1],
        K_plasma[2],
    )

    dH_dR_initial = dH["dH_dR"]
    dH_dZ_initial = dH["dH_dZ"]
    dH_dKR_initial = dH["dH_dKR"]
    dH_dKzeta_initial = dH["dH_dKzeta"]
    dH_dKZ_initial = dH["dH_dKZ"]

    Psi_3D_plasma = find_Psi_3D_plasma_continuous(
        Psi_vacuum_3D,
        dH_dKR_initial,
        dH_dKzeta_initial,
        dH_dKZ_initial,
        dH_dR_initial,
        dH_dZ_initial,
        field.d_poloidal_flux_dR(q_R, q_Z, delta_R),
        field.d_poloidal_flux_dZ(q_R, q_Z, delta_Z),
    )

    return K_plasma, Psi_3D_plasma


# -----------------


# Functions (analysis)
# These are not strictly necessary for beam tracing, but useful for analysis of DBS
def find_dbhat_dR(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z):
    # \fract{d b_hat}{d R}
    B_R_plus_R = np.squeeze(find_B_R(q_R + delta_R, q_Z))
    B_T_plus_R = np.squeeze(find_B_T(q_R + delta_R, q_Z))
    B_Z_plus_R = np.squeeze(find_B_Z(q_R + delta_R, q_Z))

    B_R_minus_R = np.squeeze(find_B_R(q_R - delta_R, q_Z))
    B_T_minus_R = np.squeeze(find_B_T(q_R - delta_R, q_Z))
    B_Z_minus_R = np.squeeze(find_B_Z(q_R - delta_R, q_Z))

    B_magnitude_plus = np.sqrt(B_R_plus_R**2 + B_T_plus_R**2 + B_Z_plus_R**2)
    b_hat_plus = np.array([B_R_plus_R, B_T_plus_R, B_Z_plus_R]) / B_magnitude_plus

    B_magnitude_minus = np.sqrt(B_R_minus_R**2 + B_T_minus_R**2 + B_Z_minus_R**2)
    b_hat_minus = np.array([B_R_minus_R, B_T_minus_R, B_Z_minus_R]) / B_magnitude_minus

    dbhat_dR = (b_hat_plus - b_hat_minus) / (2 * delta_R)

    return dbhat_dR


def find_dbhat_dZ(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z):
    # \fract{d b_hat}{d R}
    B_R_plus_Z = np.squeeze(find_B_R(q_R, q_Z + delta_Z))
    B_T_plus_Z = np.squeeze(find_B_T(q_R, q_Z + delta_Z))
    B_Z_plus_Z = np.squeeze(find_B_Z(q_R, q_Z + delta_Z))

    B_R_minus_Z = np.squeeze(
        find_B_R(
            q_R,
            q_Z - delta_Z,
        )
    )
    B_T_minus_Z = np.squeeze(
        find_B_T(
            q_R,
            q_Z - delta_Z,
        )
    )
    B_Z_minus_Z = np.squeeze(
        find_B_Z(
            q_R,
            q_Z - delta_Z,
        )
    )

    B_magnitude_plus = np.sqrt(B_R_plus_Z**2 + B_T_plus_Z**2 + B_Z_plus_Z**2)
    b_hat_plus = np.array([B_R_plus_Z, B_T_plus_Z, B_Z_plus_Z]) / B_magnitude_plus

    B_magnitude_minus = np.sqrt(B_R_minus_Z**2 + B_T_minus_Z**2 + B_Z_minus_Z**2)
    b_hat_minus = np.array([B_R_minus_Z, B_T_minus_Z, B_Z_minus_Z]) / B_magnitude_minus

    dbhat_dZ = (b_hat_plus - b_hat_minus) / (2 * delta_Z)

    return dbhat_dZ


# def find_g_magnitude(q_R,q_Z,K_R,K_zeta,K_Z,launch_angular_frequency,mode_flag,delta_K_R,delta_K_zeta,delta_K_Z,
#                     interp_poloidal_flux,find_density_1D,find_B_R,find_B_T,find_B_Z): # Finds the magnitude of the group velocity. This method is slow, do not use in main loop.\
#    dH_dKR   = find_dH_dKR(
#                           q_R,q_Z,
#                           K_R,K_zeta,K_Z,
#                           launch_angular_frequency,mode_flag,delta_K_R,
#                           interp_poloidal_flux,find_density_1D,
#                           find_B_R,find_B_T,find_B_Z
#                          )
#    dH_dKzeta = find_dH_dKzeta(
#                               q_R,q_Z,
#                               K_R,K_zeta,K_Z,
#                               launch_angular_frequency,mode_flag,delta_K_zeta,
#                               interp_poloidal_flux,find_density_1D,
#                               find_B_R,find_B_T,find_B_Z
#                              )
#    dH_dKZ    = find_dH_dKZ(
#                            q_R,q_Z,
#                            K_R,K_zeta,K_Z,
#                            launch_angular_frequency,mode_flag,delta_K_Z,
#                            interp_poloidal_flux,find_density_1D,
#                            find_B_R,find_B_T,find_B_Z
#                           )
#    g_magnitude = (q_R**2 * dH_dKzeta**2 + dH_dKR**2 + dH_dKZ**2)**0.5
#    return g_magnitude


def find_D(
    K_magnitude,
    launch_angular_frequency,
    epsilon_para,
    epsilon_perp,
    epsilon_g,
    theta_m,
):
    # Finds the dispersion

    wavenumber_K0 = launch_angular_frequency / constants.c
    n_ref_index = K_magnitude / wavenumber_K0
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)

    D_11_component = epsilon_perp - n_ref_index**2 * sin_theta_m**2
    D_22_component = epsilon_perp - n_ref_index**2
    D_bb_component = epsilon_para - n_ref_index**2 * cos_theta_m**2
    D_12_component = epsilon_g
    D_1b_component = n_ref_index**2 * sin_theta_m * cos_theta_m

    return (
        D_11_component,
        D_22_component,
        D_bb_component,
        D_12_component,
        D_1b_component,
    )


def find_H_Cardano(
    K_magnitude,
    launch_angular_frequency,
    epsilon_para,
    epsilon_perp,
    epsilon_g,
    theta_m,
):
    # This function is designed to be evaluated in post-procesing, hence it uses different inputs from the usual H

    (
        D_11_component,
        D_22_component,
        D_bb_component,
        D_12_component,
        D_1b_component,
    ) = find_D(
        K_magnitude,
        launch_angular_frequency,
        epsilon_para,
        epsilon_perp,
        epsilon_g,
        theta_m,
    )

    h_2_coefficient = -D_11_component - D_22_component - D_bb_component
    h_1_coefficient = (
        D_11_component * D_bb_component
        + D_11_component * D_22_component
        + D_22_component * D_bb_component
        - D_12_component**2
        - D_1b_component**2
    )
    h_0_coefficient = (
        D_22_component * D_1b_component**2
        + D_bb_component * D_12_component**2
        - D_11_component * D_22_component * D_bb_component
    )

    h_t_coefficient = (
        -2 * h_2_coefficient**3
        + 9 * h_2_coefficient * h_1_coefficient
        - 27 * h_0_coefficient
        + 3
        * np.sqrt(3)
        * np.sqrt(
            4 * h_2_coefficient**3 * h_0_coefficient
            - h_2_coefficient**2 * h_1_coefficient**2
            - 18 * h_2_coefficient * h_1_coefficient * h_0_coefficient
            + 4 * h_1_coefficient**3
            + 27 * h_0_coefficient**2
            + 0j  # to make the argument of the np.sqrt complex, so that the sqrt evaluates negative functions
        )
    ) ** (1 / 3)

    H_1_Cardano = (
        h_t_coefficient / (3 * 2 ** (1 / 3))
        - 2 ** (1 / 3)
        * (3 * h_1_coefficient - h_2_coefficient**2)
        / (3 * h_t_coefficient)
        - h_2_coefficient / 3
    )
    H_2_Cardano = (
        -(1 - 1j * np.sqrt(3)) / (6 * 2 ** (1 / 3)) * h_t_coefficient
        + (1 + 1j * np.sqrt(3))
        * (3 * h_1_coefficient - h_2_coefficient**2)
        / (3 * 2 ** (2 / 3) * h_t_coefficient)
        - h_2_coefficient / 3
    )
    H_3_Cardano = (
        -(1 + 1j * np.sqrt(3)) / (6 * 2 ** (1 / 3)) * h_t_coefficient
        + (1 - 1j * np.sqrt(3))
        * (3 * h_1_coefficient - h_2_coefficient**2)
        / (3 * 2 ** (2 / 3) * h_t_coefficient)
        - h_2_coefficient / 3
    )
    return H_1_Cardano, H_2_Cardano, H_3_Cardano


def find_ST_terms(
    tau_array,
    K_magnitude_array,
    k_perp_1_bs,
    g_magnitude_Cardano,
    g_magnitude_output,
    theta_m_output,
    M_w_inv_xx_output,
):
    ## Calculates how far into the 'large mismatch' ordering we are
    d_theta_m_d_tau = np.gradient(theta_m_output, tau_array)
    d_K_d_tau = np.gradient(K_magnitude_array, tau_array)
    d_tau_B_d_tau_C = (
        g_magnitude_Cardano / g_magnitude_output
    )  # d tau_Booker / d tau_Cardano

    theta_m_min_idx = np.argmin(abs(theta_m_output))
    # delta_kperp1_ST = k_perp_1_bs - k_perp_1_bs[theta_m_min_idx]

    G_full = (
        (
            d_K_d_tau * g_magnitude_output
            - K_magnitude_array**2 * d_theta_m_d_tau**2 * M_w_inv_xx_output
        )
        * d_tau_B_d_tau_C**2
    ) ** (-1)
    G_term1 = (d_K_d_tau * g_magnitude_output * d_tau_B_d_tau_C**2) ** (-1)
    G_term2 = (
        K_magnitude_array**2
        * d_theta_m_d_tau**2
        * M_w_inv_xx_output
        * G_term1**2
        * d_tau_B_d_tau_C**2
    ) ** (-1)
    indicator = np.divide(
        (G_term1 + G_term2 - G_full),
        (0.5 * (G_term1 + G_term2 + G_full)),
        out=np.zeros_like(tau_array),
        where=(0.5 * (G_term1 + G_term2 + G_full)) != 0,
    )
    return indicator, indicator[theta_m_min_idx]


# ----------------------------------


# Functions (for runs with only ray tracing and no beam tracing)
## Not used yet, still being written
def find_quick_output(ray_parameters_2D, K_zeta_initial, find_B_R, find_B_T, find_B_Z):
    """
    Finds mismatch at cut-off location
    Cut-off location where K is minimised
    """
    q_R = np.real(ray_parameters_2D[0, :])
    q_Z = np.real(ray_parameters_2D[1, :])
    K_R = np.real(ray_parameters_2D[2, :])
    K_Z = np.real(ray_parameters_2D[3, :])
    K_magnitude = np.sqrt(K_R**2 + (K_zeta_initial / q_R) ** 2 + K_Z**2)

    # cutoff_index = np.argwhere
    quick_output = 0

    return quick_output


# ----------------------------------


# Functions (circular Gaussian beam in vacuum)
def find_Rayleigh_length(
    waist, wavenumber
):  # Finds the size of the waist (assumes vacuum propagation and circular beam)
    Rayleigh_length = 0.5 * wavenumber * waist**2
    return Rayleigh_length


def find_waist(
    width, wavenumber, curvature
):  # Finds the size of the waist (assumes vacuum propagation and circular beam)
    waist = width / np.sqrt(1 + curvature**2 * width**4 * wavenumber**2 / 4)
    return waist


def find_distance_from_waist(
    width, wavenumber, curvature
):  # Finds how far you are from the waist (assumes vacuum propagation and circular beam)
    waist = width / np.sqrt(1 + curvature**2 * width**4 * wavenumber**2 / 4)
    distance_from_waist = np.sign(curvature) * np.sqrt(
        (width**2 - waist**2) * waist**2 * wavenumber**2 / 4
    )
    return distance_from_waist


# def propagate_circular_beam(distance,wavenumber,w0):
#     """
#     w0 : Width of beam waist


#     Note that the curvature in this function returns has units of inverse length.
#     """
#     z_R = find_Rayleigh_length(w0, wavenumber)
#     widths = w0 * np.sqrt(1+(distance/z_R)**2)
#     curvatures = distance / (distance**2 +z_R**2)
#     return widths, curvatures
def propagate_circular_beam(width, curvature, propagation_distance, freq_GHz):
    """
    w0 : Width of beam waist
    """
    wavenumber = freq_GHz_to_wavenumber(freq_GHz)

    w0 = find_waist(width, wavenumber, curvature)
    z0 = find_distance_from_waist(width, wavenumber, curvature)
    z_R = find_Rayleigh_length(w0, wavenumber)

    z = z0 + propagation_distance

    widths = w0 * np.sqrt(1 + (z / z_R) ** 2)
    curvatures = z / (z**2 + z_R**2)
    return widths, curvatures


def modify_beam(width, curvature, freq_GHz, z0_shift, w0_shift):
    """
    positive z0_shift moves the waist in the direction of -infinity
    """
    wavenumber = freq_GHz_to_wavenumber(freq_GHz)

    w0_i = find_waist(width, wavenumber, curvature)
    z0_i = find_distance_from_waist(width, wavenumber, curvature)

    w0_f = w0_i + w0_shift
    z0_f = z0_i + z0_shift

    # Propagate from the new waist back to the initial location
    w_f, curv_f = propagate_circular_beam(w0_f, 0, z0_f, freq_GHz)

    return w_f, curv_f


# ----------------------------------


# Functions (general Gaussian beam in vacuum)
def propagate_beam(Psi_w_initial_cartesian, propagation_distance, freq_GHz):
    """
    Uses the vacuum solution of the beam tracing equations to propagate the beam
    Works for arbitrary Gaussian beam in vacuum
    """
    wavenumber_K0 = freq_GHz_to_wavenumber(freq_GHz)

    Psi_w_inv_initial_cartersian = find_inverse_2D(Psi_w_initial_cartesian)

    Psi_w_inv_final_cartersian = (
        propagation_distance / (wavenumber_K0) * np.eye(2)
        + Psi_w_inv_initial_cartersian
    )

    Psi_w_final_cartesian = find_inverse_2D(Psi_w_inv_final_cartersian)

    return Psi_w_final_cartesian


def find_widths_and_curvatures(Psi_xx, Psi_xy, Psi_yy, K_magnitude, theta_m, theta):
    """
    Calculates beam widths and curvatures from components of Psi_w

    Equations (15) and (16) of VH Hall-Chen et al., PPCF (2022) https://doi.org/10.1088/1361-6587/ac57a1

    Parameters
    ----------
    Psi_xx : TYPE
        DESCRIPTION.
    Psi_xy : TYPE
        DESCRIPTION.
    Psi_yy : TYPE
        DESCRIPTION.
    K_magnitude : TYPE
        DESCRIPTION.
    theta_m : TYPE
        Mismatch angle, in radians. Equation (109) of the above paper
    theta : TYPE
        Equation (107) of the above paper.

    Returns
    -------
    widths : TYPE
        DESCRIPTION.
    Psi_w_imag_eigvecs : TYPE
        DESCRIPTION.
    curvatures : TYPE
        DESCRIPTION.
    Psi_w_real_eigvecs : TYPE
        DESCRIPTION.

    """
    Psi_w_real = np.array(np.real([[Psi_xx, Psi_xy], [Psi_xy, Psi_yy]]))
    Psi_w_imag = np.array(np.imag([[Psi_xx, Psi_xy], [Psi_xy, Psi_yy]]))

    Psi_w_real_eigvals, Psi_w_real_eigvecs = np.linalg.eig(Psi_w_real)
    Psi_w_imag_eigvals, Psi_w_imag_eigvecs = np.linalg.eig(Psi_w_imag)
    # The normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]

    widths = np.sqrt(2 / Psi_w_imag_eigvals)
    # curvature = 1/radius_of_curvature
    curvatures = Psi_w_real_eigvals / K_magnitude * (np.cos(theta_m + theta)) ** 2

    return widths, Psi_w_imag_eigvecs, curvatures, Psi_w_real_eigvecs


# ----------------------------------


# Functions (Debugging)


def find_dB_dR_CFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z):
    r"""Finds :math:`\frac{d B}{d R}`, where B is the magnitude of the B field"""

    B_R_plus_R = np.squeeze(find_B_R(q_R + delta_R, q_Z))
    B_T_plus_R = np.squeeze(find_B_T(q_R + delta_R, q_Z))
    B_Z_plus_R = np.squeeze(find_B_Z(q_R + delta_R, q_Z))

    B_R_minus_R = np.squeeze(find_B_R(q_R - delta_R, q_Z))
    B_T_minus_R = np.squeeze(find_B_T(q_R - delta_R, q_Z))
    B_Z_minus_R = np.squeeze(find_B_Z(q_R - delta_R, q_Z))

    B_magnitude_plus = np.sqrt(B_R_plus_R**2 + B_T_plus_R**2 + B_Z_plus_R**2)

    B_magnitude_minus = np.sqrt(B_R_minus_R**2 + B_T_minus_R**2 + B_Z_minus_R**2)

    dB_dZ = (B_magnitude_plus - B_magnitude_minus) / (2 * delta_R)
    return dB_dZ


def find_dB_dZ_CFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z):
    r"""Finds :math:`\frac{d B}{d Z}`, where B is the magnitude of the B field"""

    B_R_plus_Z = np.squeeze(find_B_R(q_R, q_Z + delta_Z))
    B_T_plus_Z = np.squeeze(find_B_T(q_R, q_Z + delta_Z))
    B_Z_plus_Z = np.squeeze(find_B_Z(q_R, q_Z + delta_Z))

    B_R_minus_Z = np.squeeze(
        find_B_R(
            q_R,
            q_Z - delta_Z,
        )
    )
    B_T_minus_Z = np.squeeze(
        find_B_T(
            q_R,
            q_Z - delta_Z,
        )
    )
    B_Z_minus_Z = np.squeeze(
        find_B_Z(
            q_R,
            q_Z - delta_Z,
        )
    )

    B_magnitude_plus = np.sqrt(B_R_plus_Z**2 + B_T_plus_Z**2 + B_Z_plus_Z**2)

    B_magnitude_minus = np.sqrt(B_R_minus_Z**2 + B_T_minus_Z**2 + B_Z_minus_Z**2)

    dB_dZ = (B_magnitude_plus - B_magnitude_minus) / (2 * delta_Z)
    return dB_dZ


def find_d2B_dR2_CFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z):
    r"""Finds :math:`\frac{d^2 B}{d R^2}`, where B is the magnitude of the B field"""

    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)

    B_R_plus = np.squeeze(find_B_R(q_R + delta_R, q_Z))
    B_T_plus = np.squeeze(find_B_T(q_R + delta_R, q_Z))
    B_Z_plus = np.squeeze(find_B_Z(q_R + delta_R, q_Z))
    B_magnitude_plus = np.sqrt(B_R_plus**2 + B_T_plus**2 + B_Z_plus**2)

    B_R_minus = np.squeeze(find_B_R(q_R - delta_R, q_Z))
    B_T_minus = np.squeeze(find_B_T(q_R - delta_R, q_Z))
    B_Z_minus = np.squeeze(find_B_Z(q_R - delta_R, q_Z))
    B_magnitude_minus = np.sqrt(B_R_minus**2 + B_T_minus**2 + B_Z_minus**2)

    d2B_dR2 = (
        (1) * B_magnitude_minus + (-2) * B_magnitude_0 + (1) * B_magnitude_plus
    ) / (delta_R**2)
    return d2B_dR2


def find_d2B_dZ2_CFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z):
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)

    B_R_plus = np.squeeze(find_B_R(q_R, q_Z + delta_Z))
    B_T_plus = np.squeeze(find_B_T(q_R, q_Z + delta_Z))
    B_Z_plus = np.squeeze(find_B_Z(q_R, q_Z + delta_Z))
    B_magnitude_plus = np.sqrt(B_R_plus**2 + B_T_plus**2 + B_Z_plus**2)

    B_R_minus = np.squeeze(find_B_R(q_R, q_Z - delta_Z))
    B_T_minus = np.squeeze(find_B_T(q_R, q_Z - delta_Z))
    B_Z_minus = np.squeeze(find_B_Z(q_R, q_Z - delta_Z))
    B_magnitude_minus = np.sqrt(B_R_minus**2 + B_T_minus**2 + B_Z_minus**2)

    d2B_dZ2 = (
        (1) * B_magnitude_minus + (-2) * B_magnitude_0 + (1) * B_magnitude_plus
    ) / (delta_Z**2)
    return d2B_dZ2


# def find_d2B_dR_dZ_CFD(q_R, q_Z, delta_R, delta_Z, find_B_R, find_B_T, find_B_Z):
#
#    dB_dZ_0 = find_dB_dZ_FFD(q_R,
#                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
#    dB_dZ_1 = find_dB_dZ_FFD(q_R+delta_R,
#                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
#    dB_dZ_2 = find_dB_dZ_FFD(q_R+2*delta_R,
#                             q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
#    d2B_dR_dZ = ( (-3/2)*dB_dZ_0 + (2)*dB_dZ_1 + (-1/2)*dB_dZ_2 ) / (delta_R)
#
##    return d2B_dR_dZ
#
#
#
#
#
#
#    B_plus_R_plus_Z   = find_H(q_R+delta_R, q_Z+delta_Z)
#    B_plus_R_minus_Z  = find_H(q_R+delta_R, q_Z-delta_Z)
#    B_minus_R_plus_Z  = find_H(q_R-delta_R, q_Z+delta_Z)
#    B_minus_R_minus_Z = find_H(q_R-delta_R, q_Z-delta_Z)
#    d2B_dR_dZ = (B_plus_R_plus_Z - B_plus_R_minus_Z - B_minus_R_plus_Z + B_minus_R_minus_Z) / (4 * delta_R * delta_Z)


def find_dB_dR_FFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z):
    r"""Finds :math:`\frac{d B}{d Z}`, where B is the magnitude of the B field"""
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)

    B_R_1 = np.squeeze(find_B_R(q_R + delta_R, q_Z))
    B_T_1 = np.squeeze(find_B_T(q_R + delta_R, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R + delta_R, q_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)

    B_R_2 = np.squeeze(find_B_R(q_R + 2 * delta_R, q_Z))
    B_T_2 = np.squeeze(find_B_T(q_R + 2 * delta_R, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R + 2 * delta_R, q_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)

    dB_dR = (
        (-3 / 2) * B_magnitude_0 + (2) * B_magnitude_1 + (-1 / 2) * B_magnitude_2
    ) / (delta_R)
    return dB_dR


def find_dB_dZ_FFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z):
    r"""Finds :math:`\frac{d B}{d Z}`, where B is the magnitude of the B field"""

    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)

    B_R_1 = np.squeeze(find_B_R(q_R, q_Z + delta_Z))
    B_T_1 = np.squeeze(find_B_T(q_R, q_Z + delta_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R, q_Z + delta_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)

    B_R_2 = np.squeeze(find_B_R(q_R, q_Z + 2 * delta_Z))
    B_T_2 = np.squeeze(find_B_T(q_R, q_Z + 2 * delta_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R, q_Z + 2 * delta_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)

    dB_dZ = (
        (-3 / 2) * B_magnitude_0 + (2) * B_magnitude_1 + (-1 / 2) * B_magnitude_2
    ) / (delta_Z)
    return dB_dZ


def find_d2B_dR2_FFD(q_R, q_Z, delta_R, find_B_R, find_B_T, find_B_Z):
    r"""Finds :math:`\frac{d^2 B}{d R^2}`, where B is the magnitude of the B field"""

    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)

    B_R_1 = np.squeeze(find_B_R(q_R + delta_R, q_Z))
    B_T_1 = np.squeeze(find_B_T(q_R + delta_R, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R + delta_R, q_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)

    B_R_2 = np.squeeze(find_B_R(q_R + 2 * delta_R, q_Z))
    B_T_2 = np.squeeze(find_B_T(q_R + 2 * delta_R, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R + 2 * delta_R, q_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)

    B_R_3 = np.squeeze(find_B_R(q_R + 3 * delta_R, q_Z))
    B_T_3 = np.squeeze(find_B_T(q_R + 3 * delta_R, q_Z))
    B_Z_3 = np.squeeze(find_B_Z(q_R + 3 * delta_R, q_Z))
    B_magnitude_3 = np.sqrt(B_R_3**2 + B_T_3**2 + B_Z_3**2)

    d2B_dR2 = (
        (2) * B_magnitude_0
        + (-5) * B_magnitude_1
        + (4) * B_magnitude_2
        + (-1) * B_magnitude_3
    ) / (delta_R**2)
    return d2B_dR2


def find_d2B_dZ2_FFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z):
    B_R_0 = np.squeeze(find_B_R(q_R, q_Z))
    B_T_0 = np.squeeze(find_B_T(q_R, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_R, q_Z))
    B_magnitude_0 = np.sqrt(B_R_0**2 + B_T_0**2 + B_Z_0**2)

    B_R_1 = np.squeeze(find_B_R(q_R, q_Z + delta_Z))
    B_T_1 = np.squeeze(find_B_T(q_R, q_Z + delta_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_R, q_Z + delta_Z))
    B_magnitude_1 = np.sqrt(B_R_1**2 + B_T_1**2 + B_Z_1**2)

    B_R_2 = np.squeeze(find_B_R(q_R, q_Z + 2 * delta_Z))
    B_T_2 = np.squeeze(find_B_T(q_R, q_Z + 2 * delta_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_R, q_Z + 2 * delta_Z))
    B_magnitude_2 = np.sqrt(B_R_2**2 + B_T_2**2 + B_Z_2**2)

    B_R_3 = np.squeeze(find_B_R(q_R, q_Z + 3 * delta_Z))
    B_T_3 = np.squeeze(find_B_T(q_R, q_Z + 3 * delta_Z))
    B_Z_3 = np.squeeze(find_B_Z(q_R, q_Z + 3 * delta_Z))
    B_magnitude_3 = np.sqrt(B_R_3**2 + B_T_3**2 + B_Z_3**2)

    d2B_dZ2 = (
        (2) * B_magnitude_0
        + (-5) * B_magnitude_1
        + (4) * B_magnitude_2
        + (-1) * B_magnitude_3
    ) / (delta_Z**2)

    return d2B_dZ2


def find_d2B_dR_dZ_FFD(q_R, q_Z, delta_R, delta_Z, find_B_R, find_B_T, find_B_Z):
    dB_dZ_0 = find_dB_dZ_FFD(q_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
    dB_dZ_1 = find_dB_dZ_FFD(q_R + delta_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z)
    dB_dZ_2 = find_dB_dZ_FFD(
        q_R + 2 * delta_R, q_Z, delta_Z, find_B_R, find_B_T, find_B_Z
    )
    d2B_dR_dZ = ((-3 / 2) * dB_dZ_0 + (2) * dB_dZ_1 + (-1 / 2) * dB_dZ_2) / (delta_R)

    return d2B_dR_dZ


# ----------------------------------


# Functions (Launch angle)
"""
Written by Neal Crocker, added to Scotty by Valerian.
Converts mirror angles of the MAST DBS to launch angles (genray)
Genray -> Scotty/Torbeam: poloidal and toroidal launch angles have opposite signs
"""


def use_deg_args(func, *args, **kwargs):
    "transform args from degrees to radians before calling"  # doc string
    return func(*[a * np.pi / 180 for a in args], **kwargs)


def tilt_trns_RZ_make(t):
    "tilts R ([1,0,0]) in to Z ([0,0,1]) using angle in radians"  # doc string
    ctilt = np.cos(t)
    stilt = np.sin(t)
    return np.array([[ctilt, 0, stilt], [0, 1, 0], [-stilt, 0, ctilt]])


def tilt_trns_TZ_make(t):
    "tilts T ([1,0,0]) in to Z ([0,0,1]) using angle in radians"  # doc string
    ctilt = np.cos(t)
    stilt = np.sin(t)
    return np.array([[1, 0, 0], [0, ctilt, -stilt], [0, stilt, ctilt]])


def rot_trns_make(r):
    "rotates R ([1,0,0]) into T ([0,1,0]) using angle in radians"  # doc string
    crot = np.cos(r)
    srot = np.sin(r)
    return np.array([[crot, -srot, 0], [srot, crot, 0], [0, 0, 1]])


def mirrornorm_make_with_rot_tilt_operators(r, t):
    "makes norm vector for mirror using rot and tilt operators given angles in radians"  # doc string
    mirrornorm0 = np.array([0, 1, 0])
    tilt_trns = tilt_trns_TZ_make(t)
    rot_trns = rot_trns_make(np.pi / 4 + r)
    mirrornorm = rot_trns @ (tilt_trns @ mirrornorm0)
    return mirrornorm


def mirrornorm_make(r, t):
    "makes norm vector for mirror directly using angles in radians"  # doc string
    mirrornorm = np.array(
        [
            *(np.cos(t) * np.array([-np.sin(r + np.pi / 4), np.cos(r + np.pi / 4)])),
            np.sin(t),
        ]
    )
    return mirrornorm


def reflector_make_from_mirrornorm(mirrornorm):
    "makes relector matrix for mirror using norm vector for mirror"  # doc string
    reflector = np.eye(3) - 2 * mirrornorm[:, np.newaxis] @ mirrornorm[np.newaxis, :]
    return reflector


def reflector_make(r, t):
    "makes relector matrix for mirror using angles in radians"  # doc string
    mirrornorm = mirrornorm_make(r, t)
    reflector = reflector_make_from_mirrornorm(mirrornorm)
    return reflector


def genray_angles_from_mirror_angles(
    rot_ang_deg: ArrayLike,
    tilt_ang_deg: ArrayLike,
    offset_for_window_norm_to_R: ArrayLike = 3.0,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute the GENRAY-equivalent angles from mirror angles

    ``rot_ang_deg = 0``, ``tilt_ang_deg = 0`` implies beam propagates
    in negative window normal direction.

    Can get from ``rot_ang_deg, tilt_ang_deg`` by using toroidal and
    poloidal "genray angles" (``tor_ang, pol_ang``) in IDL savefile
    log. The angles were calculated according to the logic::

        tor_ang = 3.0 - rot_ang_deg
        pol_ang = tilt_ang_deg


    Parameters
    ----------
    rot_ang_deg:
        Steering mirror rotation angle in degrees
    tilt_ang_deg:
        Steering mirror tilt angle in degrees
    offset_for_window_norm_to_R:
        Toroidal angle in degrees of window normal to major radial
        position, such that for ``rot_ang_deg = 0``, ``tilt_ang_deg =
        0`` beam would have toroidal angle given by
        ``offset_for_window_norm_to_R``.

    """

    beam0 = np.array([0, -1, 0])
    rdeg = rot_ang_deg if np.isscalar(rot_ang_deg) else rot_ang_deg.flat[0]
    tdeg = tilt_ang_deg if np.isscalar(tilt_ang_deg) else tilt_ang_deg.flat[0]
    reflector = use_deg_args(reflector_make, rdeg, tdeg)
    beam_windowframe = reflector @ beam0
    # rotate from window frame to port frame:
    beam = use_deg_args(rot_trns_make, -offset_for_window_norm_to_R) @ beam_windowframe

    tor_ang_genray = np.arctan2(
        beam[1], -beam[0]
    )  # +offset_for_window_norm_to_R*np.pi/180
    pol_ang_genray = np.arctan2(beam[2], np.sqrt(beam[0] ** 2 + beam[1] ** 2))
    tor_ang_genray_deg, pol_ang_genray_deg = (
        tor_ang_genray * 180 / np.pi,
        pol_ang_genray * 180 / np.pi,
    )

    return tor_ang_genray_deg, pol_ang_genray_deg


# ----------------------------------


def make_array_3x3(array):
    r"""Convert a 2x2 array into a 3x3 by appending zeros on the outside:

    .. math::

        \begin{pmatrix}
            a & b \\
            c & d \\
        \end{pmatrix}
        \Rightarrow
        \begin{pmatrix}
            a & b & 0 \\
            c & d & 0 \\
            0 & 0 & 0 \\
        \end{pmatrix}
    """
    if array.shape != (2, 2):
        raise ValueError(f"Expected array shape to be (2, 2), got {array.shape}")

    return np.append(np.append(array, [[0, 0]], axis=0), [[0], [0], [0]], axis=1)


def K_magnitude(
    K_R: ArrayLike, K_zeta: ArrayLike, K_Z: ArrayLike, q_R: ArrayLike
) -> ArrayLike:
    r"""Returns the magnitude of the wavevector :math:`\mathbf{K}`:

    .. math::

        |K| = \sqrt{\mathbf{K}\cdot\mathbf{K}}
            = \sqrt{K_R^2 + (K_\zeta / q_R)^2 + K_Z^2}
    """
    return np.sqrt(K_R**2 + (K_zeta / q_R) ** 2 + K_Z**2)
