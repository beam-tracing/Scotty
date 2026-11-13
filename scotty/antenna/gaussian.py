# -*- coding: utf-8 -*-
# Copyright 2018 - 2026, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

r"""Gaussian beams in vacuum: the launched beam's :math:`\Psi_w`, and its
waist, Rayleigh length and propagation through vacuum"""

from typing import Tuple, cast

import numpy as np

from scotty.fun_general import find_inverse_2D, freq_GHz_to_wavenumber
from scotty.typing import ComplexFloatArray


def circular_gaussian_Psi_w(
    wavenumber: float, width: float, curvature: float
) -> ComplexFloatArray:
    r""":math:`\Psi_w` of a circular Gaussian beam with width :math:`W` and
    wavefront curvature :math:`1 / R`, :math:`\Psi_w = (K_0 / R + 2i / W^2)
    \mathbb{1}`, in the beam frame (first row and column along :math:`\hat{y}`,
    second along :math:`\hat{x}`)"""
    return (wavenumber * curvature + 2j / width**2) * np.eye(2)


def find_Rayleigh_length(waist: float, wavenumber: float):
    """
    Finds the Rayleigh length (assumes vacuum propagation and circular beam)

    Parameters
    ----------
    waist : float
    wavenumber : float

    Returns
    ----------
    rayleigh_length : float
    """
    return 0.5 * wavenumber * waist**2


def find_waist(width: float, wavenumber: float, curvature: float):
    """
    Finds the size of the waist assuming vacuum propagation and circular beam

    Parameters
    ----------
    width : float
    wavenumber : float
    curvature : float

    Returns
    ----------
    beam_waist : float
    """
    return width / np.sqrt(1 + curvature**2 * width**4 * wavenumber**2 / 4)


def find_distance_from_waist(width: float, wavenumber: float, curvature: float):
    """
    Finds how far (signed distance) you are from the waist (assumes vacuum propagation and circular beam)

    Parameters
    ----------
    width : float
    wavenumber : float
    curvature : float

    Returns
    ----------
    distance_from_waist : float
        Signed distance from the waist; negative for distance to, positive for distance
        after waist
    """
    waist_sq = find_waist(width, wavenumber, curvature) ** 2
    return np.sign(curvature) * np.sqrt(
        (width**2 - waist_sq) * waist_sq * wavenumber**2 / 4
    )


def propagate_beam(
    Psi_w_cartesian: ComplexFloatArray, propagation_distance: float, freq_GHz: float
) -> ComplexFloatArray:
    """
    Uses the vacuum solution of the beam tracing equations to propagate the beam
    (works for arbitrary Gaussian beam in vacuum)

    Parameters
    ----------
    Psi_w_cartesian : (2,2) ComplexFloatArray
    propagation_distance : float
    freq_GHz : float

    Returns
    ----------
    Psi_w_cartesian_output : (2,2) ComplexFloatArray
    """
    wavenumber_K0 = freq_GHz_to_wavenumber(freq_GHz)

    Psi_w_inv_cartesian = find_inverse_2D(Psi_w_cartesian)

    Psi_w_inv_final_cartesian = (
        propagation_distance / (wavenumber_K0) * np.eye(2) + Psi_w_inv_cartesian
    )

    Psi_w_final_cartesian = find_inverse_2D(Psi_w_inv_final_cartesian)

    return cast(ComplexFloatArray, Psi_w_final_cartesian)


def propagate_circular_beam(
    width: float, curvature: float, propagation_distance: float, freq_GHz: float
) -> Tuple[float, float]:
    """
    Directly uses the width and curvature formulae to calculate circular
    beam propagation (assumes vacuum propagation and circular beam)

    Parameters
    ----------
    width : float
    curvature : float
    propagation_distance : float
    freq_GHZ : float

    Returns
    ----------
    width : float
    curvature : float
    """
    wavenumber = freq_GHz_to_wavenumber(freq_GHz)

    w0 = find_waist(width, wavenumber, curvature)
    z0 = find_distance_from_waist(width, wavenumber, curvature)
    z_R = find_Rayleigh_length(w0, wavenumber)

    z = z0 + propagation_distance

    widths = w0 * np.sqrt(1 + (z / z_R) ** 2)
    curvatures = z / (z**2 + z_R**2)
    return widths, curvatures
