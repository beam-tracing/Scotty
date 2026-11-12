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


def _rotation_2D(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def astigmatic_gaussian_Psi_w(
    wavenumber: float,
    widths: Tuple[float, float],
    curvatures: Tuple[float, float],
    width_angle: float = 0.0,
    curvature_angle: float = 0.0,
) -> ComplexFloatArray:
    r""":math:`\Psi_w` of an elliptical or astigmatic Gaussian beam, in the
    launch beam frame :math:`(\hat{x}, \hat{y})` (see
    :func:`launch_beam_frame`), ordered ``[[Psi_xx, Psi_xy], [Psi_xy, Psi_yy]]``.

    The beam's widths are ``widths[0]`` along :math:`\hat{e}_1 = \cos\alpha_w
    \hat{x} + \sin\alpha_w \hat{y}` and ``widths[1]`` along :math:`\hat{e}_2 =
    -\sin\alpha_w \hat{x} + \cos\alpha_w \hat{y}`, with :math:`\alpha_w` =
    ``width_angle`` (radians, from :math:`\hat{x}` towards :math:`\hat{y}`,
    so positive about :math:`\hat{g}`). The curvatures :math:`1/R` (m⁻¹,
    negative for a converging beam, as ``launch_beam_curvature``) are along the
    axes rotated by ``curvature_angle`` in the same way.

    .. math::

        \mathrm{Im}\,\Psi_w = R(\alpha_w)\,\mathrm{diag}(2/W_1^2, 2/W_2^2)\,R(\alpha_w)^T,
        \quad
        \mathrm{Re}\,\Psi_w = K_0\,R(\alpha_c)\,\mathrm{diag}(1/R_1, 1/R_2)\,R(\alpha_c)^T

    Equal widths and curvatures give :func:`circular_gaussian_Psi_w`, whatever
    the angles."""
    W1, W2 = widths
    if W1 <= 0 or W2 <= 0:
        raise ValueError(f"The widths must be positive, but got {widths}")
    rot_w = _rotation_2D(width_angle)
    rot_c = _rotation_2D(curvature_angle)
    im_part = rot_w @ np.diag([2 / W1**2, 2 / W2**2]) @ rot_w.T
    re_part = wavenumber * (rot_c @ np.diag(curvatures) @ rot_c.T)
    return cast(ComplexFloatArray, re_part + 1j * im_part)


def principal_widths(Psi_w: ComplexFloatArray) -> Tuple[np.ndarray, np.ndarray]:
    r"""The principal widths :math:`W = \sqrt{2/\lambda}` of a 2×2
    :math:`\Psi_w`, from the eigenvalues :math:`\lambda` of
    :math:`\mathrm{Im}\,\Psi_w` (ascending widths), and the unit vectors along
    them (columns, in the frame :math:`\Psi_w` is given in)"""
    eigvals, eigvecs = np.linalg.eigh(np.imag(Psi_w))
    # eigh sorts the eigenvalues ascending, so the widths descending: reverse
    return np.sqrt(2 / eigvals[::-1]), eigvecs[:, ::-1]


def check_launch_Psi_w(Psi_w) -> ComplexFloatArray:
    """Checks a user's launch :math:`\\Psi_w`: 2×2, symmetric (not Hermitian),
    and :math:`\\mathrm{Im}\\,\\Psi_w` positive definite, so that the beam is
    confined in both directions. Returns it as a complex array"""
    Psi_w = np.asarray(Psi_w, dtype=complex)
    if Psi_w.shape != (2, 2):
        raise ValueError(
            f"`launch_beam_Psi_w` must have shape (2, 2), but has {Psi_w.shape}"
        )
    scale = np.abs(Psi_w).max()
    if abs(Psi_w[0, 1] - Psi_w[1, 0]) > 1e-12 * scale:
        raise ValueError(
            "`launch_beam_Psi_w` must be symmetric (Psi_xy = Psi_yx, with no "
            f"complex conjugate), but got {Psi_w}"
        )
    Psi_w = 0.5 * (Psi_w + Psi_w.T)
    if np.linalg.eigvalsh(np.imag(Psi_w)).min() <= 0:
        raise ValueError(
            "Im(`launch_beam_Psi_w`) must be positive definite (a beam confined "
            f"in both directions), but got {Psi_w}"
        )
    return cast(ComplexFloatArray, Psi_w)


def launch_beam_waist_matrix(Psi_w: ComplexFloatArray) -> np.ndarray:
    r""":math:`-2\,\mathrm{Im}(\Psi_w^{-1})`, a vacuum invariant (in a fixed
    frame) whose eigenvalues are the squared principal widths at the waist,
    if the beam has one. For a circular beam it is :math:`W_0^2 \mathbb{1}`,
    with :math:`W_0` from :func:`find_waist`"""
    return -2 * np.imag(find_inverse_2D(Psi_w))


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
