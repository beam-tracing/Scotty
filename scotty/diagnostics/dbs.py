# -*- coding: utf-8 -*-
# Copyright 2018 - 2026, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Doppler backscattering (DBS): the filter functions and the backscattered
wavenumbers along the beam, stored in the output group ``diagnostics/dbs``.
Moved from `scotty.analysis`"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import constants as const
from scipy.integrate import trapezoid
import xarray as xr

from scotty.antenna.gaussian import find_waist
from scotty.checks import VALID_FIELDS
from scotty.data_io import CARTESIAN_VECTOR_COMPONENTS
from scotty.fun_general import (
    dot,
    find_electron_mass,
    make_unit_vector_from_cross_product,
)
from scotty.hamiltonian import Hamiltonian
from scotty.profile_fit import ProfileFitLike

log = logging.getLogger(__name__)


def dbs_analysis(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    hamiltonian: Hamiltonian,
    hamiltonian_other: Hamiltonian,
    field: VALID_FIELDS,
    density_fit: ProfileFitLike,
    temperature_fit: Optional[ProfileFitLike],
) -> Tuple[dict, dict]:

    log.info(f"""\n
    ##################################################
    #
    # DOPPLER BACKSCATTERING ANALYSIS ROUTINE
    #
    ##################################################
    """)

    log.debug(f"Performing analysis on ray-tracing results")

    cart = bool(inputs["geometry"] == "cartesian")
    btf = not bool(inputs["ray_tracing_flag"])
    tau = np.array(solver_output["tau"][()])
    len_tau = len(tau)
    beam_path_length = np.array(analysis["beam_path_length"][()])
    dbs_analysis_dict = {}

    ##################################################
    log.debug("Find cos(theta) and d(theta)/d(tau)")
    ##################################################

    y_hat_cart = np.array(analysis["y_hat_cartesian"][()])  # (N,3)
    g_hat_cart = np.array(analysis["g_hat_cartesian"][()])  # (N,3)
    x_hat_cart = np.array(analysis["x_hat_cartesian"][()])  # (N,3)
    b_hat_cart = np.array(analysis["b_hat_cartesian"][()])  # (N,3)
    k_perp1_hat = make_unit_vector_from_cross_product(y_hat_cart, b_hat_cart)
    sin_theta = -dot(x_hat_cart, k_perp1_hat)
    theta = np.sign(sin_theta) * np.arcsin(np.abs(sin_theta))
    cos_theta = np.cos(theta)

    dbs_analysis_dict.update(
        {
            "k_perp1_hat": (["tau", "col_cart"], k_perp1_hat),
            "sin_theta": (["tau"], sin_theta),
            "cos_theta": (["tau"], cos_theta),
            "theta": (["tau"], theta),  # in radians
            "d_theta_d_tau": (["tau"], np.gradient(theta, tau)),
        }
    )

    ##################################################
    log.debug("Find k_perp_bs components")
    ##################################################

    theta_m = np.array(analysis["theta_m"][()])  # (N,)
    dpolflux_dX = np.array(analysis["dpolflux_dX"][()])  # (N,)
    dpolflux_dY = np.array(analysis["dpolflux_dY"][()])  # (N,)
    dpolflux_dZ = np.array(analysis["dpolflux_dZ"][()])  # (N,)
    K_mag = np.array(solver_output["K_output_magnitude"][()])  # (N,)
    normal_vec = np.stack((dpolflux_dX, dpolflux_dY, dpolflux_dZ), axis=1)  # (N,3)
    normal_mag = np.linalg.norm(normal_vec, axis=1)  # (N,)
    normal_hat = normal_vec / normal_mag[:, np.newaxis]  # (N,3)
    binormal_hat = make_unit_vector_from_cross_product(b_hat_cart, normal_hat)  # (N,3)
    k_perp1_bs = -2 * K_mag * np.cos(theta + theta_m) / cos_theta  # (N,)
    # normal_hat and binormal_hat span the plane perpendicular to b_hat, which
    # contains k_perp1_hat, so these add in quadrature to k_perp1_bs; tested in
    # tests/test_analysis_outputs.py
    k_perp1_bs_normal = k_perp1_bs * dot(k_perp1_hat, normal_hat)
    k_perp1_bs_binormal = k_perp1_bs * dot(k_perp1_hat, binormal_hat)

    dbs_analysis_dict.update(
        {
            "normal_hat": (["tau", "col_cart"], normal_hat),
            "binormal_hat": (["tau", "col_cart"], binormal_hat),
            "k_perp1_bs": (["tau"], k_perp1_bs),
            "k_perp1_bs_normal": (["tau"], k_perp1_bs_normal),
            "k_perp1_bs_binormal": (["tau"], k_perp1_bs_binormal),
        }
    )

    ##################################################
    log.debug("Find grad_bhat stuff")
    ##################################################

    grad_bhat = np.array(analysis["grad_bhat_cartesian"][()])  # (N,3,3)
    xhat_dot_grad_bhat_dot_xhat = dot(x_hat_cart, dot(grad_bhat, x_hat_cart))
    xhat_dot_grad_bhat_dot_yhat = dot(x_hat_cart, dot(grad_bhat, y_hat_cart))
    xhat_dot_grad_bhat_dot_ghat = dot(x_hat_cart, dot(grad_bhat, g_hat_cart))
    yhat_dot_grad_bhat_dot_xhat = dot(y_hat_cart, dot(grad_bhat, x_hat_cart))
    yhat_dot_grad_bhat_dot_yhat = dot(y_hat_cart, dot(grad_bhat, y_hat_cart))
    yhat_dot_grad_bhat_dot_ghat = dot(y_hat_cart, dot(grad_bhat, g_hat_cart))

    dbs_analysis_dict.update(
        {
            "xhat_dot_grad_bhat_dot_xhat": (["tau"], xhat_dot_grad_bhat_dot_xhat),
            "xhat_dot_grad_bhat_dot_yhat": (["tau"], xhat_dot_grad_bhat_dot_yhat),
            "xhat_dot_grad_bhat_dot_ghat": (["tau"], xhat_dot_grad_bhat_dot_ghat),
            "yhat_dot_grad_bhat_dot_xhat": (["tau"], yhat_dot_grad_bhat_dot_xhat),
            "yhat_dot_grad_bhat_dot_yhat": (["tau"], yhat_dot_grad_bhat_dot_yhat),
            "yhat_dot_grad_bhat_dot_ghat": (["tau"], yhat_dot_grad_bhat_dot_ghat),
        }
    )

    if btf:
        ##################################################
        log.debug("Find wavenumber resolution and mismatch attenuation")
        ##################################################

        Psi_xx = np.array(analysis["Psi_xx"][()])
        Psi_xy = np.array(analysis["Psi_xy"][()])
        Psi_yy = np.array(analysis["Psi_yy"][()])
        M_xx = Psi_xx + (k_perp1_bs / 2) * xhat_dot_grad_bhat_dot_ghat
        M_xy = Psi_xy + (k_perp1_bs / 2) * yhat_dot_grad_bhat_dot_ghat
        M_yy = Psi_yy
        det_M_w = M_xx * M_yy - M_xy**2
        M_w_inv_xx = M_yy / det_M_w
        M_w_inv_xy = -M_xy / det_M_w
        M_w_inv_yy = M_xx / det_M_w
        delta_k_perp2 = 2 * np.sqrt(-1 / np.imag(M_w_inv_yy))
        # The k_perp2 that is backscattered, where the exponent of the k_perp2
        # integrand, -2 Im(M_xy^-1) K theta_m k_perp2 + Im(M_yy^-1) k_perp2^2 / 2,
        # peaks; zero at zero mismatch. It's along u2 = b_hat x g_hat / |b_hat x
        # g_hat|, which is y_hat, as k_perp1 is along k_perp1_hat (u1)
        k_perp2_bs = 2 * K_mag * theta_m * np.imag(M_w_inv_xy) / np.imag(M_w_inv_yy)
        k_perp2_bs_normal = k_perp2_bs * dot(y_hat_cart, normal_hat)
        k_perp2_bs_binormal = k_perp2_bs * dot(y_hat_cart, binormal_hat)

        dbs_analysis_dict.update(
            {
                "M_xx": (["tau"], M_xx),
                "M_xy": (["tau"], M_xy),
                "M_yy": (["tau"], M_yy),
                "det_M_w": (["tau"], det_M_w),
                "M_w_inv_xx": (["tau"], M_w_inv_xx),
                "M_w_inv_xy": (["tau"], M_w_inv_xy),
                "M_w_inv_yy": (["tau"], M_w_inv_yy),
                "delta_k_perp2": (["tau"], delta_k_perp2),
                "k_perp2_bs": (["tau"], k_perp2_bs),
                "k_perp2_bs_normal": (["tau"], k_perp2_bs_normal),
                "k_perp2_bs_binormal": (["tau"], k_perp2_bs_binormal),
            }
        )

    else:
        log.debug(
            "No beam-tracing results; skipping wavenumber resolution and mismatch attenuation"
        )

    ##################################################
    log.debug(f"Finding the polarisation piece (filterfun_p)")
    ##################################################

    zero = np.zeros(len_tau, dtype=np.float64)
    epsilon_para = np.array(analysis["epsilon_para"][()])
    epsilon_perp = np.array(analysis["epsilon_perp"][()])
    epsilon_g = np.array(analysis["epsilon_g"][()])
    e_hat = np.array(analysis["e_hat"][()])
    e_hat_conj = np.conjugate(e_hat)
    n_e = np.array(analysis["electron_density"][()])
    w_launch = hamiltonian.angular_frequency
    temperature = analysis.get("electron_temperature")
    if temperature is not None:
        # a plain array, like the inputs above, so that filterfun_p is not a DataArray
        temperature = np.array(temperature[()])

    # equilibrium dielectric tensor - identity matrix, i.e.
    # \bm{\epsilon}_{eq} - \bm{1}
    epsilon_minus_identity = np.block(
        [
            [[epsilon_perp], [1j * epsilon_g], [zero]],
            [[-1j * epsilon_g], [epsilon_perp], [zero]],
            [[zero], [zero], [epsilon_para]],
        ]
    ).T - np.eye(3)

    # Avoids dividing a small number by another small number,
    # leading to a big number because of numerical errors or something
    filterfun_p_unnormalised = np.divide(
        np.abs(dot(e_hat_conj, dot(epsilon_minus_identity, e_hat))) ** 2,
        (n_e * 1e19) ** 2,
        out=np.zeros_like(n_e),
        where=(n_e > 1e-6),
    )

    filterfun_p = (
        w_launch**2 * const.epsilon_0 * find_electron_mass(temperature) / const.e**2
    ) ** 2 * filterfun_p_unnormalised

    dbs_analysis_dict.update(
        {
            "filterfun_p": (["tau"], filterfun_p),
            "int_filterfun_p_dl": ([], trapezoid(filterfun_p, beam_path_length)),
        }
    )

    ##################################################
    log.debug(f"Finding the ray piece (filterfun_r)")
    ##################################################

    g_mag_Cardano = np.array(analysis["g_magnitude_Cardano"][()])
    filterfun_r = (2 * const.c / w_launch) ** 2 / g_mag_Cardano**2
    # needed by filterfun_b and delta_ne_sq; delta_ne_sq is also computed for ray tracing only
    launch_wavenumber = hamiltonian.wavenumber

    dbs_analysis_dict.update(
        {
            "filterfun_r": (["tau"], filterfun_r),
            "int_filterfun_r_dl": ([], trapezoid(filterfun_r, beam_path_length)),
        }
    )

    if btf:
        ##################################################
        log.debug(f"Finding the beam piece (filterfun_b)")
        ##################################################

        det_Im_Psi_w = np.imag(Psi_xx) * np.imag(Psi_yy) - np.imag(Psi_xy) ** 2
        if str(inputs["launch_flag"].values) == "plasma":
            # filterfun_b is normalised by the waist of the beam the antenna
            # launches; a plasma launch has no antenna beam
            log.warning(
                "`launch_flag` = 'plasma': there is no launched beam, so "
                "`beam_waist`, and with it filterfun_b and filterfun_all, are NaN"
            )
            beam_waist = np.nan
        else:
            launch_beam_width = float(inputs["launch_beam_width"][()])
            launch_beam_curv = float(inputs["launch_beam_curvature"][()])
            beam_waist = find_waist(
                launch_beam_width, launch_wavenumber, launch_beam_curv
            )
        filterfun_b = (beam_waist * det_Im_Psi_w) / (
            np.sqrt(2) * np.abs(det_M_w) * np.sqrt(-np.imag(M_w_inv_yy))
        )

        dbs_analysis_dict.update(
            {
                "det_Im_Psi_w": (["tau"], det_Im_Psi_w),
                "beam_waist": beam_waist,
                "filterfun_b": (["tau"], filterfun_b),
                "int_filterfun_b_dl": ([], trapezoid(filterfun_b, beam_path_length)),
            }
        )

    else:
        log.debug("No beam-tracing results; skipping beam piece (filterfun_b)")

    if btf:
        ##################################################
        log.debug(f"Finding the mismatch piece (filterfun_m)")
        ##################################################

        delta_theta_m = (
            np.sqrt(
                np.imag(M_w_inv_yy)
                / (
                    (np.imag(M_w_inv_xy)) ** 2
                    - np.imag(M_w_inv_xx) * np.imag(M_w_inv_yy)
                )
            )
            / K_mag
        )
        filterfun_m = np.exp(-2 * (theta_m / delta_theta_m) ** 2)

        dbs_analysis_dict.update(
            {
                "delta_theta_m": (["tau"], delta_theta_m),
                "filterfun_m": (["tau"], filterfun_m),
                "int_filterfun_m_dl": ([], trapezoid(filterfun_m, beam_path_length)),
            }
        )

    else:
        log.debug("No beam-tracing results; skipping mismatch piece (filterfun_m)")

    ##################################################
    log.debug(f"Finding the model turbulence spectrum (delta_ne_sq)")
    ##################################################

    # Not a filter function: a model of the spectral density of the density
    # fluctuations at the backscattered k_perp1, a power law normalised to 1 at
    # k_perp1 = -2 K0. It multiplies the filter function in the integrand of the
    # backscattered power

    spectrum_power_laws = ["-10/3", "-13/3"]
    delta_ne_sq = {
        f"delta_ne_sq_{power}": np.array(
            (k_perp1_bs / (-2 * launch_wavenumber)) ** eval(power)
        )
        for power in spectrum_power_laws
    }

    dbs_analysis_dict.update(
        {
            "spectrum_power_laws": spectrum_power_laws,
            **{f"{k}": (["tau"], v) for k, v in delta_ne_sq.items()},
            **{
                f"int_{k}_dl": ([], trapezoid(v, beam_path_length))
                for k, v in delta_ne_sq.items()
            },
        }
    )

    if btf:
        ##################################################
        log.debug(
            "Finding the filter function (filterfun_p * filterfun_r * filterfun_b * filterfun_m)"
        )
        ##################################################

        # Without the turbulence spectrum, which is not part of the filter
        # function: the backscattered power is the integral of filterfun_all
        # times delta_ne_sq along the ray
        filterfun_all = filterfun_p * filterfun_r * filterfun_b * filterfun_m

        dbs_analysis_dict.update(
            {
                "filterfun_all": (["tau"], filterfun_all),
                "int_filterfun_all_dl": (
                    [],
                    trapezoid(filterfun_all, beam_path_length),
                ),
            }
        )

    else:
        log.debug(
            "No beam-tracing results; skipping the filter function (filterfun_p * filterfun_r * filterfun_b * filterfun_m)"
        )

    dbs_analysis_coords = {
        "tau": np.array(solver_output["tau"][()]),
        "beam_path_length": (["tau"], beam_path_length),
        "beam_path_length_wrt_cutoff": (
            ["tau"],
            np.array(analysis["beam_path_length_wrt_cutoff"][()]),
        ),
        "row_cart": CARTESIAN_VECTOR_COMPONENTS,
        "col_cart": CARTESIAN_VECTOR_COMPONENTS,
    }

    # netCDF complains if keys contian '/' so
    # iterate along the keys and replace those with '_'
    def _replace_slash_with_underscore(d: dict) -> dict:
        return {k.replace("/", "_"): v for k, v in d.items()}

    dbs_analysis_dict_clean = _replace_slash_with_underscore(dbs_analysis_dict)

    return dbs_analysis_dict_clean, dbs_analysis_coords
