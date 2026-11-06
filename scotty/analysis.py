# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

import logging
import numpy as np
from scotty.checks import VALID_FIELDS, check_H_output
from scotty.data_io import CARTESIAN_VECTOR_COMPONENTS, SOLUTION_COMPONENTS
from scotty.derivatives import derivative
from scotty.fun_general import (
    find_H_Cardano_eigh,
    find_H_Cardano_formula,
    find_beam_widths_and_curvs,
    find_g_cartesian,
    find_normalised_angular_freqs,
    make_unit_vector_from_cross_product,
    dot,
)
from scotty.hamiltonian import DielectricTensor, Hamiltonian, convert_hessians, hessians
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike, FloatArray
from typing import Optional, Tuple
import xarray as xr

log = logging.getLogger(__name__)

# e_hat and the eigenvectors of D are in the field-aligned frame (u1, u2, b) in
# which D is written, eqs. (34)-(41) of Hall-Chen, Parra & Hillesheim (2022):
# u1 is the unit vector perpendicular to b_hat in the plane of g and b_hat
# (which also contains K), and u2 = b_hat x g / |b_hat x g| is perpendicular
# to that plane. This is neither the Cartesian lab frame (col_cart) nor the
# beam frame (x, y, g)
FIELD_ALIGNED_COMPONENTS = ["u1", "u2", "b"]

# Transverse directions of the beam frame (x, y, g), for Psi_w
BEAM_TRANSVERSE_COMPONENTS = ["x", "y"]


def find_mode_index_Cardano(abs_H_Cardano: FloatArray) -> int:
    """The Cardano root that the ray follows: the one with the smallest total
    :math:`|H|` along it. `abs_H_Cardano` has shape (3, N).

    Near the plasma edge, two of the roots are close together, and the step
    error of the solvers can leave the ray nearer the other one at some points
    without the ray having changed root, so these are warned about, not fatal.
    `scotty.checks.check_H_output` then checks the size of :math:`|H|` on the
    root chosen, which catches a ray that really has left it.

    A NaN counts as far from zero: only the roots with the fewest NaNs are
    considered, so that a root isn't favoured for having fewer points in its total.
    """
    nan_counts = np.count_nonzero(np.isnan(abs_H_Cardano), axis=1)
    totals = np.nansum(abs_H_Cardano, axis=1)
    mode_index = int(np.argmin(np.where(nan_counts > nan_counts.min(), np.inf, totals)))
    abs_H = np.nan_to_num(abs_H_Cardano, nan=np.inf)
    elsewhere = np.argmin(abs_H, axis=0) != mode_index
    if np.any(elsewhere):
        log.warning(
            f"{np.count_nonzero(elsewhere)} of {elsewhere.size} points along the ray "
            f"are closer to another root of H_Cardano than to root {mode_index}, "
            f"where |H| on this root is up to {np.max(abs_H[mode_index, elsewhere]):.2e}. "
            "If that is small, it is step error near the plasma edge: consider a "
            "tighter `rtol`"
        )
    return mode_index


def basic_analysis(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    hamiltonian: Hamiltonian,
    hamiltonian_other: Hamiltonian,
    field: VALID_FIELDS,
    density_fit: ProfileFitLike,
    temperature_fit: Optional[ProfileFitLike],
) -> Tuple[dict, dict]:
    r"""Unless stated otherwise, all calculations are done in
    cartesian coordinates"""

    log.info(f"""\n
    ##################################################
    #
    # BASIC ANALYSIS ROUTINE
    #
    ##################################################
    """)

    log.debug(f"Performing analysis on ray-tracing results")

    cart = bool(inputs["geometry"] == "cartesian")
    btf = not bool(inputs["ray_tracing_flag"])
    tau = np.array(solver_output["tau"][()])
    len_tau = len(tau)
    analysis_dict = {}

    ##################################################
    log.debug("Position and wavevector stuff")
    ##################################################

    q_vec_cart = np.array(solver_output["q_output_cartesian"][()])  # (N,3)
    q_vec_cyld = np.array(solver_output["q_output_cylindrical"][()])  # (N,3)
    K_vec_cart = np.array(solver_output["K_output_cartesian"][()])  # (N,3)
    K_vec_cyld = np.array(solver_output["K_output_cylindrical"][()])  # (N,3)
    K_mag = np.array(solver_output["K_output_magnitude"][()])  # (N,)

    coidx = np.argmin(K_mag)
    q_X, q_Y, q_Z = q_vec_cart.T
    dL_along_beam_path = np.sqrt(
        np.diff(q_X) ** 2 + np.diff(q_Y) ** 2 + np.diff(q_Z) ** 2
    )
    beam_path_length = np.append(0, np.cumsum(dL_along_beam_path))
    beam_path_length_wrt_cutoff = beam_path_length - beam_path_length[coidx]

    analysis_dict.update(
        {
            "beam_path_length": (["tau"], beam_path_length),
            "beam_path_length_wrt_cutoff": (["tau"], beam_path_length_wrt_cutoff),
            "cutoff_index": ([], coidx),
        }
    )

    ##################################################
    log.debug("Booker Hamiltonian stuff")
    ##################################################

    if cart:
        q_vec, K_vec = q_vec_cart, K_vec_cart  # (N,3) each
    else:
        q_vec, K_vec = q_vec_cyld, K_vec_cyld  # (N,3) each
    q_vec_T, K_vec_T = q_vec.T, K_vec.T
    H_Booker = hamiltonian(*q_vec_T, *K_vec_T)
    check_H_output(H_Booker)

    dH = hamiltonian.derivatives(q_vec_T, K_vec_T, second_order=btf)

    analysis_dict.update(
        {
            "H_Booker": (["tau"], H_Booker),  # (N,) # type: ignore
            "H_Booker_other": (
                ["tau"],
                hamiltonian_other(*q_vec_T, *K_vec_T),
            ),  # (N,) # type: ignore
        }
    )

    ##################################################
    log.debug("Magnetic field and poloidal flux stuff")
    ##################################################

    if cart:
        # field.all?
        B_X = field.B_X(*q_vec_T)  # (N,)
        B_Y = field.B_Y(*q_vec_T)  # (N,)
        B_Z = field.B_Z(*q_vec_T)  # (N,)
        B_mag = field.magnitude(*q_vec_T)  # (N,)
        b_hat_cart = field.unitvector(*q_vec_T)  # (N,3)
        dbhat_dX = derivative(
            field.unitvector_cartesian,
            dims="X",
            args={"X": q_vec_T[0], "Y": q_vec_T[1], "Z": q_vec_T[2]},
            spacings=hamiltonian.delta_X,
        )  # (N,3)
        dbhat_dY = derivative(
            field.unitvector_cartesian,
            dims="Y",
            args={"X": q_vec_T[0], "Y": q_vec_T[1], "Z": q_vec_T[2]},
            spacings=hamiltonian.delta_Y,
        )  # (N,3)
        dbhat_dZ = derivative(
            field.unitvector_cartesian,
            dims="Z",
            args={"X": q_vec_T[0], "Y": q_vec_T[1], "Z": q_vec_T[2]},
            spacings=hamiltonian.delta_Z,
        )  # (N,3)
        grad_bhat_cart = np.stack((dbhat_dX, dbhat_dY, dbhat_dZ), axis=1)  # (N,3,3)

        polflux = field.polflux(*q_vec_T)  # (N,)
        dp_dX = field.d_polflux_dX(*q_vec_T, hamiltonian.delta_X)  # (N,) # type: ignore
        dp_dY = field.d_polflux_dY(*q_vec_T, hamiltonian.delta_Y)  # (N,) # type: ignore
        dp_dZ = field.d_polflux_dZ(*q_vec_T, hamiltonian.delta_Z)  # (N,) # type: ignore
    else:
        # B_R = field.B_R(*q_vec_T)
        # B_T = field.B_T(*q_vec_T)
        # B_Z = field.B_Z(*q_vec_T)
        # B_mag = field.magnitude(*q_vec_T)
        # b_hat = field.unitvector(*q_vec_T)
        # dbhat_dR = derivative(field.unitvector, dims="R", args={"R": q_vec_T[0], "_": q_vec_T[1], "Z": q_vec_T[2]}, spacings=hamiltonian.delta_R)
        # dbhat_dZ = derivative(field.unitvector, dims="Z", args={"R": q_vec_T[0], "_": q_vec_T[1], "Z": q_vec_T[2]}, spacings=hamiltonian.delta_Z)
        # grad_bhat = np.zeros([len_tau, 3, 3])
        # grad_bhat[:,0,:] = dbhat_dR
        # grad_bhat[:,2,:] = dbhat_dZ
        # grad_bhat[:,1,0] = -B_T / (B_mag * q_vec_T[0])
        # grad_bhat[:,1,1] = B_R / (B_mag * q_vec_T[0])
        B_X = field.B_X(*q_vec_T)
        B_Y = field.B_Y(*q_vec_T)
        B_Z = field.B_Z(*q_vec_T)
        B_mag = field.magnitude(*q_vec_T)
        # unitvector_cartesian takes Cartesian (X, Y, Z), not the
        # cylindrical (R, zeta, Z) in q_vec_T
        b_hat_cart = field.unitvector_cartesian(*q_vec_cart.T)
        dbhat_dX = derivative(
            field.unitvector_cartesian,
            dims="X",
            args={"X": q_vec_cart.T[0], "Y": q_vec_cart.T[1], "Z": q_vec_cart.T[2]},
            spacings=hamiltonian.delta_R,
        )
        dbhat_dY = derivative(
            field.unitvector_cartesian,
            dims="Y",
            args={"X": q_vec_cart.T[0], "Y": q_vec_cart.T[1], "Z": q_vec_cart.T[2]},
            spacings=hamiltonian.delta_R,
        )
        dbhat_dZ = derivative(
            field.unitvector_cartesian,
            dims="Z",
            args={"X": q_vec_cart.T[0], "Y": q_vec_cart.T[1], "Z": q_vec_cart.T[2]},
            spacings=hamiltonian.delta_Z,
        )
        grad_bhat_cart = np.stack((dbhat_dX, dbhat_dY, dbhat_dZ), axis=1)

        polflux = field.polflux(*q_vec_T)
        dp_dX = field.d_polflux_dX(*q_vec_T, hamiltonian.delta_R)  # type: ignore
        dp_dY = field.d_polflux_dY(*q_vec_T, hamiltonian.delta_R)  # type: ignore
        dp_dZ = field.d_polflux_dZ(*q_vec_T, hamiltonian.delta_Z)  # type: ignore

    analysis_dict.update(
        {
            "B_X": (["tau"], B_X),
            "B_Y": (["tau"], B_Y),
            "B_Z": (["tau"], B_Z),
            "B_magnitude": (["tau"], B_mag),
            "b_hat_cartesian": (["tau", "col_cart"], b_hat_cart),
            "dbhat_dX": (["tau", "row_cart"], dbhat_dX),
            "dbhat_dY": (["tau", "row_cart"], dbhat_dY),
            "dbhat_dZ": (["tau", "row_cart"], dbhat_dZ),
            "grad_bhat_cartesian": (["tau", "row_cart", "col_cart"], grad_bhat_cart),
            "polflux": (["tau"], polflux),
            "dpolflux_dX": (["tau"], dp_dX),
            "dpolflux_dY": (["tau"], dp_dY),
            "dpolflux_dZ": (["tau"], dp_dZ),
        }
    )

    ##################################################
    log.debug("Plasma properties along path")
    ##################################################

    n_e = density_fit(polflux)  # (N,)
    T_e = temperature_fit(polflux) if temperature_fit else None
    omega_launch = float(inputs["launch_angular_frequency"][()])
    epsilon = DielectricTensor(
        launch_angular_freq=omega_launch,
        B_magnitude=B_mag,
        electron_density=n_e,
        temperature=T_e,
    )

    (
        norm_omega_pe,
        norm_omega_ce,
        norm_omega_LH,
        norm_omega_RH,
        norm_omega_UH,
    ) = find_normalised_angular_freqs(
        launch_angular_freq=omega_launch,
        B_total=B_mag,
        electron_density=n_e,
        temperature=T_e,
    )

    analysis_dict.update(
        {
            "electron_density": (["tau"], n_e),
            **({"electron_temperature": (["tau"], T_e)} if T_e is not None else {}),
            "epsilon_bb": (["tau"], epsilon.e_bb),
            "epsilon_para": (["tau"], epsilon.e_bb),
            "epsilon_11": (["tau"], epsilon.e_11),
            "epsilon_perp": (["tau"], epsilon.e_11),
            "epsilon_12": (["tau"], epsilon.e_12),
            "epsilon_g": (["tau"], epsilon.e_12),
            "normalised_omega_pe": (["tau"], norm_omega_pe),
            "normalised_omega_ce": (["tau"], norm_omega_ce),
            "normalised_omega_LH": (["tau"], norm_omega_LH),
            "normalised_omega_RH": (["tau"], norm_omega_RH),
            "normalised_omega_UH": (["tau"], norm_omega_UH),
        }
    )

    ##################################################
    log.debug("Ray properties along path")
    ##################################################

    sin_theta_m = np.sum(b_hat_cart * K_vec_cart / K_mag[:, np.newaxis], axis=1)
    theta_m = np.sign(sin_theta_m) * np.arcsin(np.abs(sin_theta_m))

    g_vec_cart, g_mag, g_hat_cart = find_g_cartesian(cart, *q_vec_T, dH)  # type: ignore
    y_hat_cart = make_unit_vector_from_cross_product(b_hat_cart, g_hat_cart)
    x_hat_cart = make_unit_vector_from_cross_product(y_hat_cart, g_hat_cart)
    d_xhat_d_tau = np.gradient(x_hat_cart, tau, axis=0)
    d_xhat_d_tau_dot_yhat = dot(d_xhat_d_tau, y_hat_cart)

    d_ghat_d_tau = np.gradient(g_hat_cart, tau, axis=0)
    ray_curvature_kappa = d_ghat_d_tau / g_mag[:, None]
    kappa_dot_xhat = dot(ray_curvature_kappa, x_hat_cart)
    kappa_dot_yhat = dot(ray_curvature_kappa, y_hat_cart)
    kappa_dot_ghat = dot(
        ray_curvature_kappa, g_hat_cart
    )  # this should be zero; good to check

    analysis_dict.update(
        {
            "sin_theta_m": (["tau"], sin_theta_m),
            "theta_m": (["tau"], theta_m),
            "g_vector_cartesian": (["tau", "col_cart"], g_vec_cart),
            "g_magnitude": (["tau"], g_mag),
            "g_hat_cartesian": (["tau", "col_cart"], g_hat_cart),
            "y_hat_cartesian": (["tau", "col_cart"], y_hat_cart),
            "x_hat_cartesian": (["tau", "col_cart"], x_hat_cart),
            "d_xhat_d_tau": (["tau", "col_cart"], d_xhat_d_tau),
            "d_xhat_d_tau_dot_yhat": (["tau"], d_xhat_d_tau_dot_yhat),
            "d_ghat_d_tau": (["tau", "col_cart"], d_ghat_d_tau),
            "ray_curvature_kappa": (["tau", "col_cart"], ray_curvature_kappa),
            "ray_curvature_kappa_magnitude": (
                ["tau"],
                np.linalg.norm(ray_curvature_kappa, axis=1),
            ),
            "kappa_dot_xhat": (["tau"], kappa_dot_xhat),
            "kappa_dot_yhat": (["tau"], kappa_dot_yhat),
            "kappa_dot_ghat": (["tau"], kappa_dot_ghat),
        }
    )

    ##################################################
    log.debug("H_Cardano dispersion relation (analytic)")
    ##################################################

    H_Cardano_1, H_Cardano_2, H_Cardano_3 = find_H_Cardano_formula(
        launch_angular_freq=omega_launch,
        K_magnitude=K_mag,
        epsilon_para=epsilon.e_bb,
        epsilon_perp=epsilon.e_11,
        epsilon_g=epsilon.e_12,
        theta_m=theta_m,
    )

    # The root is the one that vanishes along the ray, judged away from its
    # ends. In vacuum (zero density) two of the roots vanish together, so
    # which of them is smallest there is down to rounding: judge on the
    # points in the plasma. With `launch_flag` = "vacuum_numerical", much of
    # the ray is in vacuum
    middle = np.zeros(len_tau, dtype=bool)
    middle[len_tau // 10 : -len_tau // 10] = True
    in_plasma = middle & (np.asarray(n_e) > 0)
    judged = in_plasma if np.any(in_plasma) else middle
    mode_index_Cardano = find_mode_index_Cardano(
        np.abs(np.stack([H_Cardano_1, H_Cardano_2, H_Cardano_3]))[:, judged]
    )
    H_Cardano_solution = (H_Cardano_1, H_Cardano_2, H_Cardano_3)[mode_index_Cardano]

    check_H_output(H_Cardano_solution)

    def _H_Cardano(K_X: ArrayLike, K_Y: ArrayLike, K_Z: ArrayLike) -> ArrayLike:
        # theta_m depends on the direction of K, so it must be recomputed
        # from the K being evaluated here. Holding it at its value on the
        # ray would only differentiate H through |K|, making grad_K H
        # parallel to K instead of to the group velocity
        K_magnitude_trial = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
        sin_theta_m_trial = (
            b_hat_cart[:, 0] * K_X + b_hat_cart[:, 1] * K_Y + b_hat_cart[:, 2] * K_Z
        ) / K_magnitude_trial
        # The Cardano formula works in complex arithmetic, but its roots are
        # eigenvalues of a Hermitian matrix: any imaginary part is rounding.
        # Without np.real it made g_magnitude_Cardano, filterfun_r and filterfun_all
        # complex-typed
        return np.real(
            find_H_Cardano_formula(
                launch_angular_freq=omega_launch,
                K_magnitude=K_magnitude_trial,
                epsilon_para=epsilon.e_bb,
                epsilon_perp=epsilon.e_11,
                epsilon_g=epsilon.e_12,
                theta_m=np.arcsin(sin_theta_m_trial),
            )[mode_index_Cardano]
        )

    def _grad_H_Cardano(direction: str, spacing: float) -> ArrayLike:
        return derivative(
            func=_H_Cardano,
            dims=direction,
            args={
                "K_X": K_vec_cart.T[0],
                "K_Y": K_vec_cart.T[1],
                "K_Z": K_vec_cart.T[2],
            },
            spacings=spacing,
        )

    g_X_Cardano = _grad_H_Cardano(
        "K_X", float(inputs["delta_K_X" if cart else "delta_K_R"][()])
    )
    g_Y_Cardano = _grad_H_Cardano(
        "K_Y", float(inputs["delta_K_Y" if cart else "delta_K_zeta"][()])
    )
    g_Z_Cardano = _grad_H_Cardano("K_Z", float(inputs["delta_K_Z"][()]))
    g_magnitude_Cardano = np.sqrt(g_X_Cardano**2 + g_Y_Cardano**2 + g_Z_Cardano**2)
    g_hat_Cardano = (
        np.stack([g_X_Cardano, g_Y_Cardano, g_Z_Cardano], axis=-1)
        / g_magnitude_Cardano[:, np.newaxis]
    )

    analysis_dict.update(
        {
            # all three roots on one array, as master (v3.2) stores them
            "H_Cardano": (
                ["tau", "solution"],
                np.vstack([H_Cardano_1, H_Cardano_2, H_Cardano_3]).T,
            ),
            "H_Cardano_1": (["tau"], H_Cardano_1),
            "H_Cardano_2": (["tau"], H_Cardano_2),
            "H_Cardano_3": (["tau"], H_Cardano_3),
            "mode_index_Cardano": ([], mode_index_Cardano),
            "H_Cardano_solution": (["tau"], H_Cardano_solution),
            "g_X_Cardano": (["tau"], g_X_Cardano),
            "g_Y_Cardano": (["tau"], g_Y_Cardano),
            "g_Z_Cardano": (["tau"], g_Z_Cardano),
            "g_magnitude_Cardano": (["tau"], g_magnitude_Cardano),
            "g_hat_Cardano": (["tau", "col_cart"], g_hat_Cardano),
        }
    )

    ##################################################
    log.debug("H_eigval dispersion relation and polarisation vector")
    ##################################################

    H_eigvals, e_eigvecs = find_H_Cardano_eigh(
        launch_angular_freq=omega_launch,
        K_magnitude=K_mag,
        epsilon_para=epsilon.e_bb,
        epsilon_perp=epsilon.e_11,
        epsilon_g=epsilon.e_12,
        theta_m=theta_m,
    )

    # eigh returns its eigenvalues in ascending order, not in the Cardano
    # order that `mode_index` refers to, so at each point take the
    # eigenvector whose eigenvalue matches the Cardano solution the beam
    # follows
    mode_idx = np.argmin(
        np.abs(H_eigvals - np.asarray(H_Cardano_solution)[:, np.newaxis]), axis=1
    )
    H_solution = np.take_along_axis(H_eigvals, mode_idx[:, np.newaxis], axis=1)[:, 0]
    e_hat = np.take_along_axis(e_eigvecs, mode_idx[:, np.newaxis, np.newaxis], axis=2)[
        :, :, 0
    ]

    analysis_dict.update(
        {
            "H_eigvals": (["tau", "solution"], H_eigvals),
            "H_eigval1": (["tau"], H_eigvals[:, 0]),
            "H_eigval2": (["tau"], H_eigvals[:, 1]),
            "H_eigval3": (["tau"], H_eigvals[:, 2]),
            "H_eigval_solution": (["tau"], H_solution),
            "e_eigvecs": (["tau", "col_field", "solution"], e_eigvecs),
            "e_eigvec1": (["tau", "col_field"], e_eigvecs[:, :, 0]),
            "e_eigvec2": (["tau", "col_field"], e_eigvecs[:, :, 1]),
            "e_eigvec3": (["tau", "col_field"], e_eigvecs[:, :, 2]),
            "e_hat": (["tau", "col_field"], e_hat),
        }
    )

    if btf:
        ##################################################
        log.debug("Performing analysis on beam-tracing results")
        log.debug("Hessians and second derivatives")
        ##################################################

        if cart:
            grad_grad_H_cart, gradK_grad_H_cart, gradK_gradK_H_cart = hessians(
                dH, cartesian=cart
            )
        else:
            grad_grad_H_cyld, gradK_grad_H_cyld, gradK_gradK_H_cyld = hessians(
                dH, cartesian=cart
            )
            grad_grad_H_cart, gradK_grad_H_cart, gradK_gradK_H_cart = convert_hessians(
                q_start=q_vec_cyld.T,
                K_start=K_vec_cart.T,
                dH=dH,
                grad_grad_H_start=grad_grad_H_cyld,
                gradK_grad_H_start=gradK_grad_H_cyld,
                gradK_gradK_H_start=gradK_gradK_H_cyld,
                start="cylindrical",
                end="cartesian",
            )

        ##################################################
        log.debug("Beam matrix (Psi_w) components in the {y, g, x} basis")
        ##################################################

        Psi_3D_output_labframe_cartesian = np.array(
            solver_output["Psi_3D_output_labframe_cartesian"][()]
        )  # (N, 3, 3)
        Psi_xx_output_beamframe_cartesian = dot(
            x_hat_cart, dot(Psi_3D_output_labframe_cartesian, x_hat_cart)
        )
        Psi_xy_output_beamframe_cartesian = dot(
            x_hat_cart, dot(Psi_3D_output_labframe_cartesian, y_hat_cart)
        )
        Psi_xg_output_beamframe_cartesian = dot(
            x_hat_cart, dot(Psi_3D_output_labframe_cartesian, g_hat_cart)
        )
        Psi_yy_output_beamframe_cartesian = dot(
            y_hat_cart, dot(Psi_3D_output_labframe_cartesian, y_hat_cart)
        )
        Psi_yg_output_beamframe_cartesian = dot(
            y_hat_cart, dot(Psi_3D_output_labframe_cartesian, g_hat_cart)
        )
        Psi_gg_output_beamframe_cartesian = dot(
            g_hat_cart, dot(Psi_3D_output_labframe_cartesian, g_hat_cart)
        )

        Psi_w_output_beamframe_cartesian = np.zeros(
            (len_tau, 2, 2), dtype=np.complex128
        )  # (N, 2, 2)
        Psi_w_output_beamframe_cartesian[:, 0, 0] = Psi_xx_output_beamframe_cartesian
        Psi_w_output_beamframe_cartesian[:, 0, 1] = Psi_w_output_beamframe_cartesian[
            :, 1, 0
        ] = Psi_xy_output_beamframe_cartesian
        Psi_w_output_beamframe_cartesian[:, 1, 1] = Psi_yy_output_beamframe_cartesian
        curv1, curv2, width1, width2 = find_beam_widths_and_curvs(
            Psi_w_output_beamframe_cartesian, K_vec_cart, g_vec_cart
        )

        analysis_dict.update(
            {
                "grad_grad_H_cartesian": (
                    ["tau", "row_cart", "col_cart"],
                    grad_grad_H_cart,
                ),
                "gradK_grad_H_cartesian": (
                    ["tau", "row_cart", "col_cart"],
                    gradK_grad_H_cart,
                ),
                "gradK_gradK_H_cartesian": (
                    ["tau", "row_cart", "col_cart"],
                    gradK_gradK_H_cart,
                ),
                "Psi_xx": (["tau"], Psi_xx_output_beamframe_cartesian),
                "Psi_xy": (["tau"], Psi_xy_output_beamframe_cartesian),
                "Psi_xg": (["tau"], Psi_xg_output_beamframe_cartesian),
                "Psi_yy": (["tau"], Psi_yy_output_beamframe_cartesian),
                "Psi_yg": (["tau"], Psi_yg_output_beamframe_cartesian),
                "Psi_gg": (["tau"], Psi_gg_output_beamframe_cartesian),
                "Psi_w": (
                    ["tau", "row_beam", "col_beam"],
                    Psi_w_output_beamframe_cartesian,
                ),
                "curvature_1": (["tau"], curv1),
                "curvature_2": (["tau"], curv2),
                "width_1": (["tau"], width1),
                "width_2": (["tau"], width2),
            }
        )

    else:
        log.debug("No beam-tracing results; skipping beam-tracing analysis")

    analysis_coords = {
        "tau": np.array(solver_output["tau"][()]),
        "row_cart": CARTESIAN_VECTOR_COMPONENTS,
        "col_cart": CARTESIAN_VECTOR_COMPONENTS,
        "solution": SOLUTION_COMPONENTS,
        "col_field": FIELD_ALIGNED_COMPONENTS,
        "row_beam": BEAM_TRANSVERSE_COMPONENTS,
        "col_beam": BEAM_TRANSVERSE_COMPONENTS,
    }

    return analysis_dict, analysis_coords


def __getattr__(name):
    # `dbs_analysis` moved to `scotty.diagnostics.dbs`; this name still works.
    # Looked up when used, since `scotty.diagnostics.dbs` imports this module
    if name == "dbs_analysis":
        from scotty.diagnostics.dbs import dbs_analysis

        return dbs_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
