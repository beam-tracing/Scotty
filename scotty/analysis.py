import logging
import numpy as np
from scipy import constants as const
from scipy.integrate import trapezoid
from scotty.checks import VALID_FIELDS, check_H_output, check_Psi_output
from scotty.data_io import CARTESIAN_VECTOR_COMPONENTS, SOLUTION_COMPONENTS
from scotty.derivatives import derivative
from scotty.fun_general import (
    find_H_Cardano_eigh,
    find_H_Cardano_formula,
    find_beam_widths_and_curvs,
    find_g_cartesian,
    find_normalised_angular_freqs,
    find_electron_mass,
    make_unit_vector_from_cross_product,
    dot,
    find_waist,
)
from scotty.hamiltonian import DielectricTensor, Hamiltonian, convert_hessians, hessians
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike
from typing import Optional, Tuple
import xarray as xr

log = logging.getLogger(__name__)


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
            "d_xhat_d_tau": (["tau"], d_xhat_d_tau_dot_yhat),
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

    abs_H_Cardano_all = np.abs(np.stack([H_Cardano_1, H_Cardano_2, H_Cardano_3]))[
        :, len_tau // 10 : -len_tau // 10
    ]
    min_H_Cardano_all = np.argmin(abs_H_Cardano_all, axis=0)
    if np.all(min_H_Cardano_all == 0):
        mode_index_Cardano, H_Cardano_solution = 0, H_Cardano_1
    elif np.all(min_H_Cardano_all == 1):
        mode_index_Cardano, H_Cardano_solution = 1, H_Cardano_2
    elif np.all(min_H_Cardano_all == 2):
        mode_index_Cardano, H_Cardano_solution = 2, H_Cardano_3
    else:
        raise ValueError(f"Unable to select correct mode index for H_Cardano")

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
        return find_H_Cardano_formula(
            launch_angular_freq=omega_launch,
            K_magnitude=K_magnitude_trial,
            epsilon_para=epsilon.e_bb,
            epsilon_perp=epsilon.e_11,
            epsilon_g=epsilon.e_12,
            theta_m=np.arcsin(sin_theta_m_trial),
        )[mode_index_Cardano]

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
            "e_eigvecs": (["tau", "col_cart", "solution"], e_eigvecs),
            "e_eigvec1": (["tau", "col_cart"], e_eigvecs[:, :, 0]),
            "e_eigvec2": (["tau", "col_cart"], e_eigvecs[:, :, 1]),
            "e_eigvec3": (["tau", "col_cart"], e_eigvecs[:, :, 2]),
            "e_hat": (["tau", "col_cart"], e_hat),
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

        check_Psi_output(
            Psi_xg_output_beamframe_cartesian,
            Psi_yg_output_beamframe_cartesian,
            Psi_gg_output_beamframe_cartesian,
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
                # "Psi_w": (["tau", #, #], Psi_w_output_beamframe_cartesian), # TODO commented out because "col/row_cart" is 3d but Psi_w is 2d
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
    }

    return analysis_dict, analysis_coords


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
    k_perp1_bs_normal = k_perp1_bs * dot(
        k_perp1_hat, normal_hat
    )  # TODO: Check that this works correctly
    k_perp1_bs_binormal = k_perp1_bs * dot(
        k_perp1_hat, binormal_hat
    )  # TODO: Check that this works correctly

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
            }
        )

    else:
        log.debug(
            "No beam-tracing results; skipping wavenumber resolution and mismatch attenuation"
        )

    ##################################################
    log.debug(f"Finding the polarisation piece (loc_p)")
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
    loc_p_unnormalised = np.divide(
        np.abs(dot(e_hat_conj, dot(epsilon_minus_identity, e_hat))) ** 2,
        (n_e * 1e19) ** 2,
        out=np.zeros_like(n_e),
        where=(n_e > 1e-6),
    )

    loc_p = (
        w_launch**2 * const.epsilon_0 * find_electron_mass(temperature) / const.e**2
    ) ** 2 * loc_p_unnormalised

    dbs_analysis_dict.update(
        {
            "loc_p": (["tau"], loc_p),
            "int_loc_p_dL": ([], trapezoid(loc_p, beam_path_length)),
        }
    )

    ##################################################
    log.debug(f"Finding the ray piece (loc_r)")
    ##################################################

    g_mag_Cardano = np.array(analysis["g_magnitude_Cardano"][()])
    loc_r = (2 * const.c / w_launch) ** 2 / g_mag_Cardano**2

    dbs_analysis_dict.update(
        {
            "loc_r": (["tau"], loc_r),
            "int_loc_r_dL": ([], trapezoid(loc_r, beam_path_length)),
        }
    )

    if btf:
        ##################################################
        log.debug(f"Finding the beam piece (loc_b)")
        ##################################################

        launch_wavenumber = hamiltonian.wavenumber
        launch_beam_width = float(inputs["launch_beam_width"][()])
        launch_beam_curv = float(inputs["launch_beam_curvature"][()])
        det_Im_Psi_w = np.imag(Psi_xx) * np.imag(Psi_yy) - np.imag(Psi_xy) ** 2
        beam_waist = find_waist(launch_beam_width, launch_wavenumber, launch_beam_curv)
        loc_b = (beam_waist * det_Im_Psi_w) / (
            np.sqrt(2) * np.abs(det_M_w) * np.sqrt(-np.imag(M_w_inv_yy))
        )

        dbs_analysis_dict.update(
            {
                "det_Im_Psi_w": (["tau"], det_Im_Psi_w),
                "beam_waist": beam_waist,
                "loc_b": (["tau"], loc_b),
                "int_loc_b_dL": ([], trapezoid(loc_b, beam_path_length)),
            }
        )

    else:
        log.debug("No beam-tracing results; skipping beam piece (loc_b)")

    if btf:
        ##################################################
        log.debug(f"Finding the mismatch piece (loc_m)")
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
        loc_m = np.exp(-2 * (theta_m / delta_theta_m) ** 2)

        dbs_analysis_dict.update(
            {
                "delta_theta_m": (["tau"], delta_theta_m),
                "loc_m": (["tau"], loc_m),
                "int_loc_m_dL": ([], trapezoid(loc_m, beam_path_length)),
            }
        )

    else:
        log.debug("No beam-tracing results; skipping beam piece (loc_b)")

    ##################################################
    log.debug(f"Finding the spectrum piece (loc_s)")
    ##################################################

    spectrum_power_laws = ["-10/3", "-13/3"]
    loc_s = {
        f"loc_s_{power}": np.array(
            (k_perp1_bs / (-2 * launch_wavenumber)) ** eval(power)
        )
        for power in spectrum_power_laws
    }

    dbs_analysis_dict.update(
        {
            "spectrum_power_laws": spectrum_power_laws,
            **{f"{k}": (["tau"], v) for k, v in loc_s.items()},
            **{
                f"int_{k}_dL": ([], trapezoid(v, beam_path_length))
                for k, v in loc_s.items()
            },
        }
    )

    if btf:
        ##################################################
        log.debug(
            f"Finding total localisation (loc_p * loc_r * loc_b * loc_m * loc_s) and backscattered power (p_r)"
        )
        ##################################################

        loc_all = {
            f"loc_all_{power}": loc_p * loc_r * loc_b * loc_m * loc_s[f"loc_s_{power}"]
            for power in spectrum_power_laws
        }

        dbs_analysis_dict.update(
            {
                **{f"{k}": (["tau"], v) for k, v in loc_all.items()},
                **{
                    f"{k}": ([], trapezoid(v, beam_path_length))
                    for k, v in loc_all.items()
                },
            }
        )

    else:
        log.debug(
            "No beam-tracing results; skipping total localisation (loc_p * loc_r * loc_b * loc_m * loc_s) and backscattered power (p_r)"
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
