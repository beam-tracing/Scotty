from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import constants
from scipy.integrate import cumtrapz
import xarray as xr

from scotty.check_output import check_output
from scotty.profile_fit import ProfileFitLike
from scotty.derivatives import derivative
from scotty.fun_general import (
    K_magnitude,
    contract_special,
    find_nearest,
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
    find_q_lab_Cartesian,
    make_unit_vector_from_cross_product,
    find_Psi_3D_lab_Cartesian,
    find_H_Cardano,
    angular_frequency_to_wavenumber,
    find_waist,
    find_D,
    find_x0,
    find_electron_mass,
)
from scotty.geometry import MagneticField
from scotty.hamiltonian import DielectricTensor, Hamiltonian, hessians
from scotty.typing import FloatArray


def save_npz(filename: Path, df: xr.Dataset) -> None:
    """Save xarray dataset to numpy .npz file"""
    np.savez(
        filename,
        **{str(k): v for k, v in df.items()},
    )


def immediate_analysis(
    solver_output: xr.Dataset,
    field: MagneticField,
    find_density_1D: ProfileFitLike,
    find_temperature_1D: Optional[ProfileFitLike],
    hamiltonian: Hamiltonian,
    K_zeta_initial: float,
    launch_angular_frequency: float,
    mode_flag: int,
    delta_R: float,
    delta_Z: float,
    delta_K_R: float,
    delta_K_zeta: float,
    delta_K_Z: float,
    Psi_3D_lab_launch: FloatArray,
    Psi_3D_lab_entry: FloatArray,
    distance_from_launch_to_entry: float,
    vacuumLaunch_flag: bool,
    output_path: Path,
    output_filename_suffix: str,
    dH: Dict[str, FloatArray],
):
    q_R_array = solver_output.q_R_array
    q_Z_array = solver_output.q_Z_array
    tau_array = solver_output.tau_array
    K_R_array = solver_output.K_R_array
    K_Z_array = solver_output.K_Z_array

    numberOfDataPoints = len(tau_array)

    poloidal_flux_output = field.poloidal_flux(q_R_array, q_Z_array)
    electron_density_output = np.asfarray(find_density_1D(poloidal_flux_output))
    temperature_output = (
        find_temperature_1D(poloidal_flux_output) if find_temperature_1D else None
    )

    dH_dR_output = dH["dH_dR"]
    dH_dZ_output = dH["dH_dZ"]
    dH_dKR_output = dH["dH_dKR"]
    dH_dKzeta_output = dH["dH_dKzeta"]
    dH_dKZ_output = dH["dH_dKZ"]

    # Calculates nabla_K H
    # Calculates g_hat
    g_hat_output = np.zeros([numberOfDataPoints, 3])
    g_magnitude_output = (
        q_R_array**2 * dH_dKzeta_output**2 + dH_dKR_output**2 + dH_dKZ_output**2
    ) ** 0.5
    g_hat_output[:, 0] = dH_dKR_output / g_magnitude_output  # g_hat_R
    g_hat_output[:, 1] = q_R_array * dH_dKzeta_output / g_magnitude_output  # g_hat_zeta
    g_hat_output[:, 2] = dH_dKZ_output / g_magnitude_output  # g_hat_Z

    # Calculates b_hat and grad_b_hat
    B_R_output = field.B_R(q_R_array, q_Z_array)
    B_T_output = field.B_T(q_R_array, q_Z_array)
    B_Z_output = field.B_Z(q_R_array, q_Z_array)
    B_magnitude = field.magnitude(q_R_array, q_Z_array)
    b_hat_output = field.unit(q_R_array, q_Z_array)

    grad_bhat_output = np.zeros([numberOfDataPoints, 3, 3])
    dbhat_dR = derivative(
        field.unit,
        dims="q_R",
        args={"q_R": q_R_array, "q_Z": q_Z_array},
        spacings=delta_R,
    )
    dbhat_dZ = derivative(
        field.unit,
        dims="q_Z",
        args={"q_R": q_R_array, "q_Z": q_Z_array},
        spacings=delta_Z,
    )

    # Transpose dbhat_dR so that it has the right shape
    grad_bhat_output[:, 0, :] = dbhat_dR
    grad_bhat_output[:, 2, :] = dbhat_dZ
    grad_bhat_output[:, 1, 0] = -B_T_output / (B_magnitude * q_R_array)
    grad_bhat_output[:, 1, 1] = B_R_output / (B_magnitude * q_R_array)

    # x_hat and y_hat
    y_hat_output = make_unit_vector_from_cross_product(b_hat_output, g_hat_output)
    x_hat_output = make_unit_vector_from_cross_product(y_hat_output, g_hat_output)

    # Components of the dielectric tensor
    epsilon = DielectricTensor(
        electron_density_output,
        launch_angular_frequency,
        B_magnitude,
        temperature_output,
    )
    epsilon_para_output = epsilon.e_bb
    epsilon_perp_output = epsilon.e_11
    epsilon_g_output = epsilon.e_12

    # Plasma and cyclotron frequencies
    normalised_plasma_freqs = find_normalised_plasma_freq(
        electron_density_output, launch_angular_frequency, temperature_output
    )
    normalised_gyro_freqs = find_normalised_gyro_freq(
        B_magnitude, launch_angular_frequency, temperature_output
    )

    # -------------------
    # Not useful for physics or data analysis
    # But good for checking whether things are working properly
    # -------------------
    #
    H_output = hamiltonian(
        q_R_array.data, q_Z_array.data, K_R_array.data, K_zeta_initial, K_Z_array.data
    )
    # Create and immediately evaluate a Hamiltonian with the opposite mode
    H_other = Hamiltonian(
        field,
        launch_angular_frequency,
        -mode_flag,
        find_density_1D,
        delta_R,
        delta_Z,
        delta_K_R,
        delta_K_zeta,
        delta_K_Z,
    )(q_R_array.data, q_Z_array.data, K_R_array.data, K_zeta_initial, K_Z_array.data)

    # -------------------
    # Sanity check. Makes sure that calculated quantities are reasonable
    # -------------------
    check_output(H_output)

    df = xr.Dataset(
        {
            "B_R_output": (["tau"], B_R_output),
            "B_T_output": (["tau"], B_T_output),
            "B_Z_output": (["tau"], B_Z_output),
            "B_magnitude": (["tau"], B_magnitude),
            "K_R_array": K_R_array,
            "K_Z_array": K_Z_array,
            "K_zeta_initial": K_zeta_initial,
            "Psi_3D_lab_launch": (["e1", "e2"], Psi_3D_lab_launch),
            "Psi_3D_output": solver_output.Psi_3D_output,
            "b_hat_output": (["tau", "e1"], b_hat_output),
            "dH_dKR_output": (["tau"], dH_dKR_output),
            "dH_dKZ_output": (["tau"], dH_dKZ_output),
            "dH_dKzeta_output": (["tau"], dH_dKzeta_output),
            "dH_dR_output": (["tau"], dH_dR_output),
            "dH_dZ_output": (["tau"], dH_dZ_output),
            "g_hat_output": (["tau", "e1"], g_hat_output),
            "g_magnitude_output": g_magnitude_output,
            "grad_bhat_output": (["tau", "e1", "e2"], grad_bhat_output),
            "q_R_array": q_R_array,
            "q_Z_array": q_Z_array,
            "q_zeta_array": solver_output.q_zeta_array,
            "tau_array": tau_array,
            "x_hat_output": (["tau", "e1"], x_hat_output),
            "y_hat_output": (["tau", "e1"], y_hat_output),
            "poloidal_flux_output": (["tau"], poloidal_flux_output),
        },
        coords={"tau": tau_array, "e1": ["R", "zeta", "Z"], "e2": ["R", "zeta", "Z"]},
    )

    if temperature_output is not None:
        df.update({"temperature_output": (["tau"], temperature_output)})

    if vacuumLaunch_flag:
        vacuum_only = {
            "Psi_3D_lab_entry": (["e1", "e2"], Psi_3D_lab_entry),
            "distance_from_launch_to_entry": distance_from_launch_to_entry,
            "dpolflux_dR_debugging": (
                ["tau"],
                field.d_poloidal_flux_dR(q_R_array, q_Z_array, delta_R),
            ),
            "dpolflux_dZ_debugging": (
                ["tau"],
                field.d_poloidal_flux_dZ(q_R_array, q_Z_array, delta_Z),
            ),
            "epsilon_para_output": (["tau"], epsilon_para_output),
            "epsilon_perp_output": (["tau"], epsilon_perp_output),
            "epsilon_g_output": (["tau"], epsilon_g_output),
            "electron_density_output": (["tau"], electron_density_output),
            "normalised_plasma_freqs": (["tau"], normalised_plasma_freqs),
            "normalised_gyro_freqs": (["tau"], normalised_gyro_freqs),
            "H_output": (["tau"], H_output),
            "H_other": (["tau"], H_other),
        }
        df.update(vacuum_only)

    save_npz(output_path / f"data_output{output_filename_suffix}", df)

    return df


def further_analysis(
    inputs: xr.Dataset,
    df: xr.Dataset,
    Psi_3D_lab_entry_cartersian: FloatArray,
    output_path: Path,
    output_filename_suffix: str,
    field: MagneticField,
    detailed_analysis_flag: bool,
    dH: Dict[str, FloatArray],
):
    # Calculates various useful stuff
    [q_X_array, q_Y_array, _] = find_q_lab_Cartesian(
        [df.q_R_array, df.q_zeta_array, df.q_Z_array]
    )
    point_spacing = np.sqrt(
        (np.diff(q_X_array)) ** 2
        + (np.diff(q_Y_array)) ** 2
        + (np.diff(df.q_Z_array)) ** 2
    )
    distance_along_line = np.cumsum(point_spacing)
    distance_along_line = np.append(0, distance_along_line)
    RZ_point_spacing = np.sqrt(
        (np.diff(df.q_Z_array)) ** 2 + (np.diff(df.q_R_array)) ** 2
    )
    RZ_distance_along_line = np.cumsum(RZ_point_spacing)
    RZ_distance_along_line = np.append(0, RZ_distance_along_line)

    # Calculates the index of the minimum magnitude of K
    # That is, finds when the beam hits the cut-off
    K_magnitude_array = np.asfarray(
        K_magnitude(df.K_R_array, df.K_zeta_initial, df.K_Z_array, df.q_R_array)
    )

    # Index of the cutoff, at the minimum value of K, use this with other arrays
    cutoff_index = find_nearest(np.abs(K_magnitude_array), 0)

    # Calcuating the angles theta and theta_m
    # B \cdot K / (abs (B) abs(K))
    sin_theta_m_analysis = (
        df.b_hat_output.sel(e1="R") * df.K_R_array
        + df.b_hat_output.sel(e1="zeta") * df.K_zeta_initial / df.q_R_array
        + df.b_hat_output.sel(e1="Z") * df.K_Z_array
    ) / K_magnitude_array

    # Assumes the mismatch angle is never smaller than -90deg or bigger than 90deg
    theta_m_output = np.sign(sin_theta_m_analysis) * np.arcsin(
        abs(sin_theta_m_analysis)
    )

    kperp1_hat_output = make_unit_vector_from_cross_product(
        df.y_hat_output, df.b_hat_output
    )
    # The negative sign is there by definition
    sin_theta_analysis = -contract_special(df.x_hat_output, kperp1_hat_output)
    # The negative sign is there by definition. Alternative way to get sin_theta
    # Assumes theta is never smaller than -90deg or bigger than 90deg
    theta_output = np.sign(sin_theta_analysis) * np.arcsin(abs(sin_theta_analysis))

    cos_theta_analysis = np.cos(theta_output)
    tan_theta_analysis = np.tan(theta_output)
    # -----

    # Calcuating the corrections to make M from Psi
    # Includes terms small in mismatch

    # The dominant value of kperp1 that is backscattered at every point
    k_perp_1_bs = (
        -2
        * K_magnitude_array
        * np.cos(theta_m_output + theta_output)
        / cos_theta_analysis
    )
    # k_perp_1_bs = -2 * K_magnitude_array # when mismatch is small

    # Converting x_hat, y_hat, and Psi_3D to Cartesians so we can contract them with each other
    cos_q_zeta = np.cos(df.q_zeta_array)
    sin_q_zeta = np.sin(df.q_zeta_array)

    def to_Cartesian(array):
        cart_array = np.empty([len(df.tau), 3])
        cart_array[:, 0] = array[:, 0] * cos_q_zeta - array[:, 1] * sin_q_zeta
        cart_array[:, 1] = array[:, 0] * cos_q_zeta + array[:, 1] * sin_q_zeta
        cart_array[:, 2] = array[:, 2]
        return cart_array

    y_hat_Cartesian = to_Cartesian(df.y_hat_output)
    x_hat_Cartesian = to_Cartesian(df.x_hat_output)
    g_hat_Cartesian = to_Cartesian(df.g_hat_output)

    Psi_3D_Cartesian = find_Psi_3D_lab_Cartesian(
        df.Psi_3D_output, df.q_R_array, df.q_zeta_array, df.K_R_array, df.K_zeta_initial
    )
    Psi_xx_output = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, x_hat_Cartesian)
    )
    Psi_xy_output = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, y_hat_Cartesian)
    )
    Psi_yy_output = contract_special(
        y_hat_Cartesian, contract_special(Psi_3D_Cartesian, y_hat_Cartesian)
    )
    Psi_xg_output = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )
    Psi_yg_output = contract_special(
        y_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )
    Psi_gg_output = contract_special(
        g_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )

    Psi_xx_entry = np.dot(
        x_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, x_hat_Cartesian[0, :]),
    )
    Psi_xy_entry = np.dot(
        x_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, y_hat_Cartesian[0, :]),
    )
    Psi_yy_entry = np.dot(
        y_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, y_hat_Cartesian[0, :]),
    )

    numberOfDataPoints = len(df.tau)
    # Calculating intermediate terms that are needed for the corrections in M
    xhat_dot_grad_bhat = contract_special(df.x_hat_output, df.grad_bhat_output)
    yhat_dot_grad_bhat = contract_special(df.y_hat_output, df.grad_bhat_output)
    ray_curvature_kappa_output = np.zeros([numberOfDataPoints, 3])
    ray_curvature_kappa_output[:, 0] = (1 / df.g_magnitude_output) * (
        np.gradient(df.g_hat_output[:, 0], df.tau_array)
        - df.g_hat_output[:, 1] * df.dH_dKzeta_output  # See notes 07 June 2021
    )
    ray_curvature_kappa_output[:, 1] = (1 / df.g_magnitude_output) * (
        np.gradient(df.g_hat_output[:, 1], df.tau_array)
        + df.g_hat_output[:, 0] * df.dH_dKzeta_output  # See notes 07 June 2021
    )
    ray_curvature_kappa_output[:, 2] = (1 / df.g_magnitude_output) * np.gradient(
        df.g_hat_output[:, 2], df.tau_array
    )
    kappa_magnitude = np.linalg.norm(ray_curvature_kappa_output, axis=-1)
    d_theta_d_tau = np.gradient(theta_output, df.tau_array)
    d_xhat_d_tau_output = np.zeros([numberOfDataPoints, 3])
    d_xhat_d_tau_output[:, 0] = (
        np.gradient(df.x_hat_output[:, 0], df.tau_array)
        - df.x_hat_output[:, 1] * df.dH_dKzeta_output
    )  # See notes 07 June 2021
    d_xhat_d_tau_output[:, 1] = (
        np.gradient(df.x_hat_output[:, 1], df.tau_array)
        + df.x_hat_output[:, 0] * df.dH_dKzeta_output
    )  # See notes 07 June 2021
    d_xhat_d_tau_output[:, 2] = np.gradient(df.x_hat_output[:, 2], df.tau_array)

    xhat_dot_grad_bhat_dot_xhat_output = contract_special(
        xhat_dot_grad_bhat, df.x_hat_output
    )
    xhat_dot_grad_bhat_dot_yhat_output = contract_special(
        xhat_dot_grad_bhat, df.y_hat_output
    )
    xhat_dot_grad_bhat_dot_ghat_output = contract_special(
        xhat_dot_grad_bhat, df.g_hat_output
    )
    yhat_dot_grad_bhat_dot_xhat_output = contract_special(
        yhat_dot_grad_bhat, df.x_hat_output
    )
    yhat_dot_grad_bhat_dot_yhat_output = contract_special(
        yhat_dot_grad_bhat, df.y_hat_output
    )
    yhat_dot_grad_bhat_dot_ghat_output = contract_special(
        yhat_dot_grad_bhat, df.g_hat_output
    )
    kappa_dot_xhat_output = contract_special(
        ray_curvature_kappa_output, df.x_hat_output
    )
    kappa_dot_yhat_output = contract_special(
        ray_curvature_kappa_output, df.y_hat_output
    )
    # This should be 0. Good to check.
    kappa_dot_ghat_output = contract_special(
        ray_curvature_kappa_output, df.g_hat_output
    )
    d_xhat_d_tau_dot_yhat_output = contract_special(
        d_xhat_d_tau_output, df.y_hat_output
    )

    # Calculates the components of M_w, only taking into consideration
    # correction terms that are not small in mismatch
    M_xx_output = Psi_xx_output + (k_perp_1_bs / 2) * xhat_dot_grad_bhat_dot_ghat_output
    M_xy_output = Psi_xy_output + (k_perp_1_bs / 2) * yhat_dot_grad_bhat_dot_ghat_output
    M_yy_output = Psi_yy_output
    # -----

    # Calculates the localisation, wavenumber resolution, and mismatch attenuation pieces
    det_M_w_analysis = M_xx_output * M_yy_output - M_xy_output**2
    M_w_inv_xx_output = M_yy_output / det_M_w_analysis
    M_w_inv_xy_output = -M_xy_output / det_M_w_analysis
    M_w_inv_yy_output = M_xx_output / det_M_w_analysis

    delta_k_perp_2 = 2 * np.sqrt(-1 / np.imag(M_w_inv_yy_output))
    delta_theta_m = np.sqrt(
        np.imag(M_w_inv_yy_output)
        / (
            (np.imag(M_w_inv_xy_output)) ** 2
            - np.imag(M_w_inv_xx_output) * np.imag(M_w_inv_yy_output)
        )
    ) / (K_magnitude_array)
    loc_m = np.exp(-2 * (theta_m_output / delta_theta_m) ** 2)

    print("polflux: ", df.poloidal_flux_output[cutoff_index])

    print("theta_m", theta_m_output[cutoff_index])
    print("delta_theta_m", delta_theta_m[cutoff_index])
    print(
        "mismatch attenuation",
        np.exp(-2 * (theta_m_output[cutoff_index] / delta_theta_m[cutoff_index]) ** 2),
    )

    # This part is used to make some nice plots when post-processing
    R_midplane_points = np.linspace(field.R_coord[0], field.R_coord[-1], 1000)
    # poloidal flux at R and z=0
    poloidal_flux_on_midplane = field.poloidal_flux(R_midplane_points, 0)

    # Calculates localisation (start)
    # Ray piece of localisation as a function of distance along ray

    H_1_Cardano_array, H_2_Cardano_array, H_3_Cardano_array = find_H_Cardano(
        K_magnitude_array,
        inputs.launch_angular_frequency.data,
        df.epsilon_para_output.data,
        df.epsilon_perp_output.data,
        df.epsilon_g_output.data,
        theta_m_output.data,
    )

    def H_cardano(K_R, K_zeta, K_Z):
        # In my experience, the H_3_Cardano expression corresponds to
        # the O mode, and the H_2_Cardano expression corresponds to
        # the X-mode.

        # ALERT: This may not always be the case! Check the output
        # figure to make sure that the appropriate solution is indeed
        # 0 along the ray
        result = find_H_Cardano(
            K_magnitude(K_R, K_zeta, K_Z, df.q_R_array),
            inputs.launch_angular_frequency,
            df.epsilon_para_output,
            df.epsilon_perp_output,
            df.epsilon_g_output,
            theta_m_output,
        )
        if inputs.mode_flag == 1:
            return result[2]
        return result[1]

    def grad_H_Cardano(direction: str, spacing: float):
        return derivative(
            H_cardano,
            direction,
            args={
                "K_R": df.K_R_array,
                "K_zeta": df.K_zeta_initial,
                "K_Z": df.K_Z_array,
            },
            spacings=spacing,
        )

    g_R_Cardano = grad_H_Cardano("K_R", inputs.delta_K_R)
    g_zeta_Cardano = grad_H_Cardano("K_zeta", inputs.delta_K_zeta)
    g_Z_Cardano = grad_H_Cardano("K_Z", inputs.delta_K_Z)
    g_magnitude_Cardano = np.sqrt(
        g_R_Cardano**2 + g_zeta_Cardano**2 + g_Z_Cardano**2
    )

    ##
    # From here on, we use the shorthand
    # loc: localisation
    # l_lc: distance from cutoff (l - l_c). Distance along the ray
    # cum: cumulative. As such, cum_loc is the cumulative integral of the localisation
    # p: polarisation
    # r: ray
    # b: beam
    # s: spectrum
    # Otherwise, variable names get really unwieldly
    ##

    # localisation_ray = g_magnitude_Cardano[0]**2/g_magnitude_Cardano**2
    # The first point of the beam may be very slightly in the plasma, so I have used the vacuum expression for the group velocity instead
    loc_r = (
        2 * constants.c / inputs.launch_angular_frequency
    ) ** 2 / g_magnitude_Cardano**2

    # Spectrum piece of localisation as a function of distance along ray
    spectrum_power_law_coefficient = 13 / 3  # Turbulence cascade
    wavenumber_K0 = angular_frequency_to_wavenumber(
        inputs.launch_angular_frequency.data
    )
    loc_s = (k_perp_1_bs / (-2 * wavenumber_K0)) ** (-spectrum_power_law_coefficient)

    # Beam piece of localisation as a function of distance along ray
    # Determinant of the imaginary part of Psi_w
    det_imag_Psi_w_analysis = (
        np.imag(Psi_xx_output) * np.imag(Psi_yy_output) - np.imag(Psi_xy_output) ** 2
    )
    # Determinant of the real part of Psi_w. Not needed for the calculation, but gives useful insight
    det_real_Psi_w_analysis = (
        np.real(Psi_xx_output) * np.real(Psi_yy_output) - np.real(Psi_xy_output) ** 2
    )

    # Assumes circular beam at launch
    beam_waist_y = find_waist(
        inputs.launch_beam_width.data, wavenumber_K0, inputs.launch_beam_curvature.data
    )

    loc_b = (
        (beam_waist_y / np.sqrt(2))
        * det_imag_Psi_w_analysis
        / (np.abs(det_M_w_analysis) * np.sqrt(-np.imag(M_w_inv_yy_output)))
    )
    # --

    # Polarisation piece of localisation as a function of distance along ray
    # Polarisation e
    # eigenvector corresponding to eigenvalue = 0 (H=0)
    # First, find the components of the tensor D
    # Refer to 21st Dec 2020 notes for more
    # Note that e \cdot e* = 1
    [
        D_11_component,
        D_22_component,
        D_bb_component,
        D_12_component,
        D_1b_component,
    ] = find_D(
        K_magnitude_array,
        inputs.launch_angular_frequency.data,
        df.epsilon_para_output,
        df.epsilon_perp_output,
        df.epsilon_g_output,
        theta_m_output,
    )

    # Dispersion tensor
    D_tensor = np.zeros([numberOfDataPoints, 3, 3], dtype="complex128")
    D_tensor[:, 0, 0] = D_11_component
    D_tensor[:, 1, 1] = D_22_component
    D_tensor[:, 2, 2] = D_bb_component
    D_tensor[:, 0, 1] = -1j * D_12_component
    D_tensor[:, 1, 0] = 1j * D_12_component
    D_tensor[:, 0, 2] = D_1b_component
    D_tensor[:, 2, 0] = D_1b_component

    H_eigvals, e_eigvecs = np.linalg.eigh(D_tensor)

    # In my experience, H_eigvals[:,1] corresponds to the O mode, and H_eigvals[:,1] corresponds to the X-mode
    # ALERT: This may not always be the case! Check the output figure to make sure that the appropriate solution is indeed 0 along the ray
    # e_hat has components e_1,e_2,e_b
    if inputs.mode_flag == 1:
        H_solver = H_eigvals[:, 1]
        e_hat_output = e_eigvecs[:, :, 1]
    elif inputs.mode_flag == -1:
        H_solver = H_eigvals[:, 0]
        e_hat_output = e_eigvecs[:, :, 0]

    # equilibrium dielectric tensor - identity matrix. \bm{\epsilon}_{eq} - \bm{1}
    epsilon_minus_identity = np.zeros([numberOfDataPoints, 3, 3], dtype="complex128")
    identity = np.ones(numberOfDataPoints)
    epsilon_minus_identity[:, 0, 0] = df.epsilon_perp_output - identity
    epsilon_minus_identity[:, 1, 1] = df.epsilon_perp_output - identity
    epsilon_minus_identity[:, 2, 2] = df.epsilon_para_output - identity
    epsilon_minus_identity[:, 0, 1] = -1j * df.epsilon_g_output
    epsilon_minus_identity[:, 1, 0] = 1j * df.epsilon_g_output

    # loc_p_unnormalised = abs(contract_special(np.conjugate(e_hat_output), contract_special(epsilon_minus_identity,e_hat_output)))**2 / (electron_density_output*10**19)**2

    # Avoids dividing a small number by another small number, leading to a big number because of numerical errors or something
    loc_p_unnormalised = np.divide(
        np.abs(
            contract_special(
                np.conjugate(e_hat_output),
                contract_special(epsilon_minus_identity, e_hat_output),
            )
        )
        ** 2,
        (df.electron_density_output * 1e19) ** 2,
        out=np.zeros_like(df.electron_density_output),
        where=df.electron_density_output > 1e-6,
    )
    loc_p = (
        inputs.launch_angular_frequency**2
        * constants.epsilon_0
        * find_electron_mass(df.get("temperature_output"))
        / constants.e**2
    ) ** 2 * loc_p_unnormalised
    # Note that loc_p is called varepsilon in my paper

    # Note that K_1 = K cos theta_m, K_2 = 0, K_b = K sin theta_m, as a result of cold plasma dispersion
    K_hat_dot_e_hat = e_hat_output[:, 0] * np.cos(theta_m_output) + e_hat_output[
        :, 2
    ] * np.sin(theta_m_output)

    K_hat_dot_e_hat_sq = np.conjugate(K_hat_dot_e_hat) * K_hat_dot_e_hat
    # --

    # TODO: Come back and see if the naming of variables makes sense and is consistent
    # Distance from cutoff
    l_lc = distance_along_line - distance_along_line[cutoff_index]

    # Combining the various localisation pieces to get some overall localisation
    loc_b_r_s = loc_b * loc_r * loc_s
    loc_b_r = loc_b * loc_r

    # Calculates localisation (relevant pieces of the Spherical Tokamak case)
    d_theta_m_d_tau = np.gradient(theta_m_output, df.tau_array)
    d_K_d_tau = np.gradient(K_magnitude_array, df.tau_array)
    # d tau_Booker / d tau_Cardano
    d_tau_B_d_tau_C = g_magnitude_Cardano / df.g_magnitude_output
    theta_m_min_idx = np.argmin(np.abs(theta_m_output).data)
    delta_kperp1_ST = k_perp_1_bs - k_perp_1_bs[theta_m_min_idx]
    G_full = (
        (
            d_K_d_tau * df.g_magnitude_output
            - K_magnitude_array**2 * d_theta_m_d_tau**2 * M_w_inv_xx_output
        )
        * d_tau_B_d_tau_C**2
    ) ** (-1)
    G_term1 = (d_K_d_tau * df.g_magnitude_output * d_tau_B_d_tau_C**2) ** (-1)
    G_term2 = (
        K_magnitude_array**2
        * d_theta_m_d_tau**2
        * M_w_inv_xx_output
        * G_term1**2
        * d_tau_B_d_tau_C**2
    ) ** (-1)

    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH)

    print("Saving analysis data")
    further_df = {
        "Psi_xx_output": (["tau"], Psi_xx_output),
        "Psi_xy_output": (["tau"], Psi_xy_output),
        "Psi_yy_output": (["tau"], Psi_yy_output),
        "Psi_xg_output": (["tau"], Psi_xg_output),
        "Psi_yg_output": (["tau"], Psi_yg_output),
        "Psi_gg_output": (["tau"], Psi_gg_output),
        "Psi_xx_entry": Psi_xx_entry,
        "Psi_xy_entry": Psi_xy_entry,
        "Psi_yy_entry": Psi_yy_entry,
        "Psi_3D_Cartesian": (["tau", "e1", "e2"], Psi_3D_Cartesian),
        "x_hat_Cartesian": (["tau", "e1"], x_hat_Cartesian),
        "y_hat_Cartesian": (["tau", "e1"], y_hat_Cartesian),
        "g_hat_Cartesian": (["tau", "e1"], g_hat_Cartesian),
        "M_xx_output": M_xx_output,
        "M_xy_output": M_xy_output,
        "M_yy_output": M_yy_output,
        "M_w_inv_xx_output": M_w_inv_xx_output,
        "M_w_inv_xy_output": M_w_inv_xy_output,
        "M_w_inv_yy_output": M_w_inv_yy_output,
        "xhat_dot_grad_bhat_dot_xhat_output": (
            ["tau"],
            xhat_dot_grad_bhat_dot_xhat_output,
        ),
        "xhat_dot_grad_bhat_dot_yhat_output": (
            ["tau"],
            xhat_dot_grad_bhat_dot_yhat_output,
        ),
        "xhat_dot_grad_bhat_dot_ghat_output": (
            ["tau"],
            xhat_dot_grad_bhat_dot_ghat_output,
        ),
        "yhat_dot_grad_bhat_dot_xhat_output": (
            ["tau"],
            yhat_dot_grad_bhat_dot_xhat_output,
        ),
        "yhat_dot_grad_bhat_dot_yhat_output": (
            ["tau"],
            yhat_dot_grad_bhat_dot_yhat_output,
        ),
        "yhat_dot_grad_bhat_dot_ghat_output": (
            ["tau"],
            yhat_dot_grad_bhat_dot_ghat_output,
        ),
        "grad_grad_H": (["tau", "e1", "e2"], grad_grad_H),
        "gradK_grad_H": (["tau", "e1", "e2"], gradK_grad_H),
        "gradK_gradK_H": (["tau", "e1", "e2"], gradK_gradK_H),
        "d_theta_d_tau": (["tau"], d_theta_d_tau),
        "d_xhat_d_tau_dot_yhat_output": (["tau"], d_xhat_d_tau_dot_yhat_output),
        "kappa_dot_xhat_output": (["tau"], kappa_dot_xhat_output),
        "kappa_dot_yhat_output": (["tau"], kappa_dot_yhat_output),
        "kappa_dot_ghat_output": (["tau"], kappa_dot_ghat_output),
        "kappa_magnitude": (["tau"], kappa_magnitude),
        "delta_k_perp_2": delta_k_perp_2,
        "delta_theta_m": delta_theta_m,
        "theta_m_output": theta_m_output,
        "RZ_distance_along_line": (["tau"], RZ_distance_along_line),
        "distance_along_line": (["tau"], distance_along_line),
        "k_perp_1_bs": k_perp_1_bs,
        "K_magnitude_array": (["tau"], K_magnitude_array),
        "cutoff_index": cutoff_index,
        "x_hat_output": df.x_hat_output,
        "y_hat_output": df.y_hat_output,
        "b_hat_output": df.b_hat_output,
        "g_hat_output": df.g_hat_output,
        "e_hat_output": (["tau", "e1"], e_hat_output),
        "H_eigvals": (["tau", "e1"], H_eigvals),
        "e_eigvecs": (["tau", "e1", "e2"], e_eigvecs),
        "H_1_Cardano_array": (["tau"], H_1_Cardano_array),
        "H_2_Cardano_array": (["tau"], H_2_Cardano_array),
        "H_3_Cardano_array": (["tau"], H_3_Cardano_array),
        "kperp1_hat_output": (["tau", "e1"], kperp1_hat_output),
        "theta_output": (["tau"], theta_output),
        "g_magnitude_Cardano": g_magnitude_Cardano,
        "poloidal_flux_on_midplane": (["R_midplane"], poloidal_flux_on_midplane),
        "loc_b": loc_b,
        "loc_p": loc_p,
        "loc_r": loc_r,
        "loc_s": loc_s,
        "loc_m": loc_m,
        "loc_b_r_s": loc_b_r_s,
        "loc_b_r": loc_b_r,
    }

    df = df.assign_coords({"R_midplane": R_midplane_points, "l_lc": (["tau"], l_lc)})
    df.update(further_df)

    if detailed_analysis_flag and (cutoff_index + 1 != len(df.tau_array)):
        detailed_df = detailed_analysis(
            df, cutoff_index, loc_b_r.data, loc_b_r_s.data, l_lc.data, wavenumber_K0
        )
        df.update(detailed_df)

    save_npz(output_path / f"analysis_output{output_filename_suffix}", df)

    return df


def detailed_analysis(
    df: xr.Dataset,
    cutoff_index: int,
    loc_b_r: FloatArray,
    loc_b_r_s: FloatArray,
    l_lc: FloatArray,
    wavenumber_K0: float,
):
    """
    Now to do some more-complex analysis of the localisation.
    This part of the code fails in some situations, hence I'm making
    it possible to skip this section
    """
    # Finds the 1/e2 values (localisation)
    loc_b_r_s_max_over_e2 = loc_b_r_s.max() / (np.e**2)
    loc_b_r_max_over_e2 = loc_b_r.max() / (np.e**2)

    # Gives the inter-e2 range (analogous to interquartile range) in l-lc
    loc_b_r_s_delta_l_1 = find_x0(
        l_lc[0:cutoff_index], loc_b_r_s[0:cutoff_index], loc_b_r_s_max_over_e2
    )
    loc_b_r_s_delta_l_2 = find_x0(
        l_lc[cutoff_index:], loc_b_r_s[cutoff_index:], loc_b_r_s_max_over_e2
    )
    # The 1/e2 distances,  (l - l_c)
    loc_b_r_s_delta_l = np.array([loc_b_r_s_delta_l_1, loc_b_r_s_delta_l_2])
    loc_b_r_s_half_width_l = (loc_b_r_s_delta_l_2 - loc_b_r_s_delta_l_1) / 2
    loc_b_r_delta_l_1 = find_x0(
        l_lc[0:cutoff_index], loc_b_r[0:cutoff_index], loc_b_r_max_over_e2
    )
    loc_b_r_delta_l_2 = find_x0(
        l_lc[cutoff_index:], loc_b_r[cutoff_index:], loc_b_r_max_over_e2
    )
    # The 1/e2 distances,  (l - l_c)
    loc_b_r_delta_l = np.array([loc_b_r_delta_l_1, loc_b_r_delta_l_2])
    loc_b_r_half_width_l = (loc_b_r_delta_l_1 - loc_b_r_delta_l_2) / 2

    # Estimates the inter-e2 range (analogous to interquartile range) in kperp1, from l-lc
    # Bear in mind that since abs(kperp1) is minimised at cutoff, one really has to use that in addition to these.
    loc_b_r_s_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index], l_lc[0:cutoff_index], loc_b_r_s_delta_l_1
    )
    loc_b_r_s_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index:], l_lc[cutoff_index:], loc_b_r_s_delta_l_2
    )
    loc_b_r_s_delta_kperp1 = np.array(
        [loc_b_r_s_delta_kperp1_1, loc_b_r_s_delta_kperp1_2]
    )
    loc_b_r_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index], l_lc[0:cutoff_index], loc_b_r_delta_l_1
    )
    loc_b_r_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index:], l_lc[cutoff_index:], loc_b_r_delta_l_2
    )
    loc_b_r_delta_kperp1 = np.array([loc_b_r_delta_kperp1_1, loc_b_r_delta_kperp1_2])

    # Calculate the cumulative integral of the localisation pieces
    cum_loc_b_r_s = cumtrapz(loc_b_r_s, df.distance_along_line, initial=0)
    cum_loc_b_r_s = cum_loc_b_r_s - max(cum_loc_b_r_s) / 2
    cum_loc_b_r = cumtrapz(loc_b_r, df.distance_along_line, initial=0)
    cum_loc_b_r = cum_loc_b_r - max(cum_loc_b_r) / 2

    # Finds the 1/e2 values (cumulative integral of localisation)
    cum_loc_b_r_s_max_over_e2 = cum_loc_b_r_s.max() * (1 - 1 / (np.e**2))
    cum_loc_b_r_max_over_e2 = cum_loc_b_r.max() * (1 - 1 / (np.e**2))

    # Gives the inter-e range (analogous to interquartile range) in l-lc
    cum_loc_b_r_s_delta_l_1 = find_x0(l_lc, cum_loc_b_r_s, -cum_loc_b_r_s_max_over_e2)
    cum_loc_b_r_s_delta_l_2 = find_x0(l_lc, cum_loc_b_r_s, cum_loc_b_r_s_max_over_e2)
    cum_loc_b_r_s_delta_l = np.array([cum_loc_b_r_s_delta_l_1, cum_loc_b_r_s_delta_l_2])
    cum_loc_b_r_s_half_width = (cum_loc_b_r_s_delta_l_2 - cum_loc_b_r_s_delta_l_1) / 2
    cum_loc_b_r_delta_l_1 = find_x0(l_lc, cum_loc_b_r, -cum_loc_b_r_max_over_e2)
    cum_loc_b_r_delta_l_2 = find_x0(l_lc, cum_loc_b_r, cum_loc_b_r_max_over_e2)
    cum_loc_b_r_delta_l = np.array([cum_loc_b_r_delta_l_1, cum_loc_b_r_delta_l_2])
    cum_loc_b_r_half_width = (cum_loc_b_r_delta_l_2 - cum_loc_b_r_delta_l_1) / 2

    # Gives the inter-e2 range (analogous to interquartile range) in kperp1.
    # Bear in mind that since abs(kperp1) is minimised at cutoff, one really has to use that in addition to these.
    cum_loc_b_r_s_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index],
        cum_loc_b_r_s[0:cutoff_index],
        -cum_loc_b_r_s_max_over_e2,
    )
    cum_loc_b_r_s_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index::],
        cum_loc_b_r_s[cutoff_index::],
        cum_loc_b_r_s_max_over_e2,
    )
    cum_loc_b_r_s_delta_kperp1 = np.array(
        [cum_loc_b_r_s_delta_kperp1_1, cum_loc_b_r_s_delta_kperp1_2]
    )
    cum_loc_b_r_delta_kperp1_1 = find_x0(
        df.k_perp_1_bs[0:cutoff_index],
        cum_loc_b_r[0:cutoff_index],
        -cum_loc_b_r_max_over_e2,
    )
    cum_loc_b_r_delta_kperp1_2 = find_x0(
        df.k_perp_1_bs[cutoff_index::],
        cum_loc_b_r[cutoff_index::],
        cum_loc_b_r_max_over_e2,
    )
    cum_loc_b_r_delta_kperp1 = np.array(
        [cum_loc_b_r_delta_kperp1_1, cum_loc_b_r_delta_kperp1_2]
    )

    # Gives the mode l-lc for backscattering
    loc_b_r_s_max_index = find_nearest(loc_b_r_s, loc_b_r_s.max())
    loc_b_r_s_max_l_lc = (
        df.distance_along_line[loc_b_r_s_max_index]
        - df.distance_along_line[cutoff_index]
    )
    loc_b_r_max_index = find_nearest(loc_b_r, loc_b_r.max())
    loc_b_r_max_l_lc = (
        df.distance_along_line[loc_b_r_max_index] - df.distance_along_line[cutoff_index]
    )

    # Gives the mean l-lc for backscattering
    cum_loc_b_r_s_mean_l_lc = (
        np.trapz(loc_b_r_s * df.distance_along_line, df.distance_along_line)
        / np.trapz(loc_b_r_s, df.distance_along_line)
        - df.distance_along_line[cutoff_index]
    )
    cum_loc_b_r_mean_l_lc = (
        np.trapz(loc_b_r * df.distance_along_line, df.distance_along_line)
        / np.trapz(loc_b_r, df.distance_along_line)
        - df.distance_along_line[cutoff_index]
    )

    # Gives the median l-lc for backscattering
    cum_loc_b_r_s_delta_l_0 = find_x0(l_lc, cum_loc_b_r_s, 0)
    cum_loc_b_r_delta_l_0 = find_x0(l_lc, cum_loc_b_r, 0)

    # Due to the divergency of the ray piece, the mode kperp1 for backscattering is exactly that at the cut-off

    # Gives the mean kperp1 for backscattering
    cum_loc_b_r_s_mean_kperp1 = np.trapz(
        loc_b_r_s * df.k_perp_1_bs, df.k_perp_1_bs
    ) / np.trapz(loc_b_r_s, df.k_perp_1_bs)
    cum_loc_b_r_mean_kperp1 = np.trapz(
        loc_b_r * df.k_perp_1_bs, df.k_perp_1_bs
    ) / np.trapz(loc_b_r, df.k_perp_1_bs)

    # Gives the median kperp1 for backscattering
    cum_loc_b_r_s_delta_kperp1_0 = find_x0(df.k_perp_1_bs, cum_loc_b_r_s, 0)
    # Only works if point is before cutoff. To fix.
    cum_loc_b_r_delta_kperp1_0 = find_x0(
        df.k_perp_1_bs[0:cutoff_index], cum_loc_b_r[0:cutoff_index], 0
    )

    # To make the plots look nice
    k_perp_1_bs_plot = np.append(-2 * wavenumber_K0, df.k_perp_1_bs)
    k_perp_1_bs_plot = np.append(k_perp_1_bs_plot, -2 * wavenumber_K0)
    cum_loc_b_r_s_plot = np.append(cum_loc_b_r_s[0], cum_loc_b_r_s)
    cum_loc_b_r_s_plot = np.append(cum_loc_b_r_s_plot, cum_loc_b_r_s[-1])
    cum_loc_b_r_plot = np.append(cum_loc_b_r[0], cum_loc_b_r)
    cum_loc_b_r_plot = np.append(cum_loc_b_r_plot, cum_loc_b_r[-1])

    # These will get added as "dimension coordinates", arrays with
    # coordinates that have the same name, because we've not specified
    # dimensions, which is perhaps not desired. What might be better
    # would be to use e.g. loc_b_r_delta_l as the coordinate for
    # loc_b_r_max_over_e2
    return {
        "loc_b_r_s_max_over_e2": loc_b_r_s_max_over_e2,
        "loc_b_r_max_over_e2": loc_b_r_max_over_e2,
        # The 1/e2 distances,  (l - l_c)
        "loc_b_r_s_delta_l": loc_b_r_s_delta_l,
        "loc_b_r_delta_l": loc_b_r_delta_l,
        # The 1/e2 distances, kperp1, estimated from (l - l_c)
        "loc_b_r_s_delta_kperp1": loc_b_r_s_delta_kperp1,
        "loc_b_r_delta_kperp1": loc_b_r_delta_kperp1,
        "cum_loc_b_r_s": cum_loc_b_r_s,
        "cum_loc_b_r": cum_loc_b_r,
        "k_perp_1_bs_plot": k_perp_1_bs_plot,
        "cum_loc_b_r_s_plot": cum_loc_b_r_s_plot,
        "cum_loc_b_r_plot": cum_loc_b_r_plot,
        "cum_loc_b_r_s_max_over_e2": cum_loc_b_r_s_max_over_e2,
        "cum_loc_b_r_max_over_e2": cum_loc_b_r_max_over_e2,
        "cum_loc_b_r_s_delta_l": cum_loc_b_r_s_delta_l,
        "cum_loc_b_r_delta_l": cum_loc_b_r_delta_l,
        "cum_loc_b_r_s_delta_kperp1": cum_loc_b_r_s_delta_kperp1,
        "cum_loc_b_r_delta_kperp1": cum_loc_b_r_delta_kperp1,
        "loc_b_r_s_max_l_lc": loc_b_r_s_max_l_lc,
        "loc_b_r_max_l_lc": loc_b_r_max_l_lc,
        "cum_loc_b_r_s_mean_l_lc": cum_loc_b_r_s_mean_l_lc,
        "cum_loc_b_r_mean_l_lc": cum_loc_b_r_mean_l_lc,  # mean l-lc
        "cum_loc_b_r_s_delta_l_0": cum_loc_b_r_s_delta_l_0,
        "cum_loc_b_r_delta_l_0": cum_loc_b_r_delta_l_0,  # median l-lc
        "cum_loc_b_r_s_mean_kperp1": cum_loc_b_r_s_mean_kperp1,
        "cum_loc_b_r_mean_kperp1": cum_loc_b_r_mean_kperp1,  # mean kperp1
        "cum_loc_b_r_s_delta_kperp1_0": cum_loc_b_r_s_delta_kperp1_0,
    }
