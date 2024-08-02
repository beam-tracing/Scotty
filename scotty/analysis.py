"""
This file analyses solver_output (q, K, Psi_3D) from a beam tracing solver.

To do:
- Separate functions into those which analyse beam tracing only from those which are DBS specific. Maybe different files? (analysis_beam.py, analysis_DBS.py, analysis_highk.py)
- Write docstrings for functions in this file
- Change everything to Cartesian coordinates before passing into the functions in this file. Especially since within the scotty framework we expect to have both Cartesian and Cylindrical solvers
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from scipy import constants
import xarray as xr

from scotty.check_output import check_output
from scotty.profile_fit import ProfileFitLike
from scotty.derivatives import derivative
from scotty.fun_general import (
    K_magnitude,
    dot,
    find_nearest,
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
    make_unit_vector_from_cross_product,
    find_Psi_3D_lab_Cartesian,
    find_H_Cardano,
    angular_frequency_to_wavenumber,
    find_waist,
    find_D,
    find_x0,
    find_electron_mass,
    cylindrical_to_cartesian,
)
from scotty.geometry import MagneticField
from scotty.hamiltonian import DielectricTensor, Hamiltonian, hessians
from scotty.typing import ArrayLike, FloatArray, PathLike


CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]


def set_vector_components_long_name(df: Union[xr.Dataset, xr.DataArray]) -> None:
    if "col" in df.coords:
        df.col.attrs["long_name"] = "Vector/matrix column component"
    if "row" in df.coords:
        df.row.attrs["long_name"] = "Matrix row component"
    if "col_cart" in df.coords:
        df.col_cart.attrs["long_name"] = "Vector/matrix column component"
    if "row_cart" in df.coords:
        df.row_cart.attrs["long_name"] = "Matrix row component"


def save_npz(filename: Path, df: xr.Dataset) -> None:
    """Save xarray dataset to numpy .npz file"""
    np.savez(
        filename,
        **{str(k): v for k, v in df.items()},
        **{str(k): v.data for k, v in df.coords.items()},
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
    dH: Dict[str, ArrayLike],
):
    q_R = solver_output.q_R
    q_Z = solver_output.q_Z
    tau = solver_output.tau
    K_R = solver_output.K_R
    K_Z = solver_output.K_Z

    numberOfDataPoints = len(tau)

    poloidal_flux = field.poloidal_flux(q_R, q_Z)

    dH_dR = dH["dH_dR"]
    dH_dZ = dH["dH_dZ"]
    dH_dKR = dH["dH_dKR"]
    dH_dKzeta = dH["dH_dKzeta"]
    dH_dKZ = dH["dH_dKZ"]

    g_magnitude = (q_R**2 * dH_dKzeta**2 + dH_dKR**2 + dH_dKZ**2) ** 0.5
    g_hat = (np.block([[dH_dKR], [q_R * dH_dKzeta], [dH_dKZ]]) / g_magnitude.data).T

    # Calculates b_hat and grad_b_hat
    B_R = field.B_R(q_R, q_Z)
    B_T = field.B_T(q_R, q_Z)
    B_Z = field.B_Z(q_R, q_Z)
    B_magnitude = field.magnitude(q_R, q_Z)
    b_hat = field.unit(q_R, q_Z)

    dbhat_dR = derivative(
        field.unit, dims="q_R", args={"q_R": q_R, "q_Z": q_Z}, spacings=delta_R
    )
    dbhat_dZ = derivative(
        field.unit, dims="q_Z", args={"q_R": q_R, "q_Z": q_Z}, spacings=delta_Z
    )

    # Transpose dbhat_dR so that it has the right shape
    grad_bhat = np.zeros([numberOfDataPoints, 3, 3])
    grad_bhat[:, 0, :] = dbhat_dR
    grad_bhat[:, 2, :] = dbhat_dZ
    grad_bhat[:, 1, 0] = -B_T / (B_magnitude * q_R)
    grad_bhat[:, 1, 1] = B_R / (B_magnitude * q_R)

    # x_hat and y_hat
    y_hat = make_unit_vector_from_cross_product(b_hat, g_hat)
    x_hat = make_unit_vector_from_cross_product(y_hat, g_hat)

    # -------------------
    # Not useful for physics or data analysis
    # But good for checking whether things are working properly
    # -------------------
    #
    # There are two definitions of H used in Scotty
    # H_Booker: H is the determinant of the dispersion tensor D. Booker quartic
    # H_Cardano: H is the zero eigenvalue of the dispersion tensor D. Can be calculated with Cardano's formula.

    H_Booker = hamiltonian(q_R.data, q_Z.data, K_R.data, K_zeta_initial, K_Z.data)
    # Create and immediately evaluate a Hamiltonian with the opposite mode
    # That is, the opposite of the discriminant in the Booker quartic
    H_Booker_other = Hamiltonian(
        field,
        launch_angular_frequency,
        -mode_flag,
        find_density_1D,
        delta_R,
        delta_Z,
        delta_K_R,
        delta_K_zeta,
        delta_K_Z,
    )(q_R.data, q_Z.data, K_R.data, K_zeta_initial, K_Z.data)

    electron_density = np.asfarray(find_density_1D(poloidal_flux))
    temperature = find_temperature_1D(poloidal_flux) if find_temperature_1D else None

    epsilon = DielectricTensor(
        electron_density, launch_angular_frequency, B_magnitude, temperature
    )

    # Plasma and cyclotron frequencies
    normalised_plasma_freqs = find_normalised_plasma_freq(
        electron_density, launch_angular_frequency, temperature
    )
    normalised_gyro_freqs = find_normalised_gyro_freq(
        B_magnitude, launch_angular_frequency, temperature
    )

    # -------------------
    # Sanity check. Makes sure that calculated quantities are reasonable
    # -------------------
    check_output(H_Booker)
    print("The final value of H_Booker is", H_Booker[-1])
    ##

    df = xr.Dataset(
        {
            "B_R": (["tau"], B_R),
            "B_T": (["tau"], B_T),
            "B_Z": (["tau"], B_Z),
            "B_magnitude": (["tau"], B_magnitude),
            "K_R": K_R,
            "K_Z": K_Z,
            "K_zeta_initial": K_zeta_initial,
            "Psi_3D_lab_launch": (["row", "col"], Psi_3D_lab_launch),
            "Psi_3D": solver_output.Psi_3D,
            "b_hat": (["tau", "col"], b_hat),
            "dH_dKR": (["tau"], dH_dKR),
            "dH_dKZ": (["tau"], dH_dKZ),
            "dH_dKzeta": (["tau"], dH_dKzeta),
            "dH_dR": (["tau"], dH_dR),
            "dH_dZ": (["tau"], dH_dZ),
            "dpolflux_dR": (
                ["tau"],
                field.d_poloidal_flux_dR(q_R, q_Z, delta_R),
            ),
            "dpolflux_dZ": (
                ["tau"],
                field.d_poloidal_flux_dZ(q_R, q_Z, delta_Z),
            ),
            "epsilon_para": (["tau"], epsilon.e_bb),
            "epsilon_perp": (["tau"], epsilon.e_11),
            "epsilon_g": (["tau"], epsilon.e_12),
            "electron_density": (["tau"], electron_density),
            "g_hat": (["tau", "col"], g_hat),
            "g_magnitude": g_magnitude,
            "grad_bhat": (["tau", "row", "col"], grad_bhat),
            "H_Booker": (["tau"], H_Booker),
            "H_Booker_other": (["tau"], H_Booker_other),
            "normalised_plasma_freqs": (["tau"], normalised_plasma_freqs),
            "normalised_gyro_freqs": (["tau"], normalised_gyro_freqs),
            "x_hat": (["tau", "col"], x_hat),
            "y_hat": (["tau", "col"], y_hat),
            "poloidal_flux": (["tau"], poloidal_flux),
            "beam": (["tau", "col"], np.vstack([q_R, solver_output.q_zeta, q_Z]).T),
        },
        coords={
            "tau": tau,
            "row": CYLINDRICAL_VECTOR_COMPONENTS,
            "col": CYLINDRICAL_VECTOR_COMPONENTS,
            "q_R": q_R,
            "q_Z": q_Z,
            "q_zeta": solver_output.q_zeta,
        },
    )
    df.tau.attrs["long_name"] = "Parameterised distance along beam"
    set_vector_components_long_name(df)

    if temperature is not None:
        df.update({"temperature": (["tau"], temperature)})

    if vacuumLaunch_flag:
        vacuum_only = {
            "Psi_3D_lab_entry": (["row", "col"], Psi_3D_lab_entry),
            "distance_from_launch_to_entry": distance_from_launch_to_entry,
        }
        df.update(vacuum_only)

    return df


def further_analysis(
    inputs: xr.Dataset,
    df: xr.Dataset,
    Psi_3D_lab_entry_cartersian: FloatArray,
    output_path: Path,
    output_filename_suffix: str,
    field: MagneticField,
    detailed_analysis_flag: bool,
    dH: Dict[str, ArrayLike],
):
    # Calculates various useful stuff
    q_X, q_Y, _ = cylindrical_to_cartesian(df.q_R, df.q_zeta, df.q_Z)
    point_spacing = np.sqrt(
        np.diff(q_X) ** 2 + np.diff(q_Y) ** 2 + np.diff(df.q_Z) ** 2
    )
    distance_along_line = np.append(0, np.cumsum(point_spacing))

    # Calculates the index of the minimum magnitude of K
    # That is, finds when the beam hits the cut-off
    K_magnitude_array = np.asfarray(
        K_magnitude(df.K_R, df.K_zeta_initial, df.K_Z, df.q_R)
    )

    # Index of the cutoff, at the minimum value of K, use this with other arrays
    cutoff_index = find_nearest(np.abs(K_magnitude_array), 0)

    # Calcuating the angles theta and theta_m
    # B \cdot K / (abs (B) abs(K))
    sin_theta_m_analysis = (
        df.b_hat.sel(col="R") * df.K_R
        + df.b_hat.sel(col="zeta") * df.K_zeta_initial / df.q_R
        + df.b_hat.sel(col="Z") * df.K_Z
    ) / K_magnitude_array

    # Assumes the mismatch angle is never smaller than -90deg or bigger than 90deg
    theta_m = np.sign(sin_theta_m_analysis) * np.arcsin(abs(sin_theta_m_analysis))

    kperp1_hat = make_unit_vector_from_cross_product(df.y_hat, df.b_hat)
    # The negative sign is there by definition
    sin_theta_analysis = -dot(df.x_hat, kperp1_hat)
    # The negative sign is there by definition. Alternative way to get sin_theta
    # Assumes theta is never smaller than -90deg or bigger than 90deg
    theta = np.sign(sin_theta_analysis) * np.arcsin(abs(sin_theta_analysis))

    cos_theta_analysis = np.cos(theta)
    # -----

    # Calcuating the corrections to make M from Psi
    # Includes terms small in mismatch

    # The dominant value of kperp1 that is backscattered at every point
    k_perp_1_bs = -2 * K_magnitude_array * np.cos(theta_m + theta) / cos_theta_analysis

    normal_vectors = np.vstack(
        (df.dpolflux_dR, np.zeros_like(df.dpolflux_dR), df.dpolflux_dZ)
    ).T
    normal_magnitudes = np.linalg.norm(normal_vectors, axis=-1)
    normal_hat = normal_vectors / normal_magnitudes[:, np.newaxis]
    binormal_hat = make_unit_vector_from_cross_product(
        df.b_hat, normal_hat
    )  # by definition, binormal vector = tangent vector cross normal vector. Follows the same sign convention as Pyrokinetics [Patel, Bhavin, et al. "Pyrokinetics-A Python library to standardise gyrokinetic analysis." Journal of open source software (2024)]

    k_perp_1_bs_normal = k_perp_1_bs * dot(
        kperp1_hat, normal_hat
    )  # TODO: Check that this works properly
    k_perp_1_bs_binormal = k_perp_1_bs * dot(
        kperp1_hat, binormal_hat
    )  # TODO: Check that this works properly

    # Converting x_hat, y_hat, and Psi_3D to Cartesians so we can contract them with each other
    cos_q_zeta = np.cos(df.q_zeta)
    sin_q_zeta = np.sin(df.q_zeta)

    def to_Cartesian(array):
        cart = np.empty([len(df.tau), 3])
        cart[:, 0] = array[:, 0] * cos_q_zeta - array[:, 1] * sin_q_zeta
        cart[:, 1] = array[:, 0] * sin_q_zeta + array[:, 1] * cos_q_zeta
        cart[:, 2] = array[:, 2]
        return cart

    y_hat_Cartesian = to_Cartesian(df.y_hat)
    x_hat_Cartesian = to_Cartesian(df.x_hat)
    g_hat_Cartesian = to_Cartesian(df.g_hat)

    Psi_3D_Cartesian = find_Psi_3D_lab_Cartesian(
        df.Psi_3D, df.q_R, df.q_zeta, df.K_R, df.K_zeta_initial
    )
    Psi_xx = dot(x_hat_Cartesian, dot(Psi_3D_Cartesian, x_hat_Cartesian))
    Psi_xy = dot(x_hat_Cartesian, dot(Psi_3D_Cartesian, y_hat_Cartesian))
    Psi_yy = dot(y_hat_Cartesian, dot(Psi_3D_Cartesian, y_hat_Cartesian))
    Psi_xg = dot(x_hat_Cartesian, dot(Psi_3D_Cartesian, g_hat_Cartesian))
    Psi_yg = dot(y_hat_Cartesian, dot(Psi_3D_Cartesian, g_hat_Cartesian))
    Psi_gg = dot(g_hat_Cartesian, dot(Psi_3D_Cartesian, g_hat_Cartesian))

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
    xhat_dot_grad_bhat = dot(df.x_hat, df.grad_bhat)
    yhat_dot_grad_bhat = dot(df.y_hat, df.grad_bhat)

    # See notes 07 June 2021
    grad_g_hat = df.g_hat.differentiate("tau")
    ray_curvature_kappa = (
        np.block(
            [
                [grad_g_hat.sel(col="R") - df.g_hat.sel(col="zeta") * df.dH_dKzeta],
                [grad_g_hat.sel(col="zeta") + df.g_hat.sel(col="R") * df.dH_dKzeta],
                [grad_g_hat.sel(col="Z")],
            ]
        )
        / df.g_magnitude.data
    ).T

    grad_x_hat = df.x_hat.differentiate("tau")
    d_xhat_d_tau = np.block(
        [
            [grad_x_hat.sel(col="R") - df.x_hat.sel(col="zeta") * df.dH_dKzeta],
            [grad_x_hat.sel(col="zeta") + df.x_hat.sel(col="R") * df.dH_dKzeta],
            [grad_x_hat.sel(col="Z")],
        ]
    ).T

    xhat_dot_grad_bhat_dot_xhat = dot(xhat_dot_grad_bhat, df.x_hat)
    xhat_dot_grad_bhat_dot_yhat = dot(xhat_dot_grad_bhat, df.y_hat)
    xhat_dot_grad_bhat_dot_ghat = dot(xhat_dot_grad_bhat, df.g_hat)
    yhat_dot_grad_bhat_dot_xhat = dot(yhat_dot_grad_bhat, df.x_hat)
    yhat_dot_grad_bhat_dot_yhat = dot(yhat_dot_grad_bhat, df.y_hat)
    yhat_dot_grad_bhat_dot_ghat = dot(yhat_dot_grad_bhat, df.g_hat)
    kappa_dot_xhat = dot(ray_curvature_kappa, df.x_hat)
    kappa_dot_yhat = dot(ray_curvature_kappa, df.y_hat)
    # This should be 0. Good to check.
    kappa_dot_ghat = dot(ray_curvature_kappa, df.g_hat)
    d_xhat_d_tau_dot_yhat = dot(d_xhat_d_tau, df.y_hat)

    # Calculates the components of M_w, only taking into consideration
    # correction terms that are not small in mismatch
    M_xx = Psi_xx + (k_perp_1_bs / 2) * xhat_dot_grad_bhat_dot_ghat
    M_xy = Psi_xy + (k_perp_1_bs / 2) * yhat_dot_grad_bhat_dot_ghat
    M_yy = Psi_yy
    # -----

    # Calculates the localisation, wavenumber resolution, and mismatch attenuation pieces
    det_M_w_analysis = M_xx * M_yy - M_xy**2
    M_w_inv_xx = M_yy / det_M_w_analysis
    M_w_inv_xy = -M_xy / det_M_w_analysis
    M_w_inv_yy = M_xx / det_M_w_analysis

    delta_k_perp_2 = 2 * np.sqrt(-1 / np.imag(M_w_inv_yy))
    delta_theta_m = np.sqrt(
        np.imag(M_w_inv_yy)
        / ((np.imag(M_w_inv_xy)) ** 2 - np.imag(M_w_inv_xx) * np.imag(M_w_inv_yy))
    ) / (K_magnitude_array)
    loc_m = np.exp(-2 * (theta_m / delta_theta_m) ** 2)

    print("polflux: ", df.poloidal_flux[cutoff_index].data)

    print("theta_m: ", theta_m[cutoff_index].data)
    print("delta_theta_m: ", delta_theta_m[cutoff_index].data)
    print(
        "mismatch attenuation: ",
        (np.exp(-2 * (theta_m[cutoff_index] / delta_theta_m[cutoff_index]) ** 2)).data,
    )

    # This part is used to make some nice plots when post-processing
    R_midplane_points = np.linspace(field.R_coord[0], field.R_coord[-1], 1000)
    # poloidal flux at R and z=0
    poloidal_flux_on_midplane = field.poloidal_flux(R_midplane_points, 0)

    # Calculates localisation (start)
    # Ray piece of localisation as a function of distance along ray

    H_1_Cardano, H_2_Cardano, H_3_Cardano = find_H_Cardano(
        K_magnitude_array,
        inputs.launch_angular_frequency.data,
        df.epsilon_para.data,
        df.epsilon_perp.data,
        df.epsilon_g.data,
        theta_m.data,
    )

    def H_cardano(K_R, K_zeta, K_Z):
        # In my experience, the H_3_Cardano expression corresponds to
        # the O mode, and the H_2_Cardano expression corresponds to
        # the X-mode.

        # ALERT: This may not always be the case! Check the output
        # figure to make sure that the appropriate solution is indeed
        # 0 along the ray
        result = find_H_Cardano(
            K_magnitude(K_R, K_zeta, K_Z, df.q_R),
            inputs.launch_angular_frequency,
            df.epsilon_para,
            df.epsilon_perp,
            df.epsilon_g,
            theta_m,
        )
        if inputs.mode_flag == 1:
            return result[2]
        return result[1]

    def grad_H_Cardano(direction: str, spacing: float):
        return derivative(
            H_cardano,
            direction,
            args={"K_R": df.K_R, "K_zeta": df.K_zeta_initial, "K_Z": df.K_Z},
            spacings=spacing,
        )

    g_R_Cardano = grad_H_Cardano("K_R", inputs.delta_K_R)
    g_zeta_Cardano = grad_H_Cardano("K_zeta", inputs.delta_K_zeta)
    g_Z_Cardano = grad_H_Cardano("K_Z", inputs.delta_K_Z)
    # This has maximum imaginary component of like 1e-16 -- should just be real?
    g_magnitude_Cardano = np.sqrt(g_R_Cardano**2 + g_zeta_Cardano**2 + g_Z_Cardano**2)

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
    det_imag_Psi_w_analysis = np.imag(Psi_xx) * np.imag(Psi_yy) - np.imag(Psi_xy) ** 2

    # Assumes circular beam at launch
    beam_waist_y = find_waist(
        inputs.launch_beam_width.data, wavenumber_K0, inputs.launch_beam_curvature.data
    )

    loc_b = (
        (beam_waist_y / np.sqrt(2))
        * det_imag_Psi_w_analysis
        / (np.abs(det_M_w_analysis) * np.sqrt(-np.imag(M_w_inv_yy)))
    )
    # --

    # Polarisation piece of localisation as a function of distance along ray
    H_eigvals, e_eigvecs = dispersion_eigenvalues(
        K_magnitude_array,
        inputs.launch_angular_frequency.data,
        df,
        numberOfDataPoints,
        theta_m,
    )
    # In my experience, H_eigvals[:,1] corresponds to the O mode, and H_eigvals[:,0] corresponds to the X-mode
    # ALERT: This may not always be the case! Check the output figure to make sure that the appropriate solution is indeed 0 along the ray
    # e_hat has components e_1,e_2,e_b
    mode_index = 1 if inputs.mode_flag == 1 else 0
    e_hat = e_eigvecs[:, :, mode_index]

    # equilibrium dielectric tensor - identity matrix. \bm{\epsilon}_{eq} - \bm{1}
    zero = np.zeros(len(df.tau))
    # fmt: off
    epsilon_minus_identity = np.block(
        [
            [[df.epsilon_perp],    [1j * df.epsilon_g], [zero]],
            [[-1j * df.epsilon_g], [df.epsilon_perp],   [zero]],
            [[zero],               [zero],              [df.epsilon_para]],
        ]
    ).T - np.eye(3)
    # fmt: on

    # Avoids dividing a small number by another small number, leading to a big number because of numerical errors or something
    loc_p_unnormalised = np.divide(
        np.abs(
            dot(
                np.conjugate(e_hat),
                dot(epsilon_minus_identity, e_hat),
            )
        )
        ** 2,
        (df.electron_density * 1e19) ** 2,
        out=np.zeros_like(df.electron_density),
        where=(df.electron_density > 1e-6).data,
    )
    loc_p = (
        inputs.launch_angular_frequency**2
        * constants.epsilon_0
        * find_electron_mass(df.get("temperature"))
        / constants.e**2
    ) ** 2 * loc_p_unnormalised
    # Note that loc_p is called varepsilon in my paper

    # --

    # TODO: Come back and see if the naming of variables makes sense and is consistent
    # Distance from cutoff
    l_lc = distance_along_line - distance_along_line[cutoff_index]

    # Combining the various localisation pieces to get some overall localisation
    loc_b_r_s = loc_b * loc_r * loc_s
    loc_b_r = loc_b * loc_r

    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH)

    further_df = {
        "Psi_xx": (["tau"], Psi_xx),
        "Psi_xy": (["tau"], Psi_xy),
        "Psi_yy": (["tau"], Psi_yy),
        "Psi_xg": (["tau"], Psi_xg),
        "Psi_yg": (["tau"], Psi_yg),
        "Psi_gg": (["tau"], Psi_gg),
        "Psi_xx_entry": Psi_xx_entry,
        "Psi_xy_entry": Psi_xy_entry,
        "Psi_yy_entry": Psi_yy_entry,
        "Psi_3D_Cartesian": (["tau", "row_cart", "col_cart"], Psi_3D_Cartesian),
        "x_hat_Cartesian": (["tau", "col_cart"], x_hat_Cartesian),
        "y_hat_Cartesian": (["tau", "col_cart"], y_hat_Cartesian),
        "g_hat_Cartesian": (["tau", "col_cart"], g_hat_Cartesian),
        "M_xx": M_xx,
        "M_xy": M_xy,
        "M_yy": M_yy,
        "M_w_inv_xx": M_w_inv_xx,
        "M_w_inv_xy": M_w_inv_xy,
        "M_w_inv_yy": M_w_inv_yy,
        "xhat_dot_grad_bhat_dot_xhat": (["tau"], xhat_dot_grad_bhat_dot_xhat),
        "xhat_dot_grad_bhat_dot_yhat": (["tau"], xhat_dot_grad_bhat_dot_yhat),
        "xhat_dot_grad_bhat_dot_ghat": (["tau"], xhat_dot_grad_bhat_dot_ghat),
        "yhat_dot_grad_bhat_dot_xhat": (["tau"], yhat_dot_grad_bhat_dot_xhat),
        "yhat_dot_grad_bhat_dot_yhat": (["tau"], yhat_dot_grad_bhat_dot_yhat),
        "yhat_dot_grad_bhat_dot_ghat": (["tau"], yhat_dot_grad_bhat_dot_ghat),
        "grad_grad_H": (["tau", "row", "col"], grad_grad_H),
        "gradK_grad_H": (["tau", "row", "col"], gradK_grad_H),
        "gradK_gradK_H": (["tau", "row", "col"], gradK_gradK_H),
        "d_theta_d_tau": (["tau"], np.gradient(theta, df.tau)),
        "d_xhat_d_tau_dot_yhat": (["tau"], d_xhat_d_tau_dot_yhat),
        "kappa_dot_xhat": (["tau"], kappa_dot_xhat),
        "kappa_dot_yhat": (["tau"], kappa_dot_yhat),
        "kappa_dot_ghat": (["tau"], kappa_dot_ghat),
        "kappa_magnitude": (["tau"], np.linalg.norm(ray_curvature_kappa, axis=-1)),
        "delta_k_perp_2": delta_k_perp_2,
        "delta_theta_m": delta_theta_m,
        "theta_m": theta_m,
        "k_perp_1_bs": k_perp_1_bs,
        "k_perp_1_bs_normal": k_perp_1_bs_normal,
        "k_perp_1_bs_binormal": k_perp_1_bs_binormal,
        "K_magnitude": (["tau"], K_magnitude_array),
        "cutoff_index": cutoff_index,
        "x_hat": df.x_hat,
        "y_hat": df.y_hat,
        "b_hat": df.b_hat,
        "g_hat": df.g_hat,
        "e_hat": (["tau", "col"], e_hat),
        "H_eigvals": (
            ["tau", "col"],
            H_eigvals,
        ),  ##TODO: the second index should be 1,2,3. Not 'col', which is R, zeta, Z
        "e_eigvecs": (["tau", "row", "col"], e_eigvecs),
        "H_1_Cardano": (["tau"], H_1_Cardano),
        "H_2_Cardano": (["tau"], H_2_Cardano),
        "H_3_Cardano": (["tau"], H_3_Cardano),
        "kperp1_hat": (["tau", "col"], kperp1_hat),
        "theta": (["tau"], theta),
        "g_magnitude_Cardano": g_magnitude_Cardano,
        "poloidal_flux_on_midplane": (["R_midplane"], poloidal_flux_on_midplane),
        "loc_b": loc_b,
        "loc_p": loc_p,
        "loc_r": loc_r,
        "loc_s": loc_s,
        "loc_m": loc_m,
        "loc_b_r_s": loc_b_r_s,
        "loc_b_r": loc_b_r,
        "beam_cartesian": (["tau", "col_cart"], np.vstack([q_X, q_Y, df.q_Z.data]).T),
    }

    RZ_point_spacing = np.sqrt((np.diff(df.q_Z)) ** 2 + (np.diff(df.q_R)) ** 2)
    RZ_distance_along_line = np.append(0, np.cumsum(RZ_point_spacing))

    df = df.assign_coords(
        {
            "R_midplane": R_midplane_points,
            "l_lc": (
                ["tau"],
                l_lc,
                {"long_name": "Distance from cutoff", "units": "m"},
            ),
            "RZ_distance_along_line": (["tau"], RZ_distance_along_line, {"units": "m"}),
            "distance_along_line": (["tau"], distance_along_line, {"units": "m"}),
            "row_cart": CARTESIAN_VECTOR_COMPONENTS,
            "col_cart": CARTESIAN_VECTOR_COMPONENTS,
            "q_X": q_X,
            "q_Y": q_Y,
        }
    )
    set_vector_components_long_name(df)
    df.update(further_df)

    if detailed_analysis_flag and (cutoff_index + 1 != len(df.tau)):
        df.update(localisation_analysis(df, cutoff_index, wavenumber_K0))

    return df


def dispersion_eigenvalues(
    K_magnitude_array: FloatArray,
    launch_angular_frequency: float,
    df: xr.Dataset,
    numberOfDataPoints: int,
    theta_m: float,
):
    """
    Polarisation e
    eigenvector corresponding to eigenvalue = 0 (H=0)

    Returns
    -------
    eigenvalues, eigenvectors

    """

    # First, find the components of the tensor D
    # Refer to 21st Dec 2020 notes for more
    # Note that e \cdot e* = 1
    D_11, D_22, D_bb, D_12, D_1b = find_D(
        K_magnitude_array,
        launch_angular_frequency,
        df.epsilon_para,
        df.epsilon_perp,
        df.epsilon_g,
        theta_m,
    )

    # Dispersion tensor
    D_tensor = np.zeros([numberOfDataPoints, 3, 3], dtype="complex128")
    D_tensor[:, 0, 0] = D_11
    D_tensor[:, 1, 1] = D_22
    D_tensor[:, 2, 2] = D_bb
    D_tensor[:, 0, 1] = -1j * D_12
    D_tensor[:, 1, 0] = 1j * D_12
    D_tensor[:, 0, 2] = D_1b
    D_tensor[:, 2, 0] = D_1b

    return np.linalg.eigh(D_tensor)


def find_e2_width(
    l_lc: xr.DataArray, array: xr.DataArray, k_perp: xr.DataArray, cutoff_index: int
):
    r"""Estimate inter-:math:`e^2` range (analogous to interquartile range) in
    :math:`l - l_c` and :math:`k_\perp`

    Returns
    -------
    max_over_e2: float
        :math:`1/e^2` value of ``array``
    delta_l: FloatArray
        inter-:math:`e^2` range in :math:`l - l_c` of ``array``
    delta_kperp1: FloatArray
        inter-:math:`e^2` range in :math:`k_\perp` of ``array``

    """

    max_over_e2 = (array.max() / (np.e**2)).data
    delta_l_1 = find_x0(l_lc[:cutoff_index], array[:cutoff_index], max_over_e2)
    delta_l_2 = find_x0(l_lc[cutoff_index:], array[cutoff_index:], max_over_e2)

    # Estimates the inter-e2 range (analogous to interquartile range) in kperp1,
    # from l-lc. Bear in mind that since abs(kperp1) is minimised at cutoff, one
    # really has to use that in addition to these.
    delta_kperp1_1 = find_x0(k_perp[:cutoff_index], l_lc[:cutoff_index], delta_l_1)
    delta_kperp1_2 = find_x0(k_perp[cutoff_index:], l_lc[cutoff_index:], delta_l_2)
    return (
        max_over_e2,
        np.array((delta_l_1, delta_l_2)),
        np.array((delta_kperp1_1, delta_kperp1_2)),
    )


def cumulative_integrate(array: xr.DataArray) -> xr.DataArray:
    """Cumulative integral of ``array`` along ``distance_along_line``, shifted
    by half the maximum"""
    cumint = array.cumulative_integrate("distance_along_line")
    return cumint - cumint.max() / 2


def max_l_lc(
    distance_along_line: xr.DataArray, array: xr.DataArray, cutoff_index: int
) -> xr.DataArray:
    max_index = array.argmax("tau")
    return distance_along_line[max_index] - distance_along_line[cutoff_index]


def mean_l_lc(
    distance_along_line: xr.DataArray, array: xr.DataArray, cutoff_index: int
) -> xr.DataArray:
    return (
        (array * distance_along_line).integrate("distance_along_line")
        / array.integrate("distance_along_line")
    ) - distance_along_line[cutoff_index]


def mean_kperp(k_perp: xr.DataArray, array: xr.DataArray) -> xr.DataArray:
    """Mean kperp1 for backscattering"""
    return np.trapz(array * k_perp, k_perp) / np.trapz(array, k_perp)


def localisation_analysis(df: xr.Dataset, cutoff_index: int, wavenumber_K0: float):
    """
    Now to do some more-complex analysis of the localisation / instrumentation function / filter function.
    This part of the code fails in some situations, hence I'm making
    it possible to skip this section
    """
    # Finds the 1/e2 values (localisation)
    loc_b_r_s_max_over_e2, loc_b_r_s_delta_l, loc_b_r_s_delta_kperp1 = find_e2_width(
        df.l_lc, df.loc_b_r_s, df.k_perp_1_bs, cutoff_index
    )
    loc_b_r_max_over_e2, loc_b_r_delta_l, loc_b_r_delta_kperp1 = find_e2_width(
        df.l_lc, df.loc_b_r, df.k_perp_1_bs, cutoff_index
    )

    # Calculate the cumulative integral of the localisation pieces
    cum_loc_b_r_s = cumulative_integrate(df.loc_b_r_s)
    cum_loc_b_r = cumulative_integrate(df.loc_b_r)

    (
        cum_loc_b_r_s_max_over_e2,
        cum_loc_b_r_s_delta_l,
        cum_loc_b_r_s_delta_kperp1,
    ) = find_e2_width(df.l_lc, cum_loc_b_r_s, df.k_perp_1_bs, cutoff_index)

    (
        cum_loc_b_r_max_over_e2,
        cum_loc_b_r_delta_l,
        cum_loc_b_r_delta_kperp1,
    ) = find_e2_width(df.l_lc, cum_loc_b_r, df.k_perp_1_bs, cutoff_index)

    # Gives the mode l-lc for backscattering
    loc_b_r_s_max_l_lc = max_l_lc(df.distance_along_line, df.loc_b_r_s, cutoff_index)
    loc_b_r_max_l_lc = max_l_lc(df.distance_along_line, df.loc_b_r, cutoff_index)

    # Gives the mean l-lc for backscattering
    cum_loc_b_r_s_mean_l_lc = mean_l_lc(
        df.distance_along_line, df.loc_b_r_s, cutoff_index
    )
    cum_loc_b_r_mean_l_lc = mean_l_lc(df.distance_along_line, df.loc_b_r, cutoff_index)

    # Gives the median l-lc for backscattering
    cum_loc_b_r_s_delta_l_0 = find_x0(df.l_lc, cum_loc_b_r_s, 0)
    cum_loc_b_r_delta_l_0 = find_x0(df.l_lc, cum_loc_b_r, 0)

    # Due to the divergency of the ray piece, the mode kperp1 for backscattering is exactly that at the cut-off

    cum_loc_b_r_s_mean_kperp1 = mean_kperp(df.k_perp_1_bs, df.loc_b_r_s)
    cum_loc_b_r_mean_kperp1 = mean_kperp(df.k_perp_1_bs, df.loc_b_r)

    # Gives the median kperp1 for backscattering
    cum_loc_b_r_s_delta_kperp1_0 = find_x0(df.k_perp_1_bs, cum_loc_b_r_s, 0)

    # Some of these will get added as "dimension coordinates", arrays with
    # coordinates that have the same name, because we've not specified
    # dimensions, which is perhaps not desired. What might be better would be to
    # use e.g. loc_b_r_delta_l as the coordinate for loc_b_r_max_over_e2
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


def open_data_input_npz(filename: PathLike) -> xr.Dataset:
    """Read a ``data_input.npz`` file created with a previous version of Scotty, v2.4.3 or earlier, returning an xarray dataset"""
    with np.load(filename) as f:
        df = xr.Dataset(
            {
                "launch_freq_GHz": f["launch_freq_GHz"],
                "launch_position": (["col"], f["launch_position"]),
                "mode_flag": f["mode_flag"],
                "poloidalFlux_grid": (["R", "Z"], f["poloidalFlux_grid"]),
            },
            coords={
                "R": f["data_R_coord"],
                "Z": f["data_Z_coord"],
                "row": CYLINDRICAL_VECTOR_COMPONENTS,
                "col": CYLINDRICAL_VECTOR_COMPONENTS,
            },
        )

    set_vector_components_long_name(df)
    return df


def open_data_output_npz(filename: PathLike) -> xr.Dataset:
    """Read a ``data_output.npz`` file created with a previous version of Scotty, v2.4.3 or earlier, returning an xarray dataset"""

    with np.load(filename, allow_pickle=True) as f:
        df = xr.Dataset(
            {
                "B_R": (["tau"], f["B_R_output"]),
                "B_T": (["tau"], f["B_T_output"]),
                "B_Z": (["tau"], f["B_Z_output"]),
                "B_magnitude": (["tau"], f["B_magnitude"]),
                "K_R": (["tau"], f["K_R_array"]),
                "K_Z": (["tau"], f["K_Z_array"]),
                "Psi_3D_lab_launch": (["row", "col"], f["Psi_3D_lab_launch"]),
                "Psi_3D": (["tau", "row", "col"], f["Psi_3D_output"]),
                "b_hat": (["tau", "col"], f["b_hat_output"]),
                "dH_dKR": (["tau"], f["dH_dKR_output"]),
                "dH_dKZ": (["tau"], f["dH_dKZ_output"]),
                "dH_dKzeta": (["tau"], f["dH_dKzeta_output"]),
                "dH_dR": (["tau"], f["dH_dR_output"]),
                "dH_dZ": (["tau"], f["dH_dZ_output"]),
                "g_hat": (["tau", "col"], f["g_hat_output"]),
                "g_magnitude": (["tau"], f["g_magnitude_output"]),
                "grad_bhat": (["tau", "row", "col"], f["grad_bhat_output"]),
                "x_hat": (["tau", "col"], f["x_hat_output"]),
                "y_hat": (["tau", "col"], f["y_hat_output"]),
                "poloidal_flux": (["tau"], f["poloidal_flux_output"]),
            },
            coords={
                "tau": f["tau_array"],
                "row": CYLINDRICAL_VECTOR_COMPONENTS,
                "col": CYLINDRICAL_VECTOR_COMPONENTS,
                "q_R": f["q_R_array"],
                "q_Z": f["q_Z_array"],
                "q_zeta": f["q_zeta_array"],
            },
        )
        if "temperature" in f:
            df.update({"temperature": (["tau"], f["temperature"])})

        if "Psi_3D_lab_entry" in f:
            df.update(
                {
                    "Psi_3D_lab_entry": (["row", "col"], f["Psi_3D_lab_entry"]),
                    "distance_from_launch_to_entry": f["distance_from_launch_to_entry"],
                    "dpolflux_dR_debugging": (["tau"], f["dpolflux_dR_debugging"]),
                    "dpolflux_dZ_debugging": (["tau"], f["dpolflux_dZ_debugging"]),
                    "epsilon_para": (["tau"], f["epsilon_para_output"]),
                    "epsilon_perp": (["tau"], f["epsilon_perp_output"]),
                    "epsilon_g": (["tau"], f["epsilon_g_output"]),
                    "electron_density": (["tau"], f["electron_density_output"]),
                    "normalised_plasma_freqs": (["tau"], f["normalised_plasma_freqs"]),
                    "normalised_gyro_freqs": (["tau"], f["normalised_gyro_freqs"]),
                    "H": (["tau"], f["H_output"]),
                    "H_other": (["tau"], f["H_other"]),
                }
            )

    df.tau.attrs["long_name"] = "Parameterised distance along beam"
    set_vector_components_long_name(df)

    return df


def open_analysis_npz(outputs: xr.Dataset, filename: PathLike) -> xr.Dataset:
    """Read a ``analysis_output.npz`` file created with a previous version of Scotty, v2.4.3 or earlier, returning an xarray dataset"""
    with np.load(filename, allow_pickle=True) as f:
        df = xr.Dataset(
            {
                "Psi_3D_Cartesian": (["tau", "row", "col"], f["Psi_3D_Cartesian"]),
                "Psi_xx": (["tau"], f["Psi_xx_output"]),
                "Psi_xy": (["tau"], f["Psi_xy_output"]),
                "Psi_yy": (["tau"], f["Psi_yy_output"]),
                "M_xx": (["tau"], f["M_xx_output"]),
                "M_xy": (["tau"], f["M_xy_output"]),
                "xhat_dot_grad_bhat_dot_xhat": (
                    ["tau"],
                    f["xhat_dot_grad_bhat_dot_xhat_output"],
                ),
                "xhat_dot_grad_bhat_dot_ghat": (
                    ["tau"],
                    f["xhat_dot_grad_bhat_dot_ghat_output"],
                ),
                "yhat_dot_grad_bhat_dot_ghat": (
                    ["tau"],
                    f["yhat_dot_grad_bhat_dot_ghat_output"],
                ),
                "d_theta_d_tau": (["tau"], f["d_theta_d_tau"]),
                "d_xhat_d_tau_dot_yhat": (["tau"], f["d_xhat_d_tau_dot_yhat_output"]),
                "kappa_dot_xhat": (["tau"], f["kappa_dot_xhat_output"]),
                "kappa_dot_yhat": (["tau"], f["kappa_dot_yhat_output"]),
                "distance_along_line": (["tau"], f["distance_along_line"]),
                "cutoff_index": f["cutoff_index"],
                "poloidal_flux_on_midplane": (
                    ["R_midplane"],
                    f["poloidal_flux_on_midplane"],
                ),
                "theta": (["tau"], f["theta_output"]),
                "theta_m": (["tau"], f["theta_m_output"]),
                "delta_theta_m": (["tau"], f["delta_theta_m"]),
                "K_magnitude": (["tau"], f["K_magnitude_array"]),
                "k_perp_1_bs": (["tau"], f["k_perp_1_bs"]),
                "loc_b_r_s": (["tau"], f["loc_b_r_s"]),
                "loc_b_r": (["tau"], f["loc_b_r"]),
            },
            coords={
                "R_midplane": f["R_midplane_points"],
                "tau": outputs.tau,
                "row": outputs.row,
                "col": outputs.col,
                "q_R": outputs.q_R,
                "q_Z": outputs.q_Z,
                "q_zeta": outputs.q_zeta,
            },
        )

        if "l_lc" in f:
            l_lc = f["l_lc"]
        else:
            l_lc = df.distance_along_line - df.distance_along_line[df.cutoff_index]

        return df.assign_coords(
            {
                "l_lc": (
                    ["tau"],
                    l_lc.data,
                    {"long_name": "Distance from cutoff", "units": "m"},
                )
            }
        )


def beam_width(
    g_hat: xr.DataArray, orthogonal_dir: FloatArray, Psi_3D: xr.DataArray
) -> xr.DataArray:
    W_vec = np.cross(g_hat, orthogonal_dir)
    W_vec_magnitude = np.linalg.norm(W_vec, axis=1)

    # Unit vector
    W_uvec = W_vec / W_vec_magnitude[:, np.newaxis]
    width = np.sqrt(2 / dot(W_uvec, dot(Psi_3D, W_uvec)).imag)
    width = xr.DataArray(W_uvec * width[:, np.newaxis], coords=g_hat.coords)
    set_vector_components_long_name(width)
    return width
