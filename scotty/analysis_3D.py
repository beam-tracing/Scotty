import logging
import numpy as np
import pathlib
from scipy.constants import constants
from scotty.analysis import (
    dispersion_eigenvalues,
    set_vector_components_long_name,
)
from scotty.checks_3D import Parameters
from scotty.check_output import check_output
from scotty.derivatives import derivative
from scotty.fun_general import (
    angular_frequency_to_wavenumber,
    dot,
    find_electron_mass,
    find_H_Cardano,
    find_nearest,
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
    find_waist,
    make_unit_vector_from_cross_product,
)
from scotty.fun_general_3D import find_H_Cardano_eig, find_beam_widths_curvs
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import DielectricTensor_3D, Hamiltonian_3D, hessians_3D
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike, FloatArray
from typing import Dict, Optional
import xarray as xr

log = logging.getLogger(__name__)

CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]

def beam_tracing_analysis_3D(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    hamiltonian: Hamiltonian_3D,
    hamiltonian_other: Hamiltonian_3D,
    field: MagneticField_3D_Cartesian,
    density_fit: ProfileFitLike,
    temperature_fit: Optional[ProfileFitLike]):

    log.info(f"""\n
        ##################################################
        #
        # RAY AND BEAM TRACING ANALYSIS ROUTINE
        #
        ##################################################
        """)
    
    log.debug(f"Performing analysis on ray tracing results")

    tau_array = np.array(solver_output.tau)
    
    # Position and wavevector stuff
    q_vec = np.array(solver_output.q_vector)
    K_vec = np.array(solver_output.K_vector)
    K_hat = np.array(solver_output.K_hat)
    K_mag = np.array(solver_output.K_magnitude)

    # Calculating (Booker) Hamiltonian stuff
    H_Booker = hamiltonian(*q_vec.T, *K_vec.T)
    H_Booker_other = hamiltonian_other(*q_vec.T, *K_vec.T)
    dH = hamiltonian.derivatives(*q_vec.T, *K_vec.T, second_order=True)
    dH_dKx, dH_dKy, dH_dKz = dH["dH_dKx"], dH["dH_dKy"], dH["dH_dKz"]
    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians_3D(dH)
    
    # Finite difference spacings
    delta_X, delta_Y, delta_Z = hamiltonian.spacings["X"], hamiltonian.spacings["Y"], hamiltonian.spacings["Z"]

    # Poloidal flux stuff
    polflux = field.polflux(*q_vec.T)
    dp_dX = field.d_polflux_dX(*q_vec.T, delta_X)
    dp_dY = field.d_polflux_dY(*q_vec.T, delta_Y)
    dp_dZ = field.d_polflux_dZ(*q_vec.T, delta_Z)

    # Magnetic field stuff
    B_X = field.B_X(*q_vec.T)
    B_Y = field.B_Y(*q_vec.T)
    B_Z = field.B_Z(*q_vec.T)
    B_mag = field.magnitude(*q_vec.T)
    b_hat = field.unitvector(*q_vec.T)
    dbhat_dX = derivative(field.unitvector, dims="X", args={"X": q_vec.T[0], "Y": q_vec.T[1], "Z": q_vec.T[2]}, spacings=delta_X)
    dbhat_dY = derivative(field.unitvector, dims="Y", args={"X": q_vec.T[0], "Y": q_vec.T[1], "Z": q_vec.T[2]}, spacings=delta_Y)
    dbhat_dZ = derivative(field.unitvector, dims="Z", args={"X": q_vec.T[0], "Y": q_vec.T[1], "Z": q_vec.T[2]}, spacings=delta_Z)
    grad_bhat = np.stack((dbhat_dX, dbhat_dY, dbhat_dZ), axis=1)

    # Calculating plasma properties along ray path
    n_e = density_fit(polflux)
    T_e = temperature_fit(polflux) if temperature_fit else None
    epsilon = DielectricTensor_3D(n_e, hamiltonian.angular_frequency, B_mag, T_e)
    normalised_gyro_freqs = find_normalised_gyro_freq(n_e, hamiltonian.angular_frequency, T_e)
    normalised_plasma_freqs = find_normalised_plasma_freq(B_mag, hamiltonian.angular_frequency, T_e)

    # Calculating ray properties along path
    sin_theta_m = np.sum(b_hat * K_hat, axis=1)
    theta_m = np.sign(sin_theta_m) * np.arcsin(np.abs(sin_theta_m))

    log.debug(f"Performing analysis on beam tracing results")

    # Constructing {y, g, x} basis
    g_vec = np.stack((dH_dKx, dH_dKy, dH_dKz), axis=1)
    g_magnitude = np.linalg.norm(g_vec, axis=1)
    g_hat = g_vec / g_magnitude[:, np.newaxis]
    y_hat = make_unit_vector_from_cross_product(b_hat, g_hat)
    x_hat = make_unit_vector_from_cross_product(y_hat, g_hat)

    # TO REMOVE -- is this really necessary?
    # Psi_xx_entry_beamframe_cartesian = dot([x_hat[0,:]], dot([Psi_3D_entry_labframe_cartesian], [x_hat[0,:]]))
    # Psi_xy_entry_beamframe_cartesian = dot([x_hat[0,:]], dot([Psi_3D_entry_labframe_cartesian], [y_hat[0,:]]))
    # Psi_yy_entry_beamframe_cartesian = dot([y_hat[0,:]], dot([Psi_3D_entry_labframe_cartesian], [y_hat[0,:]]))

    # Calculate the entries of Psi_3D_labframe_cartesian in the beam frame
    Psi_3D_labframe_cartesian = np.array(solver_output.Psi_3D_labframe_cartesian)
    Psi_xx_beamframe_cartesian = dot(x_hat, dot(Psi_3D_labframe_cartesian, x_hat))
    Psi_xy_beamframe_cartesian = dot(x_hat, dot(Psi_3D_labframe_cartesian, y_hat))
    Psi_xg_beamframe_cartesian = dot(x_hat, dot(Psi_3D_labframe_cartesian, g_hat))
    Psi_yy_beamframe_cartesian = dot(y_hat, dot(Psi_3D_labframe_cartesian, y_hat))
    Psi_yg_beamframe_cartesian = dot(y_hat, dot(Psi_3D_labframe_cartesian, g_hat))
    Psi_gg_beamframe_cartesian = dot(g_hat, dot(Psi_3D_labframe_cartesian, g_hat))

    # Calculating beam properties (curvatures, widths) along ray path
    Psi_w = np.zeros((len(Psi_3D_labframe_cartesian), 2, 2), dtype=np.complex128)
    Psi_w[:, 0, 0] = Psi_xx_beamframe_cartesian
    Psi_w[:, 0, 1] = Psi_xy_beamframe_cartesian
    Psi_w[:, 1, 0] = Psi_xy_beamframe_cartesian
    Psi_w[:, 1, 1] = Psi_yy_beamframe_cartesian
    (curvature1, curvature2, width1, width2) = find_beam_widths_curvs(Psi_w, K_vec, g_hat)

    # Calculating (Cardano) Hamiltonian stuff
    mode_index = int(solver_output.mode_index)
    H_Cardano_eigvals, e_eigvecs = find_H_Cardano_eig(
        K_magnitude=K_mag, 
        launch_angular_frequency=hamiltonian.angular_frequency,
        epsilon_para=epsilon.e_bb,
        epsilon_perp=epsilon.e_11,
        epsilon_g=epsilon.e_12,
        theta_m=theta_m)
    e_hat = e_eigvecs[:, :, mode_index]

    H_Cardano_eigvals_analytical = find_H_Cardano(
        K_magnitude=K_mag,
        launch_angular_frequency=hamiltonian.angular_frequency,
        epsilon_para=epsilon.e_bb,
        epsilon_perp=epsilon.e_11,
        epsilon_g=epsilon.e_12,
        theta_m=theta_m)







    ### TO REMOVE -- compare if dbhats above and dBhats (implemented in base class) below are the same
    from scotty.logger_3D import timer

    (dbhat_dX_test, res) = timer(field.d_bhat_dX, *q_vec.T)
    # print(f"my method took {res} s")

    (dB_X_dX_test, res1) = timer(derivative, field.B_X, dims="X", args={"X": q_vec.T[0], "Y": q_vec.T[1], "Z": q_vec.T[2]}, spacings=delta_X)
    (dB_Y_dX_test, res2) = timer(derivative, field.B_Y, dims="X", args={"X": q_vec.T[0], "Y": q_vec.T[1], "Z": q_vec.T[2]}, spacings=delta_X)
    (dB_Z_dX_test, res3) = timer(derivative, field.B_Z, dims="X", args={"X": q_vec.T[0], "Y": q_vec.T[1], "Z": q_vec.T[2]}, spacings=delta_X)
    # print(f"normal scotty takes {res1+res2+res3} s")

    b_vec = B_mag[:, np.newaxis] * b_hat
    B_X_test, B_Y_test, B_Z_test = b_vec[:, 0], b_vec[:, 1], b_vec[:, 2]

    dp_dX_test = field.d_polflux_dX_TEST(*q_vec.T)
    dp_dY_test = field.d_polflux_dY_TEST(*q_vec.T)
    dp_dZ_test = field.d_polflux_dZ_TEST(*q_vec.T)

    dB_magnitude_dX_test = derivative(field.magnitude, dims="X", args={"X": q_vec.T[0], "Y": q_vec.T[1], "Z": q_vec.T[2]}, spacings=delta_X)







    # Saving the data and updating the original dict
    df = xr.Dataset(
        {
            # H Booker and derivatives stuff
            "H_Booker": (["tau"], H_Booker),
            "H_Booker_other": (["tau"], H_Booker_other),
            **{f"{k}": (["tau"], v) for k, v in dH.items()},
            "grad_grad_H":   (["tau","row","col"], grad_grad_H),
            "gradK_grad_H":  (["tau","row","col"], gradK_grad_H),
            "gradK_gradK_H": (["tau","row","col"], gradK_gradK_H),

            # H Cardano and eigenvectors
            "H_Cardano_1": (["tau"], H_Cardano_eigvals[:, 0]),
            "H_Cardano_2": (["tau"], H_Cardano_eigvals[:, 1]),
            "H_Cardano_3": (["tau"], H_Cardano_eigvals[:, 2]),
            "e_hat_1": (["tau", "col"], e_eigvecs[:, :, 0]),
            "e_hat_2": (["tau", "col"], e_eigvecs[:, :, 1]),
            "e_hat_3": (["tau", "col"], e_eigvecs[:, :, 2]),
            "e_hat":   (["tau", "col"], e_hat),
            "H_Cardano_1_analytical": (["tau"], H_Cardano_eigvals_analytical[0]),
            "H_Cardano_2_analytical": (["tau"], H_Cardano_eigvals_analytical[1]),
            "H_Cardano_3_analytical": (["tau"], H_Cardano_eigvals_analytical[2]),

            # Poloidal flux stuff
            "polflux":     (["tau"], polflux),
            "dpolflux_dX": (["tau"], dp_dX),
            "dpolflux_dY": (["tau"], dp_dY),
            "dpolflux_dZ": (["tau"], dp_dZ),

            # Magnetic field stuff
            "B_X": (["tau"], B_X),
            "B_Y": (["tau"], B_Y),
            "B_Z": (["tau"], B_Z),
            "B_magnitude": (["tau"], B_mag),
            "b_hat":     (["tau","col"], b_hat),
            "dbhat_dX":  (["tau","col"], dbhat_dX),
            "dbhat_dY":  (["tau","col"], dbhat_dY),
            "dbhat_dZ":  (["tau","col"], dbhat_dZ),
            "grad_bhat": (["tau","row","col"], grad_bhat),

            # Plasma properties along ray path
            "electron_density": (["tau"], n_e),
            "epsilon_para": (["tau"], epsilon.e_bb),
            "epsilon_bb":   (["tau"], epsilon.e_bb),
            "epsilon_perp": (["tau"], epsilon.e_11),
            "epsilon_11":   (["tau"], epsilon.e_11),
            "epsilon_g":    (["tau"], epsilon.e_12),
            "epsilon_12":   (["tau"], epsilon.e_12),
            "normalised_gyro_freqs": (["tau"], normalised_gyro_freqs),
            "normalised_plasma_freqs": (["tau"], normalised_plasma_freqs),

            # Ray properties along path
            "theta_m": (["tau"], theta_m),

            # Basis vectors
            "g_magnitude": (["tau"], g_magnitude),
            "g_hat": (["tau","col"], g_hat),
            "x_hat": (["tau","col"], x_hat),
            "y_hat": (["tau","col"], y_hat),

            # Entries of Psi_3D_beamframe_cartesian
            "Psi_xx_beamframe_cartesian": (["tau"], Psi_xx_beamframe_cartesian),
            "Psi_xy_beamframe_cartesian": (["tau"], Psi_xy_beamframe_cartesian),
            "Psi_xg_beamframe_cartesian": (["tau"], Psi_xg_beamframe_cartesian),
            "Psi_yy_beamframe_cartesian": (["tau"], Psi_yy_beamframe_cartesian),
            "Psi_yg_beamframe_cartesian": (["tau"], Psi_yg_beamframe_cartesian),
            "Psi_gg_beamframe_cartesian": (["tau"], Psi_gg_beamframe_cartesian),

            # Beam properties
            "beam_curvature_1": (["tau"], curvature1),
            "beam_curvature_2": (["tau"], curvature2),
            "beam_width_1": (["tau"], width1),
            "beam_width_2": (["tau"], width2),





            # TO REMOVE -- for testing only
            "dbhat_dX_test": (["tau","col"], dbhat_dX_test),
            "dB_X_dX_test": (["tau"], dB_X_dX_test),
            "dB_Y_dX_test": (["tau"], dB_Y_dX_test),
            "dB_Z_dX_test": (["tau"], dB_Z_dX_test),
            "dB_magnitude_dX_test": (["tau"], dB_magnitude_dX_test),
            "B_X_test": (["tau"], B_X_test),
            "B_Y_test": (["tau"], B_Y_test),
            "B_Z_test": (["tau"], B_Z_test),
            "dp_dX_test": (["tau"], dp_dX_test),
            "dp_dY_test": (["tau"], dp_dY_test),
            "dp_dZ_test": (["tau"], dp_dZ_test),
        },
        coords = {
            "tau": (["tau"], tau_array, {"long_name": "Parametrised distance along ray"}),
            "row": CARTESIAN_VECTOR_COMPONENTS,
            "col": CARTESIAN_VECTOR_COMPONENTS,
            "q_X": q_vec.T[0],
            "q_Y": q_vec.T[1],
            "q_Z": q_vec.T[2],
        },
    )

    set_vector_components_long_name(df)

    if T_e: df.update({"temperature": (["tau"], T_e)})

    return df











# def further_analysis_3D(
#     params: Parameters,
#     inputs: xr.Dataset,
#     solver_output: xr.Dataset,
#     df: xr.Dataset,
#     Psi_3D_entry_labframe_cartesian: FloatArray,
#     output_path: pathlib.Path,
#     output_filename_suffix: str,
#     field: MagneticField_3D_Cartesian,
#     detailed_analysis_flag: bool,
#     dH: Dict[str, ArrayLike]):

#     """
#     Need to make sure the shapes of hat vectors are all correct and that they output correct stuff
#     """

#     # Getting the important data
#     q_X, q_Y, q_Z = np.array(solver_output.q_X), np.array(solver_output.q_Y), np.array(solver_output.q_Z)
#     K_X, K_Y, K_Z = np.array(solver_output.K_X), np.array(solver_output.K_Y), np.array(solver_output.K_Z)
#     K = np.stack((K_X, K_Y, K_Z), axis=1)
#     K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
#     numberOfDataPoints = len(df.tau)

#     # To calculate some properties of the ray trajectory
#     point_spacings = np.sqrt(np.diff(q_X)**2 + np.diff(q_Y)**2 + np.diff(q_Z)**2)
#     distance_along_line = np.append(0, np.cumsum(point_spacings))

#     # Find the index of the propagation at which K is the smallest
#     # This corresponds to the cut-off
#     index_of_cutoff = np.argmin(K_magnitude)

#     # Find theta
#     kperp1_hat = make_unit_vector_from_cross_product(df.y_hat, df.b_hat)
#     sin_theta = -dot(df.x_hat, kperp1_hat)
#     theta = np.sign(sin_theta) * np.arcsin(np.abs(sin_theta))
#     cos_theta = np.cos(theta)

#     # Find theta_m
#     b_hat = np.array(df.b_hat)
#     K_hat = np.column_stack((K_X, K_Y, K_Z)) / K_magnitude[:, np.newaxis]
#     sin_theta_m = np.sum(b_hat * K_hat, axis=1)
#     theta_m = np.sign(sin_theta_m) * np.arcsin(np.abs(sin_theta_m))

#     # Find the backscattered stuff
#     dpolflux_dX, dpolflux_dY, dpolflux_dZ = np.array(df.dpolflux_dX), np.array(df.dpolflux_dY), np.array(df.dpolflux_dZ)
#     dpolflux_magnitudes = np.sqrt(dpolflux_dX**2 + dpolflux_dY**2 + dpolflux_dZ**2)
#     normal_hat = np.column_stack((dpolflux_dX, dpolflux_dY, dpolflux_dZ)) / dpolflux_magnitudes[:, np.newaxis]
#     binormal_hat = make_unit_vector_from_cross_product(b_hat, normal_hat)
#     kperp1_bs = -2 * K_magnitude * np.cos(theta + theta_m) / cos_theta
#     kperp1_bs_normal   = kperp1_bs * dot(kperp1_hat, normal_hat)   # TODO: Check that this works correctly
#     kperp1_bs_binormal = kperp1_bs * dot(kperp1_hat, binormal_hat) # TODO: Check that this works correctly

#     # Find the basis vector stuff
#     g_hat = np.array(df.g_hat)
#     x_hat = np.array(df.x_hat)
#     y_hat = np.array(df.y_hat)
#     xhat_dot_grad_bhat_dot_ghat = dot(x_hat, dot(df.grad_bhat, g_hat))
#     yhat_dot_grad_bhat_dot_ghat = dot(y_hat, dot(df.grad_bhat, g_hat))

#     # Find the entries of the modified matrix M of Psi
#     Psi_xx_beamframe_cartesian = np.array(df.Psi_xx_beamframe_cartesian)
#     Psi_xy_beamframe_cartesian = np.array(df.Psi_xy_beamframe_cartesian)
#     Psi_yy_beamframe_cartesian = np.array(df.Psi_yy_beamframe_cartesian)
#     M_xx_beamframe_cartesian = Psi_xx_beamframe_cartesian + (kperp1_bs / 2)*xhat_dot_grad_bhat_dot_ghat
#     M_xy_beamframe_cartesian = Psi_xy_beamframe_cartesian + (kperp1_bs / 2)*yhat_dot_grad_bhat_dot_ghat
#     M_yy_beamframe_cartesian = Psi_yy_beamframe_cartesian

#     """
#     Find the beam width and curvature along the ray

#     TO REMOVE -- shifted to beam_tracing_analysis_3D
#     """
#     # Psi_w = np.zeros((len(Psi_3D_labframe_cartesian), 2, 2), dtype=np.complex128)
#     # Psi_w[:, 0, 0] = Psi_xx_beamframe_cartesian
#     # Psi_w[:, 0, 1] = Psi_xy_beamframe_cartesian
#     # Psi_w[:, 1, 0] = Psi_xy_beamframe_cartesian
#     # Psi_w[:, 1, 1] = Psi_yy_beamframe_cartesian
#     # (curvature1,
#     #  curvature2,
#     #  width1,
#     #  width2) = find_beam_widths_curvs(Psi_w, K, g_hat)
#     # Re_Psi_w = np.real(Psi_w)
#     # Re_Psi_w_eigvals = np.linalg.eigvalsh(Re_Psi_w)
#     # K_g_magnitude = np.sum(K * g_hat, axis=1)
#     # curvature1, curvature2 = K_g_magnitude**2 / K_magnitude**3 * Re_Psi_w_eigvals[:, 0], K_g_magnitude**2 / K_magnitude**3 * Re_Psi_w_eigvals[:, 1]
#     # Im_Psi_w = np.imag(Psi_w)
#     # Im_Psi_w_eigvals = np.linalg.eigvalsh(Im_Psi_w)
#     # width1, width2 = np.sqrt( 2 / Im_Psi_w_eigvals[:, 0] ), np.sqrt( 2 / Im_Psi_w_eigvals[:, 1] )

#     """
#     Find the wavenumber resolution and mismatch attenuation along the ray
#     """
#     det_M_w = M_xx_beamframe_cartesian*M_yy_beamframe_cartesian - M_xy_beamframe_cartesian**2
#     M_w_inv_xx =   M_yy_beamframe_cartesian / det_M_w
#     M_w_inv_xy = - M_xy_beamframe_cartesian / det_M_w
#     M_w_inv_yy =   M_xx_beamframe_cartesian / det_M_w
#     delta_kperp2 = 2 * np.sqrt(-1 / np.imag(M_w_inv_yy))

#     """
#     Find the the polarisation piece (loc_p) along the ray
#     """
#     H_eigvals, e_eigvecs = dispersion_eigenvalues(K_magnitude, inputs.launch_angular_frequency.data, df, numberOfDataPoints, theta_m)
#     # H_1_eigval, H_2_eigval, H_3_eigval = H_eigvals
#     # e_1_eigvec, e_2_eigvec, e_3_eigvec = e_eigvecs

#     # In my experience, H_eigvals[:,1] corresponds to the O mode, and H_eigvals[:,0] corresponds to the X-mode
#     # NOTE: This may not always be the case! Check the output figure to make sure that
#     # the appropriate solution is indeed 0 along the ray
#     mode_index = params.mode_index
#     # TO REMOVE -- is this still needed? since we explicitly select mode_index in launch.py now
#     # if mode_index == None: mode_index = 1 if inputs.mode_flag == 1 else 0

#     # e_hat has components e_1, e_2, e_b
#     e_hat = e_eigvecs[:, :, mode_index]

#     # equilibrium dielectric tensor - identity matrix. \bm{\epsilon}_{eq} - \bm{1}
#     zero = np.zeros(len(df.tau))
#     epsilon_minus_identity = np.block(
#         [
#             [[df.epsilon_perp],    [1j * df.epsilon_g], [zero]],
#             [[-1j * df.epsilon_g], [df.epsilon_perp],   [zero]],
#             [[zero],               [zero],              [df.epsilon_para]],
#         ]
#     ).T - np.eye(3)

#     # Avoids dividing a small number by another small number,
#     # leading to a big number because of numerical errors or something
#     loc_p_unnormalised = np.divide(
#         np.abs(dot(np.conjugate(e_hat), dot(epsilon_minus_identity, e_hat)))**2,
#         (df.electron_density * 1e19) ** 2,
#         out=np.zeros_like(df.electron_density),
#         where=(df.electron_density > 1e-6).data)
#     loc_p = (inputs.launch_angular_frequency**2 * constants.epsilon_0 * find_electron_mass(df.get("temperature")) / constants.e**2)**2 * loc_p_unnormalised

#     """
#     Find the the dispersion relation using Cardano's formula and the ray piece (loc_r) along the ray
#     """
#     H_1_Cardano, H_2_Cardano, H_3_Cardano = find_H_Cardano(
#         K_magnitude,
#         inputs.launch_angular_frequency.data,
#         df.epsilon_para.data,
#         df.epsilon_perp.data,
#         df.epsilon_g.data,
#         theta_m)

#     def H_cardano(Kx, Ky, Kz):
#         # In my experience, the H_3_Cardano expression corresponds to
#         # the O mode, and the H_2_Cardano expression corresponds to
#         # the X-mode.

#         # ALERT: This may not always be the case! Check the output
#         # figure to make sure that the appropriate solution is indeed
#         # 0 along the ray
#         eigvals = find_H_Cardano(
#             np.sqrt(Kx**2 + Ky**2 + Kz**2),
#             inputs.launch_angular_frequency.data,
#             df.epsilon_para.data,
#             df.epsilon_perp.data,
#             df.epsilon_g.data,
#             theta_m)
        
#         return eigvals[mode_index]
    
#     def grad_H_cardano(direction: str, spacing: float):
#         return derivative(H_cardano, direction, args={"Kx": K_X, "Ky": K_Y, "Kz": K_Z}, spacings=spacing,)
    
#     g_X_Cardano = grad_H_cardano("Kx", inputs.delta_K_X.data)
#     g_Y_Cardano = grad_H_cardano("Ky", inputs.delta_K_Y.data)
#     g_Z_Cardano = grad_H_cardano("Kz", inputs.delta_K_Z.data)
#     g_magnitude_Cardano = np.sqrt(g_X_Cardano**2 + g_Y_Cardano**2 + g_Z_Cardano**2)

#     loc_r = (2*constants.c / inputs.launch_angular_frequency.data)**2 / g_magnitude_Cardano**2

#     """
#     Find the beam piece (loc_b) along the ray
#     """
#     wavenumber_K0 = angular_frequency_to_wavenumber(inputs.launch_angular_frequency.data)
#     det_Im_Psi_w = np.imag(Psi_xx_beamframe_cartesian)*np.imag(Psi_yy_beamframe_cartesian) - np.imag(Psi_xy_beamframe_cartesian)**2
#     beam_waist = find_waist(inputs.launch_beam_width.data, wavenumber_K0, inputs.launch_beam_curvature.data)
#     loc_b = (beam_waist * det_Im_Psi_w) / (np.sqrt(2) * np.abs(det_M_w) * np.sqrt(-np.imag(M_w_inv_yy)))

#     """
#     Find the mismatch piece (loc_m) along the ray
#     """
#     delta_theta_m = np.sqrt(np.imag(M_w_inv_yy) / ((np.imag(M_w_inv_xy)) ** 2 - np.imag(M_w_inv_xx) * np.imag(M_w_inv_yy))) / K_magnitude
#     loc_m = np.exp(-2 * (theta_m / delta_theta_m) ** 2)

#     """
#     Find the spectrum piece (loc_s) along the ray
#     """
#     spectrum_power_law_coefficient = -13/3 # Turbulence cascade
#     loc_s_13_3 = (kperp1_bs / (-2*wavenumber_K0)) ** (spectrum_power_law_coefficient)
#     loc_s_10_3 = (kperp1_bs / (-2*wavenumber_K0)) ** (-10/3)

#     # Combining the localisation pieces to get some overall localisation
#     l_lc = distance_along_line - distance_along_line[index_of_cutoff]
#     loc_all_13_3 = loc_p * loc_r * loc_b * loc_m * loc_s_13_3
#     loc_all_10_3 = loc_p * loc_r * loc_b * loc_m * loc_s_10_3
#     grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians_3D(dH)

#     # Integrating the localisations to get the backscattered power due to the dominant k_perp
#     loc_coeff = 1/inputs.launch_angular_frequency.data**2 # (np.pi**(3/2) * constants.e**4) / ( 2*(constants.c**2)*(inputs.launch_angular_frequency.data**2)*(constants.epsilon_0**2)*(constants.m_e**2)*beam_waist )
#     power_ratio_13_3_arb_units = loc_coeff * np.trapz(loc_all_13_3, distance_along_line)
#     power_ratio_10_3_arb_units = loc_coeff * np.trapz(loc_all_10_3, distance_along_line)

#     further_df = {
#         # Important stuff
#         "cutoff_index": index_of_cutoff,
#         "arc_length": (["tau"], distance_along_line),
#         "arc_length_relative_to_cutoff": (["tau"], l_lc),

#         # Basis vectors {y, g, x}
#         "x_hat_cartesian": (["tau","col"], x_hat),
#         "y_hat_cartesian": (["tau","col"], y_hat),
#         "g_hat_cartesian": (["tau","col"], g_hat),

#         # Additional vector stuff
#         "xhat_dot_grad_bhat_dot_ghat": (["tau"], xhat_dot_grad_bhat_dot_ghat),
#         "yhat_dot_grad_bhat_dot_ghat": (["tau"], yhat_dot_grad_bhat_dot_ghat),
#         # "normal_hat": (["tau","row","col"], normal_hat), # dimension mismatch, apparently

#         # Hessians
#         "grad_grad_H":   (["tau","row","col"], grad_grad_H),
#         "gradK_grad_H":  (["tau","row","col"], gradK_grad_H),
#         "gradK_gradK_H": (["tau","row","col"], gradK_gradK_H),

#         # M in beam frame cartesian
#         "M_xx": (["tau"], M_xx_beamframe_cartesian),
#         "M_xy": (["tau"], M_xy_beamframe_cartesian),
#         "M_yy": (["tau"], M_yy_beamframe_cartesian),
#         "det_M_w": (["tau"], det_M_w),
#         "M_w_inv_xx": (["tau"], M_w_inv_xx),
#         "M_w_inv_xy": (["tau"], M_w_inv_xy),
#         "M_w_inv_yy": (["tau"], M_w_inv_yy),

#         # Cardano Hamiltonian stuff
#         "H_1_Cardano": (["tau"], H_1_Cardano),
#         "H_2_Cardano": (["tau"], H_2_Cardano),
#         "H_3_Cardano": (["tau"], H_3_Cardano),
#         "H_eigvals":  (["tau","col"], H_eigvals),       # TODO the second index should be 1,2,3; not "col" which is X,Y,Z
#         # "H_1_eigval": (["tau"], H_1_eigval),
#         # "H_2_eigval": (["tau"], H_2_eigval),
#         # "H_3_eigval": (["tau"], H_3_eigval),
#         "e_eigvecs":  (["tau","row","col"], e_eigvecs), # TODO the third index should be 1,2,3; not "col" which is X,Y,Z
#         # "e_1_eigvec": (["tau","col"], e_1_eigvec),
#         # "e_2_eigvec": (["tau","col"], e_2_eigvec),
#         # "e_3_eigvec": (["tau","col"], e_3_eigvec),
#         "e_hat":      (["tau","col"], e_hat),

#         # Localisation stuff
#         "kperp1_hat": (["tau","row","col"], e_eigvecs), # TODO the third index should be 1,2,3; not "col" which is X,Y,Z
#         "kperp1_bs":          (["tau"], kperp1_bs),
#         "kperp1_bs_normal":   (["tau"], kperp1_bs_normal),
#         "kperp1_bs_binormal": (["tau"], kperp1_bs_binormal),
#         "delta_kperp2": (["tau"], delta_kperp2),
#         "theta": (["tau"], theta),
#         "theta_m": (["tau"], theta_m),
#         "delta_theta_m": delta_theta_m,
#         "loc_p": loc_p,
#         "loc_r": loc_r,
#         "loc_b": loc_b,
#         "loc_m": loc_m,
#         "loc_s_13_3": loc_s_13_3,
#         "loc_s_10_3": loc_s_10_3,
#         "loc_all_13_3": loc_all_13_3,
#         "loc_all_10_3": loc_all_10_3,
#         "power_ratio_13_3_arb_units": power_ratio_13_3_arb_units,
#         "power_ratio_10_3_arb_units": power_ratio_10_3_arb_units,
#         "dominant_kperp1_bs": kperp1_bs[index_of_cutoff],
#     }

#     # Assign the new coordinates used in further_df to the original df
#     df.assign_coords({
#             "l_lc": (["tau"], l_lc, {"long_name": "Distance from cutoff", "units": "m"}),
#             "distance_along_line": (["tau"], distance_along_line, {"units": "m"}),
#         })
    
#     set_vector_components_long_name(df)
#     df.update(further_df)

#     return df