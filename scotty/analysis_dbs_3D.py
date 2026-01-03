import logging
import numpy as np
import scipy.constants as const
from scotty.analysis import set_vector_components_long_name
from scotty.derivatives import derivative
from scotty.fun_general import dot, make_unit_vector_from_cross_product, find_electron_mass, find_H_Cardano, find_waist
from scotty.geometry_3D import MagneticField_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D, hessians_3D
from scotty.profile_fit import ProfileFitLike
from typing import Optional
import xarray as xr

log = logging.getLogger(__name__)

CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]

def analysis_dbs(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: xr.Dataset,
    hamiltonian: Hamiltonian_3D,
    hamiltonian_other: Hamiltonian_3D,
    field: MagneticField_Cartesian,
    density_fit: ProfileFitLike,
    temperature_fit: Optional[ProfileFitLike]):

    log.info(f"""\n
        ##################################################
        #
        # DOPPLER BACKSCATTERING ANALYSIS ROUTINE
        #
        ##################################################
        """)
    
    log.debug(f"Performing doppler backscattering analysis")

    # Find the distance along the arc
    # For line integral later
    dl_along_arc = np.sqrt( np.diff(solver_output.q_X)**2 +
                            np.diff(solver_output.q_Y)**2 +
                            np.diff(solver_output.q_Z)**2 )
    distance_along_arc = np.append(0, np.cumsum(dl_along_arc))

    # Find the index of the propagation corresponding to min(|K|)
    # i.e. the traditional cut-off
    K_mag = np.array(solver_output.K_magnitude)
    cutoff_idx = np.argmin(K_mag)

    # Find cos(theta)
    x_hat = np.array(analysis.x_hat)
    y_hat = np.array(analysis.y_hat)
    b_hat = np.array(analysis.b_hat)
    kperp1_hat = make_unit_vector_from_cross_product(y_hat, b_hat)
    sin_theta = -dot(x_hat, kperp1_hat)
    theta = np.sign(sin_theta) * np.arcsin(np.abs(sin_theta))
    cos_theta = np.cos(theta)

    # Find the backscattered stuff
    theta_m = np.array(analysis.theta_m)
    dpolflux_dX = np.array(analysis.dpolflux_dX)
    dpolflux_dY = np.array(analysis.dpolflux_dY)
    dpolflux_dZ = np.array(analysis.dpolflux_dZ)
    normal_vec = np.stack((dpolflux_dX, dpolflux_dY, dpolflux_dZ), axis=1)
    normal_mag = np.linalg.norm(normal_vec, axis=1)
    normal_hat = normal_vec / normal_mag[:, np.newaxis]
    binormal_hat = make_unit_vector_from_cross_product(b_hat, normal_hat)
    kperp1_bs = -2 * K_mag * np.cos(theta + theta_m) / cos_theta
    kperp1_bs_normal   = kperp1_bs * dot(kperp1_hat, normal_hat)   # TODO: Check that this works correctly
    kperp1_bs_binormal = kperp1_bs * dot(kperp1_hat, binormal_hat) # TODO: Check that this works correctly

    # Find the basis vector stuff
    g_hat = np.array(analysis.g_hat)
    x_hat = np.array(analysis.x_hat)
    y_hat = np.array(analysis.y_hat)
    grad_bhat = np.array(analysis.grad_bhat)
    xhat_dot_grad_bhat_dot_ghat = dot(x_hat, dot(grad_bhat, g_hat))
    yhat_dot_grad_bhat_dot_ghat = dot(y_hat, dot(grad_bhat, g_hat))

    ##################################################
    log.debug(f"Finding the wavenumber resolution and mismatch attenuation")
    ##################################################

    Psi_xx_beamframe_cartesian = np.array(analysis.Psi_xx_beamframe_cartesian)
    Psi_xy_beamframe_cartesian = np.array(analysis.Psi_xy_beamframe_cartesian)
    Psi_yy_beamframe_cartesian = np.array(analysis.Psi_yy_beamframe_cartesian)
    M_xx_beamframe_cartesian = np.array(Psi_xx_beamframe_cartesian) + (kperp1_bs / 2)*xhat_dot_grad_bhat_dot_ghat
    M_xy_beamframe_cartesian = np.array(Psi_xy_beamframe_cartesian) + (kperp1_bs / 2)*yhat_dot_grad_bhat_dot_ghat
    M_yy_beamframe_cartesian = np.array(Psi_yy_beamframe_cartesian)
    det_M_w = M_xx_beamframe_cartesian*M_yy_beamframe_cartesian - M_xy_beamframe_cartesian**2
    M_w_inv_xx =   M_yy_beamframe_cartesian / det_M_w
    M_w_inv_xy = - M_xy_beamframe_cartesian / det_M_w
    M_w_inv_yy =   M_xx_beamframe_cartesian / det_M_w
    delta_kperp2 = 2 * np.sqrt(-1 / np.imag(M_w_inv_yy))

    ##################################################
    log.debug(f"Finding the polarisation piece (loc_p)")
    ##################################################

    zero = np.zeros(len(analysis.tau))
    epsilon_para = np.array(analysis.epsilon_para)
    epsilon_perp = np.array(analysis.epsilon_perp)
    epsilon_g    = np.array(analysis.epsilon_g)
    e_hat = np.array(analysis.e_hat)
    e_hat_conj = np.conjugate(e_hat)
    n_e = np.array(analysis.electron_density)
    w_launch = hamiltonian.angular_frequency
    temperature = analysis.get("temperature")

    # equilibrium dielectric tensor - identity matrix, i.e.
    # \bm{\epsilon}_{eq} - \bm{1}
    epsilon_minus_identity = np.block([
        [[epsilon_perp],    [1j * epsilon_g], [zero]],
        [[-1j * epsilon_g], [epsilon_perp],   [zero]],
        [[zero],            [zero],           [epsilon_para]],
    ]).T - np.eye(3)

    # Avoids dividing a small number by another small number,
    # leading to a big number because of numerical errors or something
    loc_p_unnormalised = np.divide(
        np.abs(dot(e_hat_conj, dot(epsilon_minus_identity, e_hat)))**2,
        (n_e * 1e19) ** 2,
        out=np.zeros_like(n_e),
        where=(n_e > 1e-6))
    
    loc_p = (w_launch**2 * const.epsilon_0 * find_electron_mass(temperature) / const.e**2)**2 * loc_p_unnormalised

    ##################################################
    log.debug(f"Finding the ray piece (loc_r)")
    ##################################################

    mode_index = int(solver_output.mode_index)
    K_X = np.array(solver_output.K_X)
    K_Y = np.array(solver_output.K_Y)
    K_Z = np.array(solver_output.K_Z)
    delta_K_X = hamiltonian.spacings["K_X"]
    delta_K_Y = hamiltonian.spacings["K_Y"]
    delta_K_Z = hamiltonian.spacings["K_Z"]

    def _H_Cardano(Kx, Ky, Kz): 
        eigvals = find_H_Cardano(np.sqrt(Kx**2 + Ky**2 + Kz**2), w_launch, epsilon_para, epsilon_perp, epsilon_g, theta_m)
        return eigvals[mode_index]
    
    g_vec_X = derivative(_H_Cardano, "Kx", args={"Kx": K_X, "Ky": K_Y, "Kz": K_Z}, spacings=delta_K_X)
    g_vec_Y = derivative(_H_Cardano, "Ky", args={"Kx": K_X, "Ky": K_Y, "Kz": K_Z}, spacings=delta_K_Y)
    g_vec_Z = derivative(_H_Cardano, "Kz", args={"Kx": K_X, "Ky": K_Y, "Kz": K_Z}, spacings=delta_K_Z)
    g_vec = np.stack((g_vec_X, g_vec_Y, g_vec_Z), axis=1)
    g_mag = np.linalg.norm(g_vec, axis=1)

    # def _H_Cardano(Kx, Ky, Kz): 
    #     eigvals = find_H_Cardano(np.sqrt(Kx**2 + Ky**2 + Kz**2), w_launch, epsilon_para, epsilon_perp, epsilon_g, theta_m)
    #     return eigvals[mode_index]
    
    # def _grad_H_Cardano(direction, spacing):
    #     return derivative(_H_Cardano, direction, args={"K_X": K_X, "K_Y": K_Y, "K_Z": K_Z}, spacings=spacing)
    
    # g_vec_X = _grad_H_Cardano("K_X", spacing=delta_K_X)
    # g_vec_Y = _grad_H_Cardano("K_Y", spacing=delta_K_Y)
    # g_vec_Z = _grad_H_Cardano("K_Z", spacing=delta_K_Z)
    # g_vec = np.stack((g_vec_X, g_vec_Y, g_vec_Z), axis=1)
    # g_mag = np.linalg.norm(g_vec, axis=1)

    loc_r = (2*const.c / w_launch)**2 / g_mag**2

    ##################################################
    log.debug(f"Finding the beam piece (loc_b)")
    ##################################################

    launch_beam_width = float(inputs.launch_beam_width)
    launch_wavenumber = hamiltonian.wavenumber_K0
    launch_beam_curv = float(inputs.launch_beam_curvature)
    det_Im_Psi_w = np.imag(Psi_xx_beamframe_cartesian) * np.imag(Psi_yy_beamframe_cartesian) - np.imag(Psi_xy_beamframe_cartesian)**2
    beam_waist = find_waist(launch_beam_width, launch_wavenumber, launch_beam_curv)

    loc_b = (beam_waist * det_Im_Psi_w) / (np.sqrt(2) * np.abs(det_M_w) * np.sqrt(-np.imag(M_w_inv_yy)))

    ##################################################
    log.debug(f"Finding the mismatch piece (loc_m)")
    ##################################################

    delta_theta_m = np.sqrt(np.imag(M_w_inv_yy) / ((np.imag(M_w_inv_xy)) ** 2 - np.imag(M_w_inv_xx) * np.imag(M_w_inv_yy))) / K_mag
    loc_m = np.exp(-2 * (theta_m / delta_theta_m) ** 2)

    ##################################################
    log.debug(f"Finding the spectrum piece (loc_s)")
    ##################################################

    spectrum_power_laws = ["-10/3", "-13/3"]
    loc_s = {f"loc_s_{power}": np.array((kperp1_bs / (-2*launch_wavenumber)) ** eval(power)) for power in spectrum_power_laws}

    ##################################################
    log.debug(f"Finding total localisation (loc_p * loc_r * loc_b * loc_m * loc_s)")
    ##################################################

    l_lc = distance_along_arc - distance_along_arc[cutoff_idx]
    loc_coeff = 1/w_launch**2 # (np.pi**(3/2) * const.e**4) / ( 2*(const.c**2)*(w_launch**2)*(const.epsilon_0**2)*(const.m_e**2)*beam_waist )
    loc_total = {f"loc_all_{power}": loc_p * loc_r * loc_b * loc_m * loc_s[f"loc_s_{power}"] for power in spectrum_power_laws}
    power_ratio = {f"power_ratio_{power}_arb_units": loc_coeff * np.trapz(loc_total[f"loc_all_{power}"], distance_along_arc) for power in spectrum_power_laws}
    
    df = {
        # Important stuff
        "cutoff_index": cutoff_idx,
        "arc_length": (["tau"], distance_along_arc),
        "arc_length_relative_to_cutoff": (["tau"], l_lc),

        # Additional vector stuff
        "g_vector_X": (["tau"], g_vec_X),
        "g_vector_Y": (["tau"], g_vec_Y),
        "g_vector_Z": (["tau"], g_vec_Z),
        "g_vector": (["tau","col"], g_vec),
        "g_magnitude": (["tau"], g_mag),
        "kperp1_hat":   (["tau","col"], kperp1_hat),
        "normal_hat":   (["tau","col"], normal_hat), # TO REMOVE -- if this causes issues, dimension mismatch maybe?
        "binormal_hat": (["tau","col"], binormal_hat),
        "xhat_dot_grad_bhat_dot_ghat": (["tau"], xhat_dot_grad_bhat_dot_ghat),
        "yhat_dot_grad_bhat_dot_ghat": (["tau"], yhat_dot_grad_bhat_dot_ghat),

        # M in beam frame cartesian
        "M_xx": (["tau"], M_xx_beamframe_cartesian),
        "M_xy": (["tau"], M_xy_beamframe_cartesian),
        "M_yy": (["tau"], M_yy_beamframe_cartesian),
        "det_M_w": (["tau"], det_M_w),
        "M_w_inv_xx": (["tau"], M_w_inv_xx),
        "M_w_inv_xy": (["tau"], M_w_inv_xy),
        "M_w_inv_yy": (["tau"], M_w_inv_yy),

        # Backscattering and localisation stuff
        "theta": (["tau"], theta),
        "kperp1_bs":          (["tau"], kperp1_bs),
        "kperp1_bs_normal":   (["tau"], kperp1_bs_normal),
        "kperp1_bs_binormal": (["tau"], kperp1_bs_binormal),
        "dominant_kperp1_bs": kperp1_bs[cutoff_idx],
        "delta_kperp2": (["tau"], delta_kperp2),
        "delta_theta_m": (["tau"], delta_theta_m),
        "loc_p": (["tau"], loc_p),
        "loc_r": (["tau"], loc_r),
        "loc_b": (["tau"], loc_b),
        "loc_m": (["tau"], loc_m),
        **{f"{k}": (["tau"], v) for k, v in loc_s.items()},
        **{f"{k}": (["tau"], v) for k, v in loc_total.items()},
        **power_ratio,
    }

    # netCDF complains if keys contian '/' so
    # iterate along the keys and replace those with '_'
    def _replace_slash_with_underscore(d):
        for k in d:
            if "/" in k:
                _temp = d.pop(k)
                _k = k.replace("/", "_")
                d[_k] = _temp
    
    _replace_slash_with_underscore(df)

    # Assign the new coordinates used in further_df to the original df
    analysis.assign_coords({
        "l_lc": (["tau"], l_lc, {"long_name": "Distance from cutoff", "units": "m"}),
        "distance_along_line": (["tau"], distance_along_arc, {"units": "m"}),
    })
    
    set_vector_components_long_name(analysis)
    analysis.update(df)

    return analysis