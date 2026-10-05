import logging
import numpy as np
from scotty.checks_v4 import VALID_FIELDS
from scotty.derivatives import derivative
from scotty.fun_general import make_unit_vector_from_cross_product, dot
from scotty.fun_general_v4 import find_H_Cardano_eigh, find_H_Cardano_formula, find_beam_widths_and_curvs, find_g_cartesian, find_normalised_angular_freqs
from scotty.hamiltonian_v4 import DielectricTensor, Hamiltonian, convert_hessians, hessians
from scotty.profile_fit import ProfileFitLike
from typing import Optional
import xarray as xr

log = logging.getLogger(__name__)

CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]

def _temp(name, value):
    if value is not None: log.info(f"{name} -- {value.shape} -- {value}")
    else:                 log.info(f"{name} -- {value} -- {value}")
    return value

def _temp_print_dict(analysis_dict: dict):
    for k, v in analysis_dict.items(): log.info(f"{k} -- {v if v is None else v[-1].shape} -- {v if v is None else v[-1]}")

def basic_analysis(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    hamiltonian: Hamiltonian,
    hamiltonian_other: Hamiltonian,
    field: VALID_FIELDS,
    density_fit: ProfileFitLike,
    temperature_fit: Optional[ProfileFitLike]):

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

    analysis_dict = {}
    cart = bool(inputs["geometry"] == "cartesian")
    btf = not bool(inputs["ray_tracing_flag"])
    tau = np.array(solver_output["tau"][()])
    len_tau = len(tau)

    ##################################################
    # Position and wavevector stuff
    ##################################################
    q_vec_cart = np.array(solver_output["q_output_cartesian"][()])   # (N,3)
    q_vec_cyld = np.array(solver_output["q_output_cylindrical"][()]) # (N,3)
    K_vec_cart = np.array(solver_output["K_output_cartesian"][()])   # (N,3)
    K_vec_cyld = np.array(solver_output["K_output_cylindrical"][()]) # (N,3)
    K_mag = np.array(solver_output["K_output_magnitude"][()]) # (N,)

    ##################################################
    # Booker Hamiltonian stuff
    ##################################################
    if cart: q_vec, K_vec = q_vec_cart, K_vec_cart # (N,3) each
    else:    q_vec, K_vec = q_vec_cyld, K_vec_cyld # (N,3) each
    q_vec_T, K_vec_T = q_vec.T, K_vec.T
    dH = hamiltonian.derivatives(q_vec_T, K_vec_T, second_order=btf)
    analysis_dict.update({
        "H_Booker": (["tau"], hamiltonian(*q_vec_T, *K_vec_T)), # (N,) # type: ignore
        "H_Booker_other": (["tau"], hamiltonian_other(*q_vec_T, *K_vec_T)), # (N,) # type: ignore
    })

    # TO REMOVE
    _temp_print_dict(analysis_dict)

    ##################################################
    # Magnetic field and poloidal flux stuff
    ##################################################
    if cart:
        # field.all?
        B_X = _temp("B_X" , field.B_X(*q_vec_T)) # (N,)
        B_Y = _temp("B_Y" , field.B_Y(*q_vec_T)) # (N,)
        B_Z = _temp("B_Z" , field.B_Z(*q_vec_T)) # (N,)
        B_mag = _temp("B_mag" , field.magnitude(*q_vec_T)) # (N,)
        b_hat_cart = _temp("b_hat" , field.unitvector(*q_vec_T)) # (N,3)
        dbhat_dX = _temp("dbhat_dX" , derivative(field.unitvector_cartesian, dims="X", args={"X": q_vec_T[0], "Y": q_vec_T[1], "Z": q_vec_T[2]}, spacings=hamiltonian.delta_X)) # (N,3)
        dbhat_dY = _temp("dbhat_dY" , derivative(field.unitvector_cartesian, dims="Y", args={"X": q_vec_T[0], "Y": q_vec_T[1], "Z": q_vec_T[2]}, spacings=hamiltonian.delta_Y)) # (N,3)
        dbhat_dZ = _temp("dbhat_dZ" , derivative(field.unitvector_cartesian, dims="Z", args={"X": q_vec_T[0], "Y": q_vec_T[1], "Z": q_vec_T[2]}, spacings=hamiltonian.delta_Z)) # (N,3)
        grad_bhat_cart = _temp("grad_bhat" , np.stack((dbhat_dX, dbhat_dY, dbhat_dZ), axis=1)) # (N,3,3)

        polflux = _temp("polflux" , field.polflux(*q_vec_T)) # (N,)
        dp_dX = _temp("dp_dX" , field.d_polflux_dX(*q_vec_T, hamiltonian.delta_X)) # (N,)
        dp_dY = _temp("dp_dY" , field.d_polflux_dY(*q_vec_T, hamiltonian.delta_Y)) # (N,)
        dp_dZ = _temp("dp_dZ" , field.d_polflux_dZ(*q_vec_T, hamiltonian.delta_Z)) # (N,)
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
        b_hat_cart = field.unitvector_cartesian(*q_vec_T)
        dbhat_dX = derivative(field.unitvector_cartesian, dims="X", args={"X": q_vec_cart.T[0], "Y": q_vec_cart.T[1], "Z": q_vec_cart.T[2]}, spacings=hamiltonian.delta_R)
        dbhat_dY = derivative(field.unitvector_cartesian, dims="Y", args={"X": q_vec_cart.T[0], "Y": q_vec_cart.T[1], "Z": q_vec_cart.T[2]}, spacings=hamiltonian.delta_R)
        dbhat_dZ = derivative(field.unitvector_cartesian, dims="Z", args={"X": q_vec_cart.T[0], "Y": q_vec_cart.T[1], "Z": q_vec_cart.T[2]}, spacings=hamiltonian.delta_Z)
        grad_bhat_cart = np.stack((dbhat_dX, dbhat_dY, dbhat_dZ), axis=1)
        
        polflux = field.polflux(*q_vec_T)
        dp_dX = field.d_polflux_dX(*q_vec_T, hamiltonian.delta_R)
        dp_dY = field.d_polflux_dY(*q_vec_T, hamiltonian.delta_R)
        dp_dZ = field.d_polflux_dZ(*q_vec_T, hamiltonian.delta_Z)

    analysis_dict.update({
        "B_X": (["tau"], B_X),
        "B_Y": (["tau"], B_Y),
        "B_Z": (["tau"], B_Z),
        "B_magnitude": (["tau"], B_mag),
        "b_hat": (["tau", "row_cart"], b_hat_cart),
        "dbhat_dX": (["tau", "row_cart"], dbhat_dX),
        "dbhat_dY": (["tau", "row_cart"], dbhat_dY),
        "dbhat_dZ": (["tau", "row_cart"], dbhat_dZ),
        "grad_bhat": (["tau", "row_cart", "col_cart"], grad_bhat_cart),
        "polflux": (["tau"], polflux),
        "dp_dX": (["tau"], dp_dX),
        "dp_dY": (["tau"], dp_dY),
        "dp_dZ": (["tau"], dp_dZ),
    })

    # TO REMOVE
    _temp_print_dict(analysis_dict)
     
    ##################################################
    # Plasma properties along path
    ##################################################
    n_e = _temp("n_e", density_fit(polflux)) # (N,)
    T_e = _temp("T_e", temperature_fit(polflux) if temperature_fit else None)
    omega_launch = float(inputs["launch_angular_frequency"][()])
    epsilon = DielectricTensor(
        launch_angular_freq = omega_launch,
        B_magnitude = B_mag,
        electron_density = n_e,
        temperature = T_e)
    
    (   norm_omega_pe,
        norm_omega_ce,
        norm_omega_LH,
        norm_omega_RH,
        norm_omega_UH,
    ) = find_normalised_angular_freqs(
        launch_angular_freq = omega_launch,
        B_total = B_mag,
        electron_density = n_e,
        temperature = T_e)

    analysis_dict.update({
        "electron_density": (["tau"], n_e),
        "electron_temperature": (["tau"], T_e) if T_e is not None else None,
        "e_bb": (["tau"], epsilon.e_bb),
        "e_11": (["tau"], epsilon.e_11),
        "e_12": (["tau"], epsilon.e_12),
        "normalised_omega_pe": (["tau"], norm_omega_pe),
        "normalised_omega_ce": (["tau"], norm_omega_ce),
        "normalised_omega_LH": (["tau"], norm_omega_LH),
        "normalised_omega_RH": (["tau"], norm_omega_RH),
        "normalised_omega_UH": (["tau"], norm_omega_UH),
    })

    # TO REMOVE
    _temp_print_dict(analysis_dict)

    ##################################################
    # Ray properties along path
    ##################################################
    sin_theta_m = np.sum(b_hat_cart * K_vec_cart/K_mag[:, np.newaxis], axis=1)
    theta_m = np.sign(sin_theta_m) * np.arcsin(np.abs(sin_theta_m))

    g_vec_cart, g_mag, g_hat_cart = find_g_cartesian(cart, *q_vec_T, dH) # type: ignore
    y_hat_cart = make_unit_vector_from_cross_product(b_hat_cart, g_hat_cart)
    x_hat_cart = make_unit_vector_from_cross_product(y_hat_cart, g_hat_cart)

    analysis_dict.update({
        "sin_theta_m": (["tau"], sin_theta_m),
        "theta_m": (["tau"], theta_m),
        "g_vector_cartesian": (["tau"], g_vec_cart),
        "g_magnitude":     (["tau"], g_mag),
        "g_hat_cartesian": (["tau"], g_hat_cart),
        "y_hat_cartesian": (["tau"], y_hat_cart),
        "x_hat_cartesian": (["tau"], x_hat_cart),
    })

    # TO REMOVE
    _temp_print_dict(analysis_dict)

    ##################################################
    # Dispersion relation and polarisation vector
    ##################################################
    H_Cardano_1, H_Cardano_2, H_Cardano_3 = find_H_Cardano_formula(
        launch_angular_freq = omega_launch,
        K_magnitude = K_mag,
        epsilon_para = epsilon.e_bb,
        epsilon_perp = epsilon.e_11,
        epsilon_g = epsilon.e_12,
        theta_m = theta_m,
    )

    H_eigvals, e_eigvecs = find_H_Cardano_eigh(
        launch_angular_freq = omega_launch,
        K_magnitude = K_mag,
        epsilon_para = epsilon.e_bb,
        epsilon_perp = epsilon.e_11,
        epsilon_g = epsilon.e_12,
        theta_m = theta_m,
    )
    H_eigval1 = H_eigvals[:,0]
    H_eigval2 = H_eigvals[:,1]
    H_eigval3 = H_eigvals[:,2]
    e_eigvec1 = e_eigvecs[:,:,0]
    e_eigvec2 = e_eigvecs[:,:,1]
    e_eigvec3 = e_eigvecs[:,:,2]

    log.debug(f"Performing analysis on beam-tracing results")
    if btf:
        ##################################################
        # Hessians and second derivatives
        ##################################################
        if cart:
            grad_grad_H_cart, gradK_grad_H_cart, gradK_gradK_H_cart = hessians(dH, cartesian=cart)
            grad_grad_H_cyld, gradK_grad_H_cyld, gradK_gradK_H_cyld = None, None, None
        else:
            grad_grad_H_cyld, gradK_grad_H_cyld, gradK_gradK_H_cyld = hessians(dH, cartesian=cart)
            grad_grad_H_cart, gradK_grad_H_cart, gradK_gradK_H_cart = None, None, None # convert_hessians(
            #     q_start = q_vec_cyld.T,
            #     dH = dH,
            #     grad_grad_H_start = grad_grad_H_cyld,
            #     gradK_grad_H_start = gradK_grad_H_cyld,
            #     gradK_gradK_H_start = gradK_gradK_H_cyld,
            #     start = "cylindrical",
            #     end = "cartesian",
            # )
        
        ##################################################
        # Beam matrix (Psi_w) components in the {y, g, x} basis
        ##################################################
        Psi_3D_output_labframe_cartesian = np.array(solver_output["Psi_3D_output_labframe_cartesian"][()])
        Psi_xx_output_beamframe_cartesian = dot(x_hat_cart, dot(Psi_3D_output_labframe_cartesian, x_hat_cart))
        Psi_xy_output_beamframe_cartesian = dot(x_hat_cart, dot(Psi_3D_output_labframe_cartesian, y_hat_cart))
        Psi_xg_output_beamframe_cartesian = dot(x_hat_cart, dot(Psi_3D_output_labframe_cartesian, g_hat_cart))
        Psi_yy_output_beamframe_cartesian = dot(y_hat_cart, dot(Psi_3D_output_labframe_cartesian, y_hat_cart))
        Psi_yg_output_beamframe_cartesian = dot(y_hat_cart, dot(Psi_3D_output_labframe_cartesian, g_hat_cart))
        Psi_gg_output_beamframe_cartesian = dot(g_hat_cart, dot(Psi_3D_output_labframe_cartesian, g_hat_cart))

        Psi_w_output_beamframe_cartesian = np.zeros((len_tau, 2, 2), dtype=np.complex128)
        Psi_w_output_beamframe_cartesian[:, 0, 0] = Psi_xx_output_beamframe_cartesian
        Psi_w_output_beamframe_cartesian[:, 0, 1] = Psi_w_output_beamframe_cartesian[:, 1, 0] = Psi_xy_output_beamframe_cartesian
        Psi_w_output_beamframe_cartesian[:, 1, 1] = Psi_yy_output_beamframe_cartesian
        curv1, curv2, width1, width2 = find_beam_widths_and_curvs(Psi_w_output_beamframe_cartesian, K_vec_cart, g_vec_cart)

    # TO REMOVE
    _temp_print_dict(analysis_dict)








    









    
    









def dbs_analysis():

    log.info(f"""\n
        ##################################################
        #
        # DOPPLER BACKSCATTERING ANALYSIS ROUTINE
        #
        ##################################################
        """)