# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scotty.fun_general import (
    find_waist,
    find_distance_from_waist,
    find_q_lab_Cartesian,
    find_nearest,
    contract_special,
)
from scotty.fun_general import (
    find_normalised_plasma_freq,
    find_normalised_gyro_freq,
    make_unit_vector_from_cross_product,
    find_vec_lab_Cartesian,
)
import math
from scipy import constants, integrate
import sys


def plot(suffix=""):
    loadfile = np.load("data_output" + suffix + ".npz")
    tau_array = loadfile["tau_array"]
    q_R_array = loadfile["q_R_array"]
    q_zeta_array = loadfile["q_zeta_array"]
    q_Z_array = loadfile["q_Z_array"]
    K_R_array = loadfile["K_R_array"]
    K_Z_array = loadfile["K_Z_array"]
    Psi_3D_output = loadfile["Psi_3D_output"]
    x_hat_output = loadfile["x_hat_output"]
    y_hat_output = loadfile["y_hat_output"]
    g_hat_output = loadfile["g_hat_output"]
    b_hat_output = loadfile["b_hat_output"]
    epsilon_para_output = loadfile["epsilon_para_output"]
    epsilon_perp_output = loadfile["epsilon_perp_output"]
    epsilon_g_output = loadfile["epsilon_g_output"]
    B_magnitude = loadfile["B_magnitude"]
    g_magnitude_output = loadfile["g_magnitude_output"]
    electron_density_output = loadfile["electron_density_output"]
    poloidal_flux_output = loadfile["poloidal_flux_output"]
    H_output = loadfile["H_output"]
    B_R_output = (loadfile["B_R_output"],)
    B_T_output = (loadfile["B_T_output"],)
    B_Z_output = (loadfile["B_Z_output"],)
    loadfile.close()

    loadfile = np.load("analysis_output" + suffix + ".npz")
    distance_along_line = loadfile["distance_along_line"]
    cutoff_index = loadfile["cutoff_index"]
    e_hat_output = loadfile["e_hat_output"]
    theta_output = loadfile["theta_output"]
    theta_m_output = loadfile["theta_m_output"]
    delta_theta_m = loadfile["delta_theta_m"]
    delta_k_perp_2 = loadfile["delta_k_perp_2"]
    y_hat_Cartesian = loadfile["y_hat_Cartesian"]
    x_hat_Cartesian = loadfile["x_hat_Cartesian"]
    K_magnitude_array = loadfile["K_magnitude_array"]
    loadfile.close()

    loadfile = np.load("data_input" + suffix + ".npz")
    poloidalFlux_grid = loadfile["poloidalFlux_grid"]
    data_R_coord = loadfile["data_R_coord"]
    data_Z_coord = loadfile["data_Z_coord"]
    launch_position = loadfile["launch_position"]
    launch_beam_width = loadfile["launch_beam_width"]
    launch_beam_curvature = loadfile["launch_beam_curvature"]
    launch_freq_GHz = loadfile["launch_freq_GHz"]
    loadfile.close()

    l_lc = (
        distance_along_line - distance_along_line[cutoff_index]
    )  # Distance from cutoff
    [q_X_array, q_Y_array, q_Z_array] = find_q_lab_Cartesian(
        np.array([q_R_array, q_zeta_array, q_Z_array])
    )
    numberOfDataPoints = np.size(q_R_array)

    def find_N(Booker_alpha, Booker_beta, Booker_gamma, mode_flag):
        N = np.sqrt(
            -(
                Booker_beta
                - mode_flag
                * np.sqrt(
                    np.maximum(
                        np.zeros_like(Booker_beta),
                        (Booker_beta**2 - 4 * Booker_alpha * Booker_gamma),
                    )
                )
                # np.sqrt(Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
            )
            / (2 * Booker_alpha)
        )

        return N

    # ---
    sin_theta_m_sq = (np.sin(theta_m_output)) ** 2
    Booker_alpha = epsilon_para_output * sin_theta_m_sq + epsilon_perp_output * (
        1 - sin_theta_m_sq
    )
    Booker_beta = -epsilon_perp_output * epsilon_para_output * (1 + sin_theta_m_sq) - (
        epsilon_perp_output**2 - epsilon_g_output**2
    ) * (1 - sin_theta_m_sq)
    Booker_gamma = epsilon_para_output * (
        epsilon_perp_output**2 - epsilon_g_output**2
    )

    N_X = find_N(Booker_alpha, Booker_beta, Booker_gamma, -1)
    N_O = find_N(Booker_alpha, Booker_beta, Booker_gamma, 1)
    # ---

    # ---
    pitch_angle = np.arctan2(B_Z_output, B_T_output)
    # ---

    # ---
    anisotropy_term = (1 / N_O) * (1 - N_O**2 / N_X**2)
    # shear_term =

    plt.figure()
    plt.plot(l_lc, anisotropy_term)


if __name__ == "__main__":
    plot()
