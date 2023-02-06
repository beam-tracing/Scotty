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
from scotty.fun_general import find_normalised_plasma_freq, find_normalised_gyro_freq
import math
from scipy import constants


def plot(suffix="_Torbeam_benchmark"):
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
    electron_density_output = loadfile["electron_density_output"]
    loadfile.close()

    loadfile = np.load("analysis_output" + suffix + ".npz")
    Psi_xx_output = loadfile["Psi_xx_output"]
    Psi_xy_output = loadfile["Psi_xy_output"]
    Psi_yy_output = loadfile["Psi_yy_output"]
    M_xx_output = loadfile["M_xx_output"]
    M_xy_output = loadfile["M_xy_output"]
    M_yy_output = loadfile["M_yy_output"]
    xhat_dot_grad_bhat_dot_xhat_output = loadfile["xhat_dot_grad_bhat_dot_xhat_output"]
    xhat_dot_grad_bhat_dot_yhat_output = loadfile["xhat_dot_grad_bhat_dot_yhat_output"]
    xhat_dot_grad_bhat_dot_ghat_output = loadfile["xhat_dot_grad_bhat_dot_ghat_output"]
    yhat_dot_grad_bhat_dot_xhat_output = loadfile["yhat_dot_grad_bhat_dot_xhat_output"]
    yhat_dot_grad_bhat_dot_yhat_output = loadfile["yhat_dot_grad_bhat_dot_yhat_output"]
    yhat_dot_grad_bhat_dot_ghat_output = loadfile["yhat_dot_grad_bhat_dot_ghat_output"]
    kappa_dot_xhat_output = loadfile["kappa_dot_xhat_output"]
    kappa_dot_yhat_output = loadfile["kappa_dot_yhat_output"]
    distance_along_line = loadfile["distance_along_line"]
    cutoff_index = loadfile["cutoff_index"]
    R_midplane_points = loadfile["R_midplane_points"]
    poloidal_flux_on_midplane = loadfile["poloidal_flux_on_midplane"]
    e_hat_output = loadfile["e_hat_output"]
    theta_output = loadfile["theta_output"]
    theta_m_output = loadfile["theta_m_output"]
    y_hat_Cartesian = loadfile["y_hat_Cartesian"]
    x_hat_Cartesian = loadfile["x_hat_Cartesian"]
    K_magnitude_array = loadfile["K_magnitude_array"]
    loadfile.close()

    loadfile = np.load("data_input" + suffix + ".npz")
    data_poloidal_flux_grid = loadfile["data_poloidal_flux_grid"]
    data_R_coord = loadfile["data_R_coord"]
    data_Z_coord = loadfile["data_Z_coord"]
    launch_position = loadfile["launch_position"]
    launch_beam_width = loadfile["launch_beam_width"]
    launch_beam_radius_of_curvature = loadfile["launch_beam_radius_of_curvature"]
    launch_freq_GHz = loadfile["launch_freq_GHz"]
    loadfile.close()

    l_lc = (
        distance_along_line - distance_along_line[cutoff_index]
    )  # Distance from cutoff
    [q_X_array, q_Y_array, q_Z_array] = find_q_lab_Cartesian(
        np.array([q_R_array, q_zeta_array, q_Z_array])
    )
    numberOfDataPoints = np.size(q_R_array)

    out_index = numberOfDataPoints

    # Importing data from output files
    # t1tor_LIB.dat, y_LIB.dat
    # Others: coef_LIB, ecdo.dat(?), t1_LIB.dat, t2_LIB.dat, t2_new_LIB.dat, reflout.dat (not currently used in this code)
    output_files_path = "D:\\Dropbox\\VHChen2018\\Code - Torbeam\\Benchmark-7\\"
    t1_LIB_filename = output_files_path + "t1_LIB.dat"
    y_LIB_filename = output_files_path + "y_LIB.dat"

    t1_LIB = np.fromfile(t1_LIB_filename, dtype=float, sep="   ")
    y_LIB = np.fromfile(y_LIB_filename, dtype=float, sep="   ")
    # --------------------------------

    # Tidying up the output data
    numberOfDataPoints = len(t1_LIB) // 6
    t1_LIB = t1_LIB.reshape(6, numberOfDataPoints, order="F")
    y_LIB = y_LIB.reshape(19, numberOfDataPoints, order="F")

    launch_freq = 2 * (math.pi) * launch_freq_GHz * 10.0**9.0
    k0 = launch_freq / (299792458 * 100)  # [k] = cm-1
    # k_perp_1_array = np.linspace(-30,-15,num=1501)
    k_perp_1_array = np.linspace(-25, -12.5, num=1001)

    r_x_array = y_LIB[0, :]
    r_y_array = y_LIB[1, :]
    r_z_array = y_LIB[2, :]

    # K_x_array = y_LIB[3,:] * k0
    # K_y_array = y_LIB[4,:] * k0
    # K_z_array = y_LIB[5,:] * k0

    # S_xx_array = y_LIB[6,:] * k0 # Re(\Psi). Responsible for curvature
    # S_xy_array = y_LIB[7,:] * k0
    # S_xz_array = y_LIB[8,:] * k0
    # S_yy_array = y_LIB[9,:] * k0
    # S_yz_array = y_LIB[10,:] * k0
    # S_zz_array = y_LIB[11,:] * k0

    # Phi_xx_array = y_LIB[12,:] * k0 # Im (\Psi). Responsible for width
    # Phi_xy_array = y_LIB[13,:] * k0
    # Phi_xz_array = y_LIB[14,:] * k0
    # Phi_yy_array = y_LIB[15,:] * k0
    # Phi_yz_array = y_LIB[16,:] * k0
    # Phi_zz_array = y_LIB[17,:] * k0

    point_spacing = (
        (np.diff(r_x_array)) ** 2
        + (np.diff(r_y_array)) ** 2
        + (np.diff(r_z_array)) ** 2
    ) ** 0.5
    Torbeam_distance_along_line = np.cumsum(point_spacing)
    Torbeam_distance_along_line = np.append(0, Torbeam_distance_along_line)
    # ---------------------------

    plt.figure()
    plt.plot(Torbeam_distance_along_line, r_x_array, label="Torbeam")
    plt.plot(distance_along_line, q_X_array, label="Scotty")
    plt.legend()

    plt.figure()
    plt.plot(Torbeam_distance_along_line, r_y_array, label="Torbeam")
    plt.plot(distance_along_line, q_Y_array, label="Scotty")
    plt.legend()

    plt.figure()
    plt.plot(Torbeam_distance_along_line, r_z_array, label="Torbeam")
    plt.plot(distance_along_line, q_Z_array, label="Scotty")
    plt.legend()

    plt.figure()
    plt.plot(r_x_array / 100, r_z_array / 100, label="Torbeam")
    plt.plot(q_X_array, q_Z_array, label="Scotty")
    plt.xlim([1.6, 2.2])
    plt.ylim([-0.5, 0])
    plt.legend()


if __name__ == "__main__":
    plot()
