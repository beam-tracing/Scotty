# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Hall-Chen

This is my attempt to get all contributions to the localisation on the same
figure, and for everything to be seen clearly.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate as interpolate
import math
from scipy import constants as constants
from scipy import integrate as integrate
import sys
from scotty.fun_general import (
    find_waist,
    find_distance_from_waist,
    find_q_lab_Cartesian,
    find_nearest,
    contract_special,
)


def plot():
    efit_time = 220  # in ms
    launch_freq_GHz = 67.5
    mirror_rotation_angle = -3.0
    mirror_tilt_angle = -4

    if launch_freq_GHz < 52.5:
        mode_string = "X"
    else:
        mode_string = "O"

    path = "D:\\Dropbox\\VHChen2020\\Code - Mismatch\\MAST_Mismatch\\Data\\Run 17\\"

    suffix = (
        "_r" + f"{mirror_rotation_angle:.1f}"
        "_t"
        + f"{mirror_tilt_angle:.1f}"
        + "_f"
        + f"{launch_freq_GHz:.1f}"
        + "_"
        + mode_string
        + "_"
        + f"{efit_time:.3g}"
        + "ms"
    )
    suffix = ""
    path = ""

    loadfile = np.load(path + "data_input" + suffix + ".npz")
    launch_freq_GHz = loadfile["launch_freq_GHz"]
    loadfile.close()

    loadfile = np.load(path + "analysis_output" + suffix + ".npz")
    # loc_piece = loadfile['loc_piece']
    cutoff_index = loadfile["cutoff_index"]
    RZ_distance_along_line = loadfile["RZ_distance_along_line"]
    distance_along_line = loadfile["distance_along_line"]
    k_perp_1_bs = loadfile["k_perp_1_bs"]
    Psi_xx_output = loadfile["Psi_xx_output"]
    Psi_xy_output = loadfile["Psi_xy_output"]
    Psi_yy_output = loadfile["Psi_yy_output"]
    M_xx_output = loadfile["M_xx_output"]
    M_xy_output = loadfile["M_xy_output"]
    M_yy_output = loadfile["M_yy_output"]
    theta_m_output = loadfile["theta_m_output"]
    K_magnitude_array = loadfile["K_magnitude_array"]
    k_perp_1_bs_plot = loadfile["k_perp_1_bs_plot"]
    loc_b = loadfile["loc_b"]
    loc_p = loadfile["loc_p"]
    loc_r = loadfile["loc_r"]
    loc_s = loadfile["loc_s"]
    loc_m = loadfile["loc_m"]
    loc_b_r_s = loadfile["loc_b_r_s"]
    # loc_b_r_s_distances = loadfile['loc_b_r_s_distances']
    # loc_b_r_s_max_over_e2 = loadfile['loc_b_r_s_max_over_e2']
    # loc_b_r_s_half_width = loadfile['loc_b_r_s_half_width']
    cum_loc_b_r_plot = loadfile["cum_loc_b_r_plot"]
    cum_loc_b_r = loadfile["cum_loc_b_r"]
    loc_b_r = loadfile["loc_b_r"]
    # loc_b_r_distances = loadfile['loc_b_r_distances']
    loc_b_r_s_max_over_e2 = loadfile["loc_b_r_s_max_over_e2"]
    loc_b_r_s_delta_l = loadfile["loc_b_r_s_delta_l"]
    cum_loc_b_r_s_max_over_e2 = loadfile["cum_loc_b_r_s_max_over_e2"]
    cum_loc_b_r_s_delta_l = loadfile["cum_loc_b_r_s_delta_l"]
    cum_loc_b_r_s_delta_kperp1 = loadfile["cum_loc_b_r_s_delta_kperp1"]
    # loc_b_r_half_width = loadfile['loc_b_r_half_width']
    cum_loc_b_r_s_plot = loadfile["cum_loc_b_r_s_plot"]
    cum_loc_b_r_s = loadfile["cum_loc_b_r_s"]
    cum_loc_b_r_s_mean_l_lc = loadfile["cum_loc_b_r_s_mean_l_lc"]
    cum_loc_b_r_s_mean_kperp1 = loadfile["cum_loc_b_r_s_mean_kperp1"]
    loc_b_r_max_over_e2 = loadfile["loc_b_r_max_over_e2"]
    loc_b_r_delta_l = loadfile["loc_b_r_delta_l"]
    cum_loc_b_r_max_over_e2 = loadfile["cum_loc_b_r_max_over_e2"]
    cum_loc_b_r_delta_l = loadfile["cum_loc_b_r_delta_l"]
    cum_loc_b_r_delta_kperp1 = loadfile["cum_loc_b_r_delta_kperp1"]
    cum_loc_b_r_mean_l_lc = loadfile["cum_loc_b_r_mean_l_lc"]
    cum_loc_b_r_mean_kperp1 = loadfile["cum_loc_b_r_mean_kperp1"]
    loadfile.close()

    loadfile = np.load(path + "data_output" + suffix + ".npz")
    g_hat_output = loadfile["g_hat_output"]
    g_magnitude_output = loadfile["g_magnitude_output"]
    q_R_array = loadfile["q_R_array"]
    q_zeta_array = loadfile["q_zeta_array"]
    q_Z_array = loadfile["q_Z_array"]
    K_R_array = loadfile["K_R_array"]
    K_zeta_initial = loadfile["K_zeta_initial"]
    K_Z_array = loadfile["K_Z_array"]
    Psi_3D_output = loadfile["Psi_3D_output"]
    loadfile.close()

    loadfile = np.load(path + "data_input" + suffix + ".npz")
    poloidalFlux_grid = loadfile["poloidalFlux_grid"]
    data_R_coord = loadfile["data_R_coord"]
    data_Z_coord = loadfile["data_Z_coord"]
    launch_position = loadfile["launch_position"]
    launch_beam_width = loadfile["launch_beam_width"]
    launch_beam_radius_of_curvature = loadfile["launch_beam_radius_of_curvature"]
    loadfile.close()

    plot_every_n_points = 1
    out_index_new = len(q_R_array)

    l_lc = distance_along_line - distance_along_line[cutoff_index]

    loc_b = loc_b / loc_b[0]
    loc_m = loc_m / loc_m[0]

    # loc_b_p_r = loc_b * loc_p * loc_r
    loc_b_r = loc_b * loc_r
    loc_b_m_r_s = loc_b * loc_m * loc_r * loc_s

    plt.figure(figsize=(4, 4), dpi=300)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.plot(l_lc, loc_b, color="b", linestyle=":", label="beam")
    plt.plot(l_lc, loc_m, color="darkorange", linestyle="-.", label="mismatch")
    plt.plot(l_lc, loc_r, color="r", linestyle="--", label="ray")
    plt.plot(l_lc, loc_s, color="g", linestyle="--", label="spectrum")
    plt.plot(l_lc, loc_b_m_r_s, "k", label="total")
    plt.yscale("log")
    plt.legend(facecolor="white", framealpha=1, edgecolor="black")
    plt.xlabel(r"$(l - l_c)$ / m")
    plt.ylabel("localisation")
    plt.savefig("localisation_675.jpg", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot()
