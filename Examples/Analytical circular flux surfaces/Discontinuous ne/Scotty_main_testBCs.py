# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian_hall-chen@ihpc.a-star.edu.sg


"""

from scotty.beam_me_up import beam_me_up
import numpy as np

from scotty.fun_general import (
    find_Booker_alpha,
    find_Booker_beta,
    find_Booker_gamma,
    freq_GHz_to_angular_frequency,
)

B_p_a = 0.3
B_T_axis = 1.0
R_axis = 1.5
minor_radius_a = 0.5
launch_freq_GHz = 35.0

kwargs_dict = {
    "poloidal_launch_angle_Torbeam": 7.0,
    "toroidal_launch_angle_Torbeam": 6.5,
    "launch_freq_GHz": launch_freq_GHz,
    "mode_flag": 1,
    "launch_beam_width": 0.04,
    "launch_beam_curvature": -0.25,
    "launch_position": np.array([2.587, 0.0, -0.0157]),
    "density_fit_parameters": None,
    # "density_fit_method": "torbeam",
    "delta_R": -1e-5,
    "delta_Z": 1e-5,
    "find_B_method": "analytical",
    "Psi_BC_flag": 'discontinuous',
    "figure_flag": False,
    "vacuum_propagation_flag": True,
    "vacuumLaunch_flag": True,
    "poloidal_flux_enter": 1.0,
    "poloidal_flux_zero_density": 1.00001,
    "B_T_axis": B_T_axis,
    "B_p_a": B_p_a,
    "R_axis": R_axis,
    "minor_radius_a": minor_radius_a,
    "output_path": "D:\\Dropbox\\VHChen2022\\Data - Scotty\\Run 11\\",
    "ne_data_path": ".",
}

edge_ne = 0.5
B_Total = np.sqrt((B_T_axis * R_axis / (R_axis + minor_radius_a)) ** 2 + B_p_a**2)

## Approximation
sin_theta_m_sq = 0
launch_angular_frequency = freq_GHz_to_angular_frequency(launch_freq_GHz)

Booker_alpha = find_Booker_alpha(
    edge_ne, B_Total, sin_theta_m_sq, launch_angular_frequency
)
Booker_beta = find_Booker_beta(
    edge_ne, B_Total, sin_theta_m_sq, launch_angular_frequency
)
Booker_gamma = find_Booker_gamma(edge_ne, B_Total, launch_angular_frequency)

N_sq = (
    -Booker_beta
    + kwargs_dict["mode_flag"]
    * np.sqrt(Booker_beta**2 - 4 * Booker_alpha * Booker_gamma)
) / (2 * Booker_alpha)
print("N", np.sqrt(N_sq))

beam_me_up(**kwargs_dict)
