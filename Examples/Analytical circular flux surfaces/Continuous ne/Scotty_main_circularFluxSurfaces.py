# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian_hall-chen@ihpc.a-star.edu.sg


"""

from scotty.beam_me_up import beam_me_up
import numpy as np


kwargs_dict = {
    "poloidal_launch_angle_Torbeam": 6.0,
    "toroidal_launch_angle_Torbeam": 0.0,
    "launch_freq_GHz": 55.0,
    "mode_flag": 1,
    "launch_beam_width": 0.04,
    "launch_beam_curvature": -0.25,
    "launch_position": np.array([2.587, 0.0, -0.0157]),
    "density_fit_parameters": np.array([4.0, 1.0]),
    "delta_R": -0.00001,
    "delta_Z": 0.00001,
    "density_fit_method": "quadratic",
    "find_B_method": "analytical",
    "Psi_BC_flag": True,
    "figure_flag": False,
    "vacuum_propagation_flag": True,
    "vacuumLaunch_flag": True,
    "poloidal_flux_enter": 1.0,
    "poloidal_flux_zero_density": 1.0,
    "B_T_axis": 1.0,
    "B_p_a": 0.1,
    "R_axis": 1.5,
    "minor_radius_a": 0.5,
}

beam_me_up(**kwargs_dict)
