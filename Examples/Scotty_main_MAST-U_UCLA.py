# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

## Initial beam properties (width, curv) are a function of frequency ##
launch_freqs_GHz       = np.array([32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50])
launch_beam_widths     = np.array([0.07596929, 0.07058289, 0.06591742, 0.0618377 , 
                                   0.05824036, 0.05504499, 0.05218812, 0.04961897])
launch_beam_curvatures = np.array([-0.74971565, -0.75653105, -0.76383412, -0.77162126, 
                                   -0.77988862, -0.78863216, -0.7978476 , -0.80753047])

"""
from scotty.beam_me_up import beam_me_up
import numpy as np

kwargs_dict = {
    "poloidal_launch_angle_Torbeam": -6.151787276,
    "toroidal_launch_angle_Torbeam": 4.3821909,
    "launch_freq_GHz": 32.5,
    "mode_flag": 1,
    "launch_beam_width": 0.07596928872724663,
    "launch_beam_curvature": -0.7497156475519201,
    "launch_position": np.array([2.278, 0.0, 0.0]),
    "ne_data_path": "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\\MAST-U\\",
    "magnetic_data_path": "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\\MAST-U\\",
    "input_filename_suffix": "_45176_400ms",
    "find_B_method": "test",
    "equil_time": 0.4,
    "shot": 45176,
    "poloidal_flux_enter": 1.06520296**2,
    "Psi_BC_flag": True,
    "figure_flag": True,
    "vacuum_propagation_flag": True,
    "vacuumLaunch_flag": True,
    "delta_R": -0.00001,
    "delta_Z": -0.00001,
    "delta_K_R": 0.1,
    "delta_K_zeta": 0.1,
    "delta_K_Z": 0.1,
    "interp_smoothing": 0.0,
    "len_tau": 1002,
    "rtol": 1e-4,
    "atol": 1e-7,
}

# sys.exit()

beam_me_up(**kwargs_dict)
