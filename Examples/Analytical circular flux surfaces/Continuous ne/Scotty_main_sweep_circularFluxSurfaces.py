# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com


For shot 29908, the EFIT++ times are efit_times = np.linspace(0.155,0.25,20)
I want efit_times[np.arange(0,10)*2 + 1]. 160ms, 170ms, ..., 250ms
"""
from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np

kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

B_p_a_sweep = np.linspace(0.0, 0.2, 5)
tor_launch_angles = np.linspace(0.0, 4, 5)

for B_p_a in B_p_a_sweep:
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

    kwargs_dict["B_p_a"] = B_p_a
    kwargs_dict["output_filename_suffix"] = "_Bpa" + f"{B_p_a:.2f}"

    beam_me_up(**kwargs_dict)

for tor_launch_angle in tor_launch_angles:
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

    kwargs_dict["toroidal_launch_angle_Torbeam"] = tor_launch_angle
    kwargs_dict["B_p_a"] = 0.0
    kwargs_dict["output_filename_suffix"] = "_t" + f"{tor_launch_angle:.2f}"

    beam_me_up(**kwargs_dict)
