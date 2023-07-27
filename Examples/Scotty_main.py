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


kwargs_dict = get_parameters_for_Scotty(
    "DBS_NSTX_MAST",
    launch_freq_GHz=400.0,
    mirror_rotation=-1.0,  # angle, in deg
    mirror_tilt=-4.0,  # angle, in deg
    find_B_method="EFITpp",  # EFITpp, UDA_saved, UDA, torbeam
    find_ne_method="poly3",
    equil_time=0.220,
    shot=29908,
    user="Valerian_laptop",
)


if kwargs_dict["launch_freq_GHz"] > 52.5:
    kwargs_dict["mode_flag"] = 1
else:
    kwargs_dict["mode_flag"] = 1

print(kwargs_dict)
print(kwargs_dict)


beam_me_up(**kwargs_dict)
