# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com


For shot 29908, the EFIT++ times are efit_times = np.linspace(0.155,0.25,20)
I want efit_times[np.arange(0,10)*2 + 1]. 160ms, 170ms, ..., 250ms
"""
from Scotty_beam_me_up import beam_me_up
from Scotty_fun_general import find_q_lab_Cartesian, find_q_lab, find_K_lab_Cartesian, find_K_lab, find_waist, find_Rayleigh_length, genray_angles_from_mirror_angles
from Scotty_fun_general import propagate_beam

from scipy import constants
import math
import numpy as np
import sys

from Scotty_init_bruv import get_parameters_for_Scotty


args_dict, kwargs_dict = get_parameters_for_Scotty(
                              'DBS_NSTX_MAST',
                              launch_freq_GHz = 50.0,
                              mirror_rotation = -4.0, # angle, in deg
                              mirror_tilt     = -4.0, # angle, in deg
                              find_B_method   = 'UDA_saved', # EFITpp, UDA_saved, UDA, torbeam
                              equil_time      = 0.200,
                              shot            = 29980,
                              user            = 'Valerian_laptop'
                             )



if args_dict['launch_freq_GHz'] > 52.5:
    args_dict['mode_flag'] = 1
else:
    args_dict['mode_flag'] = -1
  
beam_me_up(**args_dict, **kwargs_dict)
    

