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


equil_times = np.linspace(0.16,0.25,10)
# equil_times = np.array([0.190])
mirror_rotations = np.linspace(1,-6,36)
mirror_tilt = -4
launch_freqs_GHz = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0])


for equil_time in equil_times:
    for mirror_rotation in mirror_rotations:
        for launch_freq_GHz in launch_freqs_GHz:
            args_dict, kwargs_dict = get_parameters_for_Scotty(
                                          'DBS_NSTX_MAST',
                                          launch_freq_GHz = launch_freq_GHz,
                                          mirror_rotation = mirror_rotation, # angle, in deg
                                          mirror_tilt     = mirror_tilt, # angle, in deg
                                          find_B_method   = 'EFITpp', # EFITpp, UDA_saved, UDA, torbeam
                                          find_ne_method  = 'poly3',                                          
                                          equil_time      = equil_time,
                                          shot            = 29908,
                                          user            = 'Valerian_desktop'
                                         )
            
            if args_dict['launch_freq_GHz'] > 52.5:
                args_dict['mode_flag'] = 1
            else:
                args_dict['mode_flag'] = -1
    
            if args_dict['mode_flag'] == 1:
                mode_string = 'O'
            elif args_dict['mode_flag'] == -1:
                mode_string = 'X'
    
            kwargs_dict['output_filename_suffix'] = (
                                        '_r' + f'{mirror_rotation:.1f}'
                                        '_t' + f'{mirror_tilt:.1f}'
                                      + '_f' + f'{launch_freq_GHz:.1f}'
                                      + '_'  + mode_string
                                      + '_'  + f'{equil_time*1000:.3g}' + 'ms'
                                          )      
      
            beam_me_up(**args_dict, **kwargs_dict)
    

