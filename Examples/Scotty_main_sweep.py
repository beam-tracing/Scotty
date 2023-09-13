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
import numpy as np

from scotty.init_bruv import get_parameters_for_Scotty


equil_times = np.linspace(0.19,0.25,7)
# equil_times = np.array([0.195])
mirror_rotations = np.linspace(-1,6,36)
# launch_freqs_GHz = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0,55.0,57.5,60.0,62.5,67.5,70.0,72.5,75.0])
launch_freqs_GHz = np.array([47.5,50.0,55.0,57.5])

for equil_time in equil_times:
    for mirror_rotation in mirror_rotations:
        for launch_freq_GHz in launch_freqs_GHz:
            kwargs_dict = get_parameters_for_Scotty(
                                          'DBS_NSTX_MAST',
                                          launch_freq_GHz = launch_freq_GHz,
                                          mirror_rotation = mirror_rotation, # angle, in deg
                                          mirror_tilt     = 4.0, # angle, in deg
                                          find_B_method   = 'EFITpp', # EFITpp, UDA_saved, UDA, torbeam
                                          equil_time      = equil_time,
                                          shot            = 30073,
                                          user            = 'Valerian_desktop'
                                         )
            
            if kwargs_dict['launch_freq_GHz'] > 52.5:
                kwargs_dict['mode_flag'] = -1
            else:
                kwargs_dict['mode_flag'] = 1
    
            if kwargs_dict['mode_flag'] == 1:
                mode_string = 'O'
            elif kwargs_dict['mode_flag'] == -1:
                mode_string = 'X'
    
            kwargs_dict['output_filename_suffix'] = (
                                        '_r' + f'{mirror_rotation:.1f}'
                                        '_t' + f'{4.0:.1f}'
                                      + '_f' + f'{launch_freq_GHz:.1f}'
                                      + '_'  + mode_string
                                      + '_'  + f'{equil_time*1000:.3g}' + 'ms'
                                          )      
      
            beam_me_up(**kwargs_dict)
