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
import os

from Scotty_init_bruv import get_parameters_for_Scotty


launch_beam_width_scalings = np.linspace(0.90,1.1,5)
launch_beam_curvature_scalings = np.linspace(0.90,1.1,3)


equil_times = np.linspace(0.16,0.25,10)
equil_time = 0.180
mirror_rotation = -2.3 # At 210ms, the Q band rotation angles for matching range from -2.2 to -2.4
mirror_tilt = -4
# launch_freqs_GHz = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0,
#                             55.0,57.5,60.0,62.5,67.5,70.0,72.5,75.0])
launch_freqs_GHz = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0])

counter = 0
for ii, launch_freq_GHz in enumerate(launch_freqs_GHz):
    for jj, launch_beam_width_scaling in enumerate(launch_beam_width_scalings):
        for kk, launch_beam_curvature_scaling in enumerate(launch_beam_curvature_scalings):
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

            test_width =  launch_beam_width_scaling*args_dict['launch_beam_width']
         
            args_dict['launch_beam_width'] = launch_beam_width_scaling*args_dict['launch_beam_width']
            args_dict['launch_beam_curvature'] = launch_beam_curvature_scaling*args_dict['launch_beam_curvature']
            
            if args_dict['launch_freq_GHz'] > 52.5:
                args_dict['mode_flag'] = 1
            else:
                args_dict['mode_flag'] = -1
    
            if args_dict['mode_flag'] == 1:
                mode_string = 'O'
            elif args_dict['mode_flag'] == -1:
                mode_string = 'X'        

            kwargs_dict['verbose_output_flag'] = True

            kwargs_dict['output_filename_suffix'] = (
                                        '_r' + f'{mirror_rotation:.1f}'
                                      + '_t' + f'{mirror_tilt:.1f}'
                                      + '_w' + f"{args_dict['launch_beam_width']:.6f}"
                                      + '_curv' + f"{args_dict['launch_beam_curvature']:.3f}"
                                      + '_f' + f'{launch_freq_GHz:.1f}'
                                      + '_'  + mode_string
                                      + '_'  + f'{equil_time*1000:.3g}' + 'ms'
                                          )      
            kwargs_dict['figure_flag'] = False
            kwargs_dict['detailed_analysis_flag'] = False
            kwargs_dict['output_path'] = 'D:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 19\\'

            an_output = kwargs_dict['output_path'] + 'data_output' + kwargs_dict['output_filename_suffix'] + '.npz'
            # if os.path.exists(an_output):
            #     continue
            # else:   
            #     beam_me_up(**args_dict, **kwargs_dict)  

            kwargs_dict['delta_R'] = -0.00001
            kwargs_dict['delta_Z'] = -0.00001
            kwargs_dict['delta_K_R'] = 0.1
            kwargs_dict['delta_K_zeta'] = 0.1
            kwargs_dict['delta_K_Z'] = 0.1
            kwargs_dict['interp_smoothing'] = 2.0
            kwargs_dict['len_tau'] = 1002
            kwargs_dict['rtol'] = 1e-3
            kwargs_dict['atol'] = 1e-6
            
            beam_me_up(**args_dict, **kwargs_dict)

            counter = counter + 1
            # print('Sweep completion:' + str(counter) + ' of ' + str( len(equil_times)*len(mirror_rotations)*len(launch_freqs_GHz) ) )


