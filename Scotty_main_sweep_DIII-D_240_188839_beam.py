# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from Scotty_beam_me_up import beam_me_up
import numpy as np
import os
from Scotty_init_bruv import get_parameters_for_Scotty 
from Scotty_fun_general import modify_beam

shifts_z0 = np.linspace(-2.0,2.0,21)
shifts_w0 = np.linspace(-0.01,0.02,16)
pol_launch_angle = -11.4
tor_launch_angle = 2.4
# launch_freqs_GHz = np.array([55.0,57.5,60.0,62.5,65.0,67.5,70.0,72.5,75.0])
launch_freqs_GHz = np.array([65.0])


equil_time = 1900.0


for ii, launch_freq_GHz in enumerate(launch_freqs_GHz):
    for jj, shift_z0 in enumerate(shifts_z0):
        for kk, shift_w0 in enumerate(shifts_w0):
            
            args_dict, kwargs_dict = get_parameters_for_Scotty(
                                          'DBS_UCLA_DIII-D_240',
                                          launch_freq_GHz = launch_freq_GHz,
                                          find_B_method   = 'torbeam', # EFITpp, UDA_saved, UDA, torbeam
                                          equil_time      = equil_time,
                                          shot            = 188839,
                                          user            = 'Valerian_desktop'
                                         )
            
            args_dict['poloidal_launch_angle_Torbeam'] = pol_launch_angle
            args_dict['toroidal_launch_angle_Torbeam'] = tor_launch_angle
            args_dict['mode_flag'] = -1
            
            width_new, curv_new = modify_beam(args_dict['launch_beam_width'], 
                                              args_dict['launch_beam_curvature'], 
                                              launch_freq_GHz, shift_z0, shift_w0)
            args_dict['launch_beam_width'] = width_new
            args_dict['launch_beam_curvature'] = curv_new
            
            kwargs_dict['poloidal_flux_enter'] = 1.44
            kwargs_dict['input_filename_suffix'] = '_188839_1900ms'
            # kwargs_dict['output_path'] = os.path.dirname(os.path.abspath(__file__)) + '\\Output\\'
            kwargs_dict['output_path'] = 'D:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 13\\'

            kwargs_dict['delta_R'] = -0.00001
            kwargs_dict['delta_Z'] = -0.00001
            kwargs_dict['delta_K_R'] = 0.1
            kwargs_dict['delta_K_zeta'] = 0.1
            kwargs_dict['delta_K_Z'] = 0.1
            kwargs_dict['interp_smoothing'] = 2.0
            kwargs_dict['len_tau'] = 1002
            kwargs_dict['rtol'] = 1e-3
            kwargs_dict['atol'] = 1e-6
            
            if args_dict['mode_flag'] == 1:
                mode_string = 'O'
            elif args_dict['mode_flag'] == -1:
                mode_string = 'X'
                
            kwargs_dict['output_filename_suffix'] = (
                                        '_z0shift' + f'{shift_z0:.1f}'
                                        '_w0shift' + f'{shift_w0:.3f}'
                                      + '_f' + f"{args_dict['launch_freq_GHz']:.1f}"
                                      + '_'  + mode_string
                                          )      
            kwargs_dict['figure_flag'] = False
            kwargs_dict['detailed_analysis_flag'] = False

            # if ii == 0 and jj == 0 and kk == 0:
            #     kwargs_dict['verbose_output_flag'] = True
            # else:
            #     kwargs_dict['verbose_output_flag'] = False        
            kwargs_dict['verbose_output_flag'] = True

            # an_output = kwargs_dict['output_path'] + 'data_output' + kwargs_dict['output_filename_suffix'] + '.npz'
            # if os.path.exists(an_output):
            #     continue
            # else:   
            #     beam_me_up(**args_dict, **kwargs_dict) 
            beam_me_up(**args_dict, **kwargs_dict) 
