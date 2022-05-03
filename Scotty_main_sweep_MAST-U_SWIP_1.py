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


pol_launch_angles = np.linspace(-2.0,-10.0,17)
tor_launch_angles = np.linspace(0,10.0,21)
launch_freqs_GHz  = np.linspace(30,76,24)

for ii, launch_freq_GHz in enumerate(launch_freqs_GHz):
    for jj, pol_launch_angle in enumerate(pol_launch_angles):
        for kk, tor_launch_angle in enumerate(tor_launch_angles):
            args_dict, kwargs_dict = get_parameters_for_Scotty(
                                          'DBS_SWIP_MAST-U',
                                          launch_freq_GHz = launch_freq_GHz,
                                          find_B_method   = 'test', # EFITpp, UDA_saved, UDA, torbeam
                                          equil_time      = 0.780,
                                          shot            = 45304,
                                          user            = 'Valerian_desktop'
                                         )
            
            args_dict['mode_flag'] = 1
            args_dict['poloidal_launch_angle_Torbeam'] = pol_launch_angle
            args_dict['toroidal_launch_angle_Torbeam'] = tor_launch_angle
            
            kwargs_dict['poloidal_flux_enter'] = 1.22
            # kwargs_dict['output_path'] = os.path.dirname(os.path.abspath(__file__)) + '\\Output\\'
            kwargs_dict['output_path'] = 'D:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 6\\'
            kwargs_dict['density_fit_parameters'] = None
            kwargs_dict['ne_data_path'] = 'D:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST-U\\'
            kwargs_dict['input_filename_suffix'] = '_45304_780ms'
            
            if args_dict['mode_flag'] == 1:
                mode_string = 'O'
            elif args_dict['mode_flag'] == -1:
                mode_string = 'X'
                
            kwargs_dict['output_filename_suffix'] = (
                                        '_pol' + f'{pol_launch_angle:.2f}'
                                        '_tor' + f'{tor_launch_angle:.2f}'
                                      + '_f' + f"{args_dict['launch_freq_GHz']:.1f}"
                                      + '_'  + mode_string
                                          )      
            
            kwargs_dict['figure_flag'] = False
            kwargs_dict['detailed_analysis_flag'] = False
            
            if ii == 0 and jj == 0 and kk == 0:
                kwargs_dict['verbose_output_flag'] = True
            else:
                kwargs_dict['verbose_output_flag'] = False  
                

            kwargs_dict['delta_R'] = -0.00001
            kwargs_dict['delta_Z'] = -0.00001
            kwargs_dict['delta_K_R'] = 0.1
            kwargs_dict['delta_K_zeta'] = 0.1
            kwargs_dict['delta_K_Z'] = 0.1
            kwargs_dict['interp_smoothing'] = 2.0
            kwargs_dict['interp_order'] = 5
            kwargs_dict['len_tau'] = 1002
            kwargs_dict['rtol'] = 1e-4
            kwargs_dict['atol'] = 1e-7
            
            
            beam_me_up(**args_dict, **kwargs_dict)
    

