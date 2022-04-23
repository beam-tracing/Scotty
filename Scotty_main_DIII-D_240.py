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


launch_freq_GHz = 65.0
equil_time = 3000.0
poloidal_launch_angle_Torbeam = -5.0
toroidal_launch_angle_Torbeam = 2.5

args_dict, kwargs_dict = get_parameters_for_Scotty(
                              'DBS_UCLA_DIII-D_1',
                              launch_freq_GHz = launch_freq_GHz,
                              find_B_method   = 'torbeam', # EFITpp, UDA_saved, UDA, torbeam
                              equil_time      = equil_time,
                              shot            = 187103,
                              user            = 'Valerian_desktop'
                             )

args_dict['poloidal_launch_angle_Torbeam'] = poloidal_launch_angle_Torbeam
args_dict['toroidal_launch_angle_Torbeam'] = toroidal_launch_angle_Torbeam
args_dict['mode_flag'] = -1

kwargs_dict['poloidal_flux_enter'] = 1.44
kwargs_dict['input_filename_suffix'] = '_187103_3000ms'
kwargs_dict['output_path'] = os.path.dirname(os.path.abspath(__file__)) + '\\Output\\'

kwargs_dict['delta_R'] = -0.001
kwargs_dict['delta_Z'] = -0.001
kwargs_dict['delta_K_R'] = 0.1
kwargs_dict['delta_K_zeta'] = 0.1
kwargs_dict['delta_K_Z'] = 0.1


beam_me_up(**args_dict, **kwargs_dict)
    

