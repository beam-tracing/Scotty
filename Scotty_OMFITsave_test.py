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
import numpy as np

from Scotty_init_bruv import get_parameters_for_Scotty


args_dict, kwargs_dict = get_parameters_for_Scotty(
                              'DBS_NSTX_MAST',
                              launch_freq_GHz = 46.0,
                              # mirror_rotation = -4.0, # angle, in deg
                              # mirror_tilt     = -4.0, # angle, in deg
                              find_B_method   = 'test', # EFITpp, UDA_saved, UDA, torbeam
                              equil_time      = 0.510,
                              shot            = 45154,
                              user            = 'Valerian_desktop'
                             )


args_dict['poloidal_launch_angle_Torbeam']   = 6.4
args_dict['toroidal_launch_angle_Torbeam']   = -4.4
# args_dict['launch_beam_width']               = 0.0397
# args_dict['launch_beam_radius_of_curvature'] = -0.7286   
kwargs_dict['find_B_method']        = 'torbeam'
kwargs_dict['ne_data_path']        = None
kwargs_dict['magnetic_data_path']        = None


if args_dict['launch_freq_GHz'] > 52.5:
    args_dict['mode_flag'] = 1
else:
    args_dict['mode_flag'] = 1
  
np.savez('inScotty', 
          args_dict=args_dict,
          kwargs_dict=kwargs_dict
         )        

