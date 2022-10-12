# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:16:00 2022

@author: Tan ZY

Input the parameters required for Scotty_beam_me_up_CART
"""

from Scotty_beam_me_up_CART import beam_me_up

import numpy as np


kwargs_dict = dict()
args_dict = {
  'poloidal_launch_angle_Torbeam'   : None,
  'toroidal_launch_angle_Torbeam'   : None,
  'launch_freq_GHz'                 : None,
  'mode_flag'                       : None,
  'launch_beam_width'               : None,
  'launch_beam_curvature'           : None,
  'launch_position'                 : None
}   


# From DBS synthetic
args_dict['poloidal_launch_angle_Torbeam'] = 0.0
args_dict['toroidal_launch_angle_Torbeam'] = 0.0
args_dict['launch_freq_GHz'] = 55
args_dict['mode_flag'] = 1
args_dict['launch_beam_width'] = 0.04
args_dict['launch_beam_curvature'] = 1/(-4.0)
args_dict['launch_position'] = np.array([2.0, 0.0, 0.0])

kwargs_dict['vacuumLaunch_flag'] = True
kwargs_dict['find_B_method'] = 'Slab'
kwargs_dict['shot'] = 45118
kwargs_dict['equil_time'] = 0.510
kwargs_dict['vacuum_propagation_flag'] = False
kwargs_dict['Psi_BC_flag'] = False
kwargs_dict['poloidal_flux_enter'] = 1.0
# kwargs_dict['B_T_axis'] = 1.0
# kwargs_dict['B_p_a'] = 0.1
# kwargs_dict['R_axis'] = 1.5
# kwargs_dict['minor_radius_a'] = 0.5

# ------------------------ from Scotty_init_bruv -----------------------------
# args_dict['poloidal_launch_angle_Torbeam']   = 6.0
# args_dict['toroidal_launch_angle_Torbeam']   = 0.0  
# args_dict['launch_freq_GHz']                 = 55.0
# args_dict['mode_flag']                       = 1
# args_dict['launch_beam_width']               = 0.04
# args_dict['launch_beam_radius_of_curvature'] = -4.0   
# args_dict['launch_position']                 = np.array([2.2,0,0]) # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition

# ne_fit_param = np.array([4.0, 1.0])

# kwargs_dict['density_fit_parameters']  = ne_fit_param
# kwargs_dict['find_B_method']           = 'analytical'
# kwargs_dict['Psi_BC_flag']             = True
# kwargs_dict['figure_flag']             = False
# kwargs_dict['vacuum_propagation_flag'] = True
# kwargs_dict['vacuumLaunch_flag']       = True
# kwargs_dict['poloidal_flux_enter']     = ne_fit_param[1]  
# kwargs_dict['B_T_axis']                = 1.0  
# kwargs_dict['B_p_a']                   = 0.1  
# kwargs_dict['R_axis']                  = 1.5  
# kwargs_dict['minor_radius_a']          = 0.5  
#-----------------------------------------------------------------------------


beam_me_up(**args_dict, **kwargs_dict)

