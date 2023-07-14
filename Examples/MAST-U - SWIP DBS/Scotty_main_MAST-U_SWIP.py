# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty


launch_freq_GHz = 30.0
equil_time = 0.780
poloidal_launch_angle_Torbeam = -2.0
toroidal_launch_angle_Torbeam = 0.0

kwargs_dict = get_parameters_for_Scotty(
                              'DBS_SWIP_MAST-U',
                              launch_freq_GHz = launch_freq_GHz,
                              find_B_method   = 'test', # EFITpp, UDA_saved, UDA, torbeam
                              equil_time      = equil_time,
                              shot            = 45304,
                              user            = 'Valerian_desktop'
                             )

kwargs_dict['mode_flag'] = 1
kwargs_dict['poloidal_launch_angle_Torbeam'] = poloidal_launch_angle_Torbeam
kwargs_dict['toroidal_launch_angle_Torbeam'] = toroidal_launch_angle_Torbeam

kwargs_dict['poloidal_flux_enter'] = 1.22
# kwargs_dict['output_path'] = os.path.dirname(os.path.abspath(__file__)) + '\\Output\\'
kwargs_dict['density_fit_parameters'] = None
kwargs_dict['ne_data_path'] = 'D:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST-U\\'
kwargs_dict['input_filename_suffix'] = '_45304_780ms'

kwargs_dict['figure_flag'] = False

kwargs_dict['delta_R'] = -0.001
kwargs_dict['delta_Z'] = -0.001
kwargs_dict['delta_K_R'] = 0.1
kwargs_dict['delta_K_zeta'] = 0.1
kwargs_dict['delta_K_Z'] = 0.1


beam_me_up(**kwargs_dict)
