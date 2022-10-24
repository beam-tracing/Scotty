# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from scotty.beam_me_up import beam_me_up
import numpy as np


args_dict = dict([
                ('poloidal_launch_angle_Torbeam',-35.0),
                ('toroidal_launch_angle_Torbeam',-14.5), #-12.8
                ('launch_freq_GHz',90.0),
                ('mode_flag',-1),
                ('launch_beam_width',0.06323503329291348),
                ('launch_beam_curvature',-0.5535179506038995),
                ('launch_position',np.array([4.5,0.0,-0.6])),
            ])

kwargs_dict = dict([
                ('delta_R', -0.00001),
                ('delta_Z', 0.00001),
                ('find_B_method','UDA_saved'), # either UDA_saved or test_notime for JT60-SA equilibria
                ('Psi_BC_flag',True),
                ('figure_flag',True),
                ('vacuum_propagation_flag',True),
                ('vacuumLaunch_flag',True),
                ('density_fit_parameters',np.array([10.84222049, 0.17888095, 1.29493525, 5.81062798, 0.0, 0.02727005, 0.1152848, 0.94942186])),
                ('poloidal_flux_enter',1.3),
                # ('magnetic_data_path','D:\\Dropbox\\VHChen2021\\Collaborator - Daniel Carralero\\Processed data\\JT60SA_highden.npz'),
                ('magnetic_data_path','C:\\Users\\chenv\\Dropbox\\VHChen2021\\Collaborator - Daniel Carralero\\Processed data\\JT60SA_highden.npz'),
            ])



beam_me_up(**args_dict, **kwargs_dict)
    

