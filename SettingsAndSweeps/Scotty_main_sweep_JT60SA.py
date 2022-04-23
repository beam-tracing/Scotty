# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from Scotty_beam_me_up import beam_me_up
import numpy as np


args_dict = dict([
                ('poloidal_launch_angle_Torbeam',-30.0),
                ('toroidal_launch_angle_Torbeam',-10.0),
                ('launch_freq_GHz',90.0),
                ('mode_flag',-1),
                ('launch_beam_width',0.06323503329291348),
                ('launch_beam_radius_of_curvature',1 / -0.5535179506038995),
                ('launch_position',np.array([4.5,0.0,-0.6])),
            ])

kwargs_dict = dict([
                ('find_B_method','UDA_saved'),
                ('Psi_BC_flag',True),
                ('figure_flag',True),
                ('vacuum_propagation_flag',True),
                ('vacuumLaunch_flag',True),
                ('density_fit_parameters',np.array([10.84222049, 0.17888095, 1.29493525, 5.81062798, 0.0, 0.02727005, 0.1152848, 0.94942186])),
                ('poloidal_flux_enter',1.3),
                ('magnetic_data_path','D:\\Dropbox\\VHChen2021\\Collaborator - Daniel Carralero\\Processed data\\JT60SA_highden.npz'),
            ])

pol_launch_angles = np.linspace(-20,-40,41)
tor_launch_angles = np.linspace(0,-20,41)


for pol_launch_angle in pol_launch_angles:
    for tor_launch_angle in tor_launch_angles:
        args_dict['poloidal_launch_angle_Torbeam'] = pol_launch_angle
        args_dict['toroidal_launch_angle_Torbeam'] = tor_launch_angle
        
        if args_dict['mode_flag'] == 1:
            mode_string = 'O'
        elif args_dict['mode_flag'] == -1:
            mode_string = 'X'
        
        kwargs_dict['output_filename_suffix'] = (
                                    '_pol' + f'{pol_launch_angle:.1f}'
                                    '_tor' + f'{tor_launch_angle:.1f}'
                                  + '_f' + f"{args_dict['launch_freq_GHz']:.1f}"
                                  + '_'  + mode_string
                                  + '_'  + 'highdens'
                                      )      
  
        beam_me_up(**args_dict, **kwargs_dict)    

