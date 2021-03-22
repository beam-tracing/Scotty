# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import tikzplotlib


toroidal_launch_angle_Torbeam_scan = np.linspace(0.0,-10.0,101)
poloidal_launch_angle_Torbeam_scan = np.array([4.0])

launch_freq_GHz_sweep = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0])

mode_string = 'X'

## The following values are ordered
# 29904, 29905, 29906, 29908, 29909, 29910 # Shot number
#     7,     8,     9,     5,     4,     6 # As recorded in the DBS logs
#    -4,    -5,    -6,    -2,    -1,    -3 # In Jon's paper (no offset, presumably) (just the sign convention, and the 3deg from some misalignment between the mirror and flange)
#    -5,    -6,    -7,    -3,    -2,    -4 # What I am using, ie 1 deg offset
    
# --

theta_m_sweep                = np.zeros([len(launch_freq_GHz_sweep), len(poloidal_launch_angle_Torbeam_scan), len(toroidal_launch_angle_Torbeam_scan)])
delta_theta_m_sweep          = np.zeros_like(theta_m_sweep)
k_perp_1_backscattered_sweep = np.zeros_like(theta_m_sweep)




for ii in range(0,len(launch_freq_GHz_sweep)):
    for jj in range(0,len(poloidal_launch_angle_Torbeam_scan)):
        for kk in range(0,len(toroidal_launch_angle_Torbeam_scan)):
    
            launch_freq_GHz = launch_freq_GHz_sweep[ii]    
            poloidal_launch_angle_Torbeam = poloidal_launch_angle_Torbeam_scan[jj]            
            toroidal_launch_angle_Torbeam = toroidal_launch_angle_Torbeam_scan[kk]

            output_filename_string = (
                                        '_p' + f'{poloidal_launch_angle_Torbeam:.1f}'
                                      + '_t' + f'{toroidal_launch_angle_Torbeam:.1f}' 
                                      + '_f' + f'{launch_freq_GHz:.1f}'
                                      + '_'  + mode_string
                                      + '_z-1.0' 
                                      + '_r885'
                                      + '.png'
                                      )
            
    
            loadfile = np.load('analysis_output' + output_filename_string +  '.npz')
            cutoff_index = loadfile['cutoff_index']
            k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
            delta_theta_m = loadfile['delta_theta_m']
            theta_m_output = loadfile['theta_m_output']
            # localisation_beam_ray_spectrum_max_index = loadfile['localisation_beam_ray_spectrum_max_index']
            loadfile.close()
            
            theta_m_sweep[ii,jj,kk]                = theta_m_output[cutoff_index]
            delta_theta_m_sweep[ii,jj,kk]          = delta_theta_m[cutoff_index] 
            k_perp_1_backscattered_sweep[ii,jj,kk] = k_perp_1_backscattered[cutoff_index]
            
np.savez('data_mismatch',
         toroidal_launch_angle_Torbeam_scan=toroidal_launch_angle_Torbeam_scan,
         poloidal_launch_angle_Torbeam_scan=poloidal_launch_angle_Torbeam_scan,
         launch_freq_GHz_sweep=launch_freq_GHz_sweep,
         theta_m_sweep = theta_m_sweep,
         delta_theta_m_sweep=delta_theta_m_sweep,
         k_perp_1_backscattered_sweep=k_perp_1_backscattered_sweep
         )    