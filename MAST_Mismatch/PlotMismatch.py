# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import tikzplotlib

loadfile = np.load('data_DBS.npz')
shot_array = loadfile['shot_array']
times_wanted_array = loadfile['times_wanted_array']
launch_freq_GHz_sweep = loadfile['launch_freq_GHz_sweep']
power_mean_sweep = loadfile['power_mean_sweep']
power_stderr_sweep = loadfile['power_stderr_sweep']
power_welch_sweep = loadfile['power_welch_sweep']
power_valerian_sweep = loadfile['power_valerian_sweep']
power_valerian_differential_sweep = loadfile['power_valerian_differential_sweep']
loadfile.close()

loadfile = np.load('data_mismatch.npz')
toroidal_launch_angle_Torbeam_scan = loadfile['toroidal_launch_angle_Torbeam_scan']
poloidal_launch_angle_Torbeam_scan = loadfile['poloidal_launch_angle_Torbeam_scan']
launch_freq_GHz_sweep = loadfile['launch_freq_GHz_sweep']
delta_theta_m_sweep = loadfile['delta_theta_m_sweep']
k_perp_1_backscattered_sweep = loadfile['k_perp_1_backscattered_sweep']
theta_m_sweep = loadfile['theta_m_sweep']
loadfile.close()

mode_string = 'X'


## The following values are ordered
# 29904, 29905, 29906, 29908, 29909, 29910 # Shot number
#     7,     8,     9,     5,     4,     6 # As recorded in the DBS logs
#    -4,    -5,    -6,    -2,    -1,    -3 # In Jon's paper (no offset, presumably) (just the sign convention, and the 3deg from some misalignment between the mirror and flange)
#    -5,    -6,    -7,    -3,    -2,    -4 # What I am using, ie 1 deg offset

# launch_array = np.array([-5,-6,-7,-3,-2,-4])
# toroidal_launch_angle_Torbeam_scan = np.linspace(0.0,-10.0,101)
launch_indices = np.array([41,51,61,21,11,31]) + 5

plt.figure()
for ii in range(0,len(launch_freq_GHz_sweep)):
    
    launch_freq_GHz = launch_freq_GHz_sweep[ii]    
    
    plt.subplot(4,2,ii+1)
    plt.plot(np.rad2deg(theta_m_sweep[ii,0,:]),
             np.exp(-theta_m_sweep[ii,0,:]**2/(2*delta_theta_m_sweep[ii,0,:]**2)),label='Model')
    plt.plot(np.rad2deg(theta_m_sweep[ii,0,launch_indices]),power_valerian_differential_sweep[ii,:]/power_valerian_differential_sweep[ii,:].max(),'o')
    plt.title(str(launch_freq_GHz) + 'GHz')