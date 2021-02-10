# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import tikzplotlib



#loadfile = np.load('analysis_output.npz')
#cutoff_index = loadfile['cutoff_index']
#RZ_distance_along_line = loadfile['RZ_distance_along_line']
#k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
#K_magnitude_array = loadfile['K_magnitude_array']
#loadfile.close()
#
#loadfile = np.load('data_output.npz')
#tau_array = loadfile['tau_array']
#g_magnitude_output = loadfile['g_magnitude_output']
#q_R_array = loadfile['q_R_array']
#q_Z_array = loadfile['q_Z_array']
#loadfile.close()
#
#loadfile = np.load('data_input.npz')
#data_R_coord = loadfile['data_R_coord']
#data_Z_coord = loadfile['data_Z_coord']
#data_poloidal_flux_grid = loadfile['data_poloidal_flux_grid']
#loadfile.close()
#
#
#loadfile = np.load('analysis_output0.npz')
#cutoff_index = loadfile['cutoff_index']
#k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
#delta_theta_m = loadfile['delta_theta_m']
#theta_m_output = loadfile['theta_m_output']
#loadfile.close()

numberOfRuns = 51
# toroidal_launch_angle_Torbeam_scan = np.linspace(-0.5,-7.5,71)


## The following values are ordered
# 29904, 29905, 29906, 29908, 29909, 29910 # Shot number
#     7,     8,     9,     5,     4,     6 # As recorded in the DBS logs
#    -4,    -5,    -6,    -2,    -1,    -3 # In Jon's paper (no offset, presumably) (just the sign convention, and the 3deg from some misalignment between the mirror and flange)
#    -5,    -6,    -7,    -3,    -2,    -4 # What I am using, ie 1 deg offset
    
scattered_power = np.asarray([0.000373357,5.29E-06,4.50E-06,4.66E-04,1.93E-05,0.000955853])
scattered_power_stdev = np.asarray([8.73E-05,1.14E-06,2.05E-06,0.000189811,7.52E-06,0.000127163])
# --

theta_m_sweep                = np.zeros(numberOfRuns)
delta_theta_m_sweep          = np.zeros(numberOfRuns) 
k_perp_1_backscattered_sweep = np.zeros(numberOfRuns)
theta_m_sweep2               = np.zeros(numberOfRuns)
delta_theta_m_sweep2         = np.zeros(numberOfRuns) 

for ii in range(0,numberOfRuns):
    loadfile = np.load('analysis_output' + str(ii) +  '.npz')
    cutoff_index = loadfile['cutoff_index']
    k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
    delta_theta_m = loadfile['delta_theta_m']
    theta_m_output = loadfile['theta_m_output']
    # localisation_beam_ray_spectrum_max_index = loadfile['localisation_beam_ray_spectrum_max_index']
    loadfile.close()
    
    theta_m_sweep[ii]                = theta_m_output[cutoff_index]
    delta_theta_m_sweep[ii]          = delta_theta_m[cutoff_index] 
    k_perp_1_backscattered_sweep[ii] = k_perp_1_backscattered[cutoff_index]
    # theta_m_sweep2[ii]               = theta_m_output[localisation_beam_ray_spectrum_max_index]
    # delta_theta_m_sweep2[ii]         = delta_theta_m[localisation_beam_ray_spectrum_max_index] 
    
# mismatch_angle_offset = np.rad2deg(np.asarray([theta_m_sweep[43],theta_m_sweep[53],theta_m_sweep[63],theta_m_sweep[23],theta_m_sweep[13],theta_m_sweep[33]]))
mismatch_angle_offset = np.asarray([6.17073,10.24059,13.86935,-3.20812,-8.38919,1.668527])
    
k_to_choose = k_perp_1_backscattered_sweep[0]
turbulence_piece = (k_perp_1_backscattered_sweep/k_to_choose)**(-2*10/3)
plt.figure()
plt.plot(np.rad2deg(theta_m_sweep),np.exp(-(theta_m_sweep/delta_theta_m_sweep)**2),label='Model')
# plt.plot(np.rad2deg(theta_m_sweep),np.exp(-(theta_m_sweep2/delta_theta_m_sweep2)**2))
#plt.plot(np.rad2deg(theta_m_sweep),np.exp(-(theta_m_sweep/delta_theta_m_sweep)**2)*turbulence_piece)
plt.errorbar(mismatch_angle_offset, scattered_power/max(scattered_power), yerr=(scattered_power_stdev/max(scattered_power)), fmt='o',label='Data')
plt.legend()

# plt.figure()
# plt.plot(theta_m_sweep)

tikzplotlib.save("test.tex")
