# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


loadfile = np.load('analysis_output.npz')
cutoff_index = loadfile['cutoff_index']
RZ_distance_along_line = loadfile['RZ_distance_along_line']
k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
K_magnitude_array = loadfile['K_magnitude_array']
loadfile.close()

loadfile = np.load('data_output.npz')
tau_array = loadfile['tau_array']
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
q_Z_array = loadfile['q_Z_array']
loadfile.close()

loadfile = np.load('data_input.npz')
data_R_coord = loadfile['data_R_coord']
data_Z_coord = loadfile['data_Z_coord']
data_poloidal_flux_grid = loadfile['data_poloidal_flux_grid']
loadfile.close()


loadfile = np.load('analysis_output0.npz')
cutoff_index = loadfile['cutoff_index']
k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
delta_theta_m = loadfile['delta_theta_m']
theta_m_output = loadfile['theta_m_output']
loadfile.close()

plt.figure()
plt.plot(tau_array,theta_m_output)

#for ii in range(0,51):
#    loadfile = np.load('analysis_output.npz')
#    cutoff_index = loadfile['cutoff_index']
#    k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
#    delta_theta_m
#    loadfile.close()