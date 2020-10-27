# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate as interpolate


loadfile = np.load('analysis_output.npz')
localisation_piece = loadfile['localisation_piece']
cutoff_index = loadfile['cutoff_index']
RZ_distance_along_line = loadfile['RZ_distance_along_line']
distance_along_line = loadfile['distance_along_line']
k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
Psi_xx_output = loadfile['Psi_xx_output']
Psi_xy_output = loadfile['Psi_xy_output']
Psi_yy_output = loadfile['Psi_yy_output']
M_xx_output = loadfile['M_xx_output']
M_xy_output = loadfile['M_xy_output']
M_yy_output = loadfile['M_yy_output']
in_index = loadfile['in_index']
out_index = loadfile['out_index']
loadfile.close()

loadfile = np.load('data_output.npz')
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
K_R_array = loadfile['K_R_array']
K_zeta_initial = loadfile['K_zeta_initial']
K_Z_array = loadfile['K_Z_array']
loadfile.close()

K_magnitude_array = np.sqrt(K_R_array**2 + K_Z_array**2 + K_zeta_initial**2/q_R_array**2)
d_K_magnitude_array_d_tau = np.gradient(K_magnitude_array,distance_along_line)
d2_K_magnitude_array_d_tau2 = np.gradient(d_K_magnitude_array_d_tau,distance_along_line)

#plt.figure()
#plt.plot(distance_along_line,d2_K_magnitude_array_d_tau2)

distance_in = distance_along_line[in_index]-distance_along_line[cutoff_index]
distance_out = distance_along_line[out_index]-distance_along_line[cutoff_index]

localisation = (g_magnitude_output[0]/g_magnitude_output)**2

plt.figure(figsize=(11.0, 5.0))
plt.subplot(1,2,1)
plt.plot(distance_along_line-distance_along_line[cutoff_index],localisation)
plt.axvline(distance_in,color='k')
plt.axvline(distance_out,color='k')
plt.xlabel(r' (l - l_c) / m $') # x-direction
plt.title(r'$ g^2_{ant} / g^2 $')

kperp1_minus_ks1 = - d2_K_magnitude_array_d_tau2[cutoff_index] * (distance_along_line-distance_along_line[cutoff_index])**2

plt.subplot(1,2,2)
plt.plot(kperp1_minus_ks1[0:cutoff_index],localisation[0:cutoff_index],'r')
plt.plot(kperp1_minus_ks1[cutoff_index::],localisation[cutoff_index::],'g')
plt.axvline(kperp1_minus_ks1[in_index],color='r')
plt.axvline(kperp1_minus_ks1[out_index],color='g')

plt.figure()
plt.plot(distance_along_line[0:cutoff_index],k_perp_1_backscattered[0:cutoff_index],'r')
plt.plot(distance_along_line[cutoff_index::],k_perp_1_backscattered[cutoff_index::],'g')
