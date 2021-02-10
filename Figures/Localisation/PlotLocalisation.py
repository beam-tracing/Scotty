# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate as interpolate
import math
from scipy import constants as constants
import tikzplotlib

# Vectorised some of the calculation in this version
# This version does not do some of the calculations, as said calculations have been added to Scotty proper



suffix = ''

loadfile = np.load('data_input' + suffix + '.npz')
launch_freq_GHz =loadfile['launch_freq_GHz']
loadfile.close()

loadfile = np.load('analysis_output' + suffix + '.npz')
#localisation_piece = loadfile['localisation_piece']
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
theta_m_output = loadfile['theta_m_output']
K_magnitude_array = loadfile['K_magnitude_array']
localisation_b = loadfile['localisation_b']
localisation_p = loadfile['localisation_p']
localisation_r = loadfile['localisation_r']
localisation_s = loadfile['localisation_s']
localisation_b_p_r_s = loadfile['localisation_b_p_r_s']
localisation_b_p_r_s_distances = loadfile['localisation_b_p_r_s_distances']
localisation_b_p_r_s_max_over_e = loadfile['localisation_b_p_r_s_max_over_e']
localisation_b_p_r_s_half_width = loadfile['localisation_b_p_r_s_half_width']
localisation_b_p_r = loadfile['localisation_b_p_r']
localisation_b_p_r_distances = loadfile['localisation_b_p_r_distances']
localisation_b_p_r_max_over_e = loadfile['localisation_b_p_r_max_over_e']
localisation_b_p_r_half_width = loadfile['localisation_b_p_r_half_width']
loadfile.close()

loadfile = np.load('data_output' + suffix + '.npz')
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
K_R_array = loadfile['K_R_array']
K_zeta_initial = loadfile['K_zeta_initial']
K_Z_array = loadfile['K_Z_array']
loadfile.close()

print(localisation_b_p_r_s_half_width)
print(localisation_b_p_r_half_width)


plot_every_n_points = 1
out_index_new=len(q_R_array)

# localisation_b_p_r = localisation_b * localisation_p * localisation_r
localisation_b_r = localisation_b * localisation_r

# plt.figure()
# plt.subplot(2,2,1)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_r[:out_index_new])
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_b[:out_index_new])
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel(r'$g_{ant}^2 / g^2$')
# plt.subplot(2,2,2)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],k_perp_1_backscattered[:out_index_new])
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel(r'$- 2 K$')
# plt.subplot(2,2,3)
# plt.plot(k_perp_1_backscattered[:out_index_new],localisation_s[:out_index_new])
# plt.ylabel('spectrum')
# plt.xlabel(r'$- 2 K$')
# plt.subplot(2,2,4)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_b_r_s[:out_index_new])
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_r_s[:out_index_new])
# plt.plot(distance_along_line[localisation_b_r_s_index_1]-distance_along_line[cutoff_index],localisation_b_r_s[localisation_b_r_s_index_1],'ko')
# plt.plot(distance_along_line[localisation_b_r_s_index_2]-distance_along_line[cutoff_index],localisation_b_r_s[localisation_b_r_s_index_2],'ko')
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel('localisation')
# plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
# plt.savefig('localisation.jpg',dpi=150)

plt.figure()
plt.title('Beam')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_b[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_b.tex")

plt.figure()
plt.title('Polarisation')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_p[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_r2[:out_index_new])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel(r'$ pol $')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_p.tex")

plt.figure()
plt.title('Ray')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_r[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_r2[:out_index_new])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_r.tex")

plt.figure()
plt.title('Beam, polarisation, and ray')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_b_p_r[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.plot(localisation_b_p_r_distances[0],localisation_b_p_r_max_over_e,'ko', markersize=10)
plt.plot(localisation_b_p_r_distances[1],localisation_b_p_r_max_over_e,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_b_p_r.tex")

plt.figure()
plt.title('Beam and ray')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_b_r[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')

plt.figure()
plt.title('Spectrum')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_s[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_s.tex")



# plt.figure()
# plt.title('Ray and spectrum')
# plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_r_s[:out_index_new:plot_every_n_points])
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel('localisation')

plt.figure()
plt.title('Beam, polarisation, ray, and spectrum')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_b_p_r_s[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.plot(localisation_b_p_r_s_distances[0],localisation_b_p_r_s_max_over_e,'ko', markersize=10)
plt.plot(localisation_b_p_r_s_distances[1],localisation_b_p_r_s_max_over_e,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_b_r_s.tex")

# plt.figure()
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],np.sqrt(-np.imag(M_yy_inv[:out_index_new])))