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



suffix = 'many_points'

loadfile = np.load('data_input' + suffix + '.npz')
launch_freq_GHz =loadfile['launch_freq_GHz']
loadfile.close()

loadfile = np.load('analysis_output' + suffix + '.npz')
#loc_piece = loadfile['loc_piece']
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
k_perp_1_backscattered_plot = loadfile['k_perp_1_backscattered_plot']
loc_b = loadfile['loc_b']
loc_p = loadfile['loc_p']
loc_r = loadfile['loc_r']
loc_s = loadfile['loc_s']
loc_b_r_s = loadfile['loc_b_r_s']
# loc_b_r_s_distances = loadfile['loc_b_r_s_distances']
# loc_b_r_s_max_over_e2 = loadfile['loc_b_r_s_max_over_e2']
# loc_b_r_s_half_width = loadfile['loc_b_r_s_half_width']
cum_loc_b_r_plot = loadfile['cum_loc_b_r_plot']
cum_loc_b_r = loadfile['cum_loc_b_r']
loc_b_r = loadfile['loc_b_r']
# loc_b_r_distances = loadfile['loc_b_r_distances']
loc_b_r_s_max_over_e2 = loadfile['loc_b_r_s_max_over_e2']
loc_b_r_s_delta_l = loadfile['loc_b_r_s_delta_l']
cum_loc_b_r_s_max_over_e2 = loadfile['cum_loc_b_r_s_max_over_e2']
cum_loc_b_r_s_delta_l = loadfile['cum_loc_b_r_s_delta_l']
cum_loc_b_r_s_delta_kperp1 = loadfile['cum_loc_b_r_s_delta_kperp1']
# loc_b_r_half_width = loadfile['loc_b_r_half_width']
cum_loc_b_r_s_plot = loadfile['cum_loc_b_r_s_plot']
cum_loc_b_r_s = loadfile['cum_loc_b_r_s']
cum_loc_b_r_s_mean_l_lc = loadfile['cum_loc_b_r_s_mean_l_lc']
cum_loc_b_r_s_mean_kperp1 = loadfile['cum_loc_b_r_s_mean_kperp1']
loc_b_r_max_over_e2 = loadfile['loc_b_r_max_over_e2']
loc_b_r_delta_l = loadfile['loc_b_r_delta_l']
cum_loc_b_r_max_over_e2 = loadfile['cum_loc_b_r_max_over_e2']
cum_loc_b_r_delta_l = loadfile['cum_loc_b_r_delta_l']
cum_loc_b_r_delta_kperp1 = loadfile['cum_loc_b_r_delta_kperp1']
cum_loc_b_r_mean_l_lc = loadfile['cum_loc_b_r_mean_l_lc']
cum_loc_b_r_mean_kperp1 = loadfile['cum_loc_b_r_mean_kperp1']
loadfile.close()

loadfile = np.load('data_output' + suffix + '.npz')
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
K_R_array = loadfile['K_R_array']
K_zeta_initial = loadfile['K_zeta_initial']
K_Z_array = loadfile['K_Z_array']
loadfile.close()

# print(loc_b_r_s_half_width)
# print(loc_b_r_half_width)

              
plot_every_n_points = 1
out_index_new=len(q_R_array)

l_lc = distance_along_line-distance_along_line[cutoff_index]

# loc_b_r = loc_b * loc_p * loc_r
loc_b_r = loc_b * loc_r

# plt.figure()
# plt.subplot(2,2,1)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],loc_r[:out_index_new])
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],loc_b[:out_index_new])
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel(r'$g_{ant}^2 / g^2$')
# plt.subplot(2,2,2)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],k_perp_1_backscattered[:out_index_new])
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel(r'$- 2 K$')
# plt.subplot(2,2,3)
# plt.plot(k_perp_1_backscattered[:out_index_new],loc_s[:out_index_new])
# plt.ylabel('spectrum')
# plt.xlabel(r'$- 2 K$')
# plt.subplot(2,2,4)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],loc_b_r_s[:out_index_new])
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],loc_r_s[:out_index_new])
# plt.plot(distance_along_line[loc_b_r_s_index_1]-distance_along_line[cutoff_index],loc_b_r_s[loc_b_r_s_index_1],'ko')
# plt.plot(distance_along_line[loc_b_r_s_index_2]-distance_along_line[cutoff_index],loc_b_r_s[loc_b_r_s_index_2],'ko')
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel('localisation')
# plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
# plt.savefig('localisation.jpg',dpi=150)


# plt.figure()
# plt.title('Polarisation')
# plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],loc_p[:out_index_new:plot_every_n_points],linewidth=4.0)
# # plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],loc_r2[:out_index_new])
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel(r'$ pol $')
# plt.axvline(0,color='k', linestyle='dashed')
# tikzplotlib.save("loc_p.tex")

plt.figure()
plt.title('Beam')
plt.plot(l_lc[:out_index_new:plot_every_n_points],loc_b[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('Beam piece / m^{-1}')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("loc_b.tex")

plt.figure()
plt.title('Ray')
plt.plot(l_lc[:out_index_new:plot_every_n_points],loc_r[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.plot(l_lc[:out_index_new:plot_every_n_points],K_magnitude_array[0]**2/K_magnitude_array[:out_index_new:plot_every_n_points]**2,linewidth=1.0)
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],loc_r2[:out_index_new])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel(r'$g^{-2} / m^{-2}$')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("loc_r.tex")

## In general
# plt.figure()
# plt.figure()
# plt.title('Beam and ray')
# plt.plot(l_lc[:out_index_new:plot_every_n_points],loc_b_r[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.plot(loc_b_r_delta_l[0],loc_b_r_max_over_e2,'ko', markersize=10)
# plt.plot(loc_b_r_delta_l[1],loc_b_r_max_over_e2,'ko', markersize=10)
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel('localisation')
# plt.axvline(0,color='k', linestyle='dashed')
## In this particular case
plt.figure()
plt.title('Beam and ray')
plt.plot(l_lc[:out_index_new:plot_every_n_points],loc_b_r[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.plot(l_lc[0],loc_b_r[0],'ko', markersize=10)
plt.plot(loc_b_r_delta_l[1],loc_b_r_max_over_e2,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("loc_b_r.tex")


plt.figure()
plt.title('Spectrum')
plt.plot(l_lc[:out_index_new:plot_every_n_points],loc_s[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("loc_s.tex")


plt.figure()
plt.title('Beam, ray, and spectrum')
plt.plot(l_lc[:out_index_new:plot_every_n_points],loc_b_r_s[:out_index_new:plot_every_n_points],linewidth=4.0)
plt.plot(loc_b_r_s_delta_l[0],loc_b_r_s_max_over_e2,'ko', markersize=10)
plt.plot(loc_b_r_s_delta_l[1],loc_b_r_s_max_over_e2,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("loc_b_r_s.tex")

# plt.figure()
# plt.title('Beam, ray, and spectrum')
# plt.plot(k_perp_1_backscattered[:out_index_new:plot_every_n_points],loc_b_r_s[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.xlabel(r'$(l - l_c) / m$')
# plt.ylabel('localisation')
# plt.axvline(0,color='k', linestyle='dashed')

plt.figure()
plt.title('Beam and ray')
plt.plot(l_lc[:out_index_new:plot_every_n_points],cum_loc_b_r[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.axvline(cum_loc_b_r_mean_l_lc)
plt.plot(cum_loc_b_r_delta_l[0],-cum_loc_b_r_max_over_e2,'ko', markersize=10)
plt.plot(cum_loc_b_r_delta_l[1],cum_loc_b_r_max_over_e2,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
plt.axhline(0,color='k', linestyle='dashed')
tikzplotlib.save("cum_loc_b_r.tex")


plt.figure()
plt.title('Beam, ray, and spectrum')
plt.plot(l_lc[:out_index_new:plot_every_n_points],cum_loc_b_r_s[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.axvline(cum_loc_b_r_s_mean_l_lc)
plt.plot(cum_loc_b_r_s_delta_l[0],-cum_loc_b_r_s_max_over_e2,'ko', markersize=10)
plt.plot(cum_loc_b_r_s_delta_l[1],cum_loc_b_r_s_max_over_e2,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
plt.axhline(0,color='k', linestyle='dashed')
tikzplotlib.save("cum_loc_b_r_s.tex")

out_index_new=out_index_new+2

plt.figure()
plt.title('Beam and ray')
plt.plot(-k_perp_1_backscattered_plot[:out_index_new:plot_every_n_points],cum_loc_b_r_plot[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.axvline(-cum_loc_b_r_mean_kperp1)
plt.plot(-cum_loc_b_r_delta_kperp1[0],-cum_loc_b_r_max_over_e2,'ko', markersize=10)
plt.plot(-cum_loc_b_r_delta_kperp1[1],cum_loc_b_r_max_over_e2,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(min(abs(k_perp_1_backscattered_plot)),color='k', linestyle='dashed')
plt.axhline(0,color='k', linestyle='dashed')
tikzplotlib.save("cum_loc_b_r_kperp1.tex")


plt.figure()
plt.title('Beam, ray, and spectrum')
plt.plot(-k_perp_1_backscattered_plot[:out_index_new:plot_every_n_points],cum_loc_b_r_s_plot[:out_index_new:plot_every_n_points],linewidth=4.0)
# plt.axvline(-cum_loc_b_r_s_mean_kperp1)
plt.plot(-cum_loc_b_r_s_delta_kperp1[0],-cum_loc_b_r_s_max_over_e2,'ko', markersize=10)
plt.plot(-cum_loc_b_r_s_delta_kperp1[1],cum_loc_b_r_s_max_over_e2,'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(min(abs(k_perp_1_backscattered_plot)),color='k', linestyle='dashed')
plt.axhline(0,color='k', linestyle='dashed')
tikzplotlib.save("cum_loc_b_r_s_kperp1.tex")
