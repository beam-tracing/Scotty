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





def find_H_Cardano(K_magnitude,launch_angular_frequency,epsilon_para,epsilon_perp,epsilon_g,theta_m):  
    # This function is designed to be evaluated in post-procesing, hence it uses different inputs from the usual H

    wavenumber_K0 = launch_angular_frequency / constants.c
    n_ref_index = K_magnitude/wavenumber_K0
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)
    
    
    D_11_component = epsilon_perp - n_ref_index**2*sin_theta_m**2
    D_22_component = epsilon_perp - n_ref_index**2
    D_bb_component = epsilon_para - n_ref_index**2*cos_theta_m**2
    D_12_component = epsilon_g
    D_1b_component = n_ref_index**2*sin_theta_m*cos_theta_m
    
    h_2_coefficient = - D_11_component - D_22_component - D_bb_component
    h_1_coefficient = D_11_component*D_bb_component + D_11_component*D_22_component + D_22_component*D_bb_component - D_12_component**2 - D_1b_component**2
    h_0_coefficient = D_22_component*D_1b_component**2 + D_bb_component*D_12_component**2 - D_11_component*D_22_component*D_bb_component
    
    h_t_coefficient = (
                            -2*h_2_coefficient**3
                            +9*h_2_coefficient*h_1_coefficient
                            -27*h_0_coefficient
                            +3*np.sqrt(3)*np.sqrt(
                                4*h_2_coefficient**3 * h_0_coefficient
                                -h_2_coefficient**2 * h_1_coefficient**2
                                -18*h_2_coefficient * h_1_coefficient * h_0_coefficient
                                +4*h_1_coefficient**3
                                +27*h_0_coefficient**2
                                +0j #to make the argument of the np.sqrt complex, so that the sqrt evaluates negative functions
                            )
                        )**(1/3)
    
    H_1_Cardano = h_t_coefficient/(3*2**(1/3)) - 2**(1/3) *(3*h_1_coefficient - h_2_coefficient**2)/(3*h_t_coefficient) - h_2_coefficient/3
    H_2_Cardano = - (1 - 1j*np.sqrt(3))/(6*2**(1/3))*h_t_coefficient + (1 + 1j*np.sqrt(3))*(3*h_1_coefficient - h_2_coefficient**2)/(3*2**(2/3)*h_t_coefficient) - h_2_coefficient/3
    H_3_Cardano = - (1 + 1j*np.sqrt(3))/(6*2**(1/3))*h_t_coefficient + (1 - 1j*np.sqrt(3))*(3*h_1_coefficient - h_2_coefficient**2)/(3*2**(2/3)*h_t_coefficient) - h_2_coefficient/3
    return H_1_Cardano, H_2_Cardano, H_3_Cardano

def find_coefficients(K_magnitude,launch_angular_frequency,epsilon_para,epsilon_perp,epsilon_g,theta_m):  
    # This function is designed to be evaluated in post-procesing, hence it uses different inputs from the usual H

    wavenumber_K0 = launch_angular_frequency / constants.c
    n_ref_index = K_magnitude/wavenumber_K0
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)
    
    
    D_11_component = epsilon_perp - n_ref_index**2*sin_theta_m**2
    D_22_component = epsilon_perp - n_ref_index**2
    D_bb_component = epsilon_para - n_ref_index**2*cos_theta_m**2
    D_12_component = epsilon_g
    D_1b_component = n_ref_index**2*sin_theta_m*cos_theta_m
    
    h_2_coefficient = - D_11_component - D_22_component - D_bb_component
    h_1_coefficient = D_11_component*D_bb_component + D_11_component*D_22_component + D_22_component*D_bb_component - D_12_component**2 - D_1b_component**2
    h_0_coefficient = D_22_component*D_1b_component**2 + D_bb_component*D_12_component**2 - D_11_component*D_22_component*D_bb_component
    
    return h_0_coefficient, h_1_coefficient, h_2_coefficient

def find_D_matrix(K_magnitude,launch_angular_frequency,epsilon_para,epsilon_perp,epsilon_g,theta_m):
    wavenumber_K0 = launch_angular_frequency / constants.c
    n_ref_index = K_magnitude/wavenumber_K0
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)
    
    
    D_11_component = epsilon_perp - n_ref_index**2*sin_theta_m**2
    D_22_component = epsilon_perp - n_ref_index**2
    D_bb_component = epsilon_para - n_ref_index**2*cos_theta_m**2
    D_12_component = epsilon_g
    D_1b_component = n_ref_index**2*sin_theta_m*cos_theta_m
    
    D_matrix = np.zeros([3,3],dtype='complex128')
    
    D_matrix[0][0] = D_11_component
    D_matrix[1][1] = D_22_component
    D_matrix[2][2] = D_bb_component
    D_matrix[0][2] = D_1b_component
    D_matrix[2][0] = D_1b_component
    D_matrix[0][1] = -1j*D_12_component
    D_matrix[1][0] = 1j*D_12_component

    return D_matrix

def find_nearest(array,  value): #returns the index
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)



suffix = '33'

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
in_index = loadfile['in_index']
out_index = loadfile['out_index']
theta_m_output = loadfile['theta_m_output']
K_magnitude_array = loadfile['K_magnitude_array']
loadfile.close()

loadfile = np.load('data_output' + suffix + '.npz')
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
K_R_array = loadfile['K_R_array']
K_zeta_initial = loadfile['K_zeta_initial']
K_Z_array = loadfile['K_Z_array']
epsilon_para_output = loadfile['epsilon_para_output']
epsilon_perp_output = loadfile['epsilon_perp_output']
epsilon_g_output = loadfile['epsilon_g_output']
loadfile.close()

delta_K_R = 0.1 #in the same units as K_R
delta_K_zeta = 0.1 #in the same units as K_zeta
delta_K_Z = 0.1 #in the same units as K_z



launch_angular_frequency = 2*math.pi*10.0**9 * launch_freq_GHz
K_magnitude_array = np.sqrt(K_R_array**2 + K_Z_array**2 + K_zeta_initial**2/q_R_array**2)

K_magnitude_array_plus_KR     = np.sqrt((K_R_array+delta_K_R)**2 + K_Z_array**2 + K_zeta_initial**2/q_R_array**2)
K_magnitude_array_minus_KR    = np.sqrt((K_R_array-delta_K_R)**2 + K_Z_array**2 + K_zeta_initial**2/q_R_array**2)
K_magnitude_array_plus_Kzeta  = np.sqrt(K_R_array**2 + K_Z_array**2 + (K_zeta_initial+delta_K_zeta)**2/q_R_array**2)
K_magnitude_array_minus_Kzeta = np.sqrt(K_R_array**2 + K_Z_array**2 + (K_zeta_initial-delta_K_zeta)**2/q_R_array**2)
K_magnitude_array_plus_KZ     = np.sqrt(K_R_array**2 + (K_Z_array+delta_K_Z)**2 + K_zeta_initial**2/q_R_array**2)
K_magnitude_array_minus_KZ    = np.sqrt(K_R_array**2 + (K_Z_array-delta_K_Z)**2 + K_zeta_initial**2/q_R_array**2)

numberOfDataPoints = len(K_magnitude_array)

# H_1_Cardano_array = np.zeros(numberOfDataPoints,dtype='complex128')
# H_2_Cardano_array = np.zeros(numberOfDataPoints,dtype='complex128')
# H_3_Cardano_array = np.zeros(numberOfDataPoints,dtype='complex128')

# H_3_Cardano_plus_KR_array = np.zeros(numberOfDataPoints,dtype='complex128')
# H_3_Cardano_minus_KR_array = np.zeros(numberOfDataPoints,dtype='complex128')
# H_3_Cardano_plus_Kzeta_array = np.zeros(numberOfDataPoints,dtype='complex128')
# H_3_Cardano_minus_Kzeta_array = np.zeros(numberOfDataPoints,dtype='complex128')
# H_3_Cardano_plus_KZ_array = np.zeros(numberOfDataPoints,dtype='complex128')
# H_3_Cardano_minus_KZ_array = np.zeros(numberOfDataPoints,dtype='complex128')

# h_0_coefficient_array = np.zeros(numberOfDataPoints)
# h_1_coefficient_array = np.zeros(numberOfDataPoints)
# h_2_coefficient_array = np.zeros(numberOfDataPoints)

    # K_magnitude = K_magnitude_array[ii]
    # epsilon_para = epsilon_para_output[ii]
    # epsilon_perp = epsilon_perp_output[ii]
    # epsilon_g = epsilon_g_output[ii]
    # theta_m = theta_m_output[ii]
    
H_1_Cardano_array,H_2_Cardano_array,H_3_Cardano_array = find_H_Cardano(K_magnitude_array,launch_angular_frequency,epsilon_para_output,epsilon_perp_output,epsilon_g_output,theta_m_output)

_,_,H_3_Cardano_plus_KR_array     = find_H_Cardano(K_magnitude_array_plus_KR    ,launch_angular_frequency,epsilon_para_output,epsilon_perp_output,epsilon_g_output,theta_m_output)
_,_,H_3_Cardano_minus_KR_array    = find_H_Cardano(K_magnitude_array_minus_KR   ,launch_angular_frequency,epsilon_para_output,epsilon_perp_output,epsilon_g_output,theta_m_output)
_,_,H_3_Cardano_plus_Kzeta_array  = find_H_Cardano(K_magnitude_array_plus_Kzeta ,launch_angular_frequency,epsilon_para_output,epsilon_perp_output,epsilon_g_output,theta_m_output)
_,_,H_3_Cardano_minus_Kzeta_array = find_H_Cardano(K_magnitude_array_minus_Kzeta,launch_angular_frequency,epsilon_para_output,epsilon_perp_output,epsilon_g_output,theta_m_output)
_,_,H_3_Cardano_plus_KZ_array     = find_H_Cardano(K_magnitude_array_plus_KZ    ,launch_angular_frequency,epsilon_para_output,epsilon_perp_output,epsilon_g_output,theta_m_output)
_,_,H_3_Cardano_minus_KZ_array    = find_H_Cardano(K_magnitude_array_minus_KZ   ,launch_angular_frequency,epsilon_para_output,epsilon_perp_output,epsilon_g_output,theta_m_output)

g_R_Cardano    = np.real(H_3_Cardano_plus_KR_array - H_3_Cardano_minus_KR_array) / (2 * delta_K_R)
g_zeta_Cardano = np.real(H_3_Cardano_plus_Kzeta_array - H_3_Cardano_minus_Kzeta_array) / (2 * delta_K_zeta)
g_Z_Cardano    = np.real(H_3_Cardano_plus_KZ_array - H_3_Cardano_minus_KZ_array) / (2 * delta_K_Z)

g_magnitude_Cardano = np.sqrt(g_R_Cardano**2 + g_zeta_Cardano**2 + g_Z_Cardano**2)
    
plt.figure()
plt.plot(distance_along_line,abs(H_1_Cardano_array),'r')    
plt.plot(distance_along_line,abs(H_2_Cardano_array),'g')    
plt.plot(distance_along_line,abs(H_3_Cardano_array),'b')    
plt.ylim(0,1)



#d_K_magnitude_array_d_tau = np.gradient(K_magnitude_array,distance_along_line)
#d2_K_magnitude_array_d_tau2 = np.gradient(d_K_magnitude_array_d_tau,distance_along_line)
#
#kperp1_minus_ks1 = - d2_K_magnitude_array_d_tau2[cutoff_index] * (distance_along_line-distance_along_line[cutoff_index])**2


plot_every_n_points = 10


out_index_new=out_index-15

localisation_ray = (2*constants.c/launch_angular_frequency)**2/g_magnitude_Cardano**2
# localisation_ray2 = g_magnitude_output[-1]**2/g_magnitude_output**2

localisation_spectrum = ( k_perp_1_backscattered / k_perp_1_backscattered[0] )**(-13/3)

# Beam: Localisation vs distance
M_w = np.zeros([numberOfDataPoints,2,2],dtype='complex128')
M_w[:,0,0] = M_xx_output
M_w[:,1,1] = M_yy_output
M_w[:,1,0] = M_xy_output
M_w[:,0,1] = M_w[:,1,0]

Psi_w = np.zeros([numberOfDataPoints,2,2],dtype='complex128')
Psi_w[:,0,0] = Psi_xx_output
Psi_w[:,1,1] = Psi_yy_output
Psi_w[:,1,0] = Psi_xy_output
Psi_w[:,0,1] = Psi_w[:,1,0]

M_yy_inv = M_xx_output / np.linalg.det(M_w)
         
localisation_beam = np.linalg.det(np.imag(Psi_w)) / abs(np.linalg.det(M_w))

localisation_beam2 = localisation_beam * np.sqrt(-np.imag(M_yy_inv))
# --

localisation_ray_spectrum = localisation_ray * localisation_spectrum
localisation_beam_ray_spectrum = localisation_beam2 * localisation_ray * localisation_spectrum
localisation_beam_ray = localisation_beam2 * localisation_ray

# Finds the half-width (1/e)**2 and the distance the peak is from the cutoff location
localisation_beam_ray_spectrum_max_over_e = localisation_beam_ray_spectrum.max() / (np.e) # localisation_beam_ray_spectrum.max() / 2.71
localisation_beam_ray_spectrum_max_index = find_nearest(localisation_beam_ray_spectrum,localisation_beam_ray_spectrum.max())
localised_beam_ray_spectrum_distance_from_cutoff = distance_along_line[localisation_beam_ray_spectrum_max_index]-distance_along_line[cutoff_index]
localisation_beam_ray_spectrum_index_1 = find_nearest(localisation_beam_ray_spectrum[0:localisation_beam_ray_spectrum_max_index],localisation_beam_ray_spectrum_max_over_e)
localisation_beam_ray_spectrum_index_2 = localisation_beam_ray_spectrum_max_index + find_nearest(localisation_beam_ray_spectrum[localisation_beam_ray_spectrum_max_index:],localisation_beam_ray_spectrum_max_over_e)
localisation_beam_ray_spectrum_half_width = 0.5*(distance_along_line[localisation_beam_ray_spectrum_index_2] - distance_along_line[localisation_beam_ray_spectrum_index_1])

plt.figure()
plt.subplot(2,2,1)
plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_ray[:out_index_new])
plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_beam[:out_index_new])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel(r'$g_{ant}^2 / g^2$')
plt.subplot(2,2,2)
plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],k_perp_1_backscattered[:out_index_new])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel(r'$- 2 K$')
plt.subplot(2,2,3)
plt.plot(k_perp_1_backscattered[:out_index_new],localisation_spectrum[:out_index_new])
plt.ylabel('spectrum')
plt.xlabel(r'$- 2 K$')
plt.subplot(2,2,4)
plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_beam_ray_spectrum[:out_index_new])
plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_ray_spectrum[:out_index_new])
plt.plot(distance_along_line[localisation_beam_ray_spectrum_index_1]-distance_along_line[cutoff_index],localisation_beam_ray_spectrum[localisation_beam_ray_spectrum_index_1],'ko')
plt.plot(distance_along_line[localisation_beam_ray_spectrum_index_2]-distance_along_line[cutoff_index],localisation_beam_ray_spectrum[localisation_beam_ray_spectrum_index_2],'ko')
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.savefig('localisation.jpg',dpi=150)

plt.figure()
plt.title('Beam')
# plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_beam[:out_index_new:plot_every_n_points])
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_beam2[:out_index_new:plot_every_n_points])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_beam.tex")

plt.figure()
plt.title('Ray')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_ray[:out_index_new:plot_every_n_points])
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],localisation_ray2[:out_index_new])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_ray.tex")

plt.figure()
plt.title('Spectrum')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_ray_spectrum[:out_index_new:plot_every_n_points])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_spectrum.tex")

plt.figure()
plt.title('Beam and ray')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_beam_ray[:out_index_new:plot_every_n_points])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_beam_ray.tex")


plt.figure()
plt.title('Ray and spectrum')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_ray_spectrum[:out_index_new:plot_every_n_points])
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')

plt.figure()
plt.title('Beam, ray, and spectrum')
plt.plot(distance_along_line[:out_index_new:plot_every_n_points]-distance_along_line[cutoff_index],localisation_beam_ray_spectrum[:out_index_new:plot_every_n_points])
plt.plot(distance_along_line[localisation_beam_ray_spectrum_index_1]-distance_along_line[cutoff_index],localisation_beam_ray_spectrum[localisation_beam_ray_spectrum_index_1],'ko', markersize=10)
plt.plot(distance_along_line[localisation_beam_ray_spectrum_index_2]-distance_along_line[cutoff_index],localisation_beam_ray_spectrum[localisation_beam_ray_spectrum_index_2],'ko', markersize=10)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel('localisation')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("localisation_beam_ray_spectrum.tex")

# plt.figure()
# plt.plot(distance_along_line[:out_index_new]-distance_along_line[cutoff_index],np.sqrt(-np.imag(M_yy_inv[:out_index_new])))