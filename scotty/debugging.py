# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen

There's a large discontinuity in dH_dR somewhere in the core and I'm trying
to figure out why. 18 April 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from scotty.fun_general import find_waist, find_distance_from_waist,find_q_lab_Cartesian, find_nearest, contract_special
from scotty.fun_general import find_H
# import tikzplotlib
import sys

from scotty.fun_FFD import find_dH_dR, find_dH_dZ # \nabla H


suffix = ''

loadfile = np.load('data_output' + suffix + '.npz')
tau_array = loadfile['tau_array']
q_R_array = loadfile['q_R_array']
q_zeta_array = loadfile['q_zeta_array']
q_Z_array = loadfile['q_Z_array']
K_R_array = loadfile['K_R_array']
#K_zeta_array = loadfile['K_zeta_array']
K_Z_array = loadfile['K_Z_array']
Psi_3D_output = loadfile['Psi_3D_output']
#Psi_w_xx_array = loadfile['Psi_w_xx_array']
#Psi_w_xy_array = loadfile['Psi_w_xy_array']
#Psi_w_yy_array = loadfile['Psi_w_yy_array']
#Psi_3D_output = loadfile['Psi_3D_output']
#x_hat_Cartesian_output = loadfile['x_hat_Cartesian_output']
#y_hat_Cartesian_output = loadfile['y_hat_Cartesian_output']
#b_hat_Cartesian_output = loadfile['b_hat_Cartesian_output']
x_hat_output = loadfile['x_hat_output']
y_hat_output = loadfile['y_hat_output']
B_R_output = loadfile['B_R_output']
B_T_output = loadfile['B_T_output']
B_Z_output = loadfile['B_Z_output']
#B_total_output = loadfile['B_total_output']
#grad_bhat_output = loadfile['grad_bhat_output']
#g_hat_Cartesian_output = loadfile['g_hat_Cartesian_output']
g_hat_output = loadfile['g_hat_output']
#g_magnitude_output = loadfile['g_magnitude_output']
#theta_output = loadfile['theta_output']
#d_theta_d_tau_output = loadfile['d_theta_d_tau_output']
#d_theta_m_d_tau_array = loadfile['d_theta_m_d_tau_array']
#kappa_dot_xhat_output = loadfile['kappa_dot_xhat_output']
#kappa_dot_yhat_output = loadfile['kappa_dot_yhat_output']
#d_xhat_d_tau_dot_yhat_output = loadfile['d_xhat_d_tau_dot_yhat_output']
#xhat_dot_grad_bhat_dot_xhat_output = loadfile['xhat_dot_grad_bhat_dot_xhat_output']
#xhat_dot_grad_bhat_dot_yhat_output = loadfile['xhat_dot_grad_bhat_dot_yhat_output']
#xhat_dot_grad_bhat_dot_ghat_output = loadfile['xhat_dot_grad_bhat_dot_ghat_output'] 
#yhat_dot_grad_bhat_dot_xhat_output = loadfile['yhat_dot_grad_bhat_dot_xhat_output']
#yhat_dot_grad_bhat_dot_yhat_output = loadfile['yhat_dot_grad_bhat_dot_yhat_output']
#yhat_dot_grad_bhat_dot_ghat_output = loadfile['yhat_dot_grad_bhat_dot_ghat_output']
#tau_start = loadfile['tau_start']
#tau_end = loadfile['tau_end']
#tau_nu_index=loadfile['tau_nu_index']
#tau_0_index=loadfile['tau_0_index']
#K_g_array = loadfile['K_g_array']
#d_K_g_d_tau_array = loadfile['d_K_g_d_tau_array']
# B_total_output = loadfile['B_total_output']
# b_hat_output = loadfile['b_hat_output']
# gradK_grad_H_output = loadfile['gradK_grad_H_output']
# gradK_gradK_H_output = loadfile['gradK_gradK_H_output']
# grad_grad_H_output = loadfile['grad_grad_H_output']
dH_dR_output = loadfile['dH_dR_output']
dH_dZ_output = loadfile['dH_dZ_output']
dH_dKR_output = loadfile['dH_dKR_output']
dH_dKzeta_output = loadfile['dH_dKzeta_output']
dH_dKZ_output = loadfile['dH_dKZ_output']
# dB_dR_FFD_debugging = loadfile['dB_dR_FFD_debugging']
# dB_dZ_FFD_debugging = loadfile['dB_dZ_FFD_debugging']
# d2B_dR2_FFD_debugging = loadfile['d2B_dR2_FFD_debugging']
# d2B_dZ2_FFD_debugging = loadfile['d2B_dZ2_FFD_debugging']
# d2B_dR_dZ_FFD_debugging = loadfile['d2B_dR_dZ_FFD_debugging']
# poloidal_flux_debugging_1R = loadfile['poloidal_flux_debugging_1R']
# poloidal_flux_debugging_2R = loadfile['poloidal_flux_debugging_2R']
# poloidal_flux_debugging_3R = loadfile['poloidal_flux_debugging_2R']
# poloidal_flux_debugging_1Z = loadfile['poloidal_flux_debugging_1Z']
# poloidal_flux_debugging_2Z = loadfile['poloidal_flux_debugging_2Z']
# poloidal_flux_debugging_3Z = loadfile['poloidal_flux_debugging_2Z']
# poloidal_flux_debugging_2R_2Z = loadfile['poloidal_flux_debugging_2R_2Z']
# electron_density_debugging_1R = loadfile['electron_density_debugging_1R']
# electron_density_debugging_2R = loadfile['electron_density_debugging_2R']
# electron_density_debugging_3R = loadfile['electron_density_debugging_3R']
# electron_density_debugging_1Z = loadfile['electron_density_debugging_1Z']
# electron_density_debugging_2Z = loadfile['electron_density_debugging_2Z']
# electron_density_debugging_3Z = loadfile['electron_density_debugging_3Z']
# electron_density_debugging_2R_2Z = loadfile['electron_density_debugging_2R_2Z']
# electron_density_output=loadfile['electron_density_output']
poloidal_flux_output = loadfile['poloidal_flux_output']
dpolflux_dR_debugging = loadfile['dpolflux_dR_debugging']
dpolflux_dZ_debugging = loadfile['dpolflux_dZ_debugging']
# d2polflux_dR2_FFD_debugging = loadfile['d2polflux_dR2_FFD_debugging']
# d2polflux_dZ2_FFD_debugging = loadfile['d2polflux_dZ2_FFD_debugging']
epsilon_para_output = loadfile['epsilon_para_output']
epsilon_perp_output = loadfile['epsilon_perp_output']
epsilon_g_output = loadfile['epsilon_g_output']
loadfile.close()


loadfile = np.load('data_input' + suffix + '.npz')
data_poloidal_flux_grid = loadfile['poloidalFlux_grid']
data_R_coord = loadfile['data_R_coord']
data_Z_coord = loadfile['data_Z_coord']
launch_position = loadfile['launch_position']
launch_beam_width = loadfile['launch_beam_width']
# launch_beam_curvature = loadfile['launch_beam_curvature']
loadfile.close()


[q_X_array,q_Y_array,q_Z_array] = find_q_lab_Cartesian(np.array([q_R_array,q_zeta_array,q_Z_array]))
numberOfDataPoints = np.size(q_R_array)

idx = np.argmax(np.absolute(dH_dR_output))

q_R_check = q_R_array[idx]
q_zeta_check = q_zeta_array[idx]
q_Z_check = q_Z_array[idx]

delta_R = -0.001

points = 50

test_R = np.linspace(q_R_check+points*delta_R, q_R_check-points*delta_R, points*2+1)

test_H = find_H(test_R, q_Z_check, K_R_array, K_zeta_initial, K_Z_array,
                          launch_angular_frequency, mode_flag,
                          interp_poloidal_flux, find_density_1D,
                          find_B_R, find_B_T, find_B_Z
                         )