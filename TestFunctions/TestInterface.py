# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 20:23:50 2020

@author: VH Chen
"""
import numpy as np
import matplotlib.pyplot as plt

def find_Psi_3D_plasma(Psi_v_R_R, Psi_v_R_Z, Psi_v_R_zeta, 
                       Psi_v_Z_Z, Psi_v_Z_zeta, Psi_v_zeta_zeta,
                       g_R, g_Z, g_zeta,
                       dH_dR, dH_dZ,
                       d_poloidal_flux_d_R, d_poloidal_flux_d_Z):
    # When beam is entering plasma from vacuum    
    interface_matrix = np.zeros([6,6])
    interface_matrix[0][5] = 1
    interface_matrix[1][0] = d_poloidal_flux_d_Z**2
    interface_matrix[1][1] = - 2 * d_poloidal_flux_d_R * d_poloidal_flux_d_Z
    interface_matrix[1][3] = d_poloidal_flux_d_R**2
    interface_matrix[2][2] = - d_poloidal_flux_d_Z
    interface_matrix[2][4] = d_poloidal_flux_d_R
    interface_matrix[3][0] = g_R
    interface_matrix[3][1] = g_Z
    interface_matrix[3][2] = g_zeta
    interface_matrix[4][1] = g_R
    interface_matrix[4][3] = g_Z
    interface_matrix[4][4] = g_zeta
    interface_matrix[5][2] = g_R
    interface_matrix[5][4] = g_Z
    interface_matrix[5][5] = g_zeta
    
    interface_matrix_inverse = np.linalg.inv(interface_matrix)
    
    [
     Psi_p_R_R, Psi_p_R_Z, Psi_p_R_zeta, 
     Psi_p_Z_Z, Psi_p_Z_zeta, Psi_p_zeta_zeta
    ] = np.matmul (interface_matrix_inverse, [
            Psi_v_zeta_zeta, 
            Psi_v_R_R * d_poloidal_flux_d_Z**2 - 2 * Psi_v_R_Z * d_poloidal_flux_d_R * d_poloidal_flux_d_Z + Psi_v_Z_Z * d_poloidal_flux_d_R **2, 
            - Psi_v_R_zeta * d_poloidal_flux_d_Z + Psi_v_Z_zeta * d_poloidal_flux_d_R, 
            dH_dR, 
            dH_dZ, 
            0          
           ] )
    return Psi_p_R_R, Psi_p_R_Z, Psi_p_R_zeta, Psi_p_Z_Z, Psi_p_Z_zeta, Psi_p_zeta_zeta




#loadfile = np.load('data_output.npz')
#tau_array = loadfile['tau_array']
#q_R_array = loadfile['q_R_array']
#q_zeta_array = loadfile['q_zeta_array']
#q_X_array = loadfile['q_X_array']
#q_Y_array = loadfile['q_Y_array']
#q_Z_array = loadfile['q_Z_array']
#K_R_array = loadfile['K_R_array']
#K_zeta_array = loadfile['K_zeta_array']
#K_Z_array = loadfile['K_Z_array']
#K_magnitude_array = loadfile['K_magnitude_array']
#Psi_w_xx_array = loadfile['Psi_w_xx_array']
#Psi_w_xy_array = loadfile['Psi_w_xy_array']
#Psi_w_yy_array = loadfile['Psi_w_yy_array']
#Psi_3D_output = loadfile['Psi_3D_output']
#x_hat_Cartesian_output = loadfile['x_hat_Cartesian_output']
#y_hat_Cartesian_output = loadfile['y_hat_Cartesian_output']
#b_hat_Cartesian_output = loadfile['b_hat_Cartesian_output']
#x_hat_output = loadfile['x_hat_output']
#y_hat_output = loadfile['y_hat_output']
#B_total_output = loadfile['B_total_output']
#grad_bhat_output = loadfile['grad_bhat_output']
#g_hat_Cartesian_output = loadfile['g_hat_Cartesian_output']
#g_hat_output = loadfile['g_hat_output']
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
#B_total_output = loadfile['B_total_output']
#d_poloidal_flux_dR = loadfile['d_poloidal_flux_dR_output']
#d_poloidal_flux_dZ = loadfile['d_poloidal_flux_dZ_output']
#dH_dR_output = loadfile['dH_dR_output']
#dH_dZ_output = loadfile['dH_dZ_output']
#loadfile.close()
#
#loadfile = np.load('data_input.npz')
#data_poloidal_flux_grid = loadfile['data_poloidal_flux_grid']
#data_X_coord = loadfile['data_X_coord']
#data_Z_coord = loadfile['data_Z_coord']
#launch_beam_width = loadfile['launch_beam_width']
#launch_beam_curvature = loadfile['launch_beam_curvature']
#loadfile.close()


loadfile = np.load('data_output_alt.npz')
tau_array = loadfile['tau_array']
q_R_array = loadfile['q_R_array']
q_zeta_array = loadfile['q_zeta_array']
K_R_array = loadfile['K_R_array']
K_zeta_array = loadfile['K_zeta_array']
K_Z_array = loadfile['K_Z_array']
Psi_3D_output = loadfile['Psi_3D_output']
x_hat_output = loadfile['x_hat_output']
y_hat_output = loadfile['y_hat_output']
B_total_output = loadfile['B_total_output']
g_hat_output = loadfile['g_hat_output']
d_poloidal_flux_dR = loadfile['d_poloidal_flux_dR_output']
d_poloidal_flux_dZ = loadfile['d_poloidal_flux_dZ_output']
dH_dR_output = loadfile['dH_dR_output']
dH_dZ_output = loadfile['dH_dZ_output']
loadfile.close()


    
tau_to_use = int(187)
    


[Psi_p_R_R, Psi_p_R_Z, Psi_p_R_zeta, Psi_p_Z_Z, Psi_p_Z_zeta, Psi_p_zeta_zeta]= find_Psi_3D_plasma(
                       Psi_3D_output[0,0,tau_to_use], Psi_3D_output[0,2,tau_to_use], Psi_3D_output[0,1,tau_to_use], 
                       Psi_3D_output[2,2,tau_to_use], Psi_3D_output[2,1,tau_to_use], Psi_3D_output[1,1,tau_to_use],
                       g_hat_output[0,tau_to_use], g_hat_output[2,tau_to_use], g_hat_output[1,tau_to_use],
                       dH_dR_output[tau_to_use], dH_dZ_output[tau_to_use],
                       d_poloidal_flux_dR[tau_to_use], d_poloidal_flux_dZ[tau_to_use])

x_hat = x_hat_output[:,tau_to_use]
y_hat = y_hat_output[:,tau_to_use]
Psi_p_3D = np.array([
            [Psi_p_R_R   , Psi_p_R_zeta   , Psi_p_R_Z],
            [Psi_p_R_zeta, Psi_p_zeta_zeta, Psi_p_Z_zeta],
            [Psi_p_R_Z   , Psi_p_Z_zeta   , Psi_p_Z_Z  ]
            ])
Psi_v_3D = np.squeeze(Psi_3D_output[:,:,tau_to_use])

Psi_p_xx = np.matmul(x_hat,np.matmul(Psi_p_3D,x_hat))
Psi_p_xy = np.matmul(x_hat,np.matmul(Psi_p_3D,y_hat))
Psi_p_yy = np.matmul(y_hat,np.matmul(Psi_p_3D,y_hat))

Psi_v_xx = np.matmul(x_hat,np.matmul(Psi_v_3D,x_hat))
Psi_v_xy = np.matmul(x_hat,np.matmul(Psi_v_3D,y_hat))
Psi_v_yy = np.matmul(y_hat,np.matmul(Psi_v_3D,y_hat))
