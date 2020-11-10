# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from Scotty_fun_general import find_waist, find_distance_from_waist,find_q_lab_Cartesian, find_nearest, contract_special


loadfile = np.load('data_output0.npz')
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
B_total_output = loadfile['B_total_output']
loadfile.close()


loadfile = np.load('analysis_output0.npz')
Psi_xx_output = loadfile['Psi_xx_output']
Psi_xy_output = loadfile['Psi_xy_output']
Psi_yy_output = loadfile['Psi_yy_output']
M_xx_output = loadfile['M_xx_output']
M_xy_output = loadfile['M_xy_output']
M_yy_output = loadfile['M_yy_output']
xhat_dot_grad_bhat_dot_xhat_output = loadfile['xhat_dot_grad_bhat_dot_xhat_output']
xhat_dot_grad_bhat_dot_yhat_output = loadfile['xhat_dot_grad_bhat_dot_yhat_output']
xhat_dot_grad_bhat_dot_ghat_output = loadfile['xhat_dot_grad_bhat_dot_ghat_output']
yhat_dot_grad_bhat_dot_xhat_output = loadfile['yhat_dot_grad_bhat_dot_xhat_output']
yhat_dot_grad_bhat_dot_yhat_output = loadfile['yhat_dot_grad_bhat_dot_yhat_output']
yhat_dot_grad_bhat_dot_ghat_output = loadfile['yhat_dot_grad_bhat_dot_ghat_output']
kappa_dot_xhat_output = loadfile['kappa_dot_xhat_output']
kappa_dot_yhat_output = loadfile['kappa_dot_yhat_output']
delta_k_perp_2 = loadfile['delta_k_perp_2']
delta_theta_m = loadfile['delta_theta_m']
theta_m_output = loadfile['theta_m_output']
k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
cutoff_index = loadfile['cutoff_index']
K_magnitude_array = loadfile['K_magnitude_array']
poloidal_flux_on_midplane = loadfile['poloidal_flux_on_midplane']
R_midplane_points = loadfile['R_midplane_points']
Psi_3D_Cartesian = loadfile['Psi_3D_Cartesian']
loadfile.close()


loadfile = np.load('data_input0.npz')
data_poloidal_flux_grid = loadfile['data_poloidal_flux_grid']
data_R_coord = loadfile['data_R_coord']
data_Z_coord = loadfile['data_Z_coord']
launch_position = loadfile['launch_position']
launch_beam_width = loadfile['launch_beam_width']
launch_beam_curvature = loadfile['launch_beam_curvature']
loadfile.close()


[q_X_array,q_Y_array,q_Z_array] = find_q_lab_Cartesian(np.array([q_R_array,q_zeta_array,q_Z_array]))
numberOfDataPoints = np.size(q_R_array)

"""
Beam and ray path
"""
## For plotting the plasma in the toroidal plane
index_polmin = find_nearest(poloidal_flux_on_midplane,0)
R_polmin = R_midplane_points[index_polmin]
R_outboard = R_midplane_points[find_nearest(poloidal_flux_on_midplane[index_polmin:],1)+index_polmin]
index_local_polmax = find_nearest(poloidal_flux_on_midplane[0:index_polmin],10)
R_inboard = R_midplane_points[find_nearest(poloidal_flux_on_midplane[index_local_polmax:index_polmin],1)+index_local_polmax]
zeta_plot = np.linspace(-np.pi,np.pi,1001)
circle_outboard = np.zeros([1001,2])
circle_polmin = np.zeros([1001,2])
circle_inboard = np.zeros([1001,2])
circle_outboard[:,0],circle_outboard[:,1],_ = find_q_lab_Cartesian( np.array([R_outboard*np.ones_like(zeta_plot),zeta_plot,np.zeros_like(zeta_plot)]) )
circle_polmin[:,0],circle_polmin[:,1],_ = find_q_lab_Cartesian( np.array([R_polmin*np.ones_like(zeta_plot),zeta_plot,np.zeros_like(zeta_plot)]) )
circle_inboard[:,0],circle_inboard[:,1],_ = find_q_lab_Cartesian( np.array([R_inboard*np.ones_like(zeta_plot),zeta_plot,np.zeros_like(zeta_plot)]) )
##

## For plotting how the beam propagates from launch to entry
launch_position_X, launch_position_Y, launch_position_Z = find_q_lab_Cartesian(launch_position)
entry_position_X, entry_position_Y, entry_position_Z = find_q_lab_Cartesian(np.array([q_R_array[0],q_zeta_array[0],q_Z_array[0]]))
##

## For plotting the width in the RZ plane
W_vec_RZ = np.cross(g_hat_output,np.array([0,1,0]))
W_vec_RZ_magnitude = np.linalg.norm(W_vec_RZ,axis=1)
W_uvec_RZ = np.zeros_like(W_vec_RZ) # Unit vector
W_uvec_RZ[:,0] = W_vec_RZ[:,0] / W_vec_RZ_magnitude 
W_uvec_RZ[:,1] = W_vec_RZ[:,1] / W_vec_RZ_magnitude 
W_uvec_RZ[:,2] = W_vec_RZ[:,2] / W_vec_RZ_magnitude 
width_RZ = np.sqrt(2/np.imag( contract_special(W_vec_RZ,contract_special(Psi_3D_output,W_vec_RZ)) ))
W_line_RZ_1_Rpoints = q_R_array + W_uvec_RZ[:,0] * width_RZ
W_line_RZ_1_Zpoints = q_Z_array + W_uvec_RZ[:,2] * width_RZ
W_line_RZ_2_Rpoints = q_R_array - W_uvec_RZ[:,0] * width_RZ
W_line_RZ_2_Zpoints = q_Z_array - W_uvec_RZ[:,2] * width_RZ
##

## For plotting the width in the XY plane
g_hat_Cartesian = np.zeros([numberOfDataPoints,3])
g_hat_Cartesian[:,0] = g_hat_output[:,0] * np.cos(q_zeta_array ) - g_hat_output[:,1] * np.sin(q_zeta_array )
g_hat_Cartesian[:,1] = g_hat_output[:,0] * np.sin(q_zeta_array ) + g_hat_output[:,1] * np.cos(q_zeta_array )
g_hat_Cartesian[:,2] = g_hat_output[:,2]

W_vec_XY = np.cross(g_hat_output,np.array([0,0,1]))
W_vec_XY_magnitude = np.linalg.norm(W_vec_XY,axis=1)
W_vec_XY[:,0] = W_vec_XY[:,0] / W_vec_XY_magnitude
W_vec_XY[:,1] = W_vec_XY[:,1] / W_vec_XY_magnitude
W_vec_XY[:,2] = W_vec_XY[:,2] / W_vec_XY_magnitude
width_XY = np.sqrt(2/np.imag( contract_special(W_vec_XY,contract_special(Psi_3D_output,W_vec_XY)) ))
W_line_XY_1_Xpoints = q_X_array + W_vec_XY[:,0] * width_XY
W_line_XY_1_Ypoints = q_Y_array + W_vec_XY[:,1] * width_XY
W_line_XY_2_Xpoints = q_X_array - W_vec_XY[:,0] * width_XY
W_line_XY_2_Ypoints = q_Y_array - W_vec_XY[:,1] * width_XY
##

plt.figure(figsize=(3,5))
plt.title('Poloidal Plane')
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1.2,cmap='inferno')
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.plot(q_R_array,q_Z_array,'k')
plt.plot( [launch_position[0], q_R_array[0]], [launch_position[2], q_Z_array[0]],':k')
plt.plot(W_line_RZ_1_Rpoints,W_line_RZ_1_Zpoints,'k--')
plt.plot(W_line_RZ_2_Rpoints,W_line_RZ_2_Zpoints,'k--')
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])
plt.xlabel('R / m')
plt.ylabel('Z / m')
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize=(5,5))
plt.title('Toroidal Plane')
plt.plot(circle_outboard[:,0],circle_outboard[:,1],'orange')
plt.plot(circle_polmin[:,0],circle_polmin[:,1],'#00003F')
plt.plot(circle_inboard[:,0],circle_inboard[:,1],'orange')
plt.plot(q_X_array,q_Y_array,'k')
plt.plot( [launch_position_X, entry_position_X], [launch_position_Y, entry_position_Y],':k')
plt.plot(W_line_XY_1_Xpoints,W_line_XY_1_Ypoints,'k--')
plt.plot(W_line_XY_2_Xpoints,W_line_XY_2_Ypoints,'k--')
plt.xlim(-data_R_coord[-1],data_R_coord[-1])
plt.ylim(-data_R_coord[-1],data_R_coord[-1])
plt.xlabel('X / m')
plt.ylabel('Y / m')
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.gca().set_aspect('equal', adjustable='box')

# ------------------











