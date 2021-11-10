# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from Scotty_fun_general import find_waist, find_distance_from_waist,find_q_lab_Cartesian, find_nearest, contract_special
from Scotty_fun_general import find_normalised_plasma_freq, find_normalised_gyro_freq, make_unit_vector_from_cross_product, find_vec_lab_Cartesian
import tikzplotlib
import math
from scipy import constants
import sys

suffix = '_DCS'

loadfile = np.load('data_output' + suffix + '.npz')
tau_array = loadfile['tau_array']
q_R_array = loadfile['q_R_array']
q_zeta_array = loadfile['q_zeta_array']
q_Z_array = loadfile['q_Z_array']
K_R_array = loadfile['K_R_array']
K_Z_array = loadfile['K_Z_array']
Psi_3D_output = loadfile['Psi_3D_output']
x_hat_output = loadfile['x_hat_output']
y_hat_output = loadfile['y_hat_output']
g_hat_output = loadfile['g_hat_output']
b_hat_output = loadfile['b_hat_output']
# epsilon_para_output = loadfile['epsilon_para_output']
# epsilon_perp_output = loadfile['epsilon_perp_output']
# epsilon_g_output = loadfile['epsilon_g_output']
B_magnitude = loadfile['B_magnitude']
g_magnitude_output = loadfile['g_magnitude_output']
# electron_density_output = loadfile['electron_density_output']
loadfile.close()

loadfile = np.load('analysis_output' + suffix + '.npz')
Psi_3D_Cartesian = loadfile['Psi_3D_Cartesian']
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
# d_theta_d_tau = loadfile['d_theta_d_tau']
# d_xhat_d_tau_dot_yhat_output = loadfile['d_xhat_d_tau_dot_yhat_output']
kappa_dot_xhat_output = loadfile['kappa_dot_xhat_output']
kappa_dot_yhat_output = loadfile['kappa_dot_yhat_output']
distance_along_line = loadfile['distance_along_line']
cutoff_index = loadfile['cutoff_index']
R_midplane_points = loadfile['R_midplane_points']
poloidal_flux_on_midplane = loadfile['poloidal_flux_on_midplane']
# e_hat_output = loadfile['e_hat_output']
# theta_output = loadfile['theta_output']
theta_m_output = loadfile['theta_m_output']
delta_theta_m = loadfile['delta_theta_m']
delta_k_perp_2 = loadfile['delta_k_perp_2']
y_hat_Cartesian = loadfile['y_hat_Cartesian']
x_hat_Cartesian = loadfile['x_hat_Cartesian']
K_magnitude_array = loadfile['K_magnitude_array']
k_perp_1_bs = loadfile['k_perp_1_backscattered']
loc_b_r_s = loadfile['loc_b_r_s']
loc_b_r = loadfile['loc_b_r']
loadfile.close()

loadfile = np.load('data_input' + suffix + '.npz')
poloidalFlux_grid = loadfile['data_poloidal_flux_grid']
data_R_coord = loadfile['data_R_coord']
data_Z_coord = loadfile['data_Z_coord']
launch_position = loadfile['launch_position']
launch_beam_width = loadfile['launch_beam_width']
launch_beam_radius_of_curvature = loadfile['launch_beam_radius_of_curvature']
launch_freq_GHz = loadfile['launch_freq_GHz']
loadfile.close()

l_lc = distance_along_line-distance_along_line[cutoff_index] # Distance from cutoff
[q_X_array,q_Y_array,q_Z_array] = find_q_lab_Cartesian(np.array([q_R_array,q_zeta_array,q_Z_array]))
numberOfDataPoints = np.size(q_R_array)

out_index = numberOfDataPoints

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
width_RZ = np.sqrt(2/np.imag( contract_special(W_uvec_RZ,contract_special(Psi_3D_output,W_uvec_RZ)) ))
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

W_uvec_XY = make_unit_vector_from_cross_product(g_hat_Cartesian,np.array([0,0,1]))
# W_uvec_Rzeta = make_unit_vector_from_cross_product(g_hat_output,np.array([0,0,1]))
width_XY = np.sqrt(2/np.imag( contract_special(W_uvec_XY,contract_special(Psi_3D_Cartesian,W_uvec_XY)) ))
W_line_XY_1_Xpoints = q_X_array + W_uvec_XY[:,0] * width_XY
W_line_XY_1_Ypoints = q_Y_array + W_uvec_XY[:,1] * width_XY
W_line_XY_2_Xpoints = q_X_array - W_uvec_XY[:,0] * width_XY
W_line_XY_2_Ypoints = q_Y_array - W_uvec_XY[:,1] * width_XY
##

plt.figure(figsize=(5,5))
plt.title('Poloidal Plane')
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(poloidalFlux_grid), contour_levels,vmin=0,vmax=1.2,cmap='inferno')
plt.clabel(CS, inline=True, fontsize=10,inline_spacing=-5,fmt= '%1.1f',use_clabeltext=True) # Labels the flux surfaces
# plt.xlim(1.0,1.8)
# plt.ylim(-0.7,0.1)
# tikzplotlib.clean_figure() # Removes points that are outside the plot area
plt.plot(q_R_array[:out_index],q_Z_array[:out_index],'k')
plt.plot( [launch_position[0], q_R_array[0]], [launch_position[2], q_Z_array[0]],':k')
plt.plot(W_line_RZ_1_Rpoints[:out_index],W_line_RZ_1_Zpoints[:out_index],'k--')
plt.plot(W_line_RZ_2_Rpoints[:out_index],W_line_RZ_2_Zpoints[:out_index],'k--')
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])

plt.xlabel('R / m')
plt.ylabel('Z / m')
tikzplotlib.save("propagation_poloidal.tex")
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('propagation_poloidal.jpg',dpi=200)

plt.figure(figsize=(5,5))
plt.title('Toroidal Plane')
plt.plot(circle_outboard[:,0],circle_outboard[:,1],'orange')
plt.plot(circle_polmin[:,0],circle_polmin[:,1],'#00003F')
plt.plot(circle_inboard[:,0],circle_inboard[:,1],'orange')
# plt.xlim(1.0,1.8)
# plt.ylim(-0.2,0.6)
plt.plot(q_X_array[:out_index],q_Y_array[:out_index],'k')
plt.plot( [launch_position_X, entry_position_X], [launch_position_Y, entry_position_Y],':k')
plt.plot(W_line_XY_1_Xpoints[:out_index],W_line_XY_1_Ypoints[:out_index],'k--')
plt.plot(W_line_XY_2_Xpoints[:out_index],W_line_XY_2_Ypoints[:out_index],'k--')
plt.xlim(-data_R_coord[-1],data_R_coord[-1])
plt.ylim(-data_R_coord[-1],data_R_coord[-1])
plt.xlabel('X / m')
plt.ylabel('Y / m')
tikzplotlib.save("propagation_toroidal.tex")
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('propagation_toroidal.jpg',dpi=200)




plt.figure(figsize=(16,5))
plt.subplot(2,3,1)
plt.plot(l_lc,np.real(Psi_xx_output),'k')
# plt.plot(l_lc,np.real(M_xx_output),'r')
plt.title(r'Re$(\Psi_{xx})$ and Re$(M_{xx})$')
plt.xlabel(r'$l-l_c$')
plt.subplot(2,3,2)
plt.plot(l_lc,np.real(Psi_xy_output),'k')
# plt.plot(l_lc,np.real(M_xy_output),'r')
plt.title(r'Re$(\Psi_{xy})$ and Re$(M_{xy})$')
plt.xlabel(r'$l-l_c$')
plt.subplot(2,3,3)
plt.plot(l_lc,np.real(Psi_yy_output),'k')
plt.title(r'Re$(\Psi_{yy})$')
plt.xlabel(r'$l-l_c$')
plt.subplot(2,3,4)
plt.plot(l_lc,np.imag(Psi_xx_output),'k')
# plt.plot(l_lc,np.imag(M_xx_output),'r')
plt.title(r'Im$(\Psi_{xx})$ and Im$(M_{xx})$')
plt.xlabel(r'$l-l_c$')
plt.subplot(2,3,5)
plt.plot(l_lc,np.imag(Psi_xy_output),'k')
# plt.plot(l_lc,np.imag(M_xy_output),'r')
plt.title(r'Im$(\Psi_{xy})$ and Im$(M_{xy})$')
plt.xlabel(r'$l-l_c$')
plt.subplot(2,3,6)
plt.plot(l_lc,np.imag(Psi_yy_output),'k')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.title(r'Im$(\Psi_{yy})$')
plt.xlabel(r'$l-l_c$')

sys.exit()
