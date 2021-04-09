# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from Scotty_fun_general import find_waist, find_distance_from_waist,find_q_lab_Cartesian, find_nearest, contract_special
import tikzplotlib

suffix = 'benchmark'

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
b_hat_output = loadfile['b_hat_output']
gradK_grad_H_output = loadfile['gradK_grad_H_output']
gradK_gradK_H_output = loadfile['gradK_gradK_H_output']
grad_grad_H_output = loadfile['grad_grad_H_output']
dH_dR_output = loadfile['dH_dR_output']
dH_dZ_output = loadfile['dH_dZ_output']
dH_dKR_output = loadfile['dH_dKR_output']
dH_dKzeta_output = loadfile['dH_dKzeta_output']
dH_dKZ_output = loadfile['dH_dKZ_output']
dB_dR_FFD_debugging = loadfile['dB_dR_FFD_debugging']
dB_dZ_FFD_debugging = loadfile['dB_dZ_FFD_debugging']
d2B_dR2_FFD_debugging = loadfile['d2B_dR2_FFD_debugging']
d2B_dZ2_FFD_debugging = loadfile['d2B_dZ2_FFD_debugging']
d2B_dR_dZ_FFD_debugging = loadfile['d2B_dR_dZ_FFD_debugging']
poloidal_flux_debugging_1R = loadfile['poloidal_flux_debugging_1R']
poloidal_flux_debugging_2R = loadfile['poloidal_flux_debugging_2R']
poloidal_flux_debugging_3R = loadfile['poloidal_flux_debugging_2R']
poloidal_flux_debugging_1Z = loadfile['poloidal_flux_debugging_1Z']
poloidal_flux_debugging_2Z = loadfile['poloidal_flux_debugging_2Z']
poloidal_flux_debugging_3Z = loadfile['poloidal_flux_debugging_2Z']
poloidal_flux_debugging_2R_2Z = loadfile['poloidal_flux_debugging_2R_2Z']
electron_density_debugging_1R = loadfile['electron_density_debugging_1R']
electron_density_debugging_2R = loadfile['electron_density_debugging_2R']
electron_density_debugging_3R = loadfile['electron_density_debugging_3R']
electron_density_debugging_1Z = loadfile['electron_density_debugging_1Z']
electron_density_debugging_2Z = loadfile['electron_density_debugging_2Z']
electron_density_debugging_3Z = loadfile['electron_density_debugging_3Z']
electron_density_debugging_2R_2Z = loadfile['electron_density_debugging_2R_2Z']
electron_density_output=loadfile['electron_density_output']
poloidal_flux_output = loadfile['poloidal_flux_output']
dpolflux_dR_FFD_debugging = loadfile['dpolflux_dR_FFD_debugging']
dpolflux_dZ_FFD_debugging = loadfile['dpolflux_dZ_FFD_debugging']
d2polflux_dR2_FFD_debugging = loadfile['d2polflux_dR2_FFD_debugging']
d2polflux_dZ2_FFD_debugging = loadfile['d2polflux_dZ2_FFD_debugging']
epsilon_para_output = loadfile['epsilon_para_output']
epsilon_perp_output = loadfile['epsilon_perp_output']
epsilon_g_output = loadfile['epsilon_g_output']
loadfile.close()





loadfile = np.load('analysis_output' + suffix + '.npz')
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
out_index = loadfile['out_index']
K_magnitude_array = loadfile['K_magnitude_array']
poloidal_flux_on_midplane = loadfile['poloidal_flux_on_midplane']
R_midplane_points = loadfile['R_midplane_points']
Psi_3D_Cartesian = loadfile['Psi_3D_Cartesian']
distance_along_line = loadfile['distance_along_line']
x_hat_Cartesian = loadfile['x_hat_Cartesian']
y_hat_Cartesian = loadfile['y_hat_Cartesian']
# det_imag_Psi_w_analysis = loadfile['det_imag_Psi_w_analysis']
# det_real_Psi_w_analysis = loadfile['det_real_Psi_w_analysis']
# det_M_w_analysis = loadfile['det_M_w_analysis']
delta_k_perp_2 = loadfile['delta_k_perp_2'] 
delta_theta_m = loadfile['delta_theta_m'] 
loadfile.close()

loadfile = np.load('data_input' + suffix + '.npz')
data_poloidal_flux_grid = loadfile['data_poloidal_flux_grid']
data_R_coord = loadfile['data_R_coord']
data_Z_coord = loadfile['data_Z_coord']
launch_position = loadfile['launch_position']
launch_beam_width = loadfile['launch_beam_width']
launch_beam_curvature = loadfile['launch_beam_curvature']
loadfile.close()


print(tau_array[out_index])

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

plt.figure(figsize=(5,5))
plt.title('Poloidal Plane')
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels,vmin=0,vmax=1.2,cmap='inferno')
plt.clabel(CS, inline=True, fontsize=10,inline_spacing=-5,fmt= '%1.1f',use_clabeltext=True) # Labels the flux surfaces
# plt.xlim(1.0,1.8)
# plt.ylim(-0.7,0.1)
tikzplotlib.clean_figure() # Removes points that are outside the plot area
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


# ------------------

Psi_w = np.zeros([numberOfDataPoints,2,2],dtype='complex128')
Psi_w[:,0,0] = Psi_xx_output
Psi_w[:,1,1] = Psi_yy_output
Psi_w[:,1,0] = Psi_xy_output
Psi_w[:,0,1] = Psi_w[:,1,0]

Psi_imag_eigval,Psi_imag_eigvec = np.linalg.eig(np.imag(Psi_w))
Psi_real_eigval,Psi_real_eigvec = np.linalg.eig(np.real(Psi_w))

W_eigval = np.sqrt(2/Psi_imag_eigval)
curv_eigval = np.zeros_like(W_eigval)
curv_eigval[:,0] = Psi_real_eigval[:,0] / K_magnitude_array
curv_eigval[:,1] = Psi_real_eigval[:,1] / K_magnitude_array


plt.figure()
plt.subplot(1,2,1)
plt.plot(distance_along_line,Psi_imag_eigval[:,0])
plt.plot(distance_along_line,Psi_imag_eigval[:,1])
plt.subplot(1,2,2)
plt.plot(distance_along_line,Psi_real_eigval[:,0])
plt.plot(distance_along_line,Psi_real_eigval[:,1])

plt.figure()
plt.subplot(1,2,1)
plt.plot(distance_along_line,W_eigval[:,0])
plt.plot(distance_along_line,W_eigval[:,1])
plt.axvline(distance_along_line[cutoff_index],c='k')
plt.subplot(1,2,2)
plt.plot(distance_along_line,curv_eigval[:,0])
plt.plot(distance_along_line,curv_eigval[:,1])
plt.axvline(distance_along_line[cutoff_index],c='k')

# plt.figure()
# plt.subplot(1,3,1)
# plt.plot(distance_along_line,det_real_Psi_w_analysis)
# plt.axvline(distance_along_line[cutoff_index],c='k')
# plt.subplot(1,3,2)
# plt.plot(distance_along_line,det_imag_Psi_w_analysis)
# plt.axvline(distance_along_line[cutoff_index],c='k')
# plt.subplot(1,3,3)
# plt.plot(distance_along_line,abs(det_M_w_analysis))
# plt.axvline(distance_along_line[cutoff_index],c='k')

# -----------------


plt.figure()
plt.subplot(2,3,1)
plt.plot(distance_along_line,q_R_array,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('R')

plt.subplot(2,3,2)
plt.plot(distance_along_line,q_zeta_array,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('zeta')

plt.subplot(2,3,3)
plt.plot(distance_along_line,q_Z_array,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('Z')

plt.subplot(2,3,4)
plt.plot(distance_along_line,K_R_array,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('K_R')

plt.subplot(2,3,5)


plt.subplot(2,3,6)
plt.plot(distance_along_line,K_Z_array,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('K_Z')


plt.figure()
plt.subplot(2,3,1)
plt.plot(distance_along_line,dH_dR_output,'ro')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dR')

plt.subplot(2,3,2)


plt.subplot(2,3,3)
plt.plot(distance_along_line,dH_dZ_output,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dZ')

plt.subplot(2,3,4)
plt.plot(distance_along_line,dH_dKR_output,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dKR')

plt.subplot(2,3,5)
plt.plot(distance_along_line,dH_dKzeta_output,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dKzeta')

plt.subplot(2,3,6)
plt.plot(distance_along_line,dH_dKZ_output,'r')
#plt.axvline(distance_along_line1[in_index],color='k')
#plt.axvline(distance_along_line1[out_index],color='k')
plt.title('dH_dKZ')


plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,0,0],'r')
plt.subplot(3,3,2)
plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,0,1],'r')
plt.subplot(3,3,3)
plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,0,2],'r')
plt.subplot(3,3,4)
#plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,1,0],'r')
plt.subplot(3,3,5)
plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,1,1],'r')
plt.subplot(3,3,6)
plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,1,2],'r')
plt.subplot(3,3,7)
#plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,2,0],'r')
plt.subplot(3,3,8)
#plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,2,1],'r')
plt.subplot(3,3,9)
plt.plot(distance_along_line,np.imag(Psi_3D_Cartesian)[:,2,2],'r')

plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,0,0],'r')
plt.subplot(3,3,2)
plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,0,1],'r')
plt.subplot(3,3,3)
plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,0,2],'r')
plt.subplot(3,3,4)
#plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,1,0],'r')
plt.subplot(3,3,5)
plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,1,1],'r')
plt.subplot(3,3,6)
plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,1,2],'r')
plt.subplot(3,3,7)
#plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,2,0],'r')
plt.subplot(3,3,8)
#plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,2,1],'r')
plt.subplot(3,3,9)
plt.plot(distance_along_line,np.imag(Psi_3D_output)[:,2,2],'r')

plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line,gradK_grad_H_output[:,0,0],'r')
plt.subplot(3,3,2)
plt.subplot(3,3,3)
plt.plot(distance_along_line,gradK_grad_H_output[:,0,2],'r')
plt.subplot(3,3,4)
plt.plot(distance_along_line,gradK_grad_H_output[:,1,0],'r')
plt.subplot(3,3,5)
plt.subplot(3,3,6)
plt.plot(distance_along_line,gradK_grad_H_output[:,1,2],'r')
plt.subplot(3,3,7)
plt.plot(distance_along_line,gradK_grad_H_output[:,2,0],'r')
plt.subplot(3,3,8)
plt.subplot(3,3,9)
plt.title('d2H_dKZ_dZ')
plt.plot(distance_along_line,gradK_grad_H_output[:,2,2],'r')

plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line,grad_grad_H_output[:,0,0],'or')
plt.title('d2H_dR2')
plt.subplot(3,3,2)
plt.subplot(3,3,3)
plt.plot(distance_along_line,grad_grad_H_output[:,0,2],'r')
plt.title('d2H_dR_dZ')
plt.subplot(3,3,4)
plt.subplot(3,3,5)
plt.subplot(3,3,6)
plt.subplot(3,3,7)
#plt.plot(distance_along_line,grad_grad_H_output[:,2,0],'r')
plt.subplot(3,3,8)
plt.subplot(3,3,9)
plt.plot(distance_along_line,grad_grad_H_output[:,2,2],'r')
plt.title('d2H_dZ2')


plt.figure()
plt.subplot(3,3,1)
plt.plot(distance_along_line,gradK_gradK_H_output[:,0,0],'r')
plt.subplot(3,3,2)
plt.plot(distance_along_line,gradK_gradK_H_output[:,0,1],'r')
plt.subplot(3,3,3)
plt.plot(distance_along_line,gradK_gradK_H_output[:,0,2],'r')
plt.subplot(3,3,4)
plt.subplot(3,3,5)
plt.plot(distance_along_line,gradK_gradK_H_output[:,1,1],'r')
plt.subplot(3,3,6)
plt.plot(distance_along_line,gradK_gradK_H_output[:,1,2],'r')
plt.subplot(3,3,7)
plt.subplot(3,3,8)
plt.subplot(3,3,9)
plt.title('d2H_dKZ2')
plt.plot(distance_along_line,gradK_gradK_H_output[:,2,2],'r')


plt.figure(figsize=(15,5))
plt.subplot(2,3,1)
plt.plot(distance_along_line,np.real(Psi_xx_output),'k')
plt.plot(distance_along_line,np.real(M_xx_output),'r')
plt.subplot(2,3,2)
plt.plot(distance_along_line,np.real(Psi_xy_output),'k')
plt.plot(distance_along_line,np.real(M_xy_output),'r')
plt.subplot(2,3,3)
plt.plot(distance_along_line,np.real(Psi_yy_output),'k')
plt.subplot(2,3,4)
plt.plot(distance_along_line,np.imag(Psi_xx_output),'k')
plt.plot(distance_along_line,np.imag(M_xx_output),'r')
plt.subplot(2,3,5)
plt.plot(distance_along_line,np.imag(Psi_xy_output),'k')
plt.plot(distance_along_line,np.imag(M_xy_output),'r')
plt.subplot(2,3,6)
plt.plot(distance_along_line,np.imag(Psi_yy_output),'k')



plt.figure(figsize=(15,5))
plt.subplot(2,3,1)
plt.plot(distance_along_line,dB_dR_FFD_debugging,'k')
plt.title('dB_dR')
plt.subplot(2,3,2)
plt.plot(distance_along_line,dB_dZ_FFD_debugging,'k')
plt.title('dB_dZ')
plt.subplot(2,3,3)
plt.subplot(2,3,4)
plt.plot(distance_along_line,d2B_dR2_FFD_debugging,'k')
plt.title('d2B_dR2')
plt.subplot(2,3,5)
plt.plot(distance_along_line,d2B_dZ2_FFD_debugging,'k')
plt.title('d2B_dZ2')
plt.subplot(2,3,6)
plt.plot(distance_along_line,d2B_dR_dZ_FFD_debugging,'k')
plt.title('d2B_dR_dZ')

plt.figure()
plt.subplot(2,2,1)
plt.plot(distance_along_line,dpolflux_dR_FFD_debugging,'k')
plt.subplot(2,2,2)
plt.plot(distance_along_line,dpolflux_dZ_FFD_debugging,'k')
plt.subplot(2,2,3)
plt.plot(distance_along_line,d2polflux_dR2_FFD_debugging,'k')
plt.subplot(2,2,4)
plt.plot(distance_along_line,d2polflux_dZ2_FFD_debugging,'k')

plt.figure()
plt.plot(distance_along_line,poloidal_flux_debugging_3R-poloidal_flux_output,'r')

plt.figure()
plt.plot(distance_along_line,B_total_output,'r')

plt.figure()
plt.subplot(1,3,1)
plt.plot(distance_along_line,b_hat_output[:,0])
plt.subplot(1,3,2)
plt.plot(distance_along_line,b_hat_output[:,1])
plt.subplot(1,3,3)
plt.plot(distance_along_line,b_hat_output[:,2])


plt.figure()
plt.subplot(1,3,1)
plt.plot(distance_along_line,epsilon_para_output)
plt.subplot(1,3,2)
plt.plot(distance_along_line,epsilon_perp_output)
plt.subplot(1,3,3)
plt.plot(distance_along_line,epsilon_g_output)

# plt.figure()
# plt.subplot(1,3,1)
# plt.plot(distance_along_line,det_imag_Psi_w_analysis)
# plt.subplot(1,3,2)
# plt.plot(distance_along_line,abs(det_M_w_analysis))

plt.figure()
plt.subplot(1,3,1)
plt.plot(distance_along_line,delta_k_perp_2)
plt.axvline(distance_along_line[cutoff_index],c='k')
plt.subplot(1,2,2)
plt.plot(distance_along_line,delta_theta_m)
# plt.axvline(distance_along_line[cutoff_index],c='k')


# plt.figure()
# plt.plot(KZ_debugging, d2H_dKZ2_debugging)
#poloidal_flux_debugging_1R = loadfile['poloidal_flux_debugging_1R']
#poloidal_flux_debugging_2R = loadfile['poloidal_flux_debugging_2R']
#poloidal_flux_debugging_3R = loadfile['poloidal_flux_debugging_2R']
#
#g_hat_Cartesian = np.zeros_like(g_hat_output)
#g_hat_Cartesian[:,0] = g_hat_output[:,0]*np.cos(q_zeta_array ) - g_hat_output[:,1]*np.sin(q_zeta_array )
#g_hat_Cartesian[:,1] = g_hat_output[:,0]*np.sin(q_zeta_array ) + g_hat_output[:,1]*np.cos(q_zeta_array )
#g_hat_Cartesian[:,2] = g_hat_output[:,2]
#
#plt.figure()
#plt.plot(distance_along_line,Psi_imag_eigvec[:,0,0],'r')
#plt.plot(distance_along_line,Psi_imag_eigvec[:,0,1],'g')
#
#plt.figure()
#plt.plot(distance_along_line,y_hat_output[:,0],'r')
#plt.plot(distance_along_line,y_hat_output[:,1],'g')
#plt.plot(distance_along_line,y_hat_output[:,2],'b')
#
#plt.figure()
#plt.plot(distance_along_line,g_hat_Cartesian[:,0],'r')
#plt.plot(distance_along_line,g_hat_Cartesian[:,1],'g')
#plt.plot(distance_along_line,g_hat_Cartesian[:,2],'b')