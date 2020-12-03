# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from Scotty_fun import find_waist, find_distance_from_waist


loadfile = np.load('data_output.npz')
tau_array = loadfile['tau_array']
q_R_array = loadfile['q_R_array']
q_zeta_array = loadfile['q_zeta_array']
#q_X_array = loadfile['q_X_array']
#q_Y_array = loadfile['q_Y_array']
q_Z_array = loadfile['q_Z_array']
K_R_array = loadfile['K_R_array']
#K_zeta_array = loadfile['K_zeta_array']
K_Z_array = loadfile['K_Z_array']
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


loadfile = np.load('analysis_output.npz')
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
loadfile.close()


loadfile = np.load('data_input.npz')
data_poloidal_flux_grid = loadfile['data_poloidal_flux_grid']
data_R_coord = loadfile['data_R_coord']
data_Z_coord = loadfile['data_Z_coord']
launch_beam_width = loadfile['launch_beam_width']
launch_beam_curvature = loadfile['launch_beam_curvature']
loadfile.close()


#W_xx_array = np.sqrt(2/np.imag(Psi_w_xx_array))
#W_xy_array = np.sign(np.imag(Psi_w_xy_array))*np.sqrt(2/abs(np.imag(Psi_w_xy_array)))
#W_yy_array = np.sqrt(2/np.imag(Psi_w_yy_array))
#R_xx_array = K_magnitude_array/np.real(Psi_w_xx_array)
#R_xy_array = K_magnitude_array/np.real(Psi_w_xy_array)
#R_yy_array = K_magnitude_array/np.real(Psi_w_yy_array)


# Checking gradients of bhat

    #Input data from Generation_Input7
aspect_ratio = 1.5 # major_radius/minor_radius
minor_radius = 0.5 # in meters    
B_toroidal_max = 1.00 # in Tesla (?)
B_poloidal_max = 0.1 # in Tesla

major_radius = aspect_ratio * minor_radius

d_1overabsB_d_R_analytic = - (B_total_output)**(-3) * (B_poloidal_max**2*(q_R_array-major_radius)/minor_radius**2 - B_toroidal_max**2 * q_R_array**(-3) * major_radius**2) #\frac{\partial}{\partial R} \frac{1}{abs(B)}
d_1overabsB_d_z_analytic = - (B_total_output)**(-3) * (B_poloidal_max**2*q_Z_array/minor_radius**2)

d_bhat_R_d_R_analytic = B_poloidal_max * q_Z_array / minor_radius * d_1overabsB_d_R_analytic
d_bhat_T_d_R_analytic = - B_toroidal_max/B_total_output * major_radius/(q_R_array**2) + B_toroidal_max*major_radius/(q_R_array) * d_1overabsB_d_R_analytic
d_bhat_Z_d_R_analytic = 1/minor_radius * B_poloidal_max / B_total_output + B_poloidal_max*(q_R_array-major_radius)/minor_radius * d_1overabsB_d_R_analytic
d_bhat_R_d_z_analytic = B_poloidal_max/(minor_radius*B_total_output) + B_poloidal_max/(minor_radius)*q_Z_array*d_1overabsB_d_z_analytic
d_bhat_T_d_z_analytic = B_toroidal_max * major_radius/q_R_array * d_1overabsB_d_z_analytic
d_bhat_Z_d_z_analytic = B_poloidal_max * (q_R_array-major_radius) / minor_radius * d_1overabsB_d_z_analytic

numberOfDataPoints = np.size(d_bhat_Z_d_R_analytic)
grad_bhat_analytic = np.zeros([3,3,numberOfDataPoints])
grad_bhat_analytic[0,0,:] = d_bhat_R_d_R_analytic
grad_bhat_analytic[1,0,:] = d_bhat_T_d_R_analytic
grad_bhat_analytic[2,0,:] = d_bhat_Z_d_R_analytic
grad_bhat_analytic[0,2,:] = d_bhat_R_d_z_analytic
grad_bhat_analytic[1,2,:] = d_bhat_T_d_z_analytic
grad_bhat_analytic[2,2,:] = d_bhat_Z_d_z_analytic
grad_bhat_analytic[1,1,:] = B_poloidal_max *  q_Z_array / (q_R_array * minor_radius * B_total_output) #b_R / R
grad_bhat_analytic[0,1,:] = - B_toroidal_max * major_radius / (q_R_array**2 * B_total_output) #- b_zeta / R

xhat_dot_grad_bhat_dot_xhat_analytic = np.zeros(numberOfDataPoints)
xhat_dot_grad_bhat_dot_yhat_analytic = np.zeros(numberOfDataPoints)
xhat_dot_grad_bhat_dot_ghat_analytic = np.zeros(numberOfDataPoints)
yhat_dot_grad_bhat_dot_xhat_analytic = np.zeros(numberOfDataPoints)
yhat_dot_grad_bhat_dot_yhat_analytic = np.zeros(numberOfDataPoints)
yhat_dot_grad_bhat_dot_ghat_analytic = np.zeros(numberOfDataPoints)
for ii in range(0,numberOfDataPoints):
    xhat_dot_grad_bhat_dot_xhat_analytic[ii] = np.dot(x_hat_output[:,ii],np.dot(grad_bhat_analytic[:,:,ii],x_hat_output[:,ii]))
    xhat_dot_grad_bhat_dot_yhat_analytic[ii] = np.dot(x_hat_output[:,ii],np.dot(grad_bhat_analytic[:,:,ii],y_hat_output[:,ii]))
    xhat_dot_grad_bhat_dot_ghat_analytic[ii] = np.dot(x_hat_output[:,ii],np.dot(grad_bhat_analytic[:,:,ii],g_hat_output[:,ii]))
    yhat_dot_grad_bhat_dot_xhat_analytic[ii] = np.dot(y_hat_output[:,ii],np.dot(grad_bhat_analytic[:,:,ii],x_hat_output[:,ii]))
    yhat_dot_grad_bhat_dot_yhat_analytic[ii] = np.dot(y_hat_output[:,ii],np.dot(grad_bhat_analytic[:,:,ii],y_hat_output[:,ii]))
    yhat_dot_grad_bhat_dot_ghat_analytic[ii] = np.dot(y_hat_output[:,ii],np.dot(grad_bhat_analytic[:,:,ii],g_hat_output[:,ii]))


plt.figure()
plt.subplot(3,2,1)
plt.plot(tau_array,xhat_dot_grad_bhat_dot_xhat_analytic,color='k')
plt.plot(tau_array,xhat_dot_grad_bhat_dot_xhat_output,color='r')
plt.subplot(3,2,2)
plt.plot(tau_array,xhat_dot_grad_bhat_dot_yhat_analytic,color='k')
plt.plot(tau_array,xhat_dot_grad_bhat_dot_yhat_output,color='r')
plt.subplot(3,2,3)
plt.plot(tau_array,xhat_dot_grad_bhat_dot_ghat_analytic,color='k')
plt.plot(tau_array,xhat_dot_grad_bhat_dot_ghat_output,color='r')
plt.subplot(3,2,4)
plt.plot(tau_array,yhat_dot_grad_bhat_dot_xhat_analytic,color='k')
plt.plot(tau_array,yhat_dot_grad_bhat_dot_xhat_output,color='r')
plt.subplot(3,2,5)
plt.plot(tau_array,yhat_dot_grad_bhat_dot_yhat_analytic,color='k')
plt.plot(tau_array,yhat_dot_grad_bhat_dot_yhat_output,color='r')
plt.subplot(3,2,6)
plt.plot(tau_array,yhat_dot_grad_bhat_dot_ghat_analytic,color='k')
plt.plot(tau_array,yhat_dot_grad_bhat_dot_ghat_output,color='r')

plt.figure()
plt.subplot(3,3,1)
plt.plot(tau_array,grad_bhat_analytic[0,0,:],color='k')
plt.plot(tau_array,grad_bhat_output[0,0,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,2)
plt.plot(tau_array,grad_bhat_analytic[1,0,:],color='k')
plt.plot(tau_array,grad_bhat_output[1,0,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,3)
plt.plot(tau_array,grad_bhat_analytic[2,0,:],color='k')
plt.plot(tau_array,grad_bhat_output[2,0,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,4)
plt.plot(tau_array,grad_bhat_analytic[0,1,:],color='k')
plt.plot(tau_array,grad_bhat_output[0,1,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,5)
plt.plot(tau_array,grad_bhat_analytic[1,1,:],color='k')
plt.plot(tau_array,grad_bhat_output[1,1,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,6)
plt.plot(tau_array,grad_bhat_analytic[2,1,:],color='k')
plt.plot(tau_array,grad_bhat_output[2,1,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,7)
plt.plot(tau_array,grad_bhat_analytic[0,2,:],color='k')
plt.plot(tau_array,grad_bhat_output[0,2,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,8)
plt.plot(tau_array,grad_bhat_analytic[1,2,:],color='k')
plt.plot(tau_array,grad_bhat_output[1,2,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.subplot(3,3,9)
plt.plot(tau_array,grad_bhat_analytic[2,2,:],color='k')
plt.plot(tau_array,grad_bhat_output[2,2,:],color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')

# ------------


# Checking vacuum propagation. Assumes circular beam
wavenumber_K0 = K_magnitude_array[0]
point_spacing = ( (np.diff(q_X_array))**2 + (np.diff(q_Y_array))**2 + (np.diff(q_Z_array))**2 )**0.5
distance_along_line =  np.cumsum(point_spacing)
distance_along_line = np.append(0, distance_along_line)
waist = find_waist(launch_beam_width,wavenumber_K0,1/launch_beam_curvature)
distance_from_waist = find_distance_from_waist(launch_beam_width,wavenumber_K0,1/launch_beam_curvature)
Rayleigh_length = 0.5 * wavenumber_K0 * waist**2
distance_array = distance_along_line+distance_from_waist
Psi_real_array = wavenumber_K0 * distance_array / (distance_array**2 + Rayleigh_length**2)
Psi_imag_array = 2 / (waist**2 * (1 + distance_array**2 / Rayleigh_length**2) )


plt.figure()
plt.subplot(1,2,1)
plt.plot(distance_array-distance_from_waist,Psi_real_array)
plt.plot(distance_along_line,np.real(Psi_w_yy_array),'k')
plt.subplot(1,2,2)
plt.plot(distance_array-distance_from_waist,Psi_imag_array)
plt.plot(distance_along_line,np.imag(Psi_w_yy_array),'k')







# Calculations of corrections to Psi_w

## Calculating the corrections to Psi
#d_theta_d_tau = np.gradient(theta_output,tau_array)
#ray_curvature_x = g_magnitude_output*np.gradient(g_hat_Cartesian_output[0,:],tau_array)
#ray_curvature_y = g_magnitude_output*np.gradient(g_hat_Cartesian_output[1,:],tau_array)
#ray_curvature_z = g_magnitude_output*np.gradient(g_hat_Cartesian_output[2,:],tau_array)
#
## --
k1s = - 2 * K_g_array / np.cos(theta_output)

xx_k1_1 = np.sin(theta_output) / g_magnitude_output * d_theta_d_tau_output
xx_k1_2 = kappa_dot_xhat_output * np.sin(theta_output) 
xx_k1_3 = xhat_dot_grad_bhat_dot_ghat_output
xx_k1_4 = xhat_dot_grad_bhat_dot_xhat_output * np.tan(theta_output) 
xx_k2_1 = np.tan(theta_output) / g_magnitude_output * d_xhat_d_tau_dot_yhat_output
xx_k2_2 = xhat_dot_grad_bhat_dot_yhat_output / np.cos(theta_output) 
xy_k1_1 = kappa_dot_yhat_output * np.sin(theta_output) 
xy_k1_2 = np.sin(theta_output) * np.tan(theta_output) / g_magnitude_output * d_xhat_d_tau_dot_yhat_output
xy_k1_3 = yhat_dot_grad_bhat_dot_ghat_output
xy_k1_4 = yhat_dot_grad_bhat_dot_xhat_output * np.tan(theta_output) 
xy_k2_1 = yhat_dot_grad_bhat_dot_yhat_output / np.cos(theta_output) 

plt.figure()
plt.subplot(2,2,1)
plt.plot(tau_array,xx_k1_1,color='b')
plt.plot(tau_array,xx_k1_2,color='g')
plt.plot(tau_array,xx_k1_3,color='r')
plt.plot(tau_array,xx_k1_4,color='c')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.title('xx k1')
plt.subplot(2,2,2)
plt.plot(tau_array,xx_k2_1,color='b')
plt.plot(tau_array,xx_k2_2,color='g')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.title('xx k2')
plt.subplot(2,2,3)
plt.plot(tau_array,xy_k1_1,color='b')
plt.plot(tau_array,xy_k1_2,color='g')
plt.plot(tau_array,xy_k1_3,color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.title('xy k1')
plt.subplot(2,2,4)
plt.plot(tau_array,xy_k2_1,color='b')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
plt.title('xy k2')


plt.figure()
plt.subplot(2,1,1)
plt.plot(tau_array,np.real(Psi_w_xx_array),'k')
plt.plot(tau_array,k1s/2*xx_k1_1,color='b')
plt.plot(tau_array,k1s/2*xx_k1_2,color='g')
plt.plot(tau_array,k1s/2*xx_k1_3,color='r')
plt.plot(tau_array,k1s/2*xx_k1_4,color='c')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
#plt.ylim([-10**3,10**3])
plt.title('xx')
plt.subplot(2,1,2)
plt.plot(tau_array,np.real(Psi_w_xy_array),'k')
plt.plot(tau_array,k1s/2*xy_k1_1,color='b')
plt.plot(tau_array,k1s/2*xy_k1_2,color='g')
plt.plot(tau_array,k1s/2*xy_k1_3,color='r')
plt.axvline(tau_start,color='k')
plt.axvline(tau_end,color='k')
#plt.ylim([-10**3,10**3])
plt.title('xy')

plt.figure()
plt.plot(tau_array,b_hat_Cartesian_output[0,:],color='b')
plt.plot(tau_array,b_hat_Cartesian_output[1,:],color='g')
plt.plot(tau_array,b_hat_Cartesian_output[2,:],color='r')


# -------------

# Calculations of spherical vs conventional
M_w_xx_nu = Psi_w_xx_array[tau_nu_index] + k1s[tau_nu_index]/2*(xx_k1_1[tau_nu_index]+xx_k1_2[tau_nu_index]+xx_k1_3[tau_nu_index]+xx_k1_4[tau_nu_index])
M_w_xy_nu = Psi_w_xy_array[tau_nu_index] + k1s[tau_nu_index]/2*(xy_k1_1[tau_nu_index]+xy_k1_2[tau_nu_index]+xy_k1_3[tau_nu_index]) 
M_w_yy_nu = Psi_w_yy_array[tau_nu_index]
M_w_det_nu = M_w_xx_nu*M_w_yy_nu - M_w_xy_nu**2  
M_w_inverse_xx_nu = M_w_yy_nu/M_w_det_nu
      
c_full = (1 + 
        K_magnitude_array[tau_nu_index]**2 * (d_theta_m_d_tau_array[tau_nu_index])**2 * M_w_inverse_xx_nu / (
                g_magnitude_output[tau_nu_index] * d_K_g_d_tau_array[tau_nu_index]
                )
        )**(-1)
c_expanded = (1 -
        K_magnitude_array[tau_nu_index]**2 * (d_theta_m_d_tau_array[tau_nu_index])**2 * M_w_inverse_xx_nu / (
                g_magnitude_output[tau_nu_index] * d_K_g_d_tau_array[tau_nu_index]
                )
        )
c_diff = (np.imag(c_full)-np.imag(c_expanded)) / (np.imag(c_full)+np.imag(c_expanded))
print('c_diff = ', c_diff)
# -------------


Psi_w_real_array = np.array(np.real([
                    [Psi_w_xx_array,Psi_w_xy_array],
                    [Psi_w_xy_array,Psi_w_yy_array]
                    ]))
Psi_w_imag_array = np.array(np.imag([
                    [Psi_w_xx_array,Psi_w_xy_array],
                    [Psi_w_xy_array,Psi_w_yy_array]
                    ]))

point_spacing = ( (np.diff(q_X_array))**2 + (np.diff(q_Y_array))**2 + (np.diff(q_Z_array))**2 )**0.5
distance_along_line =  np.cumsum(point_spacing)
distance_along_line = np.append(0,distance_along_line)

numberOfPlotPoints=50
numberOfDataPoints=len(tau_array)
width_points_array = np.zeros([2,numberOfDataPoints])
curvature_points_array = np.zeros([2,numberOfDataPoints]) # 1/rad_of_curvature
Psi_w_real_eigvecs_array = np.zeros([2,2,numberOfDataPoints])
Psi_w_imag_eigvecs_array = np.zeros([2,2,numberOfDataPoints])
grad_bhat_array = np.zeros([3,3,numberOfDataPoints])
ellipse_points_array = np.zeros([2,numberOfPlotPoints,numberOfDataPoints])
surface_points_array = np.zeros([3,numberOfPlotPoints*numberOfDataPoints])
cos_linspace_array = np.cos(np.linspace(0,2*np.pi,numberOfPlotPoints))
sin_linspace_array = np.sin(np.linspace(0,2*np.pi,numberOfPlotPoints))
for index in range(0,numberOfDataPoints): 
    Psi_w_real = Psi_w_real_array[:,:,index]
    Psi_w_imag = Psi_w_imag_array[:,:,index]
 
    Psi_w_real_eigvals, Psi_w_real_eigvecs = np.linalg.eig(Psi_w_real)
    Psi_w_imag_eigvals, Psi_w_imag_eigvecs = np.linalg.eig(Psi_w_imag)
    #The normalized (unit “length”) eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    
    width_points_array[:,index] = np.sqrt(2/Psi_w_imag_eigvals)
    curvature_points_array[:,index] = Psi_w_real_eigvals/K_magnitude_array[index]
    Psi_w_imag_eigvecs_array[:,:,index] = Psi_w_imag_eigvecs
    Psi_w_real_eigvecs_array[:,:,index] = Psi_w_real_eigvecs
    

#isosurface_array = np.ones(np.shape([surface_points_array]))




plt.figure()
plt.subplot(1,3,1)
plt.plot(tau_array,np.imag(Psi_w_xx_array))
plt.subplot(1,3,2)
plt.plot(tau_array,np.imag(Psi_w_xy_array))
plt.subplot(1,3,3)
plt.plot(tau_array,np.imag(Psi_w_yy_array))

plt.figure()
plt.subplot(1,3,1)
plt.plot(tau_array,np.real(Psi_w_xx_array))
plt.subplot(1,3,2)
plt.plot(tau_array,np.real(Psi_w_xy_array))
plt.subplot(1,3,3)
plt.plot(tau_array,np.real(Psi_w_yy_array))

plt.figure()
plt.plot(ellipse_points_array[0,:,0],ellipse_points_array[1,:,0])
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

plt.figure()
plt.subplot(3,3,1)
plt.title('Poloidal Plane')
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_X_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.plot(q_R_array,q_Z_array,'k')
plt.xlim(data_X_coord[0],data_X_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])
plt.xlabel('R / m')
plt.ylabel('Z / m')
plt.subplot(3,3,2)
plt.plot(q_R_array*np.cos(q_zeta_array),q_R_array*np.sin(q_zeta_array),'k')
plt.subplot(3,3,3)
plt.subplot(3,3,4)
plt.plot(distance_along_line,curvature_points_array[0,:])
plt.plot(distance_along_line,curvature_points_array[1,:])
plt.ylabel('curvature / m-1')
plt.subplot(3,3,5)
plt.plot(distance_along_line,width_points_array[0,:])
plt.plot(distance_along_line,width_points_array[1,:])
plt.ylabel('width / m')
plt.subplot(3,3,6)
plt.subplot(3,3,7)
plt.subplot(3,3,8)
plt.subplot(3,3,9)

plt.figure()
plt.subplot(1,3,1)
plt.plot(q_R_array,B_total_output*b_hat_Cartesian_output[1,:])
plt.subplot(1,3,2)
plt.plot(np.sqrt((q_R_array-1.5)**2 + q_Z_array**2),np.sqrt(B_total_output**2*b_hat_Cartesian_output[0,:]**2 + B_total_output**2*b_hat_Cartesian_output[2,:]**2))















