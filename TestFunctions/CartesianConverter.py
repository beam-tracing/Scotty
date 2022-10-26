# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:18:19 2020

@author: VH Chen

Some Cartesian conversion stuff copied over from the old Scotty_beam_me_up.py
I've put them here so that I can use this in the future
Probably load the cylindrical data, run this stuff, and output some Cartesians
Useful for benchmarking with Torbeam
"""



import matplotlib.pyplot as plt
import numpy as np
from scotty.fun import find_q_lab_Cartesian, find_q_lab, find_K_lab_Cartesian, find_K_lab

loadfile = np.load('data_output.npz')
tau_array = loadfile['tau_array']
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
q_Z_array = loadfile['q_Z_array']
K_R_array = loadfile['K_R_array']
K_Z_array = loadfile['K_Z_array']
q_zeta_array = loadfile['q_zeta_array']
K_zeta_initial = loadfile['K_zeta_initial']
b_hat_output = loadfile['b_hat_output']
dH_dKR_output = loadfile['dH_dKR_output']
dH_dKzeta_output = loadfile['dH_dKzeta_output']
dH_dKZ_output = loadfile['dH_dKZ_output']
dH_dR_output = loadfile['dH_dR_output']
dH_dZ_output = loadfile['dH_dZ_output']
Psi_3D_output = loadfile['Psi_3D_output']
#distance_from_launch_to_entry = loadfile['distance_from_launch_to_entry']
loadfile.close()

loadfile = np.load('data_input.npz')
launch_position = loadfile['launch_position']
loadfile.close()


numberOfDataPoints = len(q_R_array)

# Convert r, K, Psi_3D from cylindrical to Cartesian coordinates
launch_position_Cartesian=np.zeros(3) # Note that we have Y=0, zeta=0 at launch, by definition
launch_position_Cartesian[0] = launch_position[0]
launch_position_Cartesian[1] = launch_position[1]
launch_position_Cartesian[2] = launch_position[2]
#q_X_array = q_R_array *np.cos(q_zeta_array )
#q_Y_array = q_R_array *np.sin(q_zeta_array )
#K_X_array = K_R_array*np.cos(q_zeta_array ) - K_zeta_initial*np.sin(q_zeta_array ) / q_R_array
#K_Y_array = K_R_array*np.sin(q_zeta_array ) + K_zeta_initial*np.cos(q_zeta_array ) / q_R_array
q_X_array = np.zeros(numberOfDataPoints)
q_Y_array = np.zeros(numberOfDataPoints)
K_X_array = np.zeros(numberOfDataPoints)
K_Y_array = np.zeros(numberOfDataPoints)

b_hat_Cartesian_output = np.zeros(np.shape(b_hat_output))
b_hat_Cartesian_output[:,0] = b_hat_output[:,0]*np.cos(q_zeta_array ) - b_hat_output[:,1]*np.sin(q_zeta_array )
b_hat_Cartesian_output[:,1] = b_hat_output[:,0]*np.sin(q_zeta_array ) + b_hat_output[:,1]*np.cos(q_zeta_array )
b_hat_Cartesian_output[:,2] = b_hat_output[:,2]

temp_matrix_for_Psi = np.zeros([3,3],dtype='complex128')
temp_matrix_for_grad_grad_H = np.zeros([3,3])
temp_matrix_for_gradK_grad_H = np.zeros([3,3])
temp_matrix_for_gradK_gradK_H  = np.zeros([3,3])

Psi_3D_output_Cartesian = np.zeros(np.shape(Psi_3D_output),dtype='complex128')
grad_grad_H_output_Cartesian = np.zeros(np.shape(Psi_3D_output))
gradK_grad_H_output_Cartesian = np.zeros(np.shape(Psi_3D_output))
gradK_gradK_H_output_Cartesian = np.zeros(np.shape(Psi_3D_output))

sin_mismatch_angle_array=np.zeros(numberOfDataPoints)

for step in range(0,numberOfDataPoints):
    q_lab = np.squeeze(np.array([q_R_array[step],q_zeta_array[step],q_Z_array[step]]))
    K_lab = np.squeeze(np.array([K_R_array[step],K_zeta_initial,K_Z_array[step]]))

    [q_X_array[step],q_Y_array[step],_] = find_q_lab_Cartesian(q_lab)
    [K_X_array[step],K_Y_array[step],_] = find_K_lab_Cartesian(K_lab,q_lab)
    
    temp_matrix_for_Psi[0][0] = Psi_3D_output[step][0][0]
    temp_matrix_for_Psi[0][1] = Psi_3D_output[step][0][1]/q_R_array [step] - K_zeta_initial/q_R_array[step]**2
    temp_matrix_for_Psi[0][2] = Psi_3D_output[step][0][2]
    temp_matrix_for_Psi[1][1] = Psi_3D_output[step][1][1]/q_R_array [step]**2 + K_R_array[step]/q_R_array[step]
    temp_matrix_for_Psi[1][2] = Psi_3D_output[step][1][2]/q_R_array [step]
    temp_matrix_for_Psi[2][2] = Psi_3D_output[step][2][2]
    temp_matrix_for_Psi[1][0] = temp_matrix_for_Psi[0][1]
    temp_matrix_for_Psi[2][0] = temp_matrix_for_Psi[0][2]
    temp_matrix_for_Psi[2][1] = temp_matrix_for_Psi[1][2]

    # Second derivatives of H, when converted to Cartesians, don't agree with Torbeam
    # Psi, however, is fine
    #    temp_matrix_for_grad_grad_H[0][0] = grad_grad_H_output[0][0][step] #19th Feb 2019 notes for more details
    #    temp_matrix_for_grad_grad_H[0][1] = 0
    #    temp_matrix_for_grad_grad_H[0][2] = grad_grad_H_output[0][2][step]
    #    temp_matrix_for_grad_grad_H[1][1] = 0 + dH_dR_output[step]/q_R_array [step]
    #    temp_matrix_for_grad_grad_H[1][2] = 0
    #    temp_matrix_for_grad_grad_H[2][2] = grad_grad_H_output[2][2][step]
    #    temp_matrix_for_grad_grad_H[1][0] = temp_matrix_for_grad_grad_H[0][1]
    #    temp_matrix_for_grad_grad_H[2][0] = temp_matrix_for_grad_grad_H[0][2]
    #    temp_matrix_for_grad_grad_H[2][1] = temp_matrix_for_grad_grad_H[1][2]
    #
    #    temp_matrix_for_gradK_grad_H[0][0] = gradK_grad_H_output[0][0][step]
    #    temp_matrix_for_gradK_grad_H[0][1] = gradK_grad_H_output[0][1][step]/q_R_array [step] - 0
    #    temp_matrix_for_gradK_grad_H[0][2] = gradK_grad_H_output[0][2][step]
    #    temp_matrix_for_gradK_grad_H[1][1] = gradK_grad_H_output[1][1][step]/q_R_array [step]**2 + dH_dKR_output[step]/q_R_array [step]
    #    temp_matrix_for_gradK_grad_H[1][2] = gradK_grad_H_output[1][2][step]/q_R_array [step]
    #    temp_matrix_for_gradK_grad_H[2][2] = gradK_grad_H_output[2][2][step]
    #    temp_matrix_for_gradK_grad_H[1][0] = gradK_grad_H_output[1][0][step]/q_R_array [step] - dH_dKzeta_output[step]/q_R_array [step]**2
    #    temp_matrix_for_gradK_grad_H[2][0] = gradK_grad_H_output[2][0][step]
    #    temp_matrix_for_gradK_grad_H[2][1] = gradK_grad_H_output[2][1][step]/q_R_array [step]
    #
    #    temp_matrix_for_gradK_gradK_H[0][0] = gradK_gradK_H_output[0][0][step]
    #    temp_matrix_for_gradK_gradK_H[0][1] = gradK_gradK_H_output[0][1][step]/q_R_array [step] - dH_dKzeta_output[step]/q_R_array [step]**2
    #    temp_matrix_for_gradK_gradK_H[0][2] = gradK_gradK_H_output[0][2][step]
    #    temp_matrix_for_gradK_gradK_H[1][1] = gradK_gradK_H_output[1][1][step]/q_R_array [step]**2 + dH_dKR_output[step]/q_R_array [step]
    #    temp_matrix_for_gradK_gradK_H[1][2] = gradK_gradK_H_output[1][2][step]/q_R_array [step]
    #    temp_matrix_for_gradK_gradK_H[2][2] = gradK_gradK_H_output[2][2][step]
    #    temp_matrix_for_gradK_gradK_H[1][0] = temp_matrix_for_gradK_gradK_H[0][1]
    #    temp_matrix_for_gradK_gradK_H[2][0] = temp_matrix_for_gradK_gradK_H[0][2]
    #    temp_matrix_for_gradK_gradK_H[2][1] = temp_matrix_for_gradK_gradK_H[1][2]

    rotation_matrix_xi = np.array( [
        [ np.cos(q_zeta_array[step]), -np.sin(q_zeta_array[step]), 0 ],
        [ np.sin(q_zeta_array[step]), np.cos(q_zeta_array[step]), 0 ],
        [ 0,0,1 ]
        ] )
    rotation_matrix_xi_inverse = np.transpose(rotation_matrix_xi)

    Psi_3D_output_Cartesian[step,:,:] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_Psi),rotation_matrix_xi_inverse)
    #    grad_grad_H_output_Cartesian[:,:,step] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_grad_grad_H),rotation_matrix_xi_inverse)
    #    gradK_grad_H_output_Cartesian[:,:,step] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_gradK_grad_H),rotation_matrix_xi_inverse)
    #    gradK_gradK_H_output_Cartesian[:,:,step] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_gradK_gradK_H),rotation_matrix_xi_inverse)

#        g_hat_Cartesian = g_hat_Cartesian_output[:,ii]
    b_hat = b_hat_Cartesian_output[step,:]
    K_vec = np.asarray([K_X_array[step],K_Y_array[step],K_Z_array[step]])
#        Psi_3D_lab = Psi_3D_output_Cartesian[:,:,ii]
#        # --
#        y_hat_Cartesian = np.cross(b_hat,g_hat_Cartesian) / (np.linalg.norm(np.cross(b_hat,g_hat_Cartesian)))
#        x_hat_Cartesian = np.cross(g_hat_Cartesian,y_hat_Cartesian) / (np.linalg.norm(np.cross(g_hat_Cartesian,y_hat_Cartesian) ))
#
#        y_hat_Cartesian_output[:,ii] = y_hat_Cartesian
#        x_hat_Cartesian_output[:,ii] = x_hat_Cartesian
#
#        K_x_array[ii] = np.dot(K_vec,x_hat_Cartesian)
#        K_y_array[ii] = np.dot(K_vec,y_hat_Cartesian)
#        K_g_array[ii] = np.dot(K_vec,g_hat_Cartesian)
#
    sin_mismatch_angle_array[step] = -np.dot(K_vec,b_hat) / (np.linalg.norm(K_vec)*np.linalg.norm(b_hat)) # negative sign because of how I've defined angles
#
#        Psi_w_yy_array[ii] = np.dot(y_hat_Cartesian,np.dot(Psi_3D_lab,y_hat_Cartesian))
#        Psi_w_xy_array[ii] = np.dot(y_hat_Cartesian,np.dot(Psi_3D_lab,x_hat_Cartesian))
#        Psi_w_xx_array[ii] = np.dot(x_hat_Cartesian,np.dot(Psi_3D_lab,x_hat_Cartesian))
#
#        Psi_w_det = Psi_w_yy_array[ii] * Psi_w_xx_array[ii] - Psi_w_xy_array[ii]**2
#
#        Psi_w_inverse_yy_array[ii] = Psi_w_xx_array[ii]/Psi_w_det
#        Psi_w_inverse_xy_array[ii] = -Psi_w_xy_array[ii]/Psi_w_det
#        Psi_w_inverse_xx_array[ii] = Psi_w_yy_array[ii]/Psi_w_det

    # ------------------


# Convert the gradients to Cartesian coordinates
dH_dKX_output = dH_dKR_output * np.cos(q_zeta_array ) - dH_dKzeta_output * q_R_array  * np.sin(q_zeta_array )
dH_dKY_output = dH_dKR_output * np.sin(q_zeta_array ) + dH_dKzeta_output * q_R_array  * np.cos(q_zeta_array )
dH_dKZ_output = dH_dKZ_output

dH_dX_output = dH_dR_output * np.cos(q_zeta_array )
dH_dY_output = dH_dR_output * np.sin(q_zeta_array )
dH_dZ_output = dH_dZ_output

#    dbhat_dX_output = dbhat_dR_output * np.cos(q_zeta_array )
#    dbhat_dY_output = dbhat_dR_output * np.sin(q_zeta_array )
#    dbhat_dZ_output = dbhat_dZ_output

    #d2H_dKx_dKx_output = gradK_gradK_H_output_Cartesian[0][0]
    #d2H_dKx_dKy_output = gradK_gradK_H_output_Cartesian[0][1]
    #d2H_dKx_dKZ_output = gradK_gradK_H_output_Cartesian[0][2]
    #d2H_dKy_dKy_output = gradK_gradK_H_output_Cartesian[1][1]
    #d2H_dKy_dKZ_output = gradK_gradK_H_output_Cartesian[1][2]
    #d2H_dKZ_dKZ_output = gradK_gradK_H_output_Cartesian[2][2]
    #
    #d2H_dKx_drx_output = gradK_grad_H_output_Cartesian[0][0]
    #d2H_dKx_dry_output = gradK_grad_H_output_Cartesian[0][1]
    #d2H_dKx_drz_output = gradK_grad_H_output_Cartesian[0][2]
    #d2H_dKy_drx_output = gradK_grad_H_output_Cartesian[1][0]
    #d2H_dKy_dry_output = gradK_grad_H_output_Cartesian[1][1]
    #d2H_dKy_drz_output = gradK_grad_H_output_Cartesian[1][2]
    #d2H_dKZ_drx_output = gradK_grad_H_output_Cartesian[2][0]
    #d2H_dKZ_dry_output = gradK_grad_H_output_Cartesian[2][1]
    #d2H_dKZ_drz_output = gradK_grad_H_output_Cartesian[2][2]
    #
    #d2H_drx_drx_output = grad_grad_H_output_Cartesian[0][0]
    #d2H_drx_dry_output = grad_grad_H_output_Cartesian[0][1]
    #d2H_drx_drz_output = grad_grad_H_output_Cartesian[0][2]
    #d2H_dry_dry_output = grad_grad_H_output_Cartesian[1][1]
    #d2H_dry_drz_output = grad_grad_H_output_Cartesian[1][2]
    #d2H_drz_drz_output = grad_grad_H_output_Cartesian[2][2]
    ## ------------------









loadfile = np.load('analysis_output.npz')
theta_m_output = loadfile['theta_m_output']
loadfile.close()

plt.figure()
plt.plot(np.sin(theta_m_output),'k')
plt.plot(sin_mismatch_angle_array,'r')





np.savez('data_Cartesian', 
         dH_dKX_output=dH_dKX_output,dH_dKY_output=dH_dKY_output,dH_dKZ_output=dH_dKZ_output,
         dH_dX_output=dH_dX_output,dH_dY_output=dH_dY_output,dH_dZ_output=dH_dZ_output,
         q_X_array=q_X_array,q_Y_array=q_Y_array,q_Z_array=q_Z_array,
         K_X_array=K_X_array,K_Y_array=K_Y_array,K_Z_array=K_Z_array,
         Psi_3D_output_Cartesian=Psi_3D_output_Cartesian
         )


#
#
#    ####################################################################
#    # Some plots
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.title('Poloidal Plane')
#    plt.xlabel('x / m') # x-direction
#    plt.ylabel('z / m')
#
#    contour_levels = np.linspace(0,1,11)
#    CS = plt.contour(data_X_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
#    plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
#    plt.plot(q_X_array ,q_Z_array , '--.k') # Central (reference) ray
#    #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
#    #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
#    plt.xlim(data_X_coord[0],data_X_coord[-1])
#    plt.ylim(data_Z_coord[0],data_Z_coord[-1])
#    plt.subplot(1,2,2)
#    plt.title('Toroidal Plane')
#    plt.plot(q_X_array ,q_Y_array, '--.k') # Central (reference) ray
#    plt.xlabel('x / m') # x-direction
#    plt.ylabel('y / m')
#    plt.savefig('Ray2_' + output_filename_suffix)
##    plt.close()
##    plt.figure()
##    plt.subplot(1,2,1)
##    plt.plot(tau_array,q_X_array , '--.g') # Central (reference) ray
##    plt.subplot(1,2,2)
##    plt.plot(tau_array,q_Y_array, '--.b') # Central (reference) ray
##
##    plt.figure()
##    plt.plot(tau_array,poloidal_flux_output, '--.k') # Central (reference) ray
##    plt.figure()
##    plt.plot(tau_array,normalised_plasma_freq_output, '--.k') # Central (reference) ray
#
#    ####################################################################
#    
#    
#    
#    
#
#    # Mismatch
#    g_magnitude_output = (dH_dKX_output**2 + dH_dKY_output**2 + dH_dKZ_output**2)**0.5
#    g_hat_X_output = dH_dKX_output / g_magnitude_output
#    g_hat_Y_output = dH_dKY_output / g_magnitude_output
#    g_hat_Z_output = dH_dKZ_output / g_magnitude_output
#    g_hat_Cartesian_output = np.asarray([g_hat_X_output,g_hat_Y_output,g_hat_Z_output])
#
#    sin_theta_output = -( g_hat_Cartesian_output[0,:]*b_hat_Cartesian_output[0,:]
#                        + g_hat_Cartesian_output[1,:]*b_hat_Cartesian_output[1,:]
#                        + g_hat_Cartesian_output[2,:]*b_hat_Cartesian_output[2,:]) #negative sign because of how I've defined the angles
#
#
#    cos_theta_output = np.sqrt(1 - sin_theta_output**2) #This works since theta_m is between -pi/2 to pi/2
#    theta_output = np.arcsin(sin_theta_output)
#
#    # To calculate corrections to Psi
#    d_theta_d_tau_output = np.gradient(theta_output,tau_array)
#
#    d_xhat_d_tau_output[0,:] = np.gradient(x_hat_output[0,:],tau_array)
#    d_xhat_d_tau_output[1,:] = np.gradient(x_hat_output[1,:],tau_array)
#    d_xhat_d_tau_output[2,:] = np.gradient(x_hat_output[2,:],tau_array)
#    d_xhat_d_tau_dot_yhat_output = ( d_xhat_d_tau_output[0,:]*y_hat_output[0,:]
#                                   + d_xhat_d_tau_output[1,:]*y_hat_output[1,:]
#                                   + d_xhat_d_tau_output[2,:]*y_hat_output[2,:] ) # Can't get dot product to work properly
#
#    ray_curvature_kappa_output[0,:] = (1/g_magnitude_output) * np.gradient(g_hat_output[0,:],tau_array)
#    ray_curvature_kappa_output[1,:] = (1/g_magnitude_output) * np.gradient(g_hat_output[1,:],tau_array)
#    ray_curvature_kappa_output[2,:] = (1/g_magnitude_output) * np.gradient(g_hat_output[2,:],tau_array)
#    kappa_dot_xhat_output = ( ray_curvature_kappa_output[0,:]*x_hat_output[0,:]
#                            + ray_curvature_kappa_output[1,:]*x_hat_output[1,:]
#                            + ray_curvature_kappa_output[2,:]*x_hat_output[2,:] ) # Can't get dot product to work properly
#    kappa_dot_yhat_output = ( ray_curvature_kappa_output[0,:]*y_hat_output[0,:]
#                            + ray_curvature_kappa_output[1,:]*y_hat_output[1,:]
#                            + ray_curvature_kappa_output[2,:]*y_hat_output[2,:] ) # Can't get dot product to work properly
#    K_magnitude_array = ( K_X_array**2 + K_Y_array**2 + K_Z_array**2 )**0.5
#    K_magnitude_min = min(K_magnitude_array)
#
#    K_dot_g = dH_dKX_output*K_X_array + dH_dKY_output*K_Y_array + dH_dKZ_output*K_Z_array
#    eikonal_S = np.cumsum(K_dot_g*tau_step) # TODO: Change to proper integration methods
#    integral_g_cos_theta_dtau = np.cumsum(g_magnitude_output*cos_theta_output*tau_step)
#
#    numberOfkperp1 = 100
#    k_perp_1_array = np.linspace(-3.0*K_magnitude_min,2.0*K_magnitude_min,numberOfkperp1)
#    colours_r = np.linspace(0,1,numberOfkperp1)
#    colours_g = np.zeros(numberOfkperp1)
#    colours_b = np.linspace(1,0,numberOfkperp1)
#
#
#    poloidal_flux_search = 0.95
#    search_condition = 0
#    tau_search_end_index = -1     # So the code does not fail if the beam does not leave the plasma
#    for ii in range(0,len(poloidal_flux_output)):
#        if search_condition == 0 and poloidal_flux_output[ii] <= poloidal_flux_search:
#            tau_search_start_index = ii
#            search_condition = 1
#        elif search_condition == 1 and poloidal_flux_output[ii] >= poloidal_flux_search:
#            tau_search_end_index = ii
#            search_condition = 2
#
#    phase_array = np.zeros([len(tau_array[tau_search_start_index:tau_search_end_index]),numberOfkperp1])
#    d_phase_d_tau_array = np.zeros([len(tau_array[tau_search_start_index:tau_search_end_index]),numberOfkperp1])
#    d2_phase_d2_tau_array = np.zeros([len(tau_array[tau_search_start_index:tau_search_end_index]),numberOfkperp1])
#
#    d_phase_d_tau_min_array = np.zeros(numberOfkperp1)
#    tau_min_array = np.zeros(numberOfkperp1)
#    tau_0_estimate_array = np.zeros(numberOfkperp1)
#
#    plt.figure()
#    ax1 = plt.subplot(1,3,1)
#    plt.title('phase')
#    ax2 = plt.subplot(1,3,2)
#    plt.title('d phase / d tau')
#    ax3 = plt.subplot(1,3,3)
#    plt.title('d2 phase / d tau2')
#
#    for ii in range(0,numberOfkperp1):
#        phase_array[:,ii] = 2*eikonal_S[tau_search_start_index:tau_search_end_index] + k_perp_1_array[ii]*integral_g_cos_theta_dtau[tau_search_start_index:tau_search_end_index]
#        d_phase_d_tau_array[:,ii] = np.gradient(phase_array[:,ii],tau_array[tau_search_start_index:tau_search_end_index])
#        d2_phase_d2_tau_array[:,ii] = np.gradient(d_phase_d_tau_array[:,ii],tau_array[tau_search_start_index:tau_search_end_index])
#
#        find_function = interpolate.interp1d(d2_phase_d2_tau_array[:,ii],
#                                             tau_array[tau_search_start_index:tau_search_end_index],
#                                             kind='cubic', axis=-1, copy=True, bounds_error=True,
#                                             fill_value=0, assume_sorted=False)
#        tau_0_estimate_array[ii] = find_function(0)
#
#        d_phase_d_tau_min_array[ii] = min( d_phase_d_tau_array[:,ii] )
#        tau_min_array[ii] = tau_array[ tau_search_start_index + np.argmin(d_phase_d_tau_array[:,ii]) ]
#
#        ax1.plot(tau_array[tau_search_start_index:tau_search_end_index],phase_array[:,ii],color=[colours_r[ii],colours_g[ii],colours_b[ii]])
#        ax2.plot(tau_array[tau_search_start_index:tau_search_end_index],d_phase_d_tau_array[:,ii],color=[colours_r[ii],colours_g[ii],colours_b[ii]])
#        ax3.plot(tau_array[tau_search_start_index:tau_search_end_index],d2_phase_d2_tau_array[:,ii],color=[colours_r[ii],colours_g[ii],colours_b[ii]])
#
#    find_tau_0 = interpolate.interp1d(d_phase_d_tau_min_array,
#                                        tau_min_array,
#                                        kind='cubic', axis=-1, copy=True, bounds_error=True,
#                                        fill_value=0, assume_sorted=False)
#    tau_0 = find_tau_0(0)
#    tau_0_index = find_nearest(tau_array,tau_0)
#    tau_start = tau_array[tau_search_start_index]
#    tau_end = tau_array[tau_search_end_index]
#
#    ax3.plot([tau_array[tau_search_start_index],tau_array[tau_search_end_index]],[0,0],'k')
#    ax2.plot(tau_min_array,d_phase_d_tau_min_array, 'o', markersize=5, color="black")
#    tau_turning = tau_array[ np.argmin(K_magnitude_array) ]
#    ax2.axvline(tau_0, ymin=0, ymax=1)
#    ax3.axvline(tau_0, ymin=0, ymax=1)
#    plt.savefig('Phase_' + output_filename_suffix)
#    plt.close()
#
#    plt.figure()
#    plt.subplot(2,2,1)
#    plt.title('theta')
#    plt.plot(tau_array,theta_output)
#    plt.subplot(2,2,2)
#    plt.title('g_magnitude')
#    plt.plot(tau_array,g_magnitude_output)
#    plt.subplot(2,2,3)
#    plt.title('K_dot_g')
#    plt.plot(tau_array,K_dot_g)
#    plt.subplot(2,2,4)
#    plt.title('K_magnitude')
#    plt.plot(tau_array,K_magnitude_array)
#    plt.savefig('Params_' + output_filename_suffix)
#    plt.close()
#
##    plt.figure()
##    plt.title('tau_0 estimate')
##    plt.plot(k_perp_1_array,tau_0_estimate_array,'k')
##    plt.axhline(tau_0)
#
#    find_k_0 = interpolate.interp1d(d_phase_d_tau_min_array,
#                                        k_perp_1_array,
#                                        kind='cubic', axis=-1, copy=True, bounds_error=True,
#                                        fill_value=0, assume_sorted=False)
#    k_perp_1_0 = find_k_0(0)
#
#
#    numberOfDataPoints = len(tau_array)
#    Psi_w_yy_array = np.zeros(numberOfDataPoints,dtype=complex)
#    Psi_w_xy_array = np.zeros(numberOfDataPoints,dtype=complex)
#    Psi_w_xx_array = np.zeros(numberOfDataPoints,dtype=complex)
#    Psi_w_inverse_yy_array = np.zeros(numberOfDataPoints,dtype=complex)
#    Psi_w_inverse_xy_array = np.zeros(numberOfDataPoints,dtype=complex)
#    Psi_w_inverse_xx_array = np.zeros(numberOfDataPoints,dtype=complex)
#    K_x_array = np.zeros(numberOfDataPoints)
#    K_y_array = np.zeros(numberOfDataPoints)
#    K_g_array = np.zeros(numberOfDataPoints)
#    sin_mismatch_angle_array = np.zeros(numberOfDataPoints)
#
#    # Psi_3D_output_Cartesian
#    # K_X_array
#    # I'm sure this can be vectorised, but I'm leaving it this way for now
#    for ii in range(numberOfDataPoints):
#        g_hat_Cartesian = g_hat_Cartesian_output[:,ii]
#        b_hat = b_hat_Cartesian_output[:,ii]
#        K_vec = np.asarray([K_X_array[ii],K_Y_array[ii],K_Z_array[ii]])
#        Psi_3D_lab = Psi_3D_output_Cartesian[:,:,ii]
#        # --
#        y_hat_Cartesian = np.cross(b_hat,g_hat_Cartesian) / (np.linalg.norm(np.cross(b_hat,g_hat_Cartesian)))
#        x_hat_Cartesian = np.cross(g_hat_Cartesian,y_hat_Cartesian) / (np.linalg.norm(np.cross(g_hat_Cartesian,y_hat_Cartesian) ))
#
#        y_hat_Cartesian_output[:,ii] = y_hat_Cartesian
#        x_hat_Cartesian_output[:,ii] = x_hat_Cartesian
#
#        K_x_array[ii] = np.dot(K_vec,x_hat_Cartesian)
#        K_y_array[ii] = np.dot(K_vec,y_hat_Cartesian)
#        K_g_array[ii] = np.dot(K_vec,g_hat_Cartesian)
#
#        sin_mismatch_angle_array[ii] = -np.dot(K_vec,b_hat) / (np.linalg.norm(K_vec)*np.linalg.norm(b_hat)) # negative sign because of how I've defined angles
#
#        Psi_w_yy_array[ii] = np.dot(y_hat_Cartesian,np.dot(Psi_3D_lab,y_hat_Cartesian))
#        Psi_w_xy_array[ii] = np.dot(y_hat_Cartesian,np.dot(Psi_3D_lab,x_hat_Cartesian))
#        Psi_w_xx_array[ii] = np.dot(x_hat_Cartesian,np.dot(Psi_3D_lab,x_hat_Cartesian))
#
#        Psi_w_det = Psi_w_yy_array[ii] * Psi_w_xx_array[ii] - Psi_w_xy_array[ii]**2
#
#        Psi_w_inverse_yy_array[ii] = Psi_w_xx_array[ii]/Psi_w_det
#        Psi_w_inverse_xy_array[ii] = -Psi_w_xy_array[ii]/Psi_w_det
#        Psi_w_inverse_xx_array[ii] = Psi_w_yy_array[ii]/Psi_w_det
#    cos_mismatch_angle_array = np.sqrt(1 - sin_mismatch_angle_array**2)
#
#
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.title('K_x')
#    plt.plot(tau_array,K_x_array/K_magnitude_array)
#    plt.subplot(1,2,2)
#    plt.title('K_y')
#    plt.plot(tau_array,K_y_array/K_magnitude_array)
#    plt.savefig('K_' + output_filename_suffix)
#    plt.close()
#
#    plt.figure()
#    plt.subplot(2,2,1)
#    plt.title('K')
#    plt.plot(poloidal_flux_output,K_magnitude_array)
#    plt.subplot(2,2,2)
#    plt.title('K_g')
#    plt.plot(poloidal_flux_output,K_g_array)
#    plt.subplot(2,2,3)
#    plt.title('K_x')
#    plt.plot(poloidal_flux_output,K_x_array)
#    plt.subplot(2,2,4)
#    plt.title('Mismatch')
#    plt.plot(poloidal_flux_output,sin_mismatch_angle_array)
##    plt.savefig('K_' + output_filename_suffix)
#
#
#    plt.figure()
#    plt.subplot(2,3,1)
#    plt.title('Re Psi_22')
#    plt.plot(tau_array,np.real(Psi_w_yy_array), marker='o')
#    plt.subplot(2,3,2)
#    plt.title('Re Psi_2a')
#    plt.plot(tau_array,np.real(Psi_w_xy_array), marker='o')
#    plt.subplot(2,3,3)
#    plt.title('Re Psi_aa')
#    plt.plot(tau_array,np.real(Psi_w_xx_array), marker='o')
#    plt.subplot(2,3,4)
#    plt.title('Im Psi_22')
#    plt.plot(tau_array,np.imag(Psi_w_yy_array), marker='o')
#    plt.subplot(2,3,5)
#    plt.title('Im Psi_2a')
#    plt.plot(tau_array,np.imag(Psi_w_xy_array), marker='o')
#    plt.subplot(2,3,6)
#    plt.title('Im Psi_aa')
#    plt.plot(tau_array,np.imag(Psi_w_xx_array), marker='o')
#    plt.savefig('Psi_' + output_filename_suffix)
##    plt.close()
#
#plt.figure()
#plt.subplot(3,3,1)
#plt.title('Real(Psi_R_R)')
#plt.plot(tau_array,np.real(Psi_3D_output[0,0,:]),'r')
#plt.plot(tau_array,np.imag(Psi_3D_output[0,0,:]),'b')
#plt.subplot(3,3,2)
#plt.title('Real(Psi_R_xi)')
#plt.plot(tau_array,np.real(Psi_3D_output[0,1,:]),'r')
#plt.plot(tau_array,np.imag(Psi_3D_output[0,1,:]),'b')
#plt.subplot(3,3,3)
#plt.title('Real(Psi_r_Z)')
#plt.plot(tau_array,np.real(Psi_3D_output[0,2,:]),'r')
#plt.plot(tau_array,np.imag(Psi_3D_output[0,2,:]),'b')
#plt.subplot(3,3,4)
#plt.subplot(3,3,5)
#plt.title('Real(Psi_xi_xi)')
#plt.plot(tau_array,np.real(Psi_3D_output[1,1,:]),'r')
#plt.plot(tau_array,np.imag(Psi_3D_output[1,1,:]),'b')
#plt.subplot(3,3,6)
#plt.title('Real(Psi_xi_z)')
#plt.plot(tau_array,np.real(Psi_3D_output[1,2,:]),'r')
#plt.plot(tau_array,np.imag(Psi_3D_output[1,2,:]),'b')
#plt.subplot(3,3,7)
#plt.subplot(3,3,8)
#plt.subplot(3,3,9)
#plt.title('Real(Psi_z_z)')
#plt.plot(tau_array,np.real(Psi_3D_output[2,2,:]),'r')
#plt.plot(tau_array,np.imag(Psi_3D_output[2,2,:]),'b')
#plt.savefig('Psi_lab_' + output_filename_suffix)
#plt.close()
#
#    # k_perp_1_0
#
##    K_g_before_0 = K_g_array[tau_0_index-1]
##    g_before_0 = g_magnitude_output[tau_0_index-1]
##    cos_theta_before_0 = cos_theta_output[tau_0_index-1]
##    tau_before_0 = tau_array[tau_0_index-1]
##
##    K_g_after_0 = K_g_array[tau_0_index+1]
##    g_after_0 = g_magnitude_output[tau_0_index+1]
##    cos_theta_after_0 = cos_theta_output[tau_0_index+1]
##    tau_after_0 = tau_array[tau_0_index+1]
#
#    K_magnitude_0 = K_magnitude_array[tau_0_index]
#    Psi_w_yy_0 = Psi_w_yy_array[tau_0_index]
#    Psi_w_xy_0 = Psi_w_xy_array[tau_0_index]
#    Psi_w_xx_0 = Psi_w_xx_array[tau_0_index]
#    Psi_w_inverse_yy_0 = Psi_w_inverse_yy_array[tau_0_index]
#    Psi_w_inverse_xy_0 = Psi_w_inverse_xy_array[tau_0_index]
#    Psi_w_inverse_xx_0 = Psi_w_inverse_xx_array[tau_0_index]
#    K_a_0 = K_x_array[tau_0_index]
#    K_2_0 = K_y_array[tau_0_index]
#    K_g_0 = K_g_array[tau_0_index]
#    sin_mismatch_angle_0 = sin_mismatch_angle_array[tau_0_index]
#    sin_theta_0 = sin_theta_output[tau_0_index]
#    cos_theta_0 = cos_theta_output[tau_0_index]
#    theta_0 = np.arcsin(sin_theta_0) # maybe sign error
#    g_0 = g_magnitude_output[tau_0_index]
#    g_hat_Cartesian_0 = g_hat_Cartesian_output[:,tau_0_index]
#    b_hat_0 = b_hat_Cartesian_output[:,tau_0_index]
#    K_vec_0 = np.asarray([K_X_array[tau_0_index],K_Y_array[tau_0_index],K_Z_array[tau_0_index]])
#    K_hat_0 = K_vec_0 / np.linalg.norm(K_vec_0)
#    y_hat_Cartesian_0 = np.cross(b_hat_0,g_hat_Cartesian_0) / (np.linalg.norm(np.cross(b_hat_0,g_hat_Cartesian_0)))
#    k_perp_1_hat_0 = np.cross(y_hat_Cartesian_0,b_hat_0) / (np.linalg.norm(np.cross(y_hat_Cartesian_0,b_hat_0)))
#    x_hat_Cartesian_0 = np.cross(g_hat_Cartesian_0,y_hat_Cartesian_0) / (np.linalg.norm(np.cross(g_hat_Cartesian_0,y_hat_Cartesian_0) ))
#    poloidal_flux_0 = poloidal_flux_output[tau_0_index]
#
#    np.arccos(np.dot(k_perp_1_hat_0,g_hat_Cartesian_0))
#    np.arccos(np.dot(b_hat_0,g_hat_Cartesian_0))
#    np.arcsin(np.dot(b_hat_0,g_hat_Cartesian_0))
#
#    mismatch_angle_0 = np.sign(sin_mismatch_angle_0)*np.arcsin(abs(sin_mismatch_angle_0))
#    mismatch_attenuation = 1 / (np.sqrt(2)*K_magnitude_0) * ( np.imag(Psi_w_inverse_yy_0) / ((np.imag(Psi_w_inverse_xy_0))**2 - np.imag(Psi_w_inverse_xx_0)*np.imag(Psi_w_inverse_yy_0)) )**(0.5)
#    mismatch_attenuation_wrong = 1 / (np.sqrt(2)*K_magnitude_0) * (- np.imag(1/Psi_w_xx_0) )**(-0.5)
##    mismatch_piece_wrong = np.exp(np.real(
##            -1j * K_magnitude_0**2 * mismatch_angle_0**2 / Psi_w_xx_0
##            ))
#    mismatch_piece_wrong = np.exp( - mismatch_angle_0**2 / mismatch_attenuation_wrong**2 )
#    mismatch_piece = np.exp( - mismatch_angle_0**2 / mismatch_attenuation**2 )
#
#    print("tau_0 =",tau_0)
#    print("K_magnitude_0 =",K_magnitude_0)
#    print("mismatch_piece =", mismatch_piece)
#    print("mismatch_piece_wrong =", mismatch_piece_wrong)
#    print("mismatch_angle_0 =", mismatch_angle_0)
#    print("mismatch_attenuation =", mismatch_attenuation)
#    print("mismatch_attenuation_wrong =", mismatch_attenuation_wrong)
#
#    # Calculating some quantities from what we already know
#    d_K_g_d_tau_array = np.gradient(K_g_array,tau_array)
#    theta_m_array = np.sign(sin_mismatch_angle_array)*np.arcsin(abs(sin_mismatch_angle_array))
#    d_theta_m_d_tau_array = np.gradient(theta_m_array,tau_array)
#
#    #sign_change_array = np.diff(np.sign(theta_m_array))
#    #
#    tau_nu_index = find_nearest(theta_m_array, 0) #Finds one of the zeroes. I should do this better and find all the zeroes, but that's for later
#    tau_nu = tau_array[tau_nu_index]
#    # --