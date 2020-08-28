# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

Run in Python 3, does not work in Python 2

Version history
v1 - Renaming of variables, cleaning up

Plan
Output everything to a file, and then do the analysis on that file.

1) Check that gradK_xi and such are done correctly, check that K_mag is calculated correctly when K_zeta is nonzero
2) Check that the calculation of Psi makes sense (and the rotation angle)
3) Check that K initial's calculation makes sense


Notes
- The loading of the input files was taken from integral_5 and modified.
- I should launch the beam inside the last closed flux surface
- K**2 = K_R**2 + K_z**2 + (K_zeta/r_R)**2, and K_zeta is constant (mode number). See 14 Sep 2018 notes.

Coordinates
X,Y,Z - Lab cartesian coordinates
R,zeta,Z - Lab cylindrical coordinates
x,y,g - Beam coordinates

Units
- SI units
- Distance in m
- Angles in rad
- electron cyclotron frequency positive
- K normalised such that K = 1 in vacuum. (Not implemented yet)
- Distance not normalised yet, should give it thought
- Start in vacuum, otherwise Psi_3D_beam_initial_cartersian does not get done properly
-

"""

import numpy as np
import math
from scipy import interpolate as interpolate
from scipy import constants as constants
import matplotlib.pyplot as plt
import os

from Scotty_fun import read_floats_into_list_until, find_nearest, find_H
from Scotty_fun import find_dH_dR, find_dH_dZ # \nabla H
from Scotty_fun import find_dH_dKR, find_dH_dKZ, find_dH_dKzeta # \nabla_K H
from Scotty_fun import find_d2H_dR2, find_d2H_dZ2, find_d2H_dR_dZ # \nabla \nabla H
from Scotty_fun import find_d2H_dKR2, find_d2H_dKR_dKzeta, find_d2H_dKR_dKz, find_d2H_dKzeta2, find_d2H_dKzeta_dKz, find_d2H_dKZ2 # \nabla_K \nabla_K H
from Scotty_fun import find_d2H_dKR_dR, find_d2H_dKR_dZ, find_d2H_dKzeta_dR, find_d2H_dKzeta_dZ, find_d2H_dKZ_dR, find_d2H_dKZ_dZ # \nabla_K \nabla H
from Scotty_fun import find_normalised_plasma_freq, find_normalised_gyro_freq
from Scotty_fun import find_dbhat_dR, find_dbhat_dZ

def beam_me_up(tau_step,
               numberOfTauPoints,
               saveInterval,
               poloidal_launch_angle_Torbeam,
               toroidal_launch_angle_Torbeam,
               launch_freq_GHz,
               mode_flag,
               launch_beam_width,
               launch_beam_curvature,
               launch_position,
               input_filename_suffix='',
               output_filename_suffix=''):


    numberOfDataPoints = (numberOfTauPoints) // saveInterval

    delta_R = 0.01 #in the same units as data_X_coord
    delta_Z = 0.01 #in the same units as data_Z_coord
    delta_K_R = 0.1 #in the same units as K_R
    delta_K_zeta = 0.1 #in the same units as K_zeta
    delta_K_Z = 0.1 #in the same units as K_z

    #major_radius = 0.9


    # ------------------------------




    # ------------------------------
     # Input data #
    # ------------------------------

    # # For analytical profile
    # # Specify the parameters
    #B_toroidal_max = 0.67 # in Tesla (?)
    #B_poloidal_max = 0.01 # in Tesla
    #
    #core_ne = 4.0 # units of 10^19 / m-3
    ##core_ne = 0
    #
    #aspect_ratio = 3.0 # major_radius/minor_radius
    #minor_radius = 0.5 # in meters
    #
    ## Calculates other parameters
    #major_radius = aspect_ratio * minor_radius
    #
    ## Generate ne
    #ne_data_length = 101
    #ne_data_fludata_X_coord = np.linspace(0,1,ne_data_length)
    #ne_data_density_array = np.zeros(ne_data_length)
    #ne_data_density_array = n_e_fun(ne_data_fludata_X_coord,core_ne)
    #
    #buffer_factor = 1.5
    #data_X_coord_length = 130
    #data_X_coord_start = major_radius - buffer_factor*minor_radius # in meters
    #data_X_coord_end = major_radius + buffer_factor*minor_radius
    #data_Z_coord_length = 65
    #data_Z_coord_start = -buffer_factor*minor_radius
    #data_Z_coord_end = buffer_factor*minor_radius
    #
    #data_X_coord = np.linspace(data_X_coord_start,data_X_coord_end,data_X_coord_length)
    #data_Z_coord = np.linspace(data_Z_coord_start,data_Z_coord_end,data_Z_coord_length)
    #
    #B_r = np.zeros([data_X_coord_length,data_Z_coord_length])
    #B_z = np.zeros([data_X_coord_length,data_Z_coord_length])
    #B_t = np.zeros([data_X_coord_length,data_Z_coord_length])
    #psi = np.zeros([data_X_coord_length,data_Z_coord_length])
    #
    #x_grid, z_grid = np.meshgrid(data_X_coord, data_Z_coord,indexing='ij')
    #data_B_T_grid = B_toroidal_fun(B_toroidal_max, data_X_coord, data_Z_coord, major_radius)
    #data_B_R_grid = B_r_fun(B_poloidal_max, data_X_coord, data_Z_coord, major_radius, minor_radius)
    #data_B_Z_grid = B_z_fun(B_poloidal_max, data_X_coord, data_Z_coord, major_radius, minor_radius)
    #data_poloidal_flux_grid = psi_fun(x_grid,z_grid, major_radius, minor_radius)
    ## ------------------------------



    # Experimental Profile----------

#    input_files_path = '/home/valerian/Dropbox/VHChen2018/Code - Torbeam/torbeam_ccfe_val_test/'
    #input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Code - Mismatch\\Benchmark_efit_equil_O\\'
    #input_files_path ='D:\\Dropbox\\VHChen2018\\Code - Mismatch\\Benchmark_efit_equil_X\\'
    #input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_08Apr2019\\'
    #input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Data\\Input_Files_08Apr2019\\'
    #input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Code - Mismatch\\Benchmark_efit_equil_O\\'
    #input_files_path ='D:\\Dropbox\\VHChen2018\\Code - Mismatch\\Benchmark_efit_equil\\'

    #input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Code - Torbeam\\Benchmark-7\\'
#    input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'

#    input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
#    input_files_path ='D:\\Dropbox\\VHChen2019\\Code - Scotty\\Benchmark_8\\Torbeam\\'

#    input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'

    input_files_path = os.path.dirname(os.path.abspath(__file__)) + '\\'


#    input_files_path ='C:\\Users\\chenv\\Dropbox\\VHChen2018\\Code - Torbeam\\torbeam_ccfe_val_test\\'
#    input_files_path ='D:\\Dropbox\\VHChen2018\\Code - Torbeam\\torbeam_ccfe_val_test\\'
#    input_files_path ='D:\\Dropbox\\VHChen2018\\Code - Mismatch\\Data-MAST-190ms\\'
#    input_files_path ='D:\\Dropbox\\VHChen2019\\Code - Scotty\\Benchmark_efit_equil_O\\Torbeam\\'


    # Importing data from input files
    # ne.dat, topfile
    # Others: inbeam.dat, Te.dat (not currently used in this code)
#    ne_filename = input_files_path + 'ne' +input_filename_suffix+ '_smoothed.dat'
    ne_filename = input_files_path + 'ne' +input_filename_suffix+ '.dat'

    topfile_filename = input_files_path + 'topfile' +input_filename_suffix


    ne_data = np.fromfile(ne_filename,dtype=float, sep='   ') # electron density as a function of poloidal flux label
    with open(topfile_filename) as f:
        while not 'X-coordinates' in f.readline(): pass # Start reading only from X-coords onwards
        data_X_coord = read_floats_into_list_until('Z-coordinates', f)
        data_Z_coord = read_floats_into_list_until('B_R', f)
        data_B_R_grid = read_floats_into_list_until('B_t', f)
        data_B_T_grid = read_floats_into_list_until('B_Z', f)
        data_B_Z_grid = read_floats_into_list_until('psi', f)
        data_poloidal_flux_grid = read_floats_into_list_until('you fall asleep', f)
    # ------------------------------

    # Tidying up the input data
    launch_angular_frequency = 2*math.pi*10.0**9 * launch_freq_GHz
    wavenumber_K0 = launch_angular_frequency / constants.c

    ne_data_length = int(ne_data[0])
    ne_data_density_array = ne_data[2::2] # in units of 10.0**19 m-3
    #print('Warninig: Scale factor of 1.05 used')
    ne_data_radialcoord_array = ne_data[1::2]
    ne_data_fludata_X_coord = ne_data_radialcoord_array**2
#    ne_data_fludata_X_coord = ne_data[1::2] # My new IDL file outputs the flux density directly, instead of radialcoord

    data_B_R_grid = np.transpose((np.asarray(data_B_R_grid)).reshape(len(data_Z_coord),len(data_X_coord), order='C'))
    data_B_T_grid = np.transpose((np.asarray(data_B_T_grid)).reshape(len(data_Z_coord),len(data_X_coord), order='C'))
    data_B_Z_grid = np.transpose((np.asarray(data_B_Z_grid)).reshape(len(data_Z_coord),len(data_X_coord), order='C'))
    data_poloidal_flux_grid = np.transpose((np.asarray(data_poloidal_flux_grid)).reshape(len(data_Z_coord),len(data_X_coord), order='C'))
    # -------------------


    # Interpolation functions declared
    interp_B_R = interpolate.RectBivariateSpline(data_X_coord,data_Z_coord,data_B_R_grid, bbox=[None, None, None, None], kx=3, ky=3, s=0)
    interp_B_T = interpolate.RectBivariateSpline(data_X_coord,data_Z_coord,data_B_T_grid, bbox=[None, None, None, None], kx=3, ky=3, s=0)
    interp_B_Z = interpolate.RectBivariateSpline(data_X_coord,data_Z_coord,data_B_Z_grid, bbox=[None, None, None, None], kx=3, ky=3, s=0)

    interp_poloidal_flux = interpolate.interp2d(data_X_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), kind='cubic',
                                           copy=True, bounds_error=False, fill_value=None) # Flux extrapolated outside region

    interp_density_1D = interpolate.interp1d(ne_data_fludata_X_coord, ne_data_density_array,
                                             kind='linear', axis=-1, copy=True, bounds_error=False,
                                             fill_value=0, assume_sorted=False) # density is 0 outside the LCFS, hence the fill_value

    # -------------------

    # NOT IMPLEMENTED YET
    # Calculate entry parameters from launch parameters
    # That is, find beam at start of plasma given its parameters at the antenna
    poloidal_launch_angle_Rz = (180.0+poloidal_launch_angle_Torbeam)/180.0*np.pi
    poloidal_rotation_angle = (90.0+poloidal_launch_angle_Torbeam)/180.0*np.pi
    toroidal_launch_angle_Rz = (180.0+toroidal_launch_angle_Torbeam)/180.0*np.pi

    entry_beam_width = launch_beam_width
    entry_beam_curvature = launch_beam_curvature
    entry_position = launch_position
    # -------------------






    # Initialise
    tau_array = np.zeros(numberOfDataPoints)

    q_R_array  = np.zeros(numberOfDataPoints)
    q_zeta_array  = np.zeros(numberOfDataPoints)
    q_Z_array  = np.zeros(numberOfDataPoints)

    K_R_array = np.zeros(numberOfDataPoints)
    K_zeta_array = np.zeros(numberOfDataPoints)
    K_Z_array = np.zeros(numberOfDataPoints)

    psi_array = np.zeros(numberOfDataPoints)
    # -------------------


    # -------------------

    # Initialise the arrays for the solver
        # Euler
    #bufferSize = 2
    #coefficient_array = np.array([1])
        # AB3
    bufferSize = 4
    coefficient_array = np.array([23/12,-4/3,5/12])

    q_R_buffer = np.zeros(bufferSize)
    q_zeta_buffer = np.zeros(bufferSize)
    q_Z_buffer = np.zeros(bufferSize)
    K_R_buffer = np.zeros(bufferSize)
    K_Z_buffer = np.zeros(bufferSize)

    dH_dR_buffer = np.zeros(bufferSize)
    dH_dZ_buffer = np.zeros(bufferSize)
    dH_dKR_buffer = np.zeros(bufferSize)
    dH_dKzeta_buffer = np.zeros(bufferSize)
    dH_dKZ_buffer = np.zeros(bufferSize)

    Psi_3D_buffer = np.zeros([3,3,bufferSize],dtype='complex128')
    grad_grad_H_buffer = np.zeros([3,3,bufferSize])
    gradK_grad_H_buffer = np.zeros([3,3,bufferSize])
    grad_gradK_H_buffer = np.zeros([3,3,bufferSize])
    gradK_gradK_H_buffer = np.zeros([3,3,bufferSize])

    #K_R_initial = wavenumber_K0* np.cos( -toroidal_launch_angle_Torbeam/180.0*math.pi ) * np.cos( poloidal_launch_angle_Rz ) # K_R
    #K_zeta_initial = wavenumber_K0 * np.sin( -toroidal_launch_angle_Torbeam/180.0*math.pi ) * launch_position[0]# K_zeta
    #K_Z_initial = wavenumber_K0* np.cos( -toroidal_launch_angle_Torbeam/180.0*math.pi ) * np.sin( poloidal_launch_angle_Rz ) # K_z

    K_R_initial = -wavenumber_K0* np.cos( toroidal_launch_angle_Torbeam/180.0*math.pi ) * np.cos( poloidal_launch_angle_Torbeam/180.0*math.pi ) # K_R
    K_zeta_initial = -wavenumber_K0 * np.sin( toroidal_launch_angle_Torbeam/180.0*math.pi ) * np.cos( poloidal_launch_angle_Torbeam/180.0*math.pi ) * launch_position[0]# K_zeta
    K_Z_initial = -wavenumber_K0* np.sin( poloidal_launch_angle_Torbeam/180.0*math.pi ) # K_z

#    dH_dKR_initial = np.squeeze(find_dH_dKR(entry_position[0],entry_position[2],
#                                            K_R_initial,K_zeta_initial,K_Z_initial,
#                                            launch_angular_frequency,mode_flag,delta_K_R,
#                                            interp_poloidal_flux,interp_density_1D,
#                                            interp_B_R,interp_B_T,interp_B_Z))
#    dH_dKzeta_initial = np.squeeze(find_dH_dKzeta(entry_position[0],entry_position[2],
#                                                  K_R_initial,K_zeta_initial,K_Z_initial,
#                                                  launch_angular_frequency,mode_flag,delta_K_zeta,
#                                                  interp_poloidal_flux,interp_density_1D,
#                                                  interp_B_R,interp_B_T,interp_B_Z))
#    dH_dKZ_initial = np.squeeze(find_dH_dKZ(entry_position[0],entry_position[2],
#                                            K_R_initial,K_zeta_initial,K_Z_initial,
#                                            launch_angular_frequency,mode_flag,delta_K_Z,
#                                            interp_poloidal_flux,interp_density_1D,
#                                            interp_B_R,interp_B_T,interp_B_Z))

    # Assumes vacuum
    # Refer to 08 Dec 2019 notes for the Psi_gg component
    Psi_3D_beam_initial_cartersian = np.array(
            [
            [ wavenumber_K0/entry_beam_curvature+2j*entry_beam_width**-2, 0, 0 ],
            [ 0, wavenumber_K0/entry_beam_curvature+2j*entry_beam_width**-2, 0 ],
            [ 0, 0                                                         , 0 ]
            ]
            )

    rotation_matrix_pol = np.array( [
            [ np.cos(poloidal_rotation_angle), 0, np.sin(poloidal_rotation_angle) ],
            [ 0, 1, 0 ],
            [ -np.sin(poloidal_rotation_angle), 0, np.cos(poloidal_rotation_angle) ]
            ] )

    rotation_matrix_tor = np.array( [
            [ np.cos(toroidal_launch_angle_Torbeam/180.0*math.pi), np.sin(toroidal_launch_angle_Torbeam/180.0*math.pi), 0 ],
            [ -np.sin(toroidal_launch_angle_Torbeam/180.0*math.pi), np.cos(toroidal_launch_angle_Torbeam/180.0*math.pi), 0 ],
            [ 0,0,1 ]
            ] )

    rotation_matrix = np.matmul(rotation_matrix_pol,rotation_matrix_tor)
    rotation_matrix_inverse = np.transpose(rotation_matrix)

    Psi_3D_lab_initial_cartersian = np.matmul( rotation_matrix_inverse, np.matmul(Psi_3D_beam_initial_cartersian, rotation_matrix) )

    # Convert to cylindrical coordinates
    Psi_3D_lab_initial = np.zeros([3,3],dtype='complex128')
    Psi_3D_lab_initial[0][0] = Psi_3D_lab_initial_cartersian[0][0]
    Psi_3D_lab_initial[1][1] = Psi_3D_lab_initial_cartersian[1][1]* entry_position[0]**2 - K_R_initial*entry_position[0]
    Psi_3D_lab_initial[2][2] = Psi_3D_lab_initial_cartersian[2][2]

    #Psi_3D_lab_initial[0][1] = Psi_3D_lab_initial_cartersian[0][1]
    Psi_3D_lab_initial[0][1] = Psi_3D_lab_initial_cartersian[0][1]* entry_position[0] + K_zeta_initial / entry_position[0]
    Psi_3D_lab_initial[1][0] = Psi_3D_lab_initial[0][1]

    Psi_3D_lab_initial[0][2] = Psi_3D_lab_initial_cartersian[0][2]
    Psi_3D_lab_initial[2][0] = Psi_3D_lab_initial[0][2]
    Psi_3D_lab_initial[1][2] = Psi_3D_lab_initial_cartersian[1][2]* entry_position[0]
    Psi_3D_lab_initial[2][1] = Psi_3D_lab_initial[1][2]

    current_marker = 1
    tau_current = tau_array[0]
    output_counter = 0


    for index in range(0,bufferSize):
        q_R_buffer[index] = entry_position[0] #
        q_zeta_buffer[index] = entry_position[1] #
        q_Z_buffer[index] = entry_position[2] # r_Z
        K_R_buffer[index] = K_R_initial # K_R
        K_Z_buffer[index] = K_Z_initial # K_z
        Psi_3D_buffer[:,:,index] = Psi_3D_lab_initial

    # -------------------

    # -------------------

    # Initialise the results arrays
    dH_dKR_output = np.zeros(numberOfDataPoints)
    dH_dKzeta_output = np.zeros(numberOfDataPoints)
    dH_dKZ_output = np.zeros(numberOfDataPoints)
    dH_dR_output = np.zeros(numberOfDataPoints)
    dH_dZ_output = np.zeros(numberOfDataPoints)

    grad_grad_H_output = np.zeros([3,3,numberOfDataPoints])
    gradK_grad_H_output = np.zeros([3,3,numberOfDataPoints])
    gradK_gradK_H_output = np.zeros([3,3,numberOfDataPoints])
    Psi_3D_output = np.zeros([3,3,numberOfDataPoints],dtype='complex128')

    x_hat_Cartesian_output = np.zeros([3,numberOfDataPoints])
    y_hat_Cartesian_output = np.zeros([3,numberOfDataPoints])
    b_hat_Cartesian_output = np.zeros([3,numberOfDataPoints])

    g_hat_output = np.zeros([3,numberOfDataPoints])
    b_hat_output = np.zeros([3,numberOfDataPoints])
    y_hat_output = np.zeros([3,numberOfDataPoints])
    x_hat_output = np.zeros([3,numberOfDataPoints])

    grad_bhat_output = np.zeros([3,3,numberOfDataPoints])

    xhat_dot_grad_bhat_dot_xhat_output = np.zeros(numberOfDataPoints)
    xhat_dot_grad_bhat_dot_yhat_output = np.zeros(numberOfDataPoints)
    xhat_dot_grad_bhat_dot_ghat_output = np.zeros(numberOfDataPoints)
    yhat_dot_grad_bhat_dot_xhat_output = np.zeros(numberOfDataPoints)
    yhat_dot_grad_bhat_dot_yhat_output = np.zeros(numberOfDataPoints)
    yhat_dot_grad_bhat_dot_ghat_output = np.zeros(numberOfDataPoints)

    d_theta_d_tau_output         = np.zeros(numberOfDataPoints)
    d_xhat_d_tau_output          = np.zeros([3,numberOfDataPoints])
    d_xhat_d_tau_dot_yhat_output = np.zeros(numberOfDataPoints)
    ray_curvature_kappa_output   = np.zeros([3,numberOfDataPoints])
    kappa_dot_xhat_output        = np.zeros(numberOfDataPoints)
    kappa_dot_yhat_output        = np.zeros(numberOfDataPoints)


    B_total_output = np.zeros(numberOfDataPoints)

    H_output = np.zeros(numberOfDataPoints)
    Booker_alpha_output = np.zeros(numberOfDataPoints)
    Booker_beta_output = np.zeros(numberOfDataPoints)
    Booker_gamma_output = np.zeros(numberOfDataPoints)
    epsilon_para_output = np.zeros(numberOfDataPoints)
    epsilon_perp_output = np.zeros(numberOfDataPoints)
    epsilon_g_output = np.zeros(numberOfDataPoints)
    poloidal_flux_output = np.zeros(numberOfDataPoints)
    normalised_gyro_freq_output = np.zeros(numberOfDataPoints)
    normalised_plasma_freq_output = np.zeros(numberOfDataPoints)

    electron_density_output = np.zeros(numberOfDataPoints)

#    eigenvalues_output = np.zeros([3,numberOfDataPoints],dtype='complex128')
#    eigenvectors_output = np.zeros([3,3,numberOfDataPoints],dtype='complex128')

    # -------------------

    # Propagate the beam
        # This loop should start with current_marker = 1
        # The initial conditions are thus at [0], which makes sense
    for step in range(1,numberOfTauPoints+1):
        poloidal_flux = interp_poloidal_flux(q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1])
        electron_density = interp_density_1D(poloidal_flux)
    #    K_zeta = q_R_buffer[current_marker-1] * K_zeta_initial
        K_zeta = K_zeta_initial

        # \nabla H
        dH_dR_buffer[current_marker-1] = find_dH_dR(
                                                q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                                                K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                                                launch_angular_frequency,mode_flag,delta_R,
                                                interp_poloidal_flux,interp_density_1D,
                                                interp_B_R,interp_B_T,interp_B_Z
                                                )
        dH_dZ_buffer[current_marker-1] = find_dH_dZ(
                                                q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                                                K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                                                launch_angular_frequency,mode_flag,delta_Z,
                                                interp_poloidal_flux,interp_density_1D,
                                                interp_B_R,interp_B_T,interp_B_Z
                                                )

        # \nabla_K H
        dH_dKR_buffer[current_marker-1]    = find_dH_dKR(
                                                q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                                                K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                                                launch_angular_frequency,mode_flag,delta_K_R,
                                                interp_poloidal_flux,interp_density_1D,
                                                interp_B_R,interp_B_T,interp_B_Z
                                                )
        dH_dKzeta_buffer[current_marker-1] = find_dH_dKzeta(
                                                q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                                                K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                                                launch_angular_frequency,mode_flag,delta_K_zeta,
                                                interp_poloidal_flux,interp_density_1D,
                                                interp_B_R,interp_B_T,interp_B_Z
                                                )
        dH_dKZ_buffer[current_marker-1]    = find_dH_dKZ(
                                                q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                                                K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                                                launch_angular_frequency,mode_flag,delta_K_Z,
                                                interp_poloidal_flux,interp_density_1D,
                                                interp_B_R,interp_B_T,interp_B_Z
                                                )

        # \nabla \nabla H
        d2H_dR2   = find_d2H_dR2(
                            q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                            K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                            launch_angular_frequency,mode_flag,delta_R,
                            interp_poloidal_flux,interp_density_1D,
                            interp_B_R,interp_B_T,interp_B_Z
                            )
        d2H_dR_dZ = find_d2H_dR_dZ(
                            q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                            K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                            launch_angular_frequency,mode_flag,delta_R,delta_Z,
                            interp_poloidal_flux,interp_density_1D,
                            interp_B_R,interp_B_T,interp_B_Z
                            )
        d2H_dZ2   = find_d2H_dZ2(
                            q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                            K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                            launch_angular_frequency,mode_flag,delta_Z,
                            interp_poloidal_flux,interp_density_1D,
                            interp_B_R,interp_B_T,interp_B_Z
                            )
        grad_grad_H_buffer[:,:,current_marker-1] = np.squeeze(np.array([
            [d2H_dR2 ,  0, d2H_dR_dZ],
            [0       ,  0, 0        ],
            [d2H_dR_dZ, 0, d2H_dZ2  ]
            ]))

        # \nabla_K \nabla H
        d2H_dKR_dR    = find_d2H_dKR_dR(
                            q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                            K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                            launch_angular_frequency,mode_flag,delta_K_R,delta_R,
                            interp_poloidal_flux,interp_density_1D,
                            interp_B_R,interp_B_T,interp_B_Z
                            )
        d2H_dKZ_dZ    = find_d2H_dKZ_dZ(
                            q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                            K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                            launch_angular_frequency,mode_flag,delta_K_Z,delta_Z,
                            interp_poloidal_flux,interp_density_1D,
                            interp_B_R,interp_B_T,interp_B_Z
                            )
        d2H_dKR_dZ    = find_d2H_dKR_dZ(
                            q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                            K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                            launch_angular_frequency,mode_flag,delta_K_R,delta_Z,
                            interp_poloidal_flux,interp_density_1D,
                            interp_B_R,interp_B_T,interp_B_Z
                            )
        d2H_dKzeta_dZ = find_d2H_dKzeta_dZ(
                             q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                             K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                             launch_angular_frequency,mode_flag,delta_K_zeta,delta_Z,
                             interp_poloidal_flux,interp_density_1D,
                             interp_B_R,interp_B_T,interp_B_Z
                             )
        d2H_dKzeta_dR = find_d2H_dKzeta_dR(
                             q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                             K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                             launch_angular_frequency,mode_flag,delta_K_zeta,delta_R,
                             interp_poloidal_flux,interp_density_1D,
                             interp_B_R,interp_B_T,interp_B_Z
                             )
        d2H_dKZ_dR    = find_d2H_dKZ_dR(
                            q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                            K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                            launch_angular_frequency,mode_flag,delta_K_Z,delta_R,
                            interp_poloidal_flux,interp_density_1D,
                            interp_B_R,interp_B_T,interp_B_Z
                            )
        gradK_grad_H_buffer[:,:,current_marker-1] = np.squeeze(np.array([
            [d2H_dKR_dR,    0, d2H_dKR_dZ   ],
            [d2H_dKzeta_dR, 0, d2H_dKzeta_dZ],
            [d2H_dKZ_dR,    0, d2H_dKZ_dZ   ]
            ]))
        grad_gradK_H_buffer[:,:,current_marker-1] = np.transpose(gradK_grad_H_buffer[:,:,current_marker-1])

        # \nabla_K \nabla_K H
        d2H_dKR2 = find_d2H_dKR2(
                             q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                             K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                             launch_angular_frequency,mode_flag,delta_K_R,
                             interp_poloidal_flux,interp_density_1D,
                             interp_B_R,interp_B_T,interp_B_Z
                             )
        d2H_dKzeta2 = find_d2H_dKzeta2(
                                q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                                K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                                launch_angular_frequency,mode_flag,delta_K_zeta,
                             interp_poloidal_flux,interp_density_1D,
                             interp_B_R,interp_B_T,interp_B_Z
                                )
        d2H_dKZ2 = find_d2H_dKZ2(
                              q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                              K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                              launch_angular_frequency,mode_flag,delta_Z,
                              interp_poloidal_flux,interp_density_1D,
                              interp_B_R,interp_B_T,interp_B_Z
                              )
        d2H_dKR_dKzeta = find_d2H_dKR_dKzeta(
                               q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                               K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                               launch_angular_frequency,mode_flag,delta_K_R,delta_K_zeta,
                               interp_poloidal_flux,interp_density_1D,
                               interp_B_R,interp_B_T,interp_B_Z
                               )
        d2H_dKR_dKz = find_d2H_dKR_dKz(
                              q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                              K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                              launch_angular_frequency,mode_flag,delta_K_R,delta_K_Z,
                              interp_poloidal_flux,interp_density_1D,
                              interp_B_R,interp_B_T,interp_B_Z
                              )
        d2H_dKzeta_dKz = find_d2H_dKzeta_dKz(
                               q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                               K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                               launch_angular_frequency,mode_flag,delta_K_zeta,delta_K_Z,
                               interp_poloidal_flux,interp_density_1D,
                               interp_B_R,interp_B_T,interp_B_Z
                               )
        gradK_gradK_H_buffer[:,:,current_marker-1] = np.squeeze(np.array([
            [d2H_dKR2      , d2H_dKR_dKzeta, d2H_dKR_dKz   ],
            [d2H_dKR_dKzeta, d2H_dKzeta2   , d2H_dKzeta_dKz],
            [d2H_dKR_dKz   , d2H_dKzeta_dKz, d2H_dKZ2      ]
            ]))



        q_R_buffer[current_marker] = q_R_buffer[current_marker-1]
        q_zeta_buffer[current_marker] = q_zeta_buffer[current_marker-1]
        q_Z_buffer[current_marker] = q_Z_buffer[current_marker-1]
        K_R_buffer[current_marker] = K_R_buffer[current_marker-1]
        K_Z_buffer[current_marker] = K_Z_buffer[current_marker-1]

        Psi_3D_buffer[:,:,current_marker] = Psi_3D_buffer[:,:,current_marker-1]



        for coefficient_index in range(0,bufferSize-1):
            q_R_buffer[current_marker] += coefficient_array[coefficient_index] * dH_dKR_buffer[current_marker-1-coefficient_index] * tau_step
            q_zeta_buffer[current_marker] += coefficient_array[coefficient_index] * dH_dKzeta_buffer[current_marker-1-coefficient_index] * tau_step
            q_Z_buffer[current_marker] += coefficient_array[coefficient_index] * dH_dKZ_buffer[current_marker-1-coefficient_index] * tau_step
            K_R_buffer[current_marker] += -coefficient_array[coefficient_index] * dH_dR_buffer[current_marker-1-coefficient_index] * tau_step
            K_Z_buffer[current_marker] += -coefficient_array[coefficient_index] * dH_dZ_buffer[current_marker-1-coefficient_index] * tau_step

            Psi_3D_buffer[:,:,current_marker] += coefficient_array[coefficient_index] * (
                    - grad_grad_H_buffer[:,:,current_marker-1-coefficient_index]
                    - np.matmul(
                                Psi_3D_buffer[:,:,current_marker-1-coefficient_index],
                                gradK_grad_H_buffer[:,:,current_marker-1-coefficient_index]
                                )
                    - np.matmul(
                                grad_gradK_H_buffer[:,:,current_marker-1-coefficient_index],
                                Psi_3D_buffer[:,:,current_marker-1-coefficient_index]
                                )
                    - np.matmul(np.matmul(
                                Psi_3D_buffer[:,:,current_marker-1-coefficient_index],
                                gradK_gradK_H_buffer[:,:,current_marker-1-coefficient_index]),
                                Psi_3D_buffer[:,:,current_marker-1-coefficient_index]
                                )
                    )* tau_step




        if step % saveInterval == 0: # Write data to output arrays
            K_magnitude = np.sqrt(K_R_buffer[current_marker-1]**2 + K_Z_buffer[current_marker-1]**2)

            tau_array[output_counter] = tau_current

            q_R_array [output_counter] = q_R_buffer[current_marker-1]
            q_zeta_array [output_counter] = q_zeta_buffer[current_marker-1]
            q_Z_array [output_counter] = q_Z_buffer[current_marker-1]
            K_R_array[output_counter] = K_R_buffer[current_marker-1]
            K_Z_array[output_counter] = K_Z_buffer[current_marker-1]

            dH_dKR_output[output_counter] = dH_dKR_buffer[current_marker-1]
            dH_dKzeta_output[output_counter] = dH_dKzeta_buffer[current_marker-1]
            dH_dKZ_output[output_counter] = dH_dKZ_buffer[current_marker-1]
            dH_dR_output[output_counter] = dH_dR_buffer[current_marker-1]
            dH_dZ_output[output_counter] = dH_dZ_buffer[current_marker-1]

            Psi_3D_output[:,:,output_counter] = Psi_3D_buffer[:,:,current_marker-1]
            grad_grad_H_output[:,:,output_counter] = grad_grad_H_buffer[:,:,current_marker-1]
            gradK_grad_H_output[:,:,output_counter] = gradK_grad_H_buffer[:,:,current_marker-1]
            gradK_gradK_H_output[:,:,output_counter] = gradK_gradK_H_buffer[:,:,current_marker-1]

            electron_density_output[output_counter] = electron_density
            B_R = np.squeeze(interp_B_R(q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1]))
            B_T = np.squeeze(interp_B_T(q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1]))
            B_Z = np.squeeze(interp_B_Z(q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1]))

            B_total = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
            b_hat = np.array([B_R,B_T,B_Z]) / B_total
            g_magnitude = (dH_dKzeta_output[output_counter]**2 + dH_dKzeta_output[output_counter]**2 + dH_dKZ_output[output_counter]**2)**0.5
            g_hat_R     = dH_dKR_output[output_counter]    / g_magnitude
            g_hat_zeta  = dH_dKzeta_output[output_counter] / g_magnitude
            g_hat_Z     = dH_dKZ_output[output_counter]    / g_magnitude
            g_hat = np.asarray([g_hat_R,g_hat_zeta,g_hat_Z])
            y_hat = np.cross(b_hat,g_hat) / (np.linalg.norm(np.cross(b_hat,g_hat)))
            x_hat = np.cross(g_hat,y_hat) / (np.linalg.norm(np.cross(g_hat,y_hat)))

            g_hat_output[:,output_counter] = g_hat
            b_hat_output[:,output_counter] = b_hat
            B_total_output[output_counter] = B_total
            y_hat_output[:,output_counter] = y_hat
            x_hat_output[:,output_counter] = x_hat

            # Calculating the corrections to Psi_w
            dbhat_dR = find_dbhat_dR(q_R_buffer[current_marker-1], q_Z_buffer[current_marker-1], delta_R, interp_B_R, interp_B_T, interp_B_Z)
            dbhat_dZ = find_dbhat_dZ(q_R_buffer[current_marker-1], q_Z_buffer[current_marker-1], delta_Z, interp_B_R, interp_B_T, interp_B_Z)

            grad_bhat_output[:,0,output_counter] = dbhat_dR
            grad_bhat_output[:,1,output_counter] = dbhat_dZ

            xhat_dot_grad_bhat_dot_xhat_output[output_counter] = np.dot(x_hat,np.dot(grad_bhat_output[:,:,output_counter],x_hat))
            xhat_dot_grad_bhat_dot_yhat_output[output_counter] = np.dot(x_hat,np.dot(grad_bhat_output[:,:,output_counter],y_hat))
            xhat_dot_grad_bhat_dot_ghat_output[output_counter] = np.dot(x_hat,np.dot(grad_bhat_output[:,:,output_counter],g_hat))
            yhat_dot_grad_bhat_dot_xhat_output[output_counter] = np.dot(y_hat,np.dot(grad_bhat_output[:,:,output_counter],x_hat))
            yhat_dot_grad_bhat_dot_yhat_output[output_counter] = np.dot(y_hat,np.dot(grad_bhat_output[:,:,output_counter],y_hat))
            yhat_dot_grad_bhat_dot_ghat_output[output_counter] = np.dot(y_hat,np.dot(grad_bhat_output[:,:,output_counter],g_hat))
            # --

#            K_hat = np.array([K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1]]) / K_magnitude


            H_output[output_counter] = find_H(
                    q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                    K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                    launch_angular_frequency,mode_flag,
                    interp_poloidal_flux,interp_density_1D,
                    interp_B_R,interp_B_T,interp_B_Z
                    )

    #        Booker_alpha_output[output_counter] = find_Booker_alpha(electron_density,B_total,cos_theta_sq,launch_angular_frequency)
    #        Booker_beta_output[output_counter] = find_Booker_beta(electron_density,B_total,cos_theta_sq,launch_angular_frequency)
    #        Booker_gamma_output[output_counter] = find_Booker_gamma(electron_density,B_total,launch_angular_frequency)
    #
#            epsilon_para_output[output_counter]  = find_epsilon_para(electron_density,launch_angular_frequency)
#            epsilon_perp_output[output_counter]  = find_epsilon_perp(electron_density,B_total,launch_angular_frequency)
#            epsilon_g_output[output_counter]  = find_epsilon_g(electron_density,B_total,launch_angular_frequency)

            normalised_gyro_freq_output[output_counter] = find_normalised_gyro_freq(B_total,launch_angular_frequency)
            normalised_plasma_freq_output[output_counter] = find_normalised_plasma_freq(electron_density,launch_angular_frequency)

            poloidal_flux_output[output_counter] = poloidal_flux

            output_counter += 1

        tau_current += tau_step
        current_marker = (current_marker + 1) % bufferSize
    #K_zeta_array[:] = K_zeta_initial / q_R_array [:]
    # -------------------









    # Output the data

    plt.figure()
    plt.title('Rz')
    plt.xlabel('R / m') # x-direction
    plt.ylabel('z / m')

    contour_levels = np.linspace(0,1,11)
    CS = plt.contour(data_X_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
    plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
    plt.plot(
            np.concatenate([[launch_position[0],entry_position[0]],q_R_array ]),
            np.concatenate([[launch_position[2],entry_position[2]],q_Z_array ]),
            '--.k') # Central (reference) ray
    #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
    #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
    plt.xlim(data_X_coord[0],data_X_coord[-1])
    plt.ylim(data_Z_coord[0],data_Z_coord[-1])
    plt.savefig('Ray1_' + output_filename_suffix)
    plt.close()

    #plt.figure()
    #plt.title('Toroidal Plane')
    #plt.polar(q_zeta_array ,q_R_array )
        # Plot input parameters
    #plt.figure()
    #plt.title('B_R')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, data_B_R_grid??,levels=np.linspace(np.amin(data_B_R_grid),np.amax(data_B_R_grid),20), cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('B_z')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, data_B_Z_grid,levels=np.linspace(np.amin(data_B_Z_grid),np.amax(data_B_Z_grid),20), cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('B_t')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, data_B_T_grid,levels=np.linspace(np.amin(data_B_T_grid),np.amax(data_B_T_grid),20))
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)



        # Plot output parameters


    #
    #plt.figure()
    #plt.title('Booker alpha')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, Booker_alpha_grid,
    #                         levels=np.linspace(np.amin(Booker_alpha_grid),np.amax(Booker_alpha_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Booker beta')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, Booker_beta_grid,
    #                         levels=np.linspace(np.amin(Booker_beta_grid),np.amax(Booker_beta_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Booker gamma')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, Booker_gamma_grid,
    #                         levels=np.linspace(np.amin(Booker_gamma_grid),np.amax(Booker_gamma_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Grad_R H')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, grad_R_H_grid,
    #                         levels=np.linspace(np.amin(grad_R_H_grid),np.amax(grad_R_H_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)

    #plt.figure()
    #plt.title('Grad_z H')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, grad_z_H_grid,
    #                         levels=np.linspace(np.amin(grad_z_H_grid),np.amax(grad_z_H_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    ##plt.figure()
    ##plt.title('Gradk_R H')
    ##plt.xlabel('R / cm') # x-direction
    ##plt.ylabel('z / cm')
    ##
    ##colour_ax = plt.imshow(gradK_R_H_grid, cmap='bwr', interpolation='nearest',
    ##           extent=[data_X_coord[0],data_X_coord[-1],data_Z_coord[0],data_Z_coord[-1]])
    ##plt.colorbar(colour_ax)
    ##CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    ##
    ##plt.figure()
    ##plt.title('Gradk_z H')
    ##plt.xlabel('R / cm') # x-direction
    ##plt.ylabel('z / cm')
    ##
    ##colour_ax = plt.imshow(gradK_z_H_grid, cmap='bwr', interpolation='nearest',
    ##           extent=[data_X_coord[0],data_X_coord[-1],data_Z_coord[0],data_Z_coord[-1]])
    ##plt.colorbar(colour_ax)
    ##CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    ##
    #plt.figure()
    #plt.title('Refractive index sq')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, refractive_index_sq_grid,
    #                         levels=np.linspace(np.amin(refractive_index_sq_grid),np.amax(refractive_index_sq_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Epsilon para')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, epsilon_para_grid,
    #                         levels=np.linspace(np.amin(epsilon_para_grid),np.amax(epsilon_para_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Epsilon perp')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, epsilon_perp_grid,
    #                         levels=np.linspace(np.amin(epsilon_perp_grid),np.amax(epsilon_perp_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Epsilon g')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, epsilon_g_grid,
    #                         levels=np.linspace(np.amin(epsilon_g_grid),np.amax(epsilon_g_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Plasma Frequency')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, normalised_plasma_freq_grid,
    #                         levels=np.linspace(np.amin(normalised_plasma_freq_grid),np.amax(normalised_plasma_freq_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Gyrofrequency')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #
    #colour_ax = plt.contourf(x_grid, z_grid, normalised_gyro_freq_grid,
    #                         levels=np.linspace(np.amin(normalised_gyro_freq_grid),np.amax(normalised_gyro_freq_grid),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('Discriminant')
    #plt.xlabel('R / cm') # x-direction
    #plt.ylabel('z / cm')
    #discriminant = Booker_beta_grid**2 - 4*Booker_alpha_grid*Booker_gamma_grid
    #colour_ax = plt.contourf(x_grid, z_grid, discriminant,
    #                         levels=np.linspace(np.amin(discriminant),np.amax(discriminant),20),
    #                         cmap='bwr')
    #plt.colorbar(colour_ax)
    #CS = plt.contour(x_grid, z_grid, data_poloidal_flux_grid, levels=1,vmin=1,vmax=1)
    #
    #plt.figure()
    #plt.title('r_R')
    #plt.plot(tau_array,q_R_array )
    #
    #plt.figure()
    #plt.title('r_Z')
    #plt.plot(tau_array,q_Z_array )
    #
    #plt.figure()
    #plt.title('K_R')
    #plt.plot(tau_array,K_R_array)
    #
    #plt.figure()
    #plt.title('K_z')
    #plt.plot(tau_array,K_Z_array)
    #
    #
    #
    #plt.figure()
    #plt.title('H')
    #plt.plot(tau_array,H_output,'.')
#
#    plt.figure()
#    plt.subplot(2,2,1)
#    plt.title('gradK_R_H')
#    plt.plot(tau_array,dH_dKR_output)
#    plt.subplot(2,2,2)
#    plt.title('gradK_z_H')
#    plt.plot(tau_array,dH_dKZ_output)
#    plt.subplot(2,2,3)
#    plt.title('grad_R_H')
#    plt.plot(tau_array,dH_dR_output)
#    plt.subplot(2,2,4)
#    plt.title('grad_z_H')
#    plt.plot(tau_array,dH_dZ_output)
#
    #
    #
    #
    #plt.figure()
    #plt.subplot(1,3,1)
    #plt.title('Booker_alpha_output')
    #plt.plot(tau_array,Booker_alpha_output,'.')
    #
    #plt.subplot(1,3,2)
    #plt.title('Booker_beta_output')
    #plt.plot(tau_array,Booker_beta_output,'.')
    #
    #plt.subplot(1,3,3)
    #plt.title('Booker_gamma_output')
    #plt.plot(tau_array,Booker_gamma_output,'.')
    #
    #plt.figure()
    #plt.title('')
    #plt.plot(tau_array,(Booker_beta_output - mode_flag * np.sqrt(abs(Booker_beta_output**2 - 4*Booker_alpha_output*Booker_gamma_output))) / (2 * Booker_alpha_output),'--.k')
    #
    #plt.figure()
    #plt.title('Discriminant')
    #plt.plot(tau_array,(Booker_beta_output**2 - 4*Booker_alpha_output*Booker_gamma_output),'--.k')
    #
    #plt.figure()
    #plt.subplot(1,3,1)
    #plt.title('epsilon_para_output')
    #plt.plot(tau_array,epsilon_para_output,'.')
    #
    #plt.subplot(1,3,2)
    #plt.title('epsilon_perp_output')
    #plt.plot(tau_array,epsilon_perp_output,'.')
    #
    #plt.subplot(1,3,3)
    #plt.title('epsilon_g_output')
    #plt.plot(tau_array,epsilon_g_output,'.')
    #
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.title('Plasma freq normalised')
    #plt.plot(tau_array,normalised_plasma_freq_output,'.')
    #plt.subplot(1,2,2)
    #plt.title('Gyro freq normalised')
    #plt.plot(tau_array,normalised_gyro_freq_output,'.')
    #
    #plt.figure()
    #plt.subplot(1,3,1)
    #plt.title('epsilon_para')
    #plt.plot(tau_array,1-normalised_plasma_freq_output**2,'.')
    #
    #plt.subplot(1,3,2)
    #plt.title('epsilon_perp')
    #plt.plot(tau_array,1 - normalised_plasma_freq_output**2 / (1 - normalised_gyro_freq_output**2),'.')
    #
    #plt.subplot(1,3,3)
    #plt.title('epsilon_g')
    #plt.plot(tau_array, - (normalised_plasma_freq_output**2) * normalised_gyro_freq_output / (1 - normalised_gyro_freq_output**2),'.')
    #plt.figure()
    #plt.title('')
    #plt.plot(tau_array,q_R_array ,'.')

    #
    #
    #plt.figure()
    #plt.title('ne')
    #plt.plot(ne_data_fludata_X_coord,ne_data_density_array,'.')
    #
    #plt.figure()
    #plt.title('poloidal flux')
    #plt.plot(tau_array,poloidal_flux_output,'.')

    #
    ## -------------------

    # Convert r, K, Psi_3D from cylindrical to Cartesian coordinates
    q_X_array  = q_R_array *np.cos(q_zeta_array )
    q_Y_array = q_R_array *np.sin(q_zeta_array )

    K_X_array = K_R_array*np.cos(q_zeta_array ) - K_zeta_initial*np.sin(q_zeta_array ) / q_R_array
    K_Y_array = K_R_array*np.sin(q_zeta_array ) + K_zeta_initial*np.cos(q_zeta_array ) / q_R_array

    b_hat_Cartesian_output = np.zeros(np.shape(b_hat_output))
    b_hat_Cartesian_output[0,:] = b_hat_output[0,:]*np.cos(q_zeta_array ) - b_hat_output[1,:]*np.sin(q_zeta_array )
    b_hat_Cartesian_output[1,:] = b_hat_output[0,:]*np.sin(q_zeta_array ) + b_hat_output[1,:]*np.cos(q_zeta_array )
    b_hat_Cartesian_output[2,:] = b_hat_output[2,:]
    #b_hat_Cartesian_output = b_hat_output


    temp_matrix_for_Psi = np.zeros([3,3],dtype='complex128')
    temp_matrix_for_grad_grad_H = np.zeros([3,3])
    temp_matrix_for_gradK_grad_H = np.zeros([3,3])
    temp_matrix_for_gradK_gradK_H  = np.zeros([3,3])

    Psi_3D_output_Cartesian = np.zeros(np.shape(Psi_3D_output),dtype='complex128')
    grad_grad_H_output_Cartesian = np.zeros(np.shape(Psi_3D_output))
    gradK_grad_H_output_Cartesian = np.zeros(np.shape(Psi_3D_output))
    gradK_gradK_H_output_Cartesian = np.zeros(np.shape(Psi_3D_output))

    for step in range(0,numberOfDataPoints):
        temp_matrix_for_Psi[0][0] = Psi_3D_output[0][0][step]
        temp_matrix_for_Psi[0][1] = Psi_3D_output[0][1][step]/q_R_array [step] - K_zeta_initial/q_R_array [step]**2
        temp_matrix_for_Psi[0][2] = Psi_3D_output[0][2][step]
        temp_matrix_for_Psi[1][1] = Psi_3D_output[1][1][step]/q_R_array [step]**2 + K_R_array[step]/q_R_array [step]
        temp_matrix_for_Psi[1][2] = Psi_3D_output[1][2][step]/q_R_array [step]
        temp_matrix_for_Psi[2][2] = Psi_3D_output[2][2][step]
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
            [ np.cos(q_zeta_array [step]), -np.sin(q_zeta_array [step]), 0 ],
            [ np.sin(q_zeta_array [step]), np.cos(q_zeta_array [step]), 0 ],
            [ 0,0,1 ]
            ] )
        rotation_matrix_xi_inverse = np.transpose(rotation_matrix_xi)

        Psi_3D_output_Cartesian[:,:,step] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_Psi),rotation_matrix_xi_inverse)
    #    grad_grad_H_output_Cartesian[:,:,step] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_grad_grad_H),rotation_matrix_xi_inverse)
    #    gradK_grad_H_output_Cartesian[:,:,step] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_gradK_grad_H),rotation_matrix_xi_inverse)
    #    gradK_gradK_H_output_Cartesian[:,:,step] = np.matmul(np.matmul(rotation_matrix_xi,temp_matrix_for_gradK_gradK_H),rotation_matrix_xi_inverse)

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

    # Output the data
    #output_path = ''
    #
    #input_parameters_list = []
    #input_parameters_list += ['delta_R =' + str(delta_R)]
    #input_parameters_list += ['delta_Z =' + str(delta_Z)]
    #input_parameters_list += ['delta_K_R =' + str(delta_K_R)]
    #input_parameters_list += ['delta_K_zeta =' + str(delta_K_zeta)]
    #input_parameters_list += ['delta_K_Z =' + str(delta_K_Z)]
    #input_parameters_list += ['poloidal_launch_angle_Torbeam =' + str(poloidal_launch_angle_Torbeam)]
    #input_parameters_list += ['toroidal_launch_angle_Torbeam =' + str(toroidal_launch_angle_Torbeam)]
    #input_parameters_list += ['launch_freq_GHz = 55.0 =' + str(launch_freq_GHz)]
    #input_parameters_list += ['mode_flag =' + str(mode_flag)]
    #input_parameters_list += ['launch_beam_width =' + str(launch_beam_width)]
    #input_parameters_list += ['launch_beam_curvature =' + str(launch_beam_curvature)]
    #input_parameters_list += ['launch_position[0] =' + str(launch_position[0])]
    #input_parameters_list += ['launch_position[1] =' + str(launch_position[1])]
    #input_parameters_list += ['launch_position[2] =' + str(launch_position[2])]
    #input_parameters_list += ['major_radius =' + str(major_radius)]
    #input_parameters_list += ['numberOfTauPoints =' + str(numberOfTauPoints)]
    #input_parameters_list += ['saveInterval =' + str(saveInterval)]
    #input_parameters_list += ['tau_step =' + str(tau_step)]
    ##parameters_list += [' =' + str()]
    #
    #input_parameters_file = open(output_path + 'parameters.txt','w')
    #for item in input_parameters_list:
    #    input_parameters_file.write("%s\n" % item)
    #
    #beam_parameters_file = open(output_path + 'data.txt','w')
    #beam_parameters_file.write('psi, r_Z, r_y, r_Z, K_x, K_y, K_z, S_xx, S_xy, S_xz, S_yy, S_yz, S_zz, Phi_xx, Phi_xy, Phi_zz, Phi_yy, Phi_yz, Phi_zz \n')
    #for ii in range(0, numberOfDataPoints):
    #    beam_parameters_file.write('{:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   \n'
    #                       .format(
    #                               poloidal_flux_output[ii],
    #
    #                               q_X_array [ii],q_Y_array[ii],q_Z_array [ii],
    #                               K_X_array[ii],K_Y_array[ii],K_Z_array[ii],
    #
    #                               np.asscalar(np.real(Psi_3D_output_Cartesian[0][0][ii])),
    #                               np.asscalar(np.real(Psi_3D_output_Cartesian[0][1][ii])),
    #                               np.asscalar(np.real(Psi_3D_output_Cartesian[0][2][ii])),
    #                               np.asscalar(np.real(Psi_3D_output_Cartesian[1][1][ii])),
    #                               np.asscalar(np.real(Psi_3D_output_Cartesian[1][2][ii])),
    #                               np.asscalar(np.real(Psi_3D_output_Cartesian[2][2][ii])),
    #
    #                               np.asscalar(np.imag(Psi_3D_output_Cartesian[0][0][ii])),
    #                               np.asscalar(np.imag(Psi_3D_output_Cartesian[0][1][ii])),
    #                               np.asscalar(np.imag(Psi_3D_output_Cartesian[0][2][ii])),
    #                               np.asscalar(np.imag(Psi_3D_output_Cartesian[1][1][ii])),
    #                               np.asscalar(np.imag(Psi_3D_output_Cartesian[1][2][ii])),
    #                               np.asscalar(np.imag(Psi_3D_output_Cartesian[2][2][ii])),
    #                               )
    #                    )

    #gradients_of_H_file = open(output_path + 'gradients.txt','w')
    #gradients_of_H_file.write('dH_dKx, dH_dKy, dH_dKz, dH_drx, dH_dry, dH_drz, d2H_dKx_dKx, d2H_dKx_dKy, d2H_dKx_dKz, d2H_dKy_dKy, d2H_dKy_dKz, d2H_dKZ_dKz, d2H_dKx_drx, d2H_dKx_dry, d2H_dKx_drz, d2H_dKy_drx, d2H_dKy_dry, d2H_dKy_drz, d2H_dKZ_drx, d2H_dKZ_dry, d2H_dKZ_drz, d2H_drx_drx, d2H_drx_dry, d2H_drx_drz, d2H_dry_dry, d2H_dry_drz, d2H_drz_drz \n')
    #for ii in range(0, numberOfDataPoints):
    #    gradients_of_H_file.write('{:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}   {:.8}    \n'
    #                       .format(
    #                               dH_dKX_output[ii],
    #                               dH_dKY_output[ii],
    #                               dH_dKZ_output[ii],
    #
    #                               dH_dX_output[ii],
    #                               dH_dY_output[ii],
    #                               dH_dZ_output[ii],
    #
    #                               np.asscalar(d2H_dKx_dKx_output[ii]),
    #                               np.asscalar(d2H_dKx_dKy_output[ii]),
    #                               np.asscalar(d2H_dKx_dKZ_output[ii]),
    #                               np.asscalar(d2H_dKy_dKy_output[ii]),
    #                               np.asscalar(d2H_dKy_dKZ_output[ii]),
    #                               np.asscalar(d2H_dKZ_dKZ_output[ii]),
    #
    #                               np.asscalar(d2H_dKx_drx_output[ii]),
    #                               np.asscalar(d2H_dKx_dry_output[ii]),
    #                               np.asscalar(d2H_dKx_drz_output[ii]),
    #                               np.asscalar(d2H_dKy_drx_output[ii]),
    #                               np.asscalar(d2H_dKy_dry_output[ii]),
    #                               np.asscalar(d2H_dKy_drz_output[ii]),
    #                               np.asscalar(d2H_dKZ_drx_output[ii]),
    #                               np.asscalar(d2H_dKZ_dry_output[ii]),
    #                               np.asscalar(d2H_dKZ_drz_output[ii]),
    #
    #                               np.asscalar(d2H_drx_drx_output[ii]),
    #                               np.asscalar(d2H_drx_dry_output[ii]),
    #                               np.asscalar(d2H_drx_drz_output[ii]),
    #                               np.asscalar(d2H_dry_dry_output[ii]),
    #                               np.asscalar(d2H_dry_drz_output[ii]),
    #                               np.asscalar(d2H_drz_drz_output[ii])
    #                               )
    #                    )


    # --------------------

    ####################################################################
    # Some plots
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Poloidal Plane')
    plt.xlabel('x / m') # x-direction
    plt.ylabel('z / m')

    contour_levels = np.linspace(0,1,11)
    CS = plt.contour(data_X_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
    plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
    plt.plot(q_X_array ,q_Z_array , '--.k') # Central (reference) ray
    #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
    #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
    plt.xlim(data_X_coord[0],data_X_coord[-1])
    plt.ylim(data_Z_coord[0],data_Z_coord[-1])
    plt.subplot(1,2,2)
    plt.title('Toroidal Plane')
    plt.plot(q_X_array ,q_Y_array, '--.k') # Central (reference) ray
    plt.xlabel('x / m') # x-direction
    plt.ylabel('y / m')
    plt.savefig('Ray2_' + output_filename_suffix)
#    plt.close()
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.plot(tau_array,q_X_array , '--.g') # Central (reference) ray
#    plt.subplot(1,2,2)
#    plt.plot(tau_array,q_Y_array, '--.b') # Central (reference) ray
#
#    plt.figure()
#    plt.plot(tau_array,poloidal_flux_output, '--.k') # Central (reference) ray
#    plt.figure()
#    plt.plot(tau_array,normalised_plasma_freq_output, '--.k') # Central (reference) ray

    ####################################################################
    # Mismatch
    g_magnitude_output = (dH_dKX_output**2 + dH_dKY_output**2 + dH_dKZ_output**2)**0.5
    g_hat_X_output = dH_dKX_output / g_magnitude_output
    g_hat_Y_output = dH_dKY_output / g_magnitude_output
    g_hat_Z_output = dH_dKZ_output / g_magnitude_output
    g_hat_Cartesian_output = np.asarray([g_hat_X_output,g_hat_Y_output,g_hat_Z_output])

    sin_theta_output = -( g_hat_Cartesian_output[0,:]*b_hat_Cartesian_output[0,:]
                        + g_hat_Cartesian_output[1,:]*b_hat_Cartesian_output[1,:]
                        + g_hat_Cartesian_output[2,:]*b_hat_Cartesian_output[2,:]) #negative sign because of how I've defined the angles


    cos_theta_output = np.sqrt(1 - sin_theta_output**2) #This works since theta_m is between -pi/2 to pi/2
    theta_output = np.arcsin(sin_theta_output)

    # To calculate corrections to Psi
    d_theta_d_tau_output = np.gradient(theta_output,tau_array)

    d_xhat_d_tau_output[0,:] = np.gradient(x_hat_output[0,:],tau_array)
    d_xhat_d_tau_output[1,:] = np.gradient(x_hat_output[1,:],tau_array)
    d_xhat_d_tau_output[2,:] = np.gradient(x_hat_output[2,:],tau_array)
    d_xhat_d_tau_dot_yhat_output = ( d_xhat_d_tau_output[0,:]*y_hat_output[0,:]
                                   + d_xhat_d_tau_output[1,:]*y_hat_output[1,:]
                                   + d_xhat_d_tau_output[2,:]*y_hat_output[2,:] ) # Can't get dot product to work properly

    ray_curvature_kappa_output[0,:] = (1/g_magnitude_output) * np.gradient(g_hat_output[0,:],tau_array)
    ray_curvature_kappa_output[1,:] = (1/g_magnitude_output) * np.gradient(g_hat_output[1,:],tau_array)
    ray_curvature_kappa_output[2,:] = (1/g_magnitude_output) * np.gradient(g_hat_output[2,:],tau_array)
    kappa_dot_xhat_output = ( ray_curvature_kappa_output[0,:]*x_hat_output[0,:]
                            + ray_curvature_kappa_output[1,:]*x_hat_output[1,:]
                            + ray_curvature_kappa_output[2,:]*x_hat_output[2,:] ) # Can't get dot product to work properly
    kappa_dot_yhat_output = ( ray_curvature_kappa_output[0,:]*y_hat_output[0,:]
                            + ray_curvature_kappa_output[1,:]*y_hat_output[1,:]
                            + ray_curvature_kappa_output[2,:]*y_hat_output[2,:] ) # Can't get dot product to work properly

    K_magnitude_array = ( K_X_array**2 + K_Y_array**2 + K_Z_array**2 )**0.5
    K_magnitude_min = min(K_magnitude_array)

    K_dot_g = dH_dKX_output*K_X_array + dH_dKY_output*K_Y_array + dH_dKZ_output*K_Z_array
    eikonal_S = np.cumsum(K_dot_g*tau_step) # TODO: Change to proper integration methods
    integral_g_cos_theta_dtau = np.cumsum(g_magnitude_output*cos_theta_output*tau_step)

    numberOfkperp1 = 100
    k_perp_1_array = np.linspace(-3.0*K_magnitude_min,2.0*K_magnitude_min,numberOfkperp1)
    colours_r = np.linspace(0,1,numberOfkperp1)
    colours_g = np.zeros(numberOfkperp1)
    colours_b = np.linspace(1,0,numberOfkperp1)


    poloidal_flux_search = 0.95
    search_condition = 0
    for ii in range(0,len(poloidal_flux_output)):
        if search_condition == 0 and poloidal_flux_output[ii] <= poloidal_flux_search:
            tau_search_start_index = ii
            search_condition = 1
        elif search_condition == 1 and poloidal_flux_output[ii] >= poloidal_flux_search:
            tau_search_end_index = ii
            search_condition = 2

    phase_array = np.zeros([len(tau_array[tau_search_start_index:tau_search_end_index]),numberOfkperp1])
    d_phase_d_tau_array = np.zeros([len(tau_array[tau_search_start_index:tau_search_end_index]),numberOfkperp1])
    d2_phase_d2_tau_array = np.zeros([len(tau_array[tau_search_start_index:tau_search_end_index]),numberOfkperp1])

    d_phase_d_tau_min_array = np.zeros(numberOfkperp1)
    tau_min_array = np.zeros(numberOfkperp1)
    tau_0_estimate_array = np.zeros(numberOfkperp1)

    plt.figure()
    ax1 = plt.subplot(1,3,1)
    plt.title('phase')
    ax2 = plt.subplot(1,3,2)
    plt.title('d phase / d tau')
    ax3 = plt.subplot(1,3,3)
    plt.title('d2 phase / d tau2')

    for ii in range(0,numberOfkperp1):
        phase_array[:,ii] = 2*eikonal_S[tau_search_start_index:tau_search_end_index] + k_perp_1_array[ii]*integral_g_cos_theta_dtau[tau_search_start_index:tau_search_end_index]
        d_phase_d_tau_array[:,ii] = np.gradient(phase_array[:,ii],tau_array[tau_search_start_index:tau_search_end_index])
        d2_phase_d2_tau_array[:,ii] = np.gradient(d_phase_d_tau_array[:,ii],tau_array[tau_search_start_index:tau_search_end_index])

        find_function = interpolate.interp1d(d2_phase_d2_tau_array[:,ii],
                                             tau_array[tau_search_start_index:tau_search_end_index],
                                             kind='cubic', axis=-1, copy=True, bounds_error=True,
                                             fill_value=0, assume_sorted=False)
        tau_0_estimate_array[ii] = find_function(0)

        d_phase_d_tau_min_array[ii] = min( d_phase_d_tau_array[:,ii] )
        tau_min_array[ii] = tau_array[ tau_search_start_index + np.argmin(d_phase_d_tau_array[:,ii]) ]

        ax1.plot(tau_array[tau_search_start_index:tau_search_end_index],phase_array[:,ii],color=[colours_r[ii],colours_g[ii],colours_b[ii]])
        ax2.plot(tau_array[tau_search_start_index:tau_search_end_index],d_phase_d_tau_array[:,ii],color=[colours_r[ii],colours_g[ii],colours_b[ii]])
        ax3.plot(tau_array[tau_search_start_index:tau_search_end_index],d2_phase_d2_tau_array[:,ii],color=[colours_r[ii],colours_g[ii],colours_b[ii]])

    find_tau_0 = interpolate.interp1d(d_phase_d_tau_min_array,
                                        tau_min_array,
                                        kind='cubic', axis=-1, copy=True, bounds_error=True,
                                        fill_value=0, assume_sorted=False)
    tau_0 = find_tau_0(0)
    tau_0_index = find_nearest(tau_array,tau_0)
    tau_start = tau_array[tau_search_start_index]
    tau_end = tau_array[tau_search_end_index]

    ax3.plot([tau_array[tau_search_start_index],tau_array[tau_search_end_index]],[0,0],'k')
    ax2.plot(tau_min_array,d_phase_d_tau_min_array, 'o', markersize=5, color="black")
    tau_turning = tau_array[ np.argmin(K_magnitude_array) ]
    ax2.axvline(tau_0, ymin=0, ymax=1)
    ax3.axvline(tau_0, ymin=0, ymax=1)
    plt.savefig('Phase_' + output_filename_suffix)
    plt.close()

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('theta')
    plt.plot(tau_array,theta_output)
    plt.subplot(2,2,2)
    plt.title('g_magnitude')
    plt.plot(tau_array,g_magnitude_output)
    plt.subplot(2,2,3)
    plt.title('K_dot_g')
    plt.plot(tau_array,K_dot_g)
    plt.subplot(2,2,4)
    plt.title('K_magnitude')
    plt.plot(tau_array,K_magnitude_array)
    plt.savefig('Params_' + output_filename_suffix)
    plt.close()

#    plt.figure()
#    plt.title('tau_0 estimate')
#    plt.plot(k_perp_1_array,tau_0_estimate_array,'k')
#    plt.axhline(tau_0)

    find_k_0 = interpolate.interp1d(d_phase_d_tau_min_array,
                                        k_perp_1_array,
                                        kind='cubic', axis=-1, copy=True, bounds_error=True,
                                        fill_value=0, assume_sorted=False)
    k_perp_1_0 = find_k_0(0)


    numberOfDataPoints = len(tau_array)
    Psi_w_yy_array = np.zeros(numberOfDataPoints,dtype=complex)
    Psi_w_xy_array = np.zeros(numberOfDataPoints,dtype=complex)
    Psi_w_xx_array = np.zeros(numberOfDataPoints,dtype=complex)
    Psi_w_inverse_yy_array = np.zeros(numberOfDataPoints,dtype=complex)
    Psi_w_inverse_xy_array = np.zeros(numberOfDataPoints,dtype=complex)
    Psi_w_inverse_xx_array = np.zeros(numberOfDataPoints,dtype=complex)
    K_x_array = np.zeros(numberOfDataPoints)
    K_y_array = np.zeros(numberOfDataPoints)
    K_g_array = np.zeros(numberOfDataPoints)
    sin_mismatch_angle_array = np.zeros(numberOfDataPoints)

    # Psi_3D_output_Cartesian
    # K_X_array
    # I'm sure this can be vectorised, but I'm leaving it this way for now
    for ii in range(numberOfDataPoints):
        g_hat_Cartesian = g_hat_Cartesian_output[:,ii]
        b_hat = b_hat_Cartesian_output[:,ii]
        K_vec = np.asarray([K_X_array[ii],K_Y_array[ii],K_Z_array[ii]])
        Psi_3D_lab = Psi_3D_output_Cartesian[:,:,ii]
        # --
        y_hat_Cartesian = np.cross(b_hat,g_hat_Cartesian) / (np.linalg.norm(np.cross(b_hat,g_hat_Cartesian)))
        x_hat_Cartesian = np.cross(g_hat_Cartesian,y_hat_Cartesian) / (np.linalg.norm(np.cross(g_hat_Cartesian,y_hat_Cartesian) ))

        y_hat_Cartesian_output[:,ii] = y_hat_Cartesian
        x_hat_Cartesian_output[:,ii] = x_hat_Cartesian

        K_x_array[ii] = np.dot(K_vec,x_hat_Cartesian)
        K_y_array[ii] = np.dot(K_vec,y_hat_Cartesian)
        K_g_array[ii] = np.dot(K_vec,g_hat_Cartesian)

        sin_mismatch_angle_array[ii] = -np.dot(K_vec,b_hat) / (np.linalg.norm(K_vec)*np.linalg.norm(b_hat)) # negative sign because of how I've defined angles

        Psi_w_yy_array[ii] = np.dot(y_hat_Cartesian,np.dot(Psi_3D_lab,y_hat_Cartesian))
        Psi_w_xy_array[ii] = np.dot(y_hat_Cartesian,np.dot(Psi_3D_lab,x_hat_Cartesian))
        Psi_w_xx_array[ii] = np.dot(x_hat_Cartesian,np.dot(Psi_3D_lab,x_hat_Cartesian))

        Psi_w_det = Psi_w_yy_array[ii] * Psi_w_xx_array[ii] - Psi_w_xy_array[ii]**2

        Psi_w_inverse_yy_array[ii] = Psi_w_xx_array[ii]/Psi_w_det
        Psi_w_inverse_xy_array[ii] = -Psi_w_xy_array[ii]/Psi_w_det
        Psi_w_inverse_xx_array[ii] = Psi_w_yy_array[ii]/Psi_w_det
    cos_mismatch_angle_array = np.sqrt(1 - sin_mismatch_angle_array**2)


    plt.figure()
    plt.subplot(1,2,1)
    plt.title('K_x')
    plt.plot(tau_array,K_x_array/K_magnitude_array)
    plt.subplot(1,2,2)
    plt.title('K_y')
    plt.plot(tau_array,K_y_array/K_magnitude_array)
    plt.savefig('K_' + output_filename_suffix)
    plt.close()

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('K')
    plt.plot(poloidal_flux_output,K_magnitude_array)
    plt.subplot(2,2,2)
    plt.title('K_g')
    plt.plot(poloidal_flux_output,K_g_array)
    plt.subplot(2,2,3)
    plt.title('K_x')
    plt.plot(poloidal_flux_output,K_x_array)
    plt.subplot(2,2,4)
    plt.title('Mismatch')
    plt.plot(poloidal_flux_output,sin_mismatch_angle_array)
#    plt.savefig('K_' + output_filename_suffix)


    plt.figure()
    plt.subplot(2,3,1)
    plt.title('Re Psi_22')
    plt.plot(tau_array,np.real(Psi_w_yy_array), marker='o')
    plt.subplot(2,3,2)
    plt.title('Re Psi_2a')
    plt.plot(tau_array,np.real(Psi_w_xy_array), marker='o')
    plt.subplot(2,3,3)
    plt.title('Re Psi_aa')
    plt.plot(tau_array,np.real(Psi_w_xx_array), marker='o')
    plt.subplot(2,3,4)
    plt.title('Im Psi_22')
    plt.plot(tau_array,np.imag(Psi_w_yy_array), marker='o')
    plt.subplot(2,3,5)
    plt.title('Im Psi_2a')
    plt.plot(tau_array,np.imag(Psi_w_xy_array), marker='o')
    plt.subplot(2,3,6)
    plt.title('Im Psi_aa')
    plt.plot(tau_array,np.imag(Psi_w_xx_array), marker='o')
    plt.savefig('Psi_' + output_filename_suffix)
#    plt.close()

    plt.figure()
    plt.subplot(3,3,1)
    plt.title('Real(Psi_R_R)')
    plt.plot(tau_array,np.real(Psi_3D_output[0,0,:]),'r')
    plt.plot(tau_array,np.imag(Psi_3D_output[0,0,:]),'b')
    plt.subplot(3,3,2)
    plt.title('Real(Psi_R_xi)')
    plt.plot(tau_array,np.real(Psi_3D_output[0,1,:]),'r')
    plt.plot(tau_array,np.imag(Psi_3D_output[0,1,:]),'b')
    plt.subplot(3,3,3)
    plt.title('Real(Psi_r_Z)')
    plt.plot(tau_array,np.real(Psi_3D_output[0,2,:]),'r')
    plt.plot(tau_array,np.imag(Psi_3D_output[0,2,:]),'b')
    plt.subplot(3,3,4)
    plt.subplot(3,3,5)
    plt.title('Real(Psi_xi_xi)')
    plt.plot(tau_array,np.real(Psi_3D_output[1,1,:]),'r')
    plt.plot(tau_array,np.imag(Psi_3D_output[1,1,:]),'b')
    plt.subplot(3,3,6)
    plt.title('Real(Psi_xi_z)')
    plt.plot(tau_array,np.real(Psi_3D_output[1,2,:]),'r')
    plt.plot(tau_array,np.imag(Psi_3D_output[1,2,:]),'b')
    plt.subplot(3,3,7)
    plt.subplot(3,3,8)
    plt.subplot(3,3,9)
    plt.title('Real(Psi_z_z)')
    plt.plot(tau_array,np.real(Psi_3D_output[2,2,:]),'r')
    plt.plot(tau_array,np.imag(Psi_3D_output[2,2,:]),'b')
    plt.savefig('Psi_lab_' + output_filename_suffix)
    plt.close()

    # k_perp_1_0

#    K_g_before_0 = K_g_array[tau_0_index-1]
#    g_before_0 = g_magnitude_output[tau_0_index-1]
#    cos_theta_before_0 = cos_theta_output[tau_0_index-1]
#    tau_before_0 = tau_array[tau_0_index-1]
#
#    K_g_after_0 = K_g_array[tau_0_index+1]
#    g_after_0 = g_magnitude_output[tau_0_index+1]
#    cos_theta_after_0 = cos_theta_output[tau_0_index+1]
#    tau_after_0 = tau_array[tau_0_index+1]

    K_magnitude_0 = K_magnitude_array[tau_0_index]
    Psi_w_yy_0 = Psi_w_yy_array[tau_0_index]
    Psi_w_xy_0 = Psi_w_xy_array[tau_0_index]
    Psi_w_xx_0 = Psi_w_xx_array[tau_0_index]
    Psi_w_inverse_yy_0 = Psi_w_inverse_yy_array[tau_0_index]
    Psi_w_inverse_xy_0 = Psi_w_inverse_xy_array[tau_0_index]
    Psi_w_inverse_xx_0 = Psi_w_inverse_xx_array[tau_0_index]
    K_a_0 = K_x_array[tau_0_index]
    K_2_0 = K_y_array[tau_0_index]
    K_g_0 = K_g_array[tau_0_index]
    sin_mismatch_angle_0 = sin_mismatch_angle_array[tau_0_index]
    sin_theta_0 = sin_theta_output[tau_0_index]
    cos_theta_0 = cos_theta_output[tau_0_index]
    theta_0 = np.arcsin(sin_theta_0) # maybe sign error
    g_0 = g_magnitude_output[tau_0_index]
    g_hat_Cartesian_0 = g_hat_Cartesian_output[:,tau_0_index]
    b_hat_0 = b_hat_Cartesian_output[:,tau_0_index]
    K_vec_0 = np.asarray([K_X_array[tau_0_index],K_Y_array[tau_0_index],K_Z_array[tau_0_index]])
    K_hat_0 = K_vec_0 / np.linalg.norm(K_vec_0)
    y_hat_Cartesian_0 = np.cross(b_hat_0,g_hat_Cartesian_0) / (np.linalg.norm(np.cross(b_hat_0,g_hat_Cartesian_0)))
    k_perp_1_hat_0 = np.cross(y_hat_Cartesian_0,b_hat_0) / (np.linalg.norm(np.cross(y_hat_Cartesian_0,b_hat_0)))
    x_hat_Cartesian_0 = np.cross(g_hat_Cartesian_0,y_hat_Cartesian_0) / (np.linalg.norm(np.cross(g_hat_Cartesian_0,y_hat_Cartesian_0) ))
    poloidal_flux_0 = poloidal_flux_output[tau_0_index]

    np.arccos(np.dot(k_perp_1_hat_0,g_hat_Cartesian_0))
    np.arccos(np.dot(b_hat_0,g_hat_Cartesian_0))
    np.arcsin(np.dot(b_hat_0,g_hat_Cartesian_0))

    mismath_angle_0 = np.sign(sin_mismatch_angle_0)*np.arcsin(abs(sin_mismatch_angle_0))

#    mismatch_piece = np.exp(np.real(
#            -1j / Psi_w_inverse_yy_0 * (
#                    (2*K_a_0 + k_perp_1_0*sin_theta_output_0)*Psi_w_inverse_xy_0)**2
#            -1j / 4 * (2*K_a_0 + k_perp_1_0*sin_theta_output_0)**2*Psi_w_inverse_xx_0
#            ))
#
#    mismatch_piece_reduced = np.exp(np.real(
#            -1j * K_magnitude_0**2 * mismath_angle_0**2 / Psi_w_xx_0
#            ))

    mismatch_piece = np.exp(np.real(
            -1j * K_magnitude_0**2 * mismath_angle_0**2 / Psi_w_xx_0
            ))

    print("tau_0 =",tau_0)
    print("mismatch_piece =",mismatch_piece)
    print("mismath_angle_0 =",mismath_angle_0)

    # Calculating some quantities from what we already know
    d_K_g_d_tau_array = np.gradient(K_g_array,tau_array)
    theta_m_array = np.sign(sin_mismatch_angle_array)*np.arcsin(abs(sin_mismatch_angle_array))
    d_theta_m_d_tau_array = np.gradient(theta_m_array,tau_array)

    #sign_change_array = np.diff(np.sign(theta_m_array))
    #
    tau_nu_index = find_nearest(theta_m_array, 0) #Finds one of the zeroes. I should do this better and find all the zeroes, but that's for later
    tau_nu = tau_array[tau_nu_index]
    # --


    np.savez('data_output' + output_filename_suffix, tau_array=tau_array, q_R_array=q_R_array, q_zeta_array=q_zeta_array, q_Z_array=q_Z_array,
             K_R_array=K_R_array, K_zeta_array=K_zeta_array, K_Z_array=K_Z_array, K_magnitude_array=K_magnitude_array,
             q_X_array=q_X_array,q_Y_array=q_Y_array,K_X_array=K_X_array,K_Y_array=K_Y_array,
             Psi_w_xx_array=Psi_w_xx_array, Psi_w_xy_array=Psi_w_xy_array, Psi_w_yy_array=Psi_w_yy_array,
             Psi_3D_output_Cartesian=Psi_3D_output_Cartesian,
             tau_0=tau_0, tau_start=tau_start,tau_end=tau_end,
             tau_0_index=tau_0_index,
             tau_nu=tau_nu,tau_nu_index=tau_nu_index,
             g_hat_Cartesian_output=g_hat_Cartesian_output,g_magnitude_output=g_magnitude_output,
             b_hat_Cartesian_output=b_hat_Cartesian_output,B_total_output=B_total_output,
             x_hat_Cartesian_output=x_hat_Cartesian_output,y_hat_Cartesian_output=y_hat_Cartesian_output,
             sin_mismatch_angle_array=sin_mismatch_angle_array,cos_mismatch_angle_array=cos_mismatch_angle_array,
             theta_output=theta_output,K_g_array=K_g_array,
             xhat_dot_grad_bhat_dot_xhat_output=xhat_dot_grad_bhat_dot_xhat_output,
             xhat_dot_grad_bhat_dot_yhat_output=xhat_dot_grad_bhat_dot_yhat_output,
             xhat_dot_grad_bhat_dot_ghat_output=xhat_dot_grad_bhat_dot_ghat_output,
             yhat_dot_grad_bhat_dot_xhat_output=yhat_dot_grad_bhat_dot_xhat_output,
             yhat_dot_grad_bhat_dot_yhat_output=yhat_dot_grad_bhat_dot_yhat_output,
             yhat_dot_grad_bhat_dot_ghat_output=yhat_dot_grad_bhat_dot_ghat_output,
             grad_bhat_output=grad_bhat_output,
             kappa_dot_xhat_output=kappa_dot_xhat_output,
             kappa_dot_yhat_output=kappa_dot_yhat_output,
             d_xhat_d_tau_dot_yhat_output=d_xhat_d_tau_dot_yhat_output,
             d_theta_d_tau_output=d_theta_d_tau_output,
             d_theta_m_d_tau_array=d_theta_m_d_tau_array,
             d_K_g_d_tau_array=d_K_g_d_tau_array
             )

    np.savez('data_input' + output_filename_suffix, tau_step=tau_step, data_poloidal_flux_grid=data_poloidal_flux_grid,
             data_X_coord=data_X_coord, data_Z_coord=data_Z_coord,
             poloidal_launch_angle_Torbeam=poloidal_launch_angle_Torbeam,
             toroidal_launch_angle_Torbeam=toroidal_launch_angle_Torbeam,
             launch_freq_GHz=launch_freq_GHz,
             mode_flag=mode_flag,
             launch_beam_width=launch_beam_width,
             launch_beam_curvature=launch_beam_curvature,
             launch_position=launch_position
             )

    return None
