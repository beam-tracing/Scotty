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

from Scotty_fun_general import read_floats_into_list_until, find_nearest, contract_special, find_H
from Scotty_fun_general import find_inverse_2D, find_Psi_3D_lab, find_q_lab_Cartesian, find_K_lab_Cartesian, find_K_lab, find_Psi_3D_lab_Cartesian
from Scotty_fun_general import find_normalised_plasma_freq, find_normalised_gyro_freq
from Scotty_fun_general import find_epsilon_para, find_epsilon_perp,find_epsilon_g
from Scotty_fun_general import find_dbhat_dR, find_dbhat_dZ
from Scotty_fun_general import find_d_poloidal_flux_dR, find_d_poloidal_flux_dZ,find_Psi_3D_plasma

from Scotty_fun_FFD import find_dH_dR, find_dH_dZ # \nabla H
from Scotty_fun_FFD import find_dH_dKR, find_dH_dKZ, find_dH_dKzeta # \nabla_K H
from Scotty_fun_FFD import find_d2H_dR2, find_d2H_dZ2, find_d2H_dR_dZ # \nabla \nabla H
from Scotty_fun_FFD import find_d2H_dKR2, find_d2H_dKR_dKzeta, find_d2H_dKR_dKZ, find_d2H_dKzeta2, find_d2H_dKzeta_dKZ, find_d2H_dKZ2 # \nabla_K \nabla_K H
from Scotty_fun_FFD import find_d2H_dKR_dR, find_d2H_dKR_dZ, find_d2H_dKzeta_dR, find_d2H_dKzeta_dZ, find_d2H_dKZ_dR, find_d2H_dKZ_dZ # \nabla_K \nabla H

def beam_me_up(tau_step,
               numberOfTauPoints,
               saveInterval,
               poloidal_launch_angle_Torbeam,
               toroidal_launch_angle_Torbeam,
               launch_freq_GHz,
               mode_flag,
               vacuumLaunch_flag,
               launch_beam_width,
               launch_beam_curvature,
               launch_position,
               vacuum_propagation_flag=False,
               Psi_BC_flag = False,
               poloidal_flux_enter=None,
               input_filename_suffix='',
               output_filename_suffix='',
               figure_flag=True,
               plasmaLaunch_K=np.zeros(3),
               plasmaLaunch_Psi_3D_lab_Cartesian=np.zeros([3,3])
               ):


    numberOfDataPoints = (numberOfTauPoints) // saveInterval

    delta_R = -0.0045 #in the same units as data_R_coord
    delta_Z = 0.0045 #in the same units as data_Z_coord
    delta_K_R = 0.1 #in the same units as K_R
    delta_K_zeta = 0.1 #in the same units as K_zeta
    delta_K_Z = 0.1 #in the same units as K_z

    #major_radius = 0.9

    print('Beam trace me up, Scotty!')
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
    #ne_data_fludata_R_coord = np.linspace(0,1,ne_data_length)
    #ne_data_density_array = np.zeros(ne_data_length)
    #ne_data_density_array = n_e_fun(ne_data_fludata_R_coord,core_ne)
    #
    #buffer_factor = 1.5
    #data_R_coord_length = 130
    #data_R_coord_start = major_radius - buffer_factor*minor_radius # in meters
    #data_R_coord_end = major_radius + buffer_factor*minor_radius
    #data_Z_coord_length = 65
    #data_Z_coord_start = -buffer_factor*minor_radius
    #data_Z_coord_end = buffer_factor*minor_radius
    #
    #data_R_coord = np.linspace(data_R_coord_start,data_R_coord_end,data_R_coord_length)
    #data_Z_coord = np.linspace(data_Z_coord_start,data_Z_coord_end,data_Z_coord_length)
    #
    #B_r = np.zeros([data_R_coord_length,data_Z_coord_length])
    #B_z = np.zeros([data_R_coord_length,data_Z_coord_length])
    #B_t = np.zeros([data_R_coord_length,data_Z_coord_length])
    #psi = np.zeros([data_R_coord_length,data_Z_coord_length])
    #
    #x_grid, z_grid = np.meshgrid(data_R_coord, data_Z_coord,indexing='ij')
    #data_B_T_grid = B_toroidal_fun(B_toroidal_max, data_R_coord, data_Z_coord, major_radius)
    #data_B_R_grid = B_r_fun(B_poloidal_max, data_R_coord, data_Z_coord, major_radius, minor_radius)
    #data_B_Z_grid = B_z_fun(B_poloidal_max, data_R_coord, data_Z_coord, major_radius, minor_radius)
    #data_poloidal_flux_grid = psi_fun(x_grid,z_grid, major_radius, minor_radius)
    ## ------------------------------



    # Experimental Profile----------

    input_files_path ='D:\\Dropbox\\VHChen2020\\Data\\Input_Files_29Apr2019\\'
#    input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
#    input_files_path ='D:\\Dropbox\\VHChen2019\\Code - Scotty\\Benchmark_9\\Torbeam\\'
#    input_files_path = os.path.dirname(os.path.abspath(__file__)) + '\\'



    # Importing data from input files
    # ne.dat, topfile
#     Others: inbeam.dat, Te.dat (not currently used in this code)
#    ne_filename = input_files_path + 'ne' +input_filename_suffix+ '_smoothed.dat'
    ne_filename = input_files_path + 'ne' +input_filename_suffix+ '_fitted.dat'
#    ne_filename = input_files_path + 'ne' +input_filename_suffix+ '.dat'

    topfile_filename = input_files_path + 'topfile' +input_filename_suffix


    ne_data = np.fromfile(ne_filename,dtype=float, sep='   ') # electron density as a function of poloidal flux label
    with open(topfile_filename) as f:
        while not 'X-coordinates' in f.readline(): pass # Start reading only from X-coords onwards
        data_R_coord = read_floats_into_list_until('Z-coordinates', f)
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
    ne_data_density_array = 1.05*ne_data[2::2] # in units of 10.0**19 m-3
    print('Warninig: Scale factor of 1.05 used')
    ne_data_radialcoord_array = ne_data[1::2]
    ne_data_poloidal_flux_array = ne_data_radialcoord_array**2 # Loading radial coord for now, makes it easier to benchmark with Torbeam. Hence, have to convert to poloidal flux
#    ne_data_poloidal_flux_array = ne_data[1::2] # My new IDL file outputs the flux density directly, instead of radialcoord

    data_B_R_grid = np.transpose((np.asarray(data_B_R_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
    data_B_T_grid = np.transpose((np.asarray(data_B_T_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
    data_B_Z_grid = np.transpose((np.asarray(data_B_Z_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
    data_poloidal_flux_grid = np.transpose((np.asarray(data_poloidal_flux_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
    # -------------------


    # Interpolation functions declared
    interp_B_R = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_R_grid, bbox=[None, None, None, None], kx=3, ky=3, s=0)
    interp_B_T = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_T_grid, bbox=[None, None, None, None], kx=3, ky=3, s=0)
    interp_B_Z = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_Z_grid, bbox=[None, None, None, None], kx=3, ky=3, s=0)

    interp_poloidal_flux = interpolate.interp2d(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), kind='cubic',
                                           copy=True, bounds_error=False, fill_value=None) # Flux extrapolated outside region

    interp_density_1D = interpolate.interp1d(ne_data_poloidal_flux_array, ne_data_density_array,
                                             kind='linear', axis=-1, copy=True, bounds_error=False,
                                             fill_value=0, assume_sorted=False) # density is 0 outside the LCFS, hence the fill_value














    ## -------------------
    ## Launch parameters    
    ## -------------------
    if vacuumLaunch_flag:
        print('Beam launched from outside the plasma')
        
        K_R_launch    = -wavenumber_K0 * np.cos( toroidal_launch_angle_Torbeam/180.0*math.pi ) * np.cos( poloidal_launch_angle_Torbeam/180.0*math.pi ) # K_R
        K_zeta_launch = -wavenumber_K0 * np.sin( toroidal_launch_angle_Torbeam/180.0*math.pi ) * np.cos( poloidal_launch_angle_Torbeam/180.0*math.pi ) * launch_position[0]# K_zeta
        K_Z_launch    = -wavenumber_K0 * np.sin( poloidal_launch_angle_Torbeam/180.0*math.pi ) # K_z    
        launch_K = np.array([K_R_launch,K_zeta_launch,K_Z_launch])
        
        poloidal_launch_angle_Rz = (180.0+poloidal_launch_angle_Torbeam)/180.0*np.pi
        poloidal_rotation_angle = (90.0+poloidal_launch_angle_Torbeam)/180.0*np.pi
        toroidal_launch_angle_Rz = (180.0+toroidal_launch_angle_Torbeam)/180.0*np.pi 
    
        Psi_w_beam_launch_cartersian = np.array(
                [
                [ wavenumber_K0/launch_beam_curvature+2j*launch_beam_width**(-2), 0],
                [ 0, wavenumber_K0/launch_beam_curvature+2j*launch_beam_width**(-2)]
                ]
                )    
        identity_matrix_2D = np.array(
                [
                [ 1, 0],
                [ 0, 1]
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
        
        rotation_matrix         = np.matmul(rotation_matrix_pol,rotation_matrix_tor)
        rotation_matrix_inverse = np.transpose(rotation_matrix)
        
        Psi_3D_beam_launch_cartersian = np.array([
                [ Psi_w_beam_launch_cartersian[0][0], Psi_w_beam_launch_cartersian[0][1], 0 ],
                [ Psi_w_beam_launch_cartersian[1][0], Psi_w_beam_launch_cartersian[1][1], 0 ],
                [ 0, 0, 0 ]
                ])
        Psi_3D_lab_launch_cartersian = np.matmul( rotation_matrix_inverse, np.matmul(Psi_3D_beam_launch_cartersian, rotation_matrix) )
        Psi_3D_lab_launch = find_Psi_3D_lab(Psi_3D_lab_launch_cartersian,launch_position[0],launch_position[1],K_R_launch,K_zeta_launch)
            
        if vacuum_propagation_flag:
            Psi_w_beam_inverse_launch_cartersian = find_inverse_2D(Psi_w_beam_launch_cartersian)
           
            # Finds entry point
            search_Z_end = launch_position[2] - launch_position[0]*np.tan(np.radians(poloidal_launch_angle_Torbeam))
            numberOfCoarseSearchPoints = 50
            R_coarse_search_array = np.linspace(launch_position[0],0,numberOfCoarseSearchPoints)
            Z_coarse_search_array = np.linspace(launch_position[2],search_Z_end,numberOfCoarseSearchPoints)
            poloidal_flux_coarse_search_array = np.zeros(numberOfCoarseSearchPoints)
            for ii in range(0,numberOfCoarseSearchPoints):
                poloidal_flux_coarse_search_array[ii] = interp_poloidal_flux(R_coarse_search_array[ii],Z_coarse_search_array[ii])
            meets_flux_condition_array = poloidal_flux_coarse_search_array < 0.9*poloidal_flux_enter
            dummy_array = np.array(range(numberOfCoarseSearchPoints))
            indices_inside_for_sure_array = dummy_array[meets_flux_condition_array]
            first_inside_index = indices_inside_for_sure_array[0]
            numberOfFineSearchPoints = 1000
            R_fine_search_array = np.linspace(launch_position[0],R_coarse_search_array[first_inside_index],numberOfFineSearchPoints)
            Z_fine_search_array = np.linspace(launch_position[2],Z_coarse_search_array[first_inside_index],numberOfFineSearchPoints)
            poloidal_fine_search_array = np.zeros(numberOfFineSearchPoints)
            for ii in range(0,numberOfFineSearchPoints):
                poloidal_fine_search_array[ii] = interp_poloidal_flux(R_fine_search_array[ii],Z_fine_search_array[ii])
            entry_index = find_nearest(poloidal_fine_search_array,poloidal_flux_enter)
            entry_position = np.zeros(3) # R,Z
            entry_position[0] = R_fine_search_array[entry_index]
            entry_position[1] = K_zeta_launch/K_R_launch * ( 1/launch_position[0] - 1/entry_position[0] )
            entry_position[2] = Z_fine_search_array[entry_index]
            distance_from_launch_to_entry = np.sqrt(
                                                    launch_position[0]**2 
                                                    + entry_position[0]**2 
                                                    - 2 * launch_position[0] * entry_position[0] * np.cos(entry_position[1] - launch_position[1])
                                                    + (launch_position[2] - entry_position[2])**2
                                                    )
            # Entry point found
            
            
            # Calculate entry parameters from launch parameters
            # That is, find beam at start of plasma given its parameters at the antenna
            K_lab_launch = np.array([K_R_launch,K_zeta_launch,K_Z_launch])
            K_lab_Cartesian_launch = find_K_lab_Cartesian(K_lab_launch,launch_position)
            K_lab_Cartesian_entry = K_lab_Cartesian_launch
            entry_position_Cartesian = find_q_lab_Cartesian(entry_position)
            K_lab_entry = find_K_lab(K_lab_Cartesian_entry,entry_position_Cartesian)
            
            K_R_entry    = K_lab_entry[0] # K_R
            K_zeta_entry = K_lab_entry[1]
            K_Z_entry    = K_lab_entry[2] # K_z
        
            Psi_w_beam_inverse_entry_cartersian = distance_from_launch_to_entry/(wavenumber_K0)*identity_matrix_2D + Psi_w_beam_inverse_launch_cartersian
            Psi_w_beam_entry_cartersian = find_inverse_2D(Psi_w_beam_inverse_entry_cartersian)
        
            Psi_3D_beam_entry_cartersian = np.array([
                    [ Psi_w_beam_entry_cartersian[0][0], Psi_w_beam_entry_cartersian[0][1], 0 ],
                    [ Psi_w_beam_entry_cartersian[1][0], Psi_w_beam_entry_cartersian[1][1], 0 ],
                    [ 0, 0, 0 ]
                    ]) # 'entry' is still in vacuum, so the components of Psi along g are all 0 (since \nabla H = 0)
            
        
            Psi_3D_lab_entry_cartersian = np.matmul( rotation_matrix_inverse, np.matmul(Psi_3D_beam_entry_cartersian, rotation_matrix) )
        
        
            # Convert to cylindrical coordinates
            Psi_3D_lab_entry = find_Psi_3D_lab(Psi_3D_lab_entry_cartersian,entry_position[0],entry_position[1],K_R_entry,K_zeta_entry)

            # -------------------
            # Find initial parameters in plasma
            # -------------------
            K_R_initial        = K_R_entry
            K_zeta_initial     = K_zeta_entry
            K_Z_initial        = K_Z_entry
            initial_position   = entry_position
            if Psi_BC_flag: # Use BCs
                dH_dKR_initial    = find_dH_dKR(initial_position[0], initial_position[2], K_R_initial, K_zeta_initial, K_Z_initial,
                                             launch_angular_frequency, mode_flag, delta_K_R, 
                                             interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
                dH_dKzeta_initial = find_dH_dKzeta(initial_position[0], initial_position[2], K_R_initial, K_zeta_initial, K_Z_initial,
                                             launch_angular_frequency, mode_flag, delta_K_zeta, 
                                             interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z) 
                dH_dKZ_initial    = find_dH_dKZ(initial_position[0], initial_position[2], K_R_initial, K_zeta_initial, K_Z_initial,
                                             launch_angular_frequency, mode_flag, delta_K_Z, 
                                             interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
                dH_dR_initial     = find_dH_dR(initial_position[0], initial_position[2], K_R_initial, K_zeta_initial, K_Z_initial, 
                                            launch_angular_frequency, mode_flag, delta_R, 
                                            interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
                dH_dZ_initial     = find_dH_dZ(initial_position[0], initial_position[2], K_R_initial, K_zeta_initial, K_Z_initial, 
                                            launch_angular_frequency, mode_flag, delta_Z, 
                                            interp_poloidal_flux, interp_density_1D, interp_B_R, interp_B_T, interp_B_Z)
                d_poloidal_flux_d_R_boundary = find_d_poloidal_flux_dR(initial_position[0], initial_position[2], delta_R, interp_poloidal_flux)
                d_poloidal_flux_d_Z_boundary = find_d_poloidal_flux_dZ(initial_position[0], initial_position[2], delta_R, interp_poloidal_flux)
                
                Psi_3D_lab_initial = find_Psi_3D_plasma(Psi_3D_lab_entry,
                                                        dH_dKR_initial, dH_dKzeta_initial, dH_dKZ_initial,
                                                        dH_dR_initial, dH_dZ_initial,
                                                        d_poloidal_flux_d_R_boundary, d_poloidal_flux_d_Z_boundary)
            else: # Do not use BCs
                Psi_3D_lab_initial = Psi_3D_lab_entry
            
        else: #Run solver from the launch position, no analytical vacuum propagation
            Psi_3D_lab_initial = Psi_3D_lab_launch
            K_R_initial        = K_R_launch
            K_zeta_initial     = K_zeta_launch
            K_Z_initial        = K_Z_launch
            initial_position   = launch_position            
            
            distance_from_launch_to_entry=None
            Psi_3D_lab_entry_cartersian = np.full_like(Psi_3D_lab_launch,fill_value=np.nan)
            

    else:
        print('Beam launched from inside the plasma')
        K_R_launch = plasmaLaunch_K[0]
        K_zeta_launch = plasmaLaunch_K[1]
        K_Z_launch = plasmaLaunch_K[2]
        
        Psi_3D_lab_initial = find_Psi_3D_lab(plasmaLaunch_Psi_3D_lab_Cartesian,launch_position[0],plasmaLaunch_K[0],plasmaLaunch_K[1])
        K_R_initial        = K_R_launch
        K_zeta_initial     = K_zeta_launch
        K_Z_initial        = K_Z_launch
        initial_position   = launch_position
        
#        print(K_R_initial)
#        print(K_zeta_launch)
#        print(K_Z_launch)






















    # Initialise
    tau_array = np.zeros(numberOfDataPoints)

    q_R_array  = np.zeros(numberOfDataPoints)
    q_zeta_array  = np.zeros(numberOfDataPoints)
    q_Z_array  = np.zeros(numberOfDataPoints)

    K_R_array = np.zeros(numberOfDataPoints)
    K_Z_array = np.zeros(numberOfDataPoints)
    # -------------------


    # -------------------

    # Initialise the arrays for the solver
        # Euler
    #bufferSize = 2
    #coefficient_array = np.array([1])
        # AB3
    bufferSize = 4
    coefficient_array = np.array([23/12,-4/3,5/12])

    q_R_buffer    = np.zeros(bufferSize)
    q_zeta_buffer = np.zeros(bufferSize)
    q_Z_buffer    = np.zeros(bufferSize)
    K_R_buffer    = np.zeros(bufferSize)
    K_Z_buffer    = np.zeros(bufferSize)

    dH_dR_buffer     = np.zeros(bufferSize)
    dH_dZ_buffer     = np.zeros(bufferSize)
    dH_dKR_buffer    = np.zeros(bufferSize)
    dH_dKzeta_buffer = np.zeros(bufferSize)
    dH_dKZ_buffer    = np.zeros(bufferSize)

    Psi_3D_buffer        = np.zeros([bufferSize,3,3],dtype='complex128')
    grad_grad_H_buffer   = np.zeros([bufferSize,3,3])
    gradK_grad_H_buffer  = np.zeros([bufferSize,3,3])
    grad_gradK_H_buffer  = np.zeros([bufferSize,3,3])
    gradK_gradK_H_buffer = np.zeros([bufferSize,3,3])


    current_marker = 1
    tau_current    = tau_array[0]
    output_counter = 0


    for index in range(0,bufferSize):
        q_R_buffer[index]        = initial_position[0] #
        q_zeta_buffer[index]     = initial_position[1]
        q_Z_buffer[index]        = initial_position[2] # r_Z
        K_R_buffer[index]        = K_R_initial # K_R
        K_Z_buffer[index]        = K_Z_initial # K_z
        Psi_3D_buffer[index,:,:] = Psi_3D_lab_initial

    # -------------------

    # -------------------

    # Initialise the results arrays
    dH_dKR_output    = np.zeros(numberOfDataPoints)
    dH_dKzeta_output = np.zeros(numberOfDataPoints)
    dH_dKZ_output    = np.zeros(numberOfDataPoints)
    dH_dR_output     = np.zeros(numberOfDataPoints)
    dH_dZ_output     = np.zeros(numberOfDataPoints)

    grad_grad_H_output   = np.zeros([numberOfDataPoints,3,3])
    gradK_grad_H_output  = np.zeros([numberOfDataPoints,3,3])
    gradK_gradK_H_output = np.zeros([numberOfDataPoints,3,3])
    Psi_3D_output        = np.zeros([numberOfDataPoints,3,3],dtype='complex128')

    g_hat_output = np.zeros([numberOfDataPoints,3])
    b_hat_output = np.zeros([numberOfDataPoints,3])
    y_hat_output = np.zeros([numberOfDataPoints,3])
    x_hat_output = np.zeros([numberOfDataPoints,3])

    g_magnitude_output = np.zeros(numberOfDataPoints)
    grad_bhat_output   = np.zeros([numberOfDataPoints,3,3])

#    x_hat_Cartesian_output             = np.zeros([numberOfDataPoints,3])
#    y_hat_Cartesian_output             = np.zeros([numberOfDataPoints,3])
#    b_hat_Cartesian_output             = np.zeros([numberOfDataPoints,3])
#    xhat_dot_grad_bhat_dot_xhat_output = np.zeros(numberOfDataPoints)
#    xhat_dot_grad_bhat_dot_yhat_output = np.zeros(numberOfDataPoints)
#    xhat_dot_grad_bhat_dot_ghat_output = np.zeros(numberOfDataPoints)
#    yhat_dot_grad_bhat_dot_xhat_output = np.zeros(numberOfDataPoints)
#    yhat_dot_grad_bhat_dot_yhat_output = np.zeros(numberOfDataPoints)
#    yhat_dot_grad_bhat_dot_ghat_output = np.zeros(numberOfDataPoints)
#    d_theta_d_tau_output               = np.zeros(numberOfDataPoints)
#    d_xhat_d_tau_output                = np.zeros([numberOfDataPoints,3])
#    d_xhat_d_tau_dot_yhat_output       = np.zeros(numberOfDataPoints)
#    ray_curvature_kappa_output         = np.zeros([numberOfDataPoints,3])
#    kappa_dot_xhat_output              = np.zeros(numberOfDataPoints)
#    kappa_dot_yhat_output              = np.zeros(numberOfDataPoints)


    B_total_output = np.zeros(numberOfDataPoints)

    H_output                      = np.zeros(numberOfDataPoints)
    Booker_alpha_output           = np.zeros(numberOfDataPoints)
    Booker_beta_output            = np.zeros(numberOfDataPoints)
    Booker_gamma_output           = np.zeros(numberOfDataPoints)
    epsilon_para_output           = np.zeros(numberOfDataPoints)
    epsilon_perp_output           = np.zeros(numberOfDataPoints)
    epsilon_g_output              = np.zeros(numberOfDataPoints)
    poloidal_flux_output          = np.zeros(numberOfDataPoints)
    d_poloidal_flux_dR_output     = np.zeros(numberOfDataPoints)
    d_poloidal_flux_dZ_output     = np.zeros(numberOfDataPoints)
    normalised_gyro_freq_output   = np.zeros(numberOfDataPoints)
    normalised_plasma_freq_output = np.zeros(numberOfDataPoints)

    electron_density_output = np.zeros(numberOfDataPoints)

#    eigenvalues_output = np.zeros([numberOfDataPoints,3],dtype='complex128')
#    eigenvectors_output = np.zeros([numberOfDataPoints,3,3],dtype='complex128')

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
        grad_grad_H_buffer[current_marker-1,:,:] = np.squeeze(np.array([
            [d2H_dR2.item()  , 0.0, d2H_dR_dZ.item()],
            [0.0             , 0.0, 0.0             ],
            [d2H_dR_dZ.item(), 0.0, d2H_dZ2.item()  ] #. item() to convert variable from type ndarray to float, such that the array elements all have the same type
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
        gradK_grad_H_buffer[current_marker-1,:,:] = np.squeeze(np.array([
            [d2H_dKR_dR.item(),    0.0, d2H_dKR_dZ.item()   ],
            [d2H_dKzeta_dR.item(), 0.0, d2H_dKzeta_dZ.item()],
            [d2H_dKZ_dR.item(),    0.0, d2H_dKZ_dZ.item()   ]
            ]))
        grad_gradK_H_buffer[current_marker-1,:,:] = np.transpose(gradK_grad_H_buffer[current_marker-1,:,:])

        # \nabla_K \nabla_K H
        d2H_dKR2       = find_d2H_dKR2(
                             q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                             K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                             launch_angular_frequency,mode_flag,delta_K_R,
                             interp_poloidal_flux,interp_density_1D,
                             interp_B_R,interp_B_T,interp_B_Z
                             )
        d2H_dKzeta2    = find_d2H_dKzeta2(
                                q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                                K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                                launch_angular_frequency,mode_flag,delta_K_zeta,
                             interp_poloidal_flux,interp_density_1D,
                             interp_B_R,interp_B_T,interp_B_Z
                                )
        d2H_dKZ2       = find_d2H_dKZ2(
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
        d2H_dKR_dKZ    = find_d2H_dKR_dKZ(
                              q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                              K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                              launch_angular_frequency,mode_flag,delta_K_R,delta_K_Z,
                              interp_poloidal_flux,interp_density_1D,
                              interp_B_R,interp_B_T,interp_B_Z
                              )
        d2H_dKzeta_dKZ = find_d2H_dKzeta_dKZ(
                               q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1],
                               K_R_buffer[current_marker-1],K_zeta,K_Z_buffer[current_marker-1],
                               launch_angular_frequency,mode_flag,delta_K_zeta,delta_K_Z,
                               interp_poloidal_flux,interp_density_1D,
                               interp_B_R,interp_B_T,interp_B_Z
                               )
        gradK_gradK_H_buffer[current_marker-1,:,:] = np.squeeze(np.array([
            [d2H_dKR2.item()      , d2H_dKR_dKzeta.item(), d2H_dKR_dKZ.item()   ],
            [d2H_dKR_dKzeta.item(), d2H_dKzeta2.item()   , d2H_dKzeta_dKZ.item()],
            [d2H_dKR_dKZ.item()   , d2H_dKzeta_dKZ.item(), d2H_dKZ2.item()      ]
            ]))



        q_R_buffer[current_marker]        = q_R_buffer[current_marker-1]
        q_zeta_buffer[current_marker]     = q_zeta_buffer[current_marker-1]
        q_Z_buffer[current_marker]        = q_Z_buffer[current_marker-1]
        K_R_buffer[current_marker]        = K_R_buffer[current_marker-1]
        K_Z_buffer[current_marker]        = K_Z_buffer[current_marker-1]
        Psi_3D_buffer[current_marker,:,:] = Psi_3D_buffer[current_marker-1,:,:]


        
        for coefficient_index in range(0,bufferSize-1):
            q_R_buffer[current_marker]    +=  coefficient_array[coefficient_index] * dH_dKR_buffer[current_marker-1-coefficient_index]    * tau_step
            q_zeta_buffer[current_marker] +=  coefficient_array[coefficient_index] * dH_dKzeta_buffer[current_marker-1-coefficient_index] * tau_step
            q_Z_buffer[current_marker]    +=  coefficient_array[coefficient_index] * dH_dKZ_buffer[current_marker-1-coefficient_index]    * tau_step
            K_R_buffer[current_marker]    += -coefficient_array[coefficient_index] * dH_dR_buffer[current_marker-1-coefficient_index]     * tau_step
            K_Z_buffer[current_marker]    += -coefficient_array[coefficient_index] * dH_dZ_buffer[current_marker-1-coefficient_index]     * tau_step

            Psi_3D_buffer[current_marker,:,:] += coefficient_array[coefficient_index] * (
                    - grad_grad_H_buffer[current_marker-1-coefficient_index,:,:]
                    - np.matmul(
                                Psi_3D_buffer[current_marker-1-coefficient_index,:,:],
                                gradK_grad_H_buffer[current_marker-1-coefficient_index,:,:]
                                )
                    - np.matmul(
                                grad_gradK_H_buffer[current_marker-1-coefficient_index,:,:],
                                Psi_3D_buffer[current_marker-1-coefficient_index,:,:]
                                )
                    - np.matmul(np.matmul(
                                Psi_3D_buffer[current_marker-1-coefficient_index,:,:],
                                gradK_gradK_H_buffer[current_marker-1-coefficient_index,:,:]),
                                Psi_3D_buffer[current_marker-1-coefficient_index,:,:]
                                )
                    )* tau_step




        if step % saveInterval == 0: # Write data to output arrays
            tau_array[output_counter] = tau_current

            q_R_array [output_counter]    = q_R_buffer[current_marker-1]
            q_zeta_array [output_counter] = q_zeta_buffer[current_marker-1]
            q_Z_array [output_counter]    = q_Z_buffer[current_marker-1]
            K_R_array[output_counter]     = K_R_buffer[current_marker-1]
            K_Z_array[output_counter]     = K_Z_buffer[current_marker-1]

            dH_dKR_output[output_counter]    = dH_dKR_buffer[current_marker-1]
            dH_dKzeta_output[output_counter] = dH_dKzeta_buffer[current_marker-1]
            dH_dKZ_output[output_counter]    = dH_dKZ_buffer[current_marker-1]
            dH_dR_output[output_counter]     = dH_dR_buffer[current_marker-1]
            dH_dZ_output[output_counter]     = dH_dZ_buffer[current_marker-1]

            Psi_3D_output[output_counter,:,:]        = Psi_3D_buffer[current_marker-1,:,:]
            grad_grad_H_output[output_counter,:,:]   = grad_grad_H_buffer[current_marker-1,:,:]
            gradK_grad_H_output[output_counter,:,:]  = gradK_grad_H_buffer[current_marker-1,:,:]
            gradK_gradK_H_output[output_counter,:,:] = gradK_gradK_H_buffer[current_marker-1,:,:]

            electron_density_output[output_counter] = electron_density
            B_R = np.squeeze(interp_B_R(q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1]))
            B_T = np.squeeze(interp_B_T(q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1]))
            B_Z = np.squeeze(interp_B_Z(q_R_buffer[current_marker-1],q_Z_buffer[current_marker-1]))

            B_total     = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
            b_hat       = np.array([B_R,B_T,B_Z]) / B_total
            g_magnitude = (q_R_buffer[current_marker-1]**2 * dH_dKzeta_output[output_counter]**2 + dH_dKR_output[output_counter]**2 + dH_dKZ_output[output_counter]**2)**0.5
            g_hat_R     = dH_dKR_output[output_counter]    / g_magnitude
            g_hat_zeta  = q_R_buffer[current_marker-1] * dH_dKzeta_output[output_counter] / g_magnitude
            g_hat_Z     = dH_dKZ_output[output_counter]    / g_magnitude
            g_hat = np.asarray([g_hat_R,g_hat_zeta,g_hat_Z])
            y_hat = np.cross(b_hat,g_hat) / (np.linalg.norm(np.cross(b_hat,g_hat)))
            x_hat = np.cross(g_hat,y_hat) / (np.linalg.norm(np.cross(g_hat,y_hat)))
                
            g_magnitude_output[output_counter] = g_magnitude
            g_hat_output[output_counter,:] = g_hat
            b_hat_output[output_counter,:] = b_hat
            B_total_output[output_counter] = B_total
            y_hat_output[output_counter,:] = y_hat
            x_hat_output[output_counter,:] = x_hat

            # Calculating the corrections to Psi_w
            dbhat_dR = find_dbhat_dR(q_R_buffer[current_marker-1], q_Z_buffer[current_marker-1], delta_R, interp_B_R, interp_B_T, interp_B_Z)
            dbhat_dZ = find_dbhat_dZ(q_R_buffer[current_marker-1], q_Z_buffer[current_marker-1], delta_Z, interp_B_R, interp_B_T, interp_B_Z)

            grad_bhat_output[output_counter,:,0] = dbhat_dR
            grad_bhat_output[output_counter,:,2] = dbhat_dZ
            grad_bhat_output[output_counter,1,1] = B_R / (B_total * q_R_buffer[current_marker-1])
            grad_bhat_output[output_counter,0,1] = - B_T / (B_total * q_R_buffer[current_marker-1])
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
            epsilon_para_output[output_counter]  = find_epsilon_para(electron_density,launch_angular_frequency)
            epsilon_perp_output[output_counter]  = find_epsilon_perp(electron_density,B_total,launch_angular_frequency)
            epsilon_g_output[output_counter]  = find_epsilon_g(electron_density,B_total,launch_angular_frequency)

            normalised_gyro_freq_output[output_counter]   = find_normalised_gyro_freq(B_total,launch_angular_frequency)
            normalised_plasma_freq_output[output_counter] = find_normalised_plasma_freq(electron_density,launch_angular_frequency)

            poloidal_flux_output[output_counter]      = poloidal_flux
            d_poloidal_flux_dR_output[output_counter] = find_d_poloidal_flux_dR(q_R_buffer[current_marker-1], q_Z_buffer[current_marker-1], delta_R, interp_poloidal_flux)
            d_poloidal_flux_dZ_output[output_counter] = find_d_poloidal_flux_dZ(q_R_buffer[current_marker-1], q_Z_buffer[current_marker-1], delta_R, interp_poloidal_flux)
            
            output_counter += 1

        tau_current += tau_step
        current_marker = (current_marker + 1) % bufferSize
    #K_zeta_array[:] = K_zeta_initial / q_R_array [:]
    print('Main loop complete')
    # -------------------



    ## -------------------
    ## This saves the data generated by the main loop and the input data
    ## Input data saved at this point in case something is changed between loading and the end of the main loop, this allows for comparison
    ## The rest of the data is save further down, after the analysis generates them.
    ## Just in case the analysis fails to run, at least one can get the data from the main loop
    ## -------------------
    print('Saving data')


    if vacuumLaunch_flag:
        np.savez('data_input' + output_filename_suffix, tau_step=tau_step, data_poloidal_flux_grid=data_poloidal_flux_grid,
                 data_R_coord=data_R_coord, data_Z_coord=data_Z_coord,
                 poloidal_launch_angle_Torbeam=poloidal_launch_angle_Torbeam,
                 toroidal_launch_angle_Torbeam=toroidal_launch_angle_Torbeam,
                 launch_freq_GHz=launch_freq_GHz,
                 mode_flag=mode_flag,
                 launch_beam_width=launch_beam_width,
                 launch_beam_curvature=launch_beam_curvature,
                 launch_position=launch_position,
                 launch_K=launch_K,
                 ne_data_density_array=ne_data_density_array,ne_data_radialcoord_array=ne_data_radialcoord_array
                 )    
        np.savez('data_output' + output_filename_suffix, 
                 tau_array=tau_array, q_R_array=q_R_array, q_zeta_array=q_zeta_array, q_Z_array=q_Z_array,
                 K_R_array=K_R_array, K_zeta_initial=K_zeta_initial, K_Z_array=K_Z_array,
                 Psi_3D_output=Psi_3D_output, Psi_3D_lab_launch=Psi_3D_lab_launch,
#                 Psi_3D_lab_entry=Psi_3D_lab_entry,
                 distance_from_launch_to_entry=distance_from_launch_to_entry,
                 g_hat_output=g_hat_output,g_magnitude_output=g_magnitude_output,
                 B_total_output=B_total_output,
                 x_hat_output=x_hat_output,y_hat_output=y_hat_output,
                 b_hat_output=b_hat_output,
                 grad_bhat_output=grad_bhat_output,
                 dH_dKR_output=dH_dKR_output,dH_dKzeta_output=dH_dKzeta_output,dH_dKZ_output=dH_dKZ_output,
                 dH_dR_output=dH_dR_output,dH_dZ_output=dH_dZ_output,
                 grad_grad_H_output=grad_grad_H_output,gradK_grad_H_output=gradK_grad_H_output,gradK_gradK_H_output=gradK_gradK_H_output,
                 d_poloidal_flux_dR_output=d_poloidal_flux_dR_output,
                 d_poloidal_flux_dZ_output=d_poloidal_flux_dZ_output,
                 epsilon_para_output=epsilon_para_output,epsilon_perp_output=epsilon_perp_output,epsilon_g_output=epsilon_g_output,
                 electron_density_output=electron_density_output,H_output=H_output
                 )
    else:
         np.savez('data_input' + output_filename_suffix, tau_step=tau_step, data_poloidal_flux_grid=data_poloidal_flux_grid,
                 data_R_coord=data_R_coord, data_Z_coord=data_Z_coord,
                 launch_freq_GHz=launch_freq_GHz,
                 mode_flag=mode_flag,
                 launch_position=launch_position,
                 plasmaLaunch_K=plasmaLaunch_K,
                 plasmaLaunch_Psi_3D_lab_Cartesian=plasmaLaunch_Psi_3D_lab_Cartesian,
                 ne_data_density_array=ne_data_density_array,ne_data_radialcoord_array=ne_data_radialcoord_array
                 )  
         np.savez('data_output' + output_filename_suffix, 
                  tau_array=tau_array, q_R_array=q_R_array, q_zeta_array=q_zeta_array, q_Z_array=q_Z_array,
                  K_R_array=K_R_array, K_zeta_initial=K_zeta_initial, K_Z_array=K_Z_array,
                  Psi_3D_output=Psi_3D_output, Psi_3D_lab_launch=Psi_3D_lab_launch,
                  g_hat_output=g_hat_output,g_magnitude_output=g_magnitude_output,
                  B_total_output=B_total_output,
                  x_hat_output=x_hat_output,y_hat_output=y_hat_output,
                  b_hat_output=b_hat_output,
                  grad_bhat_output=grad_bhat_output,
                  dH_dKR_output=dH_dKR_output,dH_dKzeta_output=dH_dKzeta_output,dH_dKZ_output=dH_dKZ_output,
                  dH_dR_output=dH_dR_output,dH_dZ_output=dH_dZ_output,
                  grad_grad_H_output=grad_grad_H_output,gradK_grad_H_output=gradK_grad_H_output,gradK_gradK_H_output=gradK_gradK_H_output,
                  d_poloidal_flux_dR_output=d_poloidal_flux_dR_output,
                  d_poloidal_flux_dZ_output=d_poloidal_flux_dZ_output,
                  epsilon_para_output=epsilon_para_output,epsilon_perp_output=epsilon_perp_output,epsilon_g_output=epsilon_g_output,
                  electron_density_output=electron_density_output,H_output=H_output
                  )     
        
    print('Data saved')
    # -------------------



    ## -------------------
    ## Process the data from the main loop to give a bunch of useful stuff
    ## -------------------
    print('Analysing data')
    
        # Calculates various useful stuff
    [q_X_array,q_Y_array,_] = find_q_lab_Cartesian([q_R_array,q_zeta_array,q_Z_array])
    point_spacing =  ( (np.diff(q_X_array))**2 + (np.diff(q_Y_array))**2 + (np.diff(q_Z_array))**2  )**0.5
    distance_along_line =  np.cumsum(point_spacing)
    distance_along_line = np.append(0,distance_along_line)
    RZ_point_spacing = ( (np.diff(q_Z_array))**2 + (np.diff(q_R_array))**2  )**0.5
    RZ_distance_along_line =  np.cumsum(RZ_point_spacing)
    RZ_distance_along_line = np.append(0,RZ_distance_along_line)
    
        # Calculates the index of the minimum magnitude of K
        # That is, finds when the beam hits the cut-off
    K_magnitude_array = (K_R_array**2 + K_zeta_initial**2/q_R_array**2 + K_Z_array**2)**(0.5)        
        
    cutoff_index = find_nearest(abs(K_magnitude_array),  0) # Index of the cutoff, at the minimum value of K, use this with other arrays

        # Calculates when the beam 'enters' and 'leaves' the plasma
        # Here entry and exit refer to cross the LCFS, poloidal_flux = 1.0
    in_out_poloidal_flux = 1.0
    poloidal_flux_a = poloidal_flux_output[0:cutoff_index]
    poloidal_flux_b = poloidal_flux_output[cutoff_index::]
    in_index = find_nearest(poloidal_flux_a,in_out_poloidal_flux)
    out_index = cutoff_index + find_nearest(poloidal_flux_b,in_out_poloidal_flux)


    
        # Calcuating the corrections to make M from Psi
        # Corrections that are small in mismatch are ignored in the calculation of M
        # However, these small terms are still calculated in this section, so that their actual size can be checked, if need be
    ray_curvature_kappa_output         = np.zeros([numberOfDataPoints,3])

    k_perp_1_backscattered = -2*K_magnitude_array

    y_hat_Cartesian = np.zeros([numberOfDataPoints,3])
    x_hat_Cartesian = np.zeros([numberOfDataPoints,3])
    y_hat_Cartesian[:,0] = y_hat_output[:,0]*np.cos(q_zeta_array ) - y_hat_output[:,1]*np.sin(q_zeta_array )
    y_hat_Cartesian[:,1] = y_hat_output[:,0]*np.sin(q_zeta_array ) + y_hat_output[:,1]*np.cos(q_zeta_array )
    y_hat_Cartesian[:,2] = y_hat_output[:,2]
    x_hat_Cartesian[:,0] = x_hat_output[:,0]*np.cos(q_zeta_array ) - x_hat_output[:,1]*np.sin(q_zeta_array )
    x_hat_Cartesian[:,1] = x_hat_output[:,0]*np.sin(q_zeta_array ) + x_hat_output[:,1]*np.cos(q_zeta_array )
    x_hat_Cartesian[:,2] = x_hat_output[:,2]
    
    Psi_3D_Cartesian = find_Psi_3D_lab_Cartesian(Psi_3D_output, q_R_array, q_zeta_array, K_R_array, K_zeta_initial)
    Psi_xx_output = contract_special(x_hat_Cartesian,contract_special(Psi_3D_Cartesian,x_hat_Cartesian))
    Psi_xy_output = contract_special(x_hat_Cartesian,contract_special(Psi_3D_Cartesian,y_hat_Cartesian))
    Psi_yy_output = contract_special(y_hat_Cartesian,contract_special(Psi_3D_Cartesian,y_hat_Cartesian))
    
    Psi_xx_entry = np.dot(x_hat_Cartesian[0,:],np.dot(Psi_3D_lab_entry_cartersian,x_hat_Cartesian[0,:]))
    Psi_xy_entry = np.dot(x_hat_Cartesian[0,:],np.dot(Psi_3D_lab_entry_cartersian,y_hat_Cartesian[0,:]))
    Psi_yy_entry = np.dot(y_hat_Cartesian[0,:],np.dot(Psi_3D_lab_entry_cartersian,y_hat_Cartesian[0,:]))

    xhat_dot_grad_bhat = contract_special(x_hat_output,grad_bhat_output)
    yhat_dot_grad_bhat = contract_special(y_hat_output,grad_bhat_output)
    ray_curvature_kappa_output[:,0] = (1/g_magnitude_output) * np.gradient(g_hat_output[:,0],tau_array)
    ray_curvature_kappa_output[:,1] = (1/g_magnitude_output) * np.gradient(g_hat_output[:,1],tau_array)
    ray_curvature_kappa_output[:,2] = (1/g_magnitude_output) * np.gradient(g_hat_output[:,2],tau_array)
        
    xhat_dot_grad_bhat_dot_xhat_output = contract_special(xhat_dot_grad_bhat,x_hat_output)
    xhat_dot_grad_bhat_dot_yhat_output = contract_special(xhat_dot_grad_bhat,y_hat_output)
    xhat_dot_grad_bhat_dot_ghat_output = contract_special(xhat_dot_grad_bhat,g_hat_output)
    yhat_dot_grad_bhat_dot_xhat_output = contract_special(yhat_dot_grad_bhat,x_hat_output)
    yhat_dot_grad_bhat_dot_yhat_output = contract_special(yhat_dot_grad_bhat,y_hat_output)
    yhat_dot_grad_bhat_dot_ghat_output = contract_special(yhat_dot_grad_bhat,g_hat_output)
    kappa_dot_xhat_output = contract_special(ray_curvature_kappa_output,xhat_dot_grad_bhat)
    kappa_dot_yhat_output = contract_special(ray_curvature_kappa_output,yhat_dot_grad_bhat)
    # TODO: Calculate the other small corrections
    
    M_xx_output = Psi_xx_output + (k_perp_1_backscattered/2) * xhat_dot_grad_bhat_dot_ghat_output
    M_xy_output = Psi_xy_output + (k_perp_1_backscattered/2) * yhat_dot_grad_bhat_dot_ghat_output
    M_yy_output = Psi_yy_output
    
        # Calculates the localisation, wavenumber resolution, and mismatch attenuation pieces
    det_M_w = M_xx_output*M_yy_output - M_xy_output**2
    M_w_inv_xx_output =   M_yy_output / det_M_w
    M_w_inv_xy_output = - M_xy_output / det_M_w
    M_w_inv_yy_output =   M_xx_output / det_M_w
    
    delta_k_perp_2 = np.sqrt( - 2 / np.imag(M_w_inv_yy_output) )
    delta_theta_m  = np.sqrt( 
                              np.imag(M_w_inv_yy_output) / ( (np.imag(M_w_inv_xy_output))**2 - np.imag(M_w_inv_xx_output)*np.imag(M_w_inv_yy_output) ) 
                            ) / (np.sqrt(2) * K_magnitude_array)
#    print(delta_theta_m[cutoff_index])
    
    sin_theta_m_analysis = np.zeros(numberOfDataPoints)
    sin_theta_m_analysis[:] = (b_hat_output[:,0]*K_R_array[:] + b_hat_output[:,1]*K_zeta_initial/q_R_array[:] + b_hat_output[:,2]*K_Z_array[:]) / (K_magnitude_array[:]) # B \cdot K / (abs (B) abs(K))
    theta_m_output = np.sign(sin_theta_m_analysis)*np.arcsin(abs(sin_theta_m_analysis)) # Assumes the mismatch angle is never smaller than -90deg or bigger than 90deg
#    print(theta_m_output[cutoff_index])
#    print(cutoff_index)
    
#    g_magnitude_launch = find_g_magnitude(launch_position[0],launch_position[1],K_R_launch,K_zeta_launch,K_Z_launch,
#                                          launch_angular_frequency,mode_flag,delta_K_R,delta_K_zeta,delta_K_Z,
#                                          interp_poloidal_flux,interp_density_1D,interp_B_R,interp_B_T,interp_B_Z)
#    d_K_d_tau_analysis = np.gradient(K_magnitude_array,tau_array)
#    localisation_piece = g_magnitude_launch**2/abs(g_magnitude_output*d_K_d_tau_analysis)
    # -------------------



    ## -------------------
    ## Cartesian check
    ## -------------------

























    ## -------------------
    ## This saves the data generated by the analysis after the main loop
    ## -------------------
    print('Saving analysis data')
    np.savez('analysis_output' + output_filename_suffix, 
             Psi_xx_output = Psi_xx_output, Psi_xy_output = Psi_xy_output, Psi_yy_output = Psi_yy_output,
             Psi_xx_entry=Psi_xx_entry, Psi_xy_entry=Psi_xy_entry, Psi_yy_entry=Psi_yy_entry,
             M_xx_output = M_xx_output, M_xy_output = M_xy_output, M_yy_output = M_yy_output,
             xhat_dot_grad_bhat_dot_xhat_output=xhat_dot_grad_bhat_dot_xhat_output,
             xhat_dot_grad_bhat_dot_yhat_output=xhat_dot_grad_bhat_dot_yhat_output,
             xhat_dot_grad_bhat_dot_ghat_output=xhat_dot_grad_bhat_dot_ghat_output,
             yhat_dot_grad_bhat_dot_xhat_output=yhat_dot_grad_bhat_dot_xhat_output,
             yhat_dot_grad_bhat_dot_yhat_output=yhat_dot_grad_bhat_dot_yhat_output,
             yhat_dot_grad_bhat_dot_ghat_output=yhat_dot_grad_bhat_dot_ghat_output,
             kappa_dot_xhat_output=kappa_dot_xhat_output,
             kappa_dot_yhat_output=kappa_dot_yhat_output,
             delta_k_perp_2=delta_k_perp_2, delta_theta_m=delta_theta_m,
             theta_m_output=theta_m_output,
             RZ_distance_along_line=RZ_distance_along_line,
             distance_along_line=distance_along_line,
             k_perp_1_backscattered = k_perp_1_backscattered,K_magnitude_array=K_magnitude_array,
             cutoff_index=cutoff_index,in_index=in_index,out_index=out_index
             )
#    print('Analysis data saved')
    # -------------------

    ## -------------------
    ## This saves some simple figures
    ## Allows one to quickly gain an insight into what transpired in the simulation
    ## -------------------
    if figure_flag:
        print('Making figures')
        
        # Plots the beam path on the R Z plane
        plt.figure()
        plt.title('Rz')
        plt.xlabel('R / m') # x-direction
        plt.ylabel('z / m')
    
        contour_levels = np.linspace(0,1.3,27)
        CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
        plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
        plt.plot(
                np.concatenate([[launch_position[0],initial_position[0]],q_R_array ]),
                np.concatenate([[launch_position[2],initial_position[2]],q_Z_array ]),
                '--.k') # Central (reference) ray
        #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
        #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
        plt.xlim(data_R_coord[0],data_R_coord[-1])
        plt.ylim(data_Z_coord[0],data_Z_coord[-1])
        plt.savefig('Ray1_' + output_filename_suffix)
        plt.close()
        

        print('Figures have been saved')
    ## -------------------


    return None
