# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com


For shot 29908, the EFIT++ times are efit_times = np.linspace(0.155,0.25,20)
I want efit_times[np.arange(0,10)*2 + 1]. 160ms, 170ms, ..., 250ms
"""
from Scotty_beam_me_up import beam_me_up
from Scotty_fun_general import find_q_lab_Cartesian, find_q_lab, find_K_lab_Cartesian, find_K_lab, find_waist, find_Rayleigh_length, genray_angles_from_mirror_angles

from scipy import constants
import math
import numpy as np

# from joblib import Parallel, delayed
# from numba import njit, prange


# input_filename_suffix = '_29905_190'
#input_filename_suffix = ''

poloidal_launch_angle_Torbeam = 4.0 # deg
toroidal_launch_angle_Torbeam = -2.0 # deg

# poloidal_launch_angle_Torbeam = 8.30 # deg
# toroidal_launch_angle_Torbeam = -0.31 # deg

# rotation_angles_array = np.array([7.0,8,9,5,4,6]) 
# mirror_rotation_angle_scan = np.linspace(-1,-7,31)
# mirror_rotation_angle_scan = np.linspace(-1,-7,13)
# mirror_rotation_angle = -1.3
# mirror_tilt_angle = -3.0
# mirror_tilt_angle = 0



# launch_freq_GHz_sweep = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0,55.0,57.5,60.0,62.5,67.5,70.0,72.5,75.0])
# launch_freq_GHz_sweep = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0])
launch_freq_GHz = 55.0
mode_flag = 1 # O-mode (1) or X-mode (-1)

# beam_rayleigh_distance = 0.885
# beam_waist = np.sqrt(beam_rayleigh_distance * constants.c / (np.pi * launch_freq_GHz*10**9) )

# beam_waist = 0.0392 * np.sqrt(2)
# launch_distance_from_waist = np.linspace(-beam_rayleigh_distance*10,-beam_rayleigh_distance,81)
# launch_beam_width_sweep = beam_waist * np.sqrt( 1 + (launch_distance_from_waist/beam_rayleigh_distance)**2 )
# launch_beam_radius_of_curvature_sweep = launch_distance_from_waist*( 1 + (beam_rayleigh_distance/launch_distance_from_waist)**2 )
# launch_beam_width = launch_beam_width_sweep[0] # in m
# launch_beam_radius_of_curvature = launch_beam_radius_of_curvature_sweep[0]
# launch_beam_width = 0.0554 # in m
# launch_beam_radius_of_curvature = -1.77 # in m. negative because launched BEFORE the beam waist
# launch_beam_width = beam_waist * np.sqrt(2) # in m
# launch_beam_radius_of_curvature = -2*beam_rayleigh_distance # in m. negative because launched BEFORE the beam waist

## These values are from my beam-fitting routines (but simplified)
launch_beam_width = 0.05
launch_beam_radius_of_curvature = 1 / (-0.85)


vacuumLaunch_flag = True # If true, the launch_position is in vacuum. If false, the launch_position is in plasma.

vacuum_propagation_flag = True #If true, use analytical propagation until poloidal_flux_enter is reached. If false, start propagating numerically straight away.
# poloidal_flux_enter = 1.22

Psi_BC_flag = True # This solves the boundary conditions for the 3D matrix Psi, which is necessary if there is a discontinuity in the first derivative of density (or B field)

find_B_method='efit'
# efit_time_index = 7 # 190ms
efit_times = np.linspace(155,250,20)
# efit_time_index_scan = np.array([1])
# efit_time_index_scan = np.arange(0,10)*2 + 1
efit_time_index_scan = np.arange(0,10)*2 + 1
print(efit_times[efit_time_index_scan])

# 29908
params_record = np.array([
                        # [2.3,-1.9,1.18], # 150ms
                        [2.55,-2.2,1.15], # 160ms
                        [2.8,-2.2,1.15], # 170ms
                        [3.0,-2.35,1.2], # 180ms
                        [3.5,-2.1,1.22], # 190ms
                        [3.7,-2.7,1.15], # 200ms
                        [4.2,-2.0,1.2], # 210ms
                        [4.5,-1.8,1.24], # 220ms
                        [4.8,-1.8,1.2], # 230ms
                        [5.2,-1.8,1.2], # 240ms
                        [5.2,-2.8,1.1], # 250ms
                        # [5.7,-1.9,1.15], # 260ms
                        # [5.8,-2.2,1.1], # 270ms
                        # [6.5,-1.7,1.15], # 280ms
                        # [6.6,-1.8,1.1] # 290ms
                        ]
                        )

# # Check that parameters are not unreasonable
# launch_angular_frequency = 2*math.pi*10.0**9 * launch_freq_GHz
# wavenumber_K0 = launch_angular_frequency / constants.c
# waist_size = find_waist(launch_beam_width, wavenumber_K0, 1/launch_beam_radius_of_curvature)
# print("Waist size", waist_size, sep=": ")
# Rayleigh_length = find_Rayleigh_length(waist_size, wavenumber_K0)
# print("Rayleigh length", Rayleigh_length, sep=": ")
    
# Assumes that the launch point is outside the plasma (in vacuum)
# Otherwise, Psi_3D_beam_initial_cartersian does not get calculated correctly
#launch_position_Cartesian = np.asarray([1.73899,0.0110531,-0.0598212]) 
#launch_position = find_q_lab(launch_position_Cartesian) # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
#launch_position_Cartesian_test = find_q_lab_Cartesian(launch_position)

#plasmaLaunch_K_Cartesian = np.array([-765.4855975795751,23.237807434624816,-135.9027650366779])
#plasmaLaunch_K = find_K_lab(plasmaLaunch_K_Cartesian,launch_position_Cartesian) 
#plasmaLaunch_K_Cartesian_test = find_K_lab_Cartesian(plasmaLaunch_K,launch_position)

#plasmaLaunch_Psi_3D_lab_Cartesian=np.zeros([3,3])

launch_position = np.asarray([2.43521,0,0]) # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition

# for ii in range(0,len(launch_freq_GHz_sweep)):
#     for jj in range(0,len(toroidal_launch_angle_Torbeam_scan)):
#         for kk in range(0,len(poloidal_launch_angle_Torbeam_scan)):
    
#             print('Iteration number: ' + str(ii) + ' ' + str(jj) +  ' ' + str(kk))
    
#             launch_freq_GHz = launch_freq_GHz_sweep[ii]    
#             toroidal_launch_angle_Torbeam = toroidal_launch_angle_Torbeam_scan[jj]
#             poloidal_launch_angle_Torbeam = poloidal_launch_angle_Torbeam_scan[kk]
    
#             # beam_waist = np.sqrt(beam_rayleigh_distance * constants.c / (np.pi * launch_freq_GHz*10**9) )
#             # launch_beam_width = beam_waist * np.sqrt(2) # in m
#             # launch_beam_radius_of_curvature = -2*beam_rayleigh_distance # in m. negative because launched BEFORE the beam waist
            

#             if mode_flag == 1:
#                 mode_string = 'O'
#             elif mode_flag == -1:
#                 mode_string = 'X'
            
#             output_filename_string = (
#                                         '_p' + f'{poloidal_launch_angle_Torbeam:.1f}'
#                                       + '_t' + f'{toroidal_launch_angle_Torbeam:.1f}' 
#                                       + '_f' + f'{launch_freq_GHz:.1f}'
#                                       + '_'  + mode_string
#                                       + '_z-1.0' 
#                                       + '_r885'
#                                       + '.png'
#                                       )
            
#             beam_me_up( poloidal_launch_angle_Torbeam,
#                         toroidal_launch_angle_Torbeam,
#                         launch_freq_GHz,
#                         mode_flag,
#                         vacuumLaunch_flag,
#                         launch_beam_width,
#                         launch_beam_radius_of_curvature,
#                         launch_position,
#                         find_B_method,
#                         stop_flag,
#                         vacuum_propagation_flag,
#                         Psi_BC_flag,
#                         poloidal_flux_enter,
#                         input_filename_suffix,
#                         output_filename_suffix= output_filename_string,
#                         figure_flag=True,
#                         density_fit_parameters=np.array([3.5,-2.1,1.22])
#                       )

# for ii in range(0,len(launch_freq_GHz_sweep)):
#     for jj in range(0,len(mirror_rotation_angle_scan)):
    
#         print('Iteration number: ' + str(ii) + ' ' + str(jj))
    
#         launch_freq_GHz = launch_freq_GHz_sweep[ii]    
#         mirror_rotation_angle = mirror_rotation_angle_scan[jj]
#         toroidal_launch_angle_genray, poloidal_launch_angle_genray = genray_angles_from_mirror_angles(mirror_rotation_angle,mirror_tilt_angle,offset_for_window_norm_to_R = np.rad2deg(math.atan2(125,2432)))

#         poloidal_launch_angle_Torbeam = - poloidal_launch_angle_genray
#         toroidal_launch_angle_Torbeam = - toroidal_launch_angle_genray

#         print('poloidal_launch_angle_Torbeam: ' + str(poloidal_launch_angle_Torbeam))
#         print('toroidal_launch_angle_Torbeam: ' + str(toroidal_launch_angle_Torbeam))


#         if mode_flag == 1:
#             mode_string = 'O'
#         elif mode_flag == -1:
#             mode_string = 'X'
            
#         output_filename_string = (
#                                     '_r' + f'{mirror_rotation_angle:.1f}'
#                                     '_t' + f'{mirror_tilt_angle:.1f}'
#                                   + '_f' + f'{launch_freq_GHz:.1f}'
#                                   + '_'  + mode_string
#                                       )
            
#         beam_me_up( poloidal_launch_angle_Torbeam,
#                     toroidal_launch_angle_Torbeam,
#                     launch_freq_GHz,
#                     mode_flag,
#                     vacuumLaunch_flag,
#                     launch_beam_width,
#                     launch_beam_radius_of_curvature,
#                     launch_position,
#                     efit_time_index,
#                     find_B_method,
#                     vacuum_propagation_flag,
#                     Psi_BC_flag,
#                     poloidal_flux_enter,
#                     input_filename_suffix,
#                     output_filename_suffix= output_filename_string,
#                     figure_flag=True,
#                     density_fit_parameters=np.array([3.5,-2.1,1.22])
#                     )

# @njit(parallel=True)
# def beam_us_up(mirror_rotation_angle_scan,
#                mirror_tilt_angle,
#                launch_freq_GHz_sweep,
#                mode_flag,
#                vacuumLaunch_flag,
#                launch_beam_width,
#                launch_beam_radius_of_curvature,
#                launch_position,
#                find_B_method,
#                efit_time_index_scan,
#                efit_times,
#                vacuum_propagation_flag=False,
#                Psi_BC_flag = False,
#                input_filename_suffix='',
#                plasmaLaunch_K=np.zeros(3),
#                plasmaLaunch_Psi_3D_lab_Cartesian=np.zeros([3,3]),
#                density_fit_parameters_scan=None
#                ):
#     """
#     Wrapper for beam_me_up
#     Easy parallelisation for parameter sweeps
    
#     Sweep parameters:
#         mirror_rotation_angle_scan
#         efit_time_index_scan
#         launch_freq_GHz_sweep
        
#     Apparently it's generally better to parallelise the outer loop
#     I've decided to make that sweep across launch frequencies
#     After all, that's the thing I sweep over most frequently
#     """
    
#     # First prepare some stuff
#     # I'm doing it in separate for loops so that later on, when we need to have parallel for loops, things will work
#     poloidal_launch_angle_Torbeam_sweep = np.zeros_like(mirror_rotation_angle_scan)
#     toroidal_launch_angle_Torbeam_sweep = np.zeros_like(mirror_rotation_angle_scan)
#     for kk, mirror_rotation_angle in enumerate(mirror_rotation_angle_scan):            
#         toroidal_launch_angle_genray, poloidal_launch_angle_genray = genray_angles_from_mirror_angles(mirror_rotation_angle,mirror_tilt_angle,offset_for_window_norm_to_R = np.rad2deg(math.atan2(125,2432)))
        
#         poloidal_launch_angle_Torbeam_sweep[kk] = - poloidal_launch_angle_genray
#         toroidal_launch_angle_Torbeam_sweep[kk] = - toroidal_launch_angle_genray    
    
#     if mode_flag == 1:
#         mode_string = 'O'
#     elif mode_flag == -1:
#         mode_string = 'X'    

                
#     ##
#     for ii in prange(len(launch_freq_GHz_sweep)):
#         for jj, efit_time_index in enumerate(efit_time_index_scan):
#             for kk, mirror_rotation_angle in enumerate(mirror_rotation_angle_scan):        
#                 beam_me_up( toroidal_launch_angle_Torbeam_sweep[kk],
#                             toroidal_launch_angle_Torbeam_sweep[kk],
#                             launch_freq_GHz_sweep[ii],
#                             mode_flag,
#                             vacuumLaunch_flag,
#                             launch_beam_width,
#                             launch_beam_radius_of_curvature,
#                             launch_position,
#                             find_B_method,
#                             efit_time_index,
#                             vacuum_propagation_flag,
#                             Psi_BC_flag,
#                             poloidal_flux_enter=params_record[ii,2],
#                             output_filename_suffix= (
#                                                         '_r' #+ f'{mirror_rotation_angle:.1f}'
#                                                       # + '_t' + f'{mirror_tilt_angle:.1f}'
#                                                       # + '_f' + f'{launch_freq_GHz_sweep[ii]:.1f}'
#                                                       # + '_'  + mode_string
#                                                       # + '_'  + f'{efit_times[jj]:.3g}' + 'ms'
#                                                     ),
#                             density_fit_parameters=params_record[ii,:]
#                             )    
    
#     return None

# beam_us_up(mirror_rotation_angle_scan,
#            mirror_tilt_angle,
#            launch_freq_GHz_sweep,
#            mode_flag,
#            vacuumLaunch_flag,
#            launch_beam_width,
#            launch_beam_radius_of_curvature,
#            launch_position,
#            find_B_method,
#            efit_time_index_scan,
#            efit_times,
#            vacuum_propagation_flag,
#            Psi_BC_flag,
#            density_fit_parameters_scan=params_record
#            )

# for ii, efit_time_index in enumerate(efit_time_index_scan):
#     for jj, mirror_rotation_angle in enumerate(mirror_rotation_angle_scan):
#         for kk, launch_freq_GHz in enumerate(launch_freq_GHz_sweep):
#             # ii = 1
#             # jj = 30
#             # kk = 7
#             # mirror_rotation_angle=mirror_rotation_angle_scan[30]
#             # launch_freq_GHz = launch_freq_GHz_sweep[7]
#             # efit_time_index=3
#             print('Iteration number: ' + str(ii) + ' ' + str(jj) + ' ' + str(kk))
    
#             print(mirror_rotation_angle)
    
#             toroidal_launch_angle_genray, poloidal_launch_angle_genray = genray_angles_from_mirror_angles(mirror_rotation_angle,mirror_tilt_angle,offset_for_window_norm_to_R = np.rad2deg(math.atan2(125,2432)))
    
#             poloidal_launch_angle_Torbeam = - poloidal_launch_angle_genray
#             toroidal_launch_angle_Torbeam = - toroidal_launch_angle_genray
                
#             density_fit_parameters = params_record[ii,:]
            
#             print('poloidal_launch_angle_Torbeam: ' + str(poloidal_launch_angle_Torbeam))
#             print('toroidal_launch_angle_Torbeam: ' + str(toroidal_launch_angle_Torbeam))
    
    
#             if mode_flag == 1:
#                 mode_string = 'O'
#             elif mode_flag == -1:
#                 mode_string = 'X'
#             efit_time = efit_times[efit_time_index]    
            
#             output_filename_string = (
#                                         '_r' + f'{mirror_rotation_angle:.1f}'
#                                         '_t' + f'{mirror_tilt_angle:.1f}'
#                                       + '_f' + f'{launch_freq_GHz:.1f}'
#                                       + '_'  + mode_string
#                                       + '_'  + f'{efit_time:.3g}' + 'ms'

#                                           )
                

            
#             beam_me_up( poloidal_launch_angle_Torbeam,
#                         toroidal_launch_angle_Torbeam,
#                         launch_freq_GHz,
#                         mode_flag,
#                         vacuumLaunch_flag,
#                         launch_beam_width,
#                         launch_beam_radius_of_curvature,
#                         launch_position,
#                         find_B_method,
#                         efit_time_index,
#                         vacuum_propagation_flag,
#                         Psi_BC_flag,
#                         poloidal_flux_enter=density_fit_parameters[2],
#                         output_filename_suffix= output_filename_string,
#                         figure_flag=True,
#                         density_fit_parameters=density_fit_parameters
#                         )

# toroidal_launch_angle_genray, poloidal_launch_angle_genray = genray_angles_from_mirror_angles(mirror_rotation_angle,mirror_tilt_angle,offset_for_window_norm_to_R = np.rad2deg(math.atan2(125,2432)))
    
# poloidal_launch_angle_Torbeam = -poloidal_launch_angle_genray
# toroidal_launch_angle_Torbeam = -toroidal_launch_angle_genray
           
efit_time_index = 5
density_fit_parameters = params_record[2,:]
poloidal_flux_enter = density_fit_parameters[2]

beam_me_up( poloidal_launch_angle_Torbeam,
            toroidal_launch_angle_Torbeam,
            launch_freq_GHz,
            mode_flag,
            vacuumLaunch_flag,
            launch_beam_width,
            launch_beam_radius_of_curvature,
            launch_position,
            find_B_method,
            efit_time_index,
            vacuum_propagation_flag,
            Psi_BC_flag,
            poloidal_flux_enter,
            output_filename_suffix='',
            figure_flag= True,
            density_fit_parameters=density_fit_parameters
            )# from ne_fit_radialcoord.py

# for ii in range(0,51):
#     print('Iteration number: ' + str(ii))
    
#     toroidal_launch_angle_Torbeam = toroidal_launch_angle_Torbeam_scan[ii]

#     beam_me_up(tau_max,
#             saveInterval,
#               poloidal_launch_angle_Torbeam,
#               toroidal_launch_angle_Torbeam,
#               launch_freq_GHz,
#               mode_flag,
#               vacuumLaunch_flag,
#               launch_beam_width,
#               launch_beam_radius_of_curvature,
#               launch_position,
#               find_B_method,
#               stop_flag,
#               vacuum_propagation_flag,
#               Psi_BC_flag,
#               poloidal_flux_enter,
#               input_filename_suffix,
#               output_filename_suffix=str(ii),
#               figure_flag=True,
#               density_fit_parameters=np.array([-6.78999607,6.43248856,0.96350798,0.48792645])
#               )

# for ii in range(0,81):
#     print('Iteration number: ' + str(ii))
    
#     launch_beam_width = launch_beam_width_sweep[ii]
#     launch_beam_radius_of_curvature = launch_beam_radius_of_curvature_sweep[ii]

#     beam_me_up(tau_step,
#               numberOfTauPoints,
#               saveInterval,
#               poloidal_launch_angle_Torbeam,
#               toroidal_launch_angle_Torbeam,
#               launch_freq_GHz,
#               mode_flag,
#               vacuumLaunch_flag,
#               launch_beam_width,
#               launch_beam_radius_of_curvature,
#               launch_position,
#               find_B_method,
#               vacuum_propagation_flag,
#               Psi_BC_flag,
#               poloidal_flux_enter,
#               input_filename_suffix,
#               output_filename_suffix=str(ii),
#               figure_flag=True,
#               density_fit_parameters=np.array([-6.78999607,6.43248856,0.96350798,0.48792645])
#               )
