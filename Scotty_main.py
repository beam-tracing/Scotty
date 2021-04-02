# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

"""
from Scotty_beam_me_up import beam_me_up
from Scotty_fun_general import find_q_lab_Cartesian, find_q_lab, find_K_lab_Cartesian, find_K_lab, find_waist, find_Rayleigh_length, genray_angles_from_mirror_angles

from scipy import constants
import math
import numpy as np

input_filename_suffix = '_29905_190'
#input_filename_suffix = ''

poloidal_launch_angle_Torbeam = 4.0 # deg
toroidal_launch_angle_Torbeam = -4.0 # deg
# toroidal_launch_angle_Torbeam_scan = np.linspace(0.0,-16.0,65)
# poloidal_launch_angle_Torbeam_scan = np.array([5.34])

# rotation_angles_array = np.array([7.0,8,9,5,4,6]) 
mirror_rotation_angle_scan = np.linspace(2,7,11)
mirror_tilt_angle = -2.0 
# mirror_tilt_angle = 0



# launch_freq_GHz_sweep = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0,55.0,57.5,60.0,62.5,67.5,70.0,72.5,75.0])
launch_freq_GHz_sweep = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0])
# launch_freq_GHz = 55.0
mode_flag = -1 # O-mode (1) or X-mode (-1)

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
launch_beam_width = 0.072
launch_beam_radius_of_curvature = 1 / (-0.85)


vacuumLaunch_flag = True # If true, the launch_position is in vacuum. If false, the launch_position is in plasma.

vacuum_propagation_flag = True #If true, use analytical propagation until poloidal_flux_enter is reached. If false, start propagating numerically straight away.
poloidal_flux_enter = 1.22

Psi_BC_flag = True # This solves the boundary conditions for the 3D matrix Psi, which is necessary if there is a discontinuity in the first derivative of density (or B field)

find_B_method='efit'

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

for ii in range(0,len(launch_freq_GHz_sweep)):
    for jj in range(0,len(mirror_rotation_angle_scan)):
    
        print('Iteration number: ' + str(ii) + ' ' + str(jj))
    
        launch_freq_GHz = launch_freq_GHz_sweep[ii]    
        mirror_rotation_angle = mirror_rotation_angle_scan[jj]
        toroidal_launch_angle_genray, poloidal_launch_angle_genray = genray_angles_from_mirror_angles(mirror_rotation_angle,mirror_tilt_angle,offset_for_window_norm_to_R = np.rad2deg(math.atan2(125,2432)))

        poloidal_launch_angle_Torbeam = - poloidal_launch_angle_genray
        toroidal_launch_angle_Torbeam = toroidal_launch_angle_genray

        print('poloidal_launch_angle_Torbeam: ' + str(poloidal_launch_angle_Torbeam))
        print('toroidal_launch_angle_Torbeam: ' + str(toroidal_launch_angle_Torbeam))


        if mode_flag == 1:
            mode_string = 'O'
        elif mode_flag == -1:
            mode_string = 'X'
            
        output_filename_string = (
                                    '_r' + f'{mirror_rotation_angle:.1f}'
                                    '_t' + f'{mirror_tilt_angle:.1f}'
                                  + '_f' + f'{launch_freq_GHz:.1f}'
                                  + '_'  + mode_string
                                      )
            
        beam_me_up( poloidal_launch_angle_Torbeam,
                    toroidal_launch_angle_Torbeam,
                    launch_freq_GHz,
                    mode_flag,
                    vacuumLaunch_flag,
                    launch_beam_width,
                    launch_beam_radius_of_curvature,
                    launch_position,
                    find_B_method,
                    vacuum_propagation_flag,
                    Psi_BC_flag,
                    poloidal_flux_enter,
                    input_filename_suffix,
                    output_filename_suffix= output_filename_string,
                    figure_flag=True,
                    density_fit_parameters=np.array([3.5,-2.1,1.22])
                    )



# beam_me_up( poloidal_launch_angle_Torbeam,
#             toroidal_launch_angle_Torbeam,
#             launch_freq_GHz,
#             mode_flag,
#             vacuumLaunch_flag,
#             launch_beam_width,
#             launch_beam_radius_of_curvature,
#             launch_position,
#             find_B_method,
#             stop_flag,
#             vacuum_propagation_flag,
#             Psi_BC_flag,
#             poloidal_flux_enter,
#             input_filename_suffix,
#             output_filename_suffix='',
#             figure_flag= True,
#             density_fit_parameters=np.array([3.5,-2.1,1.22])
#             )# from ne_fit_radialcoord.py

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
