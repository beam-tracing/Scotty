# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

"""
from Scotty_beam_me_up import beam_me_up
from Scotty_fun_general import find_q_lab_Cartesian, find_q_lab, find_K_lab_Cartesian, find_K_lab
import numpy as np

input_filename_suffix = '_29910_190'
#input_filename_suffix = ''

tau_step = 1.0
numberOfTauPoints = 1251
saveInterval = 1  # saves every n time steps

poloidal_launch_angle_Torbeam = 4.0 # deg
toroidal_launch_angle_Torbeam = -5.25 # deg
#toroidal_launch_angle_Torbeam_scan = np.linspace(2,-6,41)

launch_freq_GHz = 55.0
mode_flag = 1 # O-mode (1) or X-mode (-1)
launch_beam_width = 0.06 # in m
launch_beam_curvature = -192 # in m. negative because launched BEFORE the beam waist
vacuumLaunch_flag = True # If true, the launch_position is in vacuum. If false, the launch_position is in plasma.


vacuum_propagation_flag = True #If true, use analytical propagation until poloidal_flux_enter is reached. If false, start propagating numerically straight away.
poloidal_flux_enter = 1.2612056746393183

Psi_BC_flag = True # This solves the boundary conditions for the 3D matrix Psi, which is necessary if there is a discontinuity in the first derivative of density (or B field)





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

beam_me_up(tau_step,
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
           vacuum_propagation_flag,
           Psi_BC_flag,
           poloidal_flux_enter,
           input_filename_suffix,
           output_filename_suffix='1',
           figure_flag= True)

#for ii in range(0,41):
#    print('Iteration number: ' + str(ii))
#    
#    toroidal_launch_angle_Torbeam = toroidal_launch_angle_Torbeam_scan[ii]
#
#    beam_me_up(tau_step,
#               numberOfTauPoints,
#               saveInterval,
#               poloidal_launch_angle_Torbeam,
#               toroidal_launch_angle_Torbeam,
#               launch_freq_GHz,
#               mode_flag,
#               vacuumLaunch_flag,
#               launch_beam_width,
#               launch_beam_curvature,
#               launch_position,
#               vacuum_propagation_flag,
#               Psi_BC_flag,
#               poloidal_flux_enter,
#               input_filename_suffix,
#               output_filename_suffix=str(ii),
#               figure_flag= True)
