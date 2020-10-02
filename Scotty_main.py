# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

"""
from Scotty_beam_me_up import beam_me_up
import numpy as np

#input_filename_suffix = '_29908_200'
input_filename_suffix = ''

tau_step = 1.0
numberOfTauPoints = 751
saveInterval = 1  # saves every n time steps

poloidal_launch_angle_Torbeam = 4.0 # deg
toroidal_launch_angle_Torbeam = -1.0 # deg
#toroidal_launch_angle_Torbeam_scan = np.linspace(-3,-8,51)

launch_freq_GHz = 55.0
mode_flag = 1 # O-mode (1) or X-mode (-1)

vacuum_propagation_flag = True #TODO: If true, use analytical propagation until poloidal_flux_enter is reached. If false, start propagating numerically straight away.
poloidal_flux_enter = 1.2 

launch_beam_width = 0.06 # in m
launch_beam_curvature = -192 # in m. negative because launched BEFORE the beam waist


# Assumes that the launch point is outside the plasma (in vacuum)
# Otherwise, Psi_3D_beam_initial_cartersian does not get calculated correctly
launch_position = np.asarray([2.43521,0,0]) # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
#launch_position = np.asarray([1.5,0,0]) # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition

beam_me_up(tau_step,
           numberOfTauPoints,
           saveInterval,
           poloidal_launch_angle_Torbeam,
           toroidal_launch_angle_Torbeam,
           launch_freq_GHz,
           mode_flag,
           launch_beam_width,
           launch_beam_curvature,
           launch_position,
           poloidal_flux_enter,
           input_filename_suffix,
           output_filename_suffix='',
           figure_flag= True)

#for ii in range(0,51):
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
#               launch_beam_width,
#               launch_beam_curvature,
#               launch_position,
#               poloidal_flux_enter,
#               input_filename_suffix,
#               output_filename_suffix=str(ii),
#               figure_flag= True)
