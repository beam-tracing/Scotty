# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import tikzplotlib



numberOfRuns = 81
# toroidal_launch_angle_Torbeam_scan = np.linspace(-0.5,-7.5,71)

beam_rayleigh_distance = 0.885
beam_waist = 0.0392
launch_distance_from_waist = np.linspace(-beam_rayleigh_distance*5,-beam_rayleigh_distance,81)
launch_beam_width_sweep = beam_waist * np.sqrt( 1 + (launch_distance_from_waist/beam_rayleigh_distance)**2 )
launch_beam_curvature_sweep = launch_distance_from_waist*( 1 + (beam_rayleigh_distance/launch_distance_from_waist)**2 )

localisation_beam_ray_spectrum_half_width_sweep = np.zeros(numberOfRuns)
localisation_beam_ray_spectrum_max_distance_from_cutoff_sweep = np.zeros(numberOfRuns)
delta_k_perp_2_sweep = np.zeros(numberOfRuns)
delta_theta_m_sweep = np.zeros(numberOfRuns)

for ii in range(0,numberOfRuns):
    loadfile = np.load('analysis_output' + str(ii) +  '.npz')
    cutoff_index = loadfile['cutoff_index']
    delta_k_perp_2 = loadfile['delta_k_perp_2']
    delta_theta_m = loadfile['delta_theta_m']
    localisation_beam_ray_spectrum_max_distance_from_cutoff = loadfile['localisation_beam_ray_spectrum_max_distance_from_cutoff']
    localisation_beam_ray_spectrum_half_width = loadfile['localisation_beam_ray_spectrum_half_width']
    loadfile.close()
    
    delta_k_perp_2_sweep[ii] = delta_k_perp_2[cutoff_index]
    delta_theta_m_sweep[ii]  = delta_theta_m[cutoff_index] 
    localisation_beam_ray_spectrum_max_distance_from_cutoff_sweep[ii] = localisation_beam_ray_spectrum_max_distance_from_cutoff
    localisation_beam_ray_spectrum_half_width_sweep[ii] = localisation_beam_ray_spectrum_half_width
    

plt.figure()
plt.plot(launch_distance_from_waist,delta_k_perp_2_sweep)
tikzplotlib.save("test.tex")

plt.figure()
plt.plot(delta_theta_m_sweep)

plt.figure()
plt.plot(localisation_beam_ray_spectrum_max_distance_from_cutoff_sweep)

plt.figure()
plt.plot(localisation_beam_ray_spectrum_half_width_sweep)

loadfile = np.load('analysis_output' + str(53) +  '.npz')
det_imag_Psi_w_analysis = loadfile['det_imag_Psi_w_analysis']
det_real_Psi_w_analysis = loadfile['det_real_Psi_w_analysis']
distance_along_line = loadfile['distance_along_line']
det_M_w_analysis = loadfile['det_M_w_analysis']
loadfile.close()

plt.figure()
plt.plot(distance_along_line,det_imag_Psi_w_analysis)
plt.plot(distance_along_line,det_real_Psi_w_analysis)


plt.figure()
plt.plot(distance_along_line,abs(det_M_w_analysis))


