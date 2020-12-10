# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

suffix = '80'
loadfile = np.load('analysis_output' + suffix + '.npz')
cutoff_index = loadfile['cutoff_index']
RZ_distance_along_line = loadfile['RZ_distance_along_line']
k_perp_1_backscattered = loadfile['k_perp_1_backscattered']
K_magnitude_array = loadfile['K_magnitude_array']
loadfile.close()

loadfile = np.load('data_output' + suffix + '.npz')
tau_array = loadfile['tau_array']
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
q_Z_array = loadfile['q_Z_array']
loadfile.close()

loadfile = np.load('data_input' + suffix + '.npz')
data_R_coord = loadfile['data_R_coord']
data_Z_coord = loadfile['data_Z_coord']
data_poloidal_flux_grid = loadfile['data_poloidal_flux_grid']
loadfile.close()

k_perp_1_s = -809.9881062270306
k_perp_mu_s = -1500
k_perp_low_s = -200
k_perp_high_s = -2400
tau_step = 1.0
start_index = 50
end_index = 600

# Phase
eikonal_S = np.cumsum(K_magnitude_array*g_magnitude_output*tau_step)
k1_s_L = k_perp_1_s * np.cumsum(g_magnitude_output*tau_step)
k1_mu_L = k_perp_mu_s * np.cumsum(g_magnitude_output*tau_step)
k1_low_L = k_perp_low_s * np.cumsum(g_magnitude_output*tau_step)
k1_high_L = k_perp_high_s * np.cumsum(g_magnitude_output*tau_step)

phase_s = 2 * eikonal_S + k1_s_L
phase_mu = 2 * eikonal_S + k1_mu_L
phase_low = 2 * eikonal_S + k1_low_L
phase_high = 2 * eikonal_S + k1_high_L

index_a = 156
index_b = 552

plt.figure(figsize=(8.0, 5.0))
plt.subplot(1,2,1)
plt.title(r'$\tau$')
plt.plot(tau_array,phase_s,'g')
plt.plot(tau_array,phase_mu,'b')
plt.plot(tau_array,phase_low,'r')
plt.plot(tau_array,phase_high,'r')

plt.plot(tau_array[cutoff_index], phase_s[cutoff_index], marker='o', markersize=10, color="g")
plt.plot(tau_array[index_a], phase_mu[index_a], marker='o', markersize=10, color="cyan")
plt.plot(tau_array[index_b], phase_mu[index_b], marker='o', markersize=10, color="cyan")
plt.title('Integrand Phase')


plt.subplot(1,2,2)
plt.title('Rz')
plt.xlabel('R / m') # x-direction
plt.ylabel('z / m')
   
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.plot(
#        np.concatenate([[launch_position[0],entry_position[0]],q_R_array ]),
#        np.concatenate([[launch_position[1],entry_position[1]],q_Z_array ]),
        q_R_array, q_Z_array,
        '--.k') # Central (reference) ray
        #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
        #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
plt.plot(q_R_array[cutoff_index], q_Z_array[cutoff_index], marker='o', markersize=10, color="g")
plt.plot(q_R_array[index_a], q_Z_array[index_a], marker='o', markersize=10, color="cyan")
plt.plot(q_R_array[index_b], q_Z_array[index_b], marker='o', markersize=10, color="cyan")
#plt.xlim(data_R_coord[0],data_R_coord[-1])
#plt.ylim(data_Z_coord[0],data_Z_coord[-1])
plt.xlim(1,1.5)
plt.ylim(-0.7,0.2)
plt.tight_layout()
plt.savefig('StationaryPhase', bbox_inches='tight')
#plt.savefig('StationaryPhase.eps', format='eps', bbox_inches='tight')
plt.savefig('StationaryPhase.jpg', format='jpg', bbox_inches='tight')







plt.figure(figsize=(8.0, 5.0))
plt.title('Rz')
plt.xlabel('R / m') # x-direction
plt.ylabel('z / m')
   
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.plot(
#        np.concatenate([[launch_position[0],entry_position[0]],q_R_array ]),
#        np.concatenate([[launch_position[1],entry_position[1]],q_Z_array ]),
        q_R_array, q_Z_array,
        '--.k') # Central (reference) ray
        #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
        #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
plt.plot(q_R_array[cutoff_index], q_Z_array[cutoff_index], marker='o', markersize=10, color="g")
plt.plot(q_R_array[index_a], q_Z_array[index_a], marker='o', markersize=10, color="cyan")
plt.plot(q_R_array[index_b], q_Z_array[index_b], marker='o', markersize=10, color="cyan")
#plt.xlim(data_R_coord[0],data_R_coord[-1])
#plt.ylim(data_Z_coord[0],data_Z_coord[-1])
plt.xlim(1,1.5)
plt.ylim(-0.7,0.2)
plt.tight_layout()
plt.savefig('StationaryPhaseAlt.jpg', format='jpg', bbox_inches='tight')




#plt.close()