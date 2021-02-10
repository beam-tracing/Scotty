# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import tikzplotlib

def find_nearest(array,  value): #returns the index
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

suffix = '80'
loadfile = np.load('analysis_output' + suffix + '.npz')
cutoff_index = loadfile['cutoff_index']
out_index = loadfile['out_index']
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
launch_position = loadfile['launch_position']
loadfile.close()

k_perp_1_s = -2*K_magnitude_array[cutoff_index]
k_perp_mu_s = -1320
k_perp_low_s = -500
k_perp_high_s = -2350
# tau_step = 1.0
# start_index = 50
# end_index = 600

# Phase
eikonal_S = cumtrapz(K_magnitude_array*g_magnitude_output,tau_array,initial=0)
k1_s_L = k_perp_1_s * cumtrapz(g_magnitude_output,tau_array,initial=0)
k1_mu_L = k_perp_mu_s * cumtrapz(g_magnitude_output,tau_array,initial=0)
k1_low_L = k_perp_low_s * cumtrapz(g_magnitude_output,tau_array,initial=0)
k1_high_L = k_perp_high_s * cumtrapz(g_magnitude_output,tau_array,initial=0)

phase_s = 2 * eikonal_S + k1_s_L
phase_mu = 2 * eikonal_S + k1_mu_L
phase_low = 2 * eikonal_S + k1_low_L
phase_high = 2 * eikonal_S + k1_high_L

index_a = find_nearest(phase_mu[:cutoff_index],phase_mu[:cutoff_index].max())
index_b = cutoff_index + find_nearest(phase_mu[cutoff_index:],phase_mu[cutoff_index:].min())


plot_every_n_points = 10
R_start_index = 60
R_end_index = 100
Z_start_index = 40
Z_end_index = 70


plt.figure(figsize=(8.0, 5.0))
plt.subplot(1,2,1)
plt.title(r'$\tau$')
plt.plot(tau_array[::plot_every_n_points],phase_s[::plot_every_n_points],'g')
plt.plot(tau_array[::plot_every_n_points],phase_mu[::plot_every_n_points],'b')
plt.plot(tau_array[::plot_every_n_points],phase_low[::plot_every_n_points],'r')
plt.plot(tau_array[::plot_every_n_points],phase_high[::plot_every_n_points],'r')

plt.plot(tau_array[cutoff_index], phase_s[cutoff_index], marker='o', markersize=10, color="g")
plt.plot(tau_array[index_a], phase_mu[index_a], marker='o', markersize=10, color="cyan")
plt.plot(tau_array[index_b], phase_mu[index_b], marker='o', markersize=10, color="cyan")
plt.title('Integrand Phase')


plt.subplot(1,2,2)
plt.title('Rz')
plt.xlabel('R / m') # x-direction
plt.ylabel('z / m')
   
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord[R_start_index:R_end_index], data_Z_coord[Z_start_index:Z_end_index], np.transpose(data_poloidal_flux_grid[R_start_index:R_end_index,Z_start_index:Z_end_index]), contour_levels,vmin=0,vmax=1.2,cmap='inferno')
plt.clabel(CS, inline=True, fontsize=10,inline_spacing=1,fmt= '%1.1f',use_clabeltext=True) # Labels the flux surfaces
plt.xlim(1.0,1.6)
plt.ylim(-0.7,0.1)
plt.plot(q_R_array[:out_index:plot_every_n_points],q_Z_array[:out_index:plot_every_n_points],'k')
plt.plot( [launch_position[0], q_R_array[0]], [launch_position[2], q_Z_array[0]],':k')

        #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
        #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
plt.plot(q_R_array[cutoff_index], q_Z_array[cutoff_index], marker='o', markersize=10, color="g")
plt.plot(q_R_array[index_a], q_Z_array[index_a], marker='o', markersize=10, color="cyan")
plt.plot(q_R_array[index_b], q_Z_array[index_b], marker='o', markersize=10, color="cyan")
#plt.xlim(data_R_coord[0],data_R_coord[-1])
#plt.ylim(data_Z_coord[0],data_Z_coord[-1])
# plt.xlim(1,1.5)
# plt.ylim(-0.7,0.2)
plt.tight_layout()
plt.savefig('StationaryPhase', bbox_inches='tight')
#plt.savefig('StationaryPhase.eps', format='eps', bbox_inches='tight')
plt.savefig('StationaryPhase.jpg', format='jpg', bbox_inches='tight')



plt.figure()
plt.title('Integrand Phase')
plt.plot(tau_array[:out_index:plot_every_n_points],phase_s[:out_index:plot_every_n_points],'g')
plt.plot(tau_array[:out_index:plot_every_n_points],phase_mu[:out_index:plot_every_n_points],'b')
plt.plot(tau_array[:out_index:plot_every_n_points],phase_low[:out_index:plot_every_n_points],'r')
plt.plot(tau_array[:out_index:plot_every_n_points],phase_high[:out_index:plot_every_n_points],'r')

plt.plot(tau_array[cutoff_index], phase_s[cutoff_index], marker='o', markersize=10, color="g")
plt.plot(tau_array[index_a], phase_mu[index_a], marker='o', markersize=10, color="cyan")
plt.plot(tau_array[index_b], phase_mu[index_b], marker='o', markersize=10, color="cyan")
tikzplotlib.save("stationary_phase1.tex")


plt.figure()
plt.title('Poloidal plane')
plt.xlabel('R / m') # x-direction
plt.ylabel('z / m')
   
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord[R_start_index:R_end_index], data_Z_coord[Z_start_index:Z_end_index], np.transpose(data_poloidal_flux_grid[R_start_index:R_end_index,Z_start_index:Z_end_index]), contour_levels,vmin=0,vmax=1.2,cmap='inferno')
plt.clabel(CS, inline=True, fontsize=10,inline_spacing=1,fmt= '%1.1f',use_clabeltext=True) # Labels the flux surfaces
plt.xlim(1.0,1.6)
plt.ylim(-0.7,0.1)
plt.plot(q_R_array[:out_index:plot_every_n_points],q_Z_array[:out_index:plot_every_n_points],'k')
plt.plot( [launch_position[0], q_R_array[0]], [launch_position[2], q_Z_array[0]],':k')

        #cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
        #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
plt.plot(q_R_array[cutoff_index], q_Z_array[cutoff_index], marker='o', markersize=10, color="g")
plt.plot(q_R_array[index_a], q_Z_array[index_a], marker='o', markersize=10, color="cyan")
plt.plot(q_R_array[index_b], q_Z_array[index_b], marker='o', markersize=10, color="cyan")
#plt.xlim(data_R_coord[0],data_R_coord[-1])
#plt.ylim(data_Z_coord[0],data_Z_coord[-1])
# plt.xlim(1,1.5)
# plt.ylim(-0.7,0.2)
tikzplotlib.save("stationary_phase2.tex")
plt.tight_layout()
plt.savefig('StationaryPhase', bbox_inches='tight')
#plt.savefig('StationaryPhase.eps', format='eps', bbox_inches='tight')
plt.savefig('StationaryPhase.jpg', format='jpg', bbox_inches='tight')