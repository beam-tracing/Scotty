# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:44:04 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolate
from scipy import constants

def read_floats_into_list_until(terminator, lines):
    # Reads the lines of a file until the string (terminator) is read
    # Currently used to read topfile
    # Written by NE Bricknell
    lst = []
    while True:
        try: line = lines.readline()
        except StopIteration: break
        if terminator in line: break
        elif not line: break
        lst.extend(map(float,  line.split()))
    return lst

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



launch_beam_width = 0.06 # in m
launch_beam_curvature = -19.2 
launch_freq_GHz = 55.0

poloidal_flux_enter = 1.0

poloidal_launch_angle_Torbeam = 4.0 # deg
launch_position = np.asarray([2.43521,0]) # q_R, q_Z. q_zeta = 0 at launch, by definition

launch_angular_frequency = 2*constants.pi*launch_freq_GHz*10**9
wavenumber_K0 = launch_angular_frequency / constants.c









input_files_path ='D:\\Dropbox\\VHChen2018\\Data\\Input_Files_29Apr2019\\'
topfile_filename = input_files_path + 'topfile' + '_29908_200'

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
data_B_R_grid = np.transpose((np.asarray(data_B_R_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_B_T_grid = np.transpose((np.asarray(data_B_T_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_B_Z_grid = np.transpose((np.asarray(data_B_Z_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_poloidal_flux_grid = np.transpose((np.asarray(data_poloidal_flux_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
# -------------------

interp_poloidal_flux = interpolate.interp2d(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), kind='cubic',
                                           copy=True, bounds_error=False, fill_value=None) # Flux extrapolated outside region



# This part is what I'm trying to get right
search_Z_end = launch_position[1] - launch_position[0]*np.tan(np.radians(poloidal_launch_angle_Torbeam))
numberOfCoarseSearchPoints = 50
R_coarse_search_array = np.linspace(launch_position[0],0,numberOfCoarseSearchPoints)
Z_coarse_search_array = np.linspace(launch_position[1],search_Z_end,numberOfCoarseSearchPoints)
poloidal_flux_coarse_search_array = np.zeros(numberOfCoarseSearchPoints)
for ii in range(0,numberOfCoarseSearchPoints):
    poloidal_flux_coarse_search_array[ii] = interp_poloidal_flux(R_coarse_search_array[ii],Z_coarse_search_array[ii])
meets_flux_condition_array = poloidal_flux_coarse_search_array < 0.9*poloidal_flux_enter
dummy_array = np.array(range(numberOfCoarseSearchPoints))
indices_inside_for_sure_array = dummy_array[meets_flux_condition_array]
first_inside_index = indices_inside_for_sure_array[0]
numberOfFineSearchPoints = 1000
R_fine_search_array = np.linspace(launch_position[0],R_coarse_search_array[first_inside_index],numberOfFineSearchPoints)
Z_fine_search_array = np.linspace(launch_position[1],Z_coarse_search_array[first_inside_index],numberOfFineSearchPoints)
poloidal_fine_search_array = np.zeros(numberOfFineSearchPoints)
for ii in range(0,numberOfFineSearchPoints):
    poloidal_fine_search_array[ii] = interp_poloidal_flux(R_fine_search_array[ii],Z_fine_search_array[ii])
entry_index = find_nearest(poloidal_fine_search_array,poloidal_flux_enter)
entry_position = np.zeros(2) # R,Z
entry_position[0] = R_fine_search_array[entry_index]
entry_position[1] = Z_fine_search_array[entry_index]
distance_from_launch_to_entry = np.sqrt((launch_position[0] - entry_position[0])**2 + (launch_position[1] - entry_position[1])**2)
# ------------------





plt.figure()
plt.title('Rz')
plt.xlabel('R / m') # x-direction
plt.ylabel('z / m')
   
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(data_poloidal_flux_grid), contour_levels)
plt.clabel(CS, inline=1, fontsize=10) # Labels the flux surfaces
plt.plot(
        (launch_position[0],0),
        (launch_position[1],search_Z_end),
'--.k') # Central (reference) ray
plt.plot(R_fine_search_array[entry_index],Z_fine_search_array[entry_index],'ro') 
#cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
#                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
plt.xlim(data_R_coord[0],data_R_coord[-1])
plt.ylim(data_Z_coord[0],data_Z_coord[-1])
        