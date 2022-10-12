# -*- coding: utf-8 -*-
"""
Created: 30/08/2022
Last Updated: 30/08/2022
@author: Tan Zheng Yang

Processes the EFIT files. Change them for Cartesian to cylindrical coordinates.
"""

import numpy as np
import math
from scipy import interpolate as interpolate

#==============================================================================
#               IMPORT EFIT DATA IN CYLINDRICAL COORDINATES
#==============================================================================
magnetic_data_path = 'test_data\\'
shot = 45118
load_file_path = magnetic_data_path + str(shot) + '_equilibrium_data.npz'

loadfile = np.load(load_file_path)

data_R_coord = loadfile['R_EFIT']
data_Z_coord = loadfile['Z_EFIT']
poloidalFlux_grid_all_times = loadfile['poloidalFlux_grid']
Bphi_grid_all_times = loadfile['Bphi_grid']
Br_grid_all_times = loadfile['Br_grid']
Bz_grid_all_times = loadfile['Bz_grid']
time_EFIT = loadfile['time_EFIT']
loadfile.close()

#---------------------------------- DEBUG -------------------------------------
# print(loadfile.files)
# print(np.shape(loadfile['poloidalFlux_grid'])) ---> shape = (160, 65, 65)
# print(np.shape(loadfile['Br_grid'])) ---> shape = (160, 65, 65)
# print(np.shape(loadfile['Bphi_grid'])) ---> shape = (160, 65, 65)
# print(np.shape(loadfile['Bz_grid'])) ---> shape = (160, 65, 65)
# print(len(loadfile['time_EFIT'])) ---> len = 160
# print(len(loadfile['R_EFIT'])) ---> len = 65
# print(len(loadfile['Z_EFIT'])) ---> len = 65
#------------------------------------------------------------------------------

#==============================================================================
#                          SET UP CARTESIAN GRID 
#==============================================================================

print('Data conversion in progress...')

delta_X = 0.1 
delta_Y = 0.1 
#delta_Z = 0.1

data_R_coord_max = np.max(data_R_coord)

data_X_coord = np.arange(-data_R_coord_max, 
                         data_R_coord_max + delta_X,
                         delta_X)

print(len(data_X_coord))

data_Y_coord = np.arange(-data_R_coord_max, 
                         data_R_coord_max + delta_Y,
                         delta_Y)

#==============================================================================
#              CONVERSION OF DATA TO CARTESIAN COORDIANTES
#==============================================================================

poloidalFlux_grid_all_times_CART = np.zeros((len(time_EFIT), 
                                             len(data_X_coord),
                                             len(data_Y_coord),
                                             len(data_Z_coord)
                                             ))

Bx_grid_all_times_CART = poloidalFlux_grid_all_times_CART.copy()

By_grid_all_times_CART = poloidalFlux_grid_all_times_CART.copy()

Bz_grid_all_times_CART = poloidalFlux_grid_all_times_CART.copy()

# From the original Scotty_beam_me_up, used for bivariate interpolating functions
interp_order = 5 
interp_smoothing = 2

# # Use a for loop to run through each point in time
for time in np.arange(len(time_EFIT)):
    
    #----------------- Poloidal Flux Interpolating Function ------------------
    
    # Poloidal flux grid for the current time
    poloidalFlux_grid_current_time = poloidalFlux_grid_all_times[time]
    
    # Bivariate interpolating function (in cylindrical coordinates). Note that
    # data_R_coord and data_Z_coord must be sorted in ascending order
    interp_poloidal_flux = interpolate.RectBivariateSpline(data_R_coord,
                                                           data_Z_coord,
                                                           poloidalFlux_grid_current_time, 
                                                           bbox = [None, None, None, None], 
                                                           kx = interp_order, 
                                                           ky = interp_order, 
                                                           s = interp_smoothing)
    
    # --------- Magnetic Field Components Interpolating Functions ------------
    
    # Magnetic field component grids for the current time
    Bphi_grid_current_time = Bphi_grid_all_times[time]
    Br_grid_current_time = Br_grid_all_times[time]
    Bz_grid_current_time = Bz_grid_all_times[time]
    
    # Interpolation functions for each of the magnetic field components
    interp_Bphi = interpolate.RectBivariateSpline(data_R_coord,
                                                  data_Z_coord,
                                                  Bphi_grid_current_time, 
                                                  bbox = [None, None, None, None], 
                                                  kx = interp_order, 
                                                  ky = interp_order, 
                                                  s = interp_smoothing)
    
    interp_Br = interpolate.RectBivariateSpline(data_R_coord,
                                                data_Z_coord,
                                                Br_grid_current_time, 
                                                bbox = [None, None, None, None], 
                                                kx = interp_order, 
                                                ky = interp_order, 
                                                s = interp_smoothing)
    
    interp_Bz = interpolate.RectBivariateSpline(data_R_coord,
                                                data_Z_coord,
                                                Bz_grid_current_time, 
                                                bbox = [None, None, None, None], 
                                                kx = interp_order, 
                                                ky = interp_order, 
                                                s = interp_smoothing)

    # ------------- Data Conversion to Cartesian Coordinates  ----------------
    
    for x_idx in np.arange(len(data_X_coord)):
        for y_idx in np.arange(len(data_Y_coord)):
            for z_idx in np.arange(len(data_Z_coord)):
                
                current_X = data_X_coord[x_idx]
                current_Y = data_Y_coord[y_idx]
                current_Z = data_Z_coord[z_idx]
                
                current_R = np.sqrt(current_X**2 + current_Y**2)
                current_zeta = np.arctan2(current_Y, current_X)
                
                # Poloidal flux conversion
                current_pol_flux = interp_poloidal_flux(current_R, current_Z)
                poloidalFlux_grid_all_times_CART[time][x_idx][y_idx][z_idx] = current_pol_flux
                
                # Magnetic flux conversion
                current_Bphi = interp_Bphi(current_R, current_Z)
                current_Br = interp_Br(current_R, current_Z)
                current_Bz = interp_Bz(current_R, current_Z)
                
                current_Bx = (-current_Bphi * np.sin(current_zeta) 
                              + current_Br * np.cos(current_zeta))
                
                current_By = (current_Bphi * np.cos(current_zeta) 
                              + current_Br * np.sin(current_zeta))
                
                Bx_grid_all_times_CART[time][x_idx][y_idx][z_idx] = current_Bx
                By_grid_all_times_CART[time][x_idx][y_idx][z_idx] = current_By
                Bz_grid_all_times_CART[time][x_idx][y_idx][z_idx] = current_Bz

#==============================================================================
#                                  SAVE FILE
#==============================================================================

#
save_file_path = magnetic_data_path + str(shot) + '_equilibrium_data_CART.npz'
np.savez(save_file_path,
         data_X_coord = data_X_coord,
         data_Y_coord = data_Y_coord,
         data_Z_coord = data_Z_coord,
         poloidalFlux_grid_all_times = poloidalFlux_grid_all_times_CART,
         Bx_grid_all_times = Bx_grid_all_times_CART,
         By_grid_all_times = By_grid_all_times_CART,
         Bz_grid_all_times = Bz_grid_all_times_CART,
         time_EFIT = time_EFIT)


print('Data conversion completed!')













