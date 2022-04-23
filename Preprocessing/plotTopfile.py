# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 05:54:07 2019

Let's try smoothing out the electron density data

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolate
from scipy import signal as signal

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

topfile_path ='D:\\Dropbox\\VHChen2021\\Data - Equilibrium\DIII-D\\'

suffix = '_187103_3000ms'
topfile_filename = topfile_path + 'topfile' + suffix

with open(topfile_filename) as f:
    while not 'X-coordinates' in f.readline(): pass # Start reading only from X-coords onwards
    data_R_coord = read_floats_into_list_until('Z-coordinates', f)
    data_Z_coord = read_floats_into_list_until('B_R', f)
    data_B_R_grid = read_floats_into_list_until('B_t', f)
    data_B_T_grid = read_floats_into_list_until('B_Z', f)
    data_B_Z_grid = read_floats_into_list_until('psi', f)
    poloidalFlux_grid = read_floats_into_list_until('you fall asleep', f)
# ------------------------------

## Converts some lists to arrays so that stuff later doesn't complain
data_R_coord = np.array(data_R_coord)
data_Z_coord = np.array(data_Z_coord)

## Row-major and column-major business (Torbeam is in Fortran and Scotty is in Python)
data_B_R_grid = np.transpose((np.asarray(data_B_R_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_B_T_grid = np.transpose((np.asarray(data_B_T_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
data_B_Z_grid = np.transpose((np.asarray(data_B_Z_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
poloidalFlux_grid = np.transpose((np.asarray(poloidalFlux_grid)).reshape(len(data_Z_coord),len(data_R_coord), order='C'))
# -------------------

plt.figure()
plt.subplot(2,2,1)
contour_levels = np.linspace(0,1,11)
CS = plt.contour(data_R_coord, data_Z_coord, np.transpose(poloidalFlux_grid), contour_levels,vmin=0,vmax=1.2,cmap='inferno')
plt.clabel(CS, inline=True, fontsize=10,inline_spacing=-5,fmt= '%1.1f',use_clabeltext=True) # Labels the flux surfaces

plt.subplot(2,2,2)
plt.contourf(data_R_coord, data_Z_coord, np.transpose(data_B_T_grid),101,cmap='inferno')

plt.subplot(2,2,3)
plt.contourf(data_R_coord, data_Z_coord, np.transpose(data_B_R_grid),101,cmap='inferno')

plt.subplot(2,2,4)
plt.contourf(data_R_coord, data_Z_coord, np.transpose(data_B_Z_grid),101,cmap='inferno')

###
# q_R_test = np.linspace(1.0,2.5,101)
# q_Z_test = np.zeros(101) + 0.1

# interp_order = 5
# interp_smoothing = 0

# interp_B_R = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_R_grid, bbox=[None, None, None, None], kx=interp_order, ky=interp_order, s=interp_smoothing)
# interp_B_T = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_T_grid, bbox=[None, None, None, None], kx=interp_order, ky=interp_order, s=interp_smoothing)
# interp_B_Z = interpolate.RectBivariateSpline(data_R_coord,data_Z_coord,data_B_Z_grid, bbox=[None, None, None, None], kx=interp_order, ky=interp_order, s=interp_smoothing)
        
# B_R = interp_B_R(q_R_test,q_Z_test)
# B_T = interp_B_T(q_R_test,q_Z_test)
# B_Z = interp_B_Z(q_R_test,q_Z_test)

# plt.figure()
# plt.plot(B_R)

# plt.figure()
# plt.plot(B_T)

# plt.figure()
# plt.plot(B_Z)