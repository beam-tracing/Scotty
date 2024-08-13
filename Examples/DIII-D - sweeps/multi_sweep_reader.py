# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 00:53:03 2024

@author: matth
"""


from matplotlib.ticker import FormatStrFormatter
import xarray as xr
import numpy as np
import datatree
import matplotlib.pyplot as plt
from math import *
import scipy
from scipy import interpolate,optimize,fmin
from scipy import constants

# extract out sweep file
path = 'C:/Users/matth/Downloads/Scotty/Examples/DIII-D - sweeps/sweep_main.h5'
dt = datatree.open_datatree(path, engine="h5netcdf")

# Example poloidal angle you want to select
selected_pol_angle = -11.4 

# Selecting the data for the given poloidal angle
levels1 = np.linspace(-1,1,1001)

# Filter out data with specific poloidal angle

# mismatch plot
selected_data = dt["outputs"].sel(pol_launch_angle=selected_pol_angle)
plt.contourf(selected_data.coords['tor_launch_angle'],selected_data.coords['frequency'],np.transpose(selected_data['mismatch_angle_plot'].values),cmap='seismic',levels = levels1, vmin= -1, vmax = 1)

# rho cutoffs
rhocutplot = selected_data['rho_cutoff_plot'].values
plt.figure()
plt.plot(selected_data.coords['tor_launch_angle'],rhocutplot)