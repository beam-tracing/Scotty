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
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

ne_start = 4
ne_end = 0.5

radialcoord_array = np.linspace(0,1.2,121)
density_array = np.zeros_like(radialcoord_array)

densities_to_load = np.linspace(4,0.5,101)
for idx, density in enumerate(densities_to_load):
    density_array[idx] = density

plt.figure()
plt.scatter(radialcoord_array,density_array,color='black')
plt.ylim([0, 4])
plt.xlim([0, 1.5])


#density_fit_array[density_fit_array<0] = 0
output_length = len(radialcoord_array)

with open('ne.dat','w') as ne_data_file:
    ne_data_file.write(f"{int(output_length)}\n") 
    for ii in range(0, output_length):
        ne_data_file.write(f'{radialcoord_array[ii]:.8e} {density_array[ii]:.8e} \n')        