# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:42:37 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt


loadfile = np.load('data_input3.npz')
ne_data_density_array = loadfile['ne_data_density_array']
ne_data_radialcoord_array = loadfile['ne_data_radialcoord_array']
loadfile.close()

ne_data_poloidal_flux_array = ne_data_radialcoord_array**2

dne_dpolflux = np.gradient(ne_data_density_array,ne_data_poloidal_flux_array)
d2ne_dpolflux2 = np.gradient(dne_dpolflux,ne_data_poloidal_flux_array)

plt.plot()
plt.subplot(1,3,1)
plt.plot(ne_data_poloidal_flux_array,ne_data_density_array)
plt.title('ne')
plt.subplot(1,3,2)
plt.plot(ne_data_poloidal_flux_array,dne_dpolflux)
plt.title('d_ne_d_polflux')
plt.subplot(1,3,3)
plt.plot(ne_data_poloidal_flux_array,d2ne_dpolflux2)
plt.title('d2_ne_d_polflux2')