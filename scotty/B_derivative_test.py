# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scotty.fun_general import find_waist, find_distance_from_waist,find_q_lab_Cartesian, find_nearest, contract_special
import tikzplotlib


suffix2 = ''

# loadfile = np.load('solver_output' + suffix2 + '.npz')
# tau_array2 = loadfile['tau_array']
# q_R_array2 = loadfile['q_R_array']
# q_zeta_array2 = loadfile['q_zeta_array']
# q_Z_array2 = loadfile['q_Z_array']
# K_R_array2 = loadfile['K_R_array']
# #K_zeta_array2 = loadfile['K_zeta_array']
# K_Z_array2 = loadfile['K_Z_array']
# Psi_3D_output2 = loadfile['Psi_3D_output']
# loadfile.close()

loadfile = np.load('data_output' + suffix2 + '.npz')
B_magnitude = loadfile['B_magnitude']
B_R_output = loadfile['B_R_output']
B_T_output = loadfile['B_T_output']
B_Z_output = loadfile['B_Z_output']
loadfile.close()

loadfile = np.load('analysis_output' + suffix2 + '.npz')
distance_along_line = loadfile['distance_along_line']
loadfile.close()

plt.figure
plt.subplot(2,3,1)
plt.plot(distance_along_line,B_R_output)
plt.subplot(2,3,2)
plt.plot(distance_along_line,B_T_output)
plt.subplot(2,3,3)
plt.plot(distance_along_line,B_Z_output)
plt.subplot(2,3,4)
plt.plot(distance_along_line,np.gradient(B_R_output,distance_along_line))
plt.subplot(2,3,5)
plt.plot(distance_along_line,np.gradient(B_T_output,distance_along_line))
plt.subplot(2,3,6)
plt.plot(distance_along_line,np.gradient(B_Z_output,distance_along_line))
# plt.subplot(1,3,3)
# plt.plot(distance_along_line,)