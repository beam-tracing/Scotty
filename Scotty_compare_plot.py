# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from Scotty_fun_general import find_waist, find_distance_from_waist,find_q_lab_Cartesian, find_nearest, contract_special

suffix1 = '9'
suffix2 = '10'

loadfile = np.load('data_output' + suffix1 + '.npz')
q_R_array1 = loadfile['q_R_array']
q_zeta_array1 = loadfile['q_zeta_array']
q_Z_array1 = loadfile['q_Z_array']
K_R_array1 = loadfile['K_R_array']
K_Z_array1 = loadfile['K_Z_array']
Psi_3D_output1 = loadfile['Psi_3D_output']
x_hat_output1 = loadfile['x_hat_output']
y_hat_output1 = loadfile['y_hat_output']
g_hat_output1 = loadfile['g_hat_output']
loadfile.close()

loadfile = np.load('analysis_output' + suffix1 + '.npz')
Psi_xx_output1 = loadfile['Psi_xx_output']
Psi_xy_output1 = loadfile['Psi_xy_output']
Psi_yy_output1 = loadfile['Psi_yy_output']
M_xx_output1 = loadfile['M_xx_output']
M_xy_output1 = loadfile['M_xy_output']
M_yy_output1 = loadfile['M_yy_output']
distance_along_line1 = loadfile['distance_along_line']
loadfile.close()


loadfile = np.load('data_output' + suffix2 + '.npz')
q_R_array2 = loadfile['q_R_array']
q_zeta_array2 = loadfile['q_zeta_array']
q_Z_array2 = loadfile['q_Z_array']
K_R_array2 = loadfile['K_R_array']
K_Z_array2 = loadfile['K_Z_array']
Psi_3D_output2 = loadfile['Psi_3D_output']
x_hat_output2 = loadfile['x_hat_output']
y_hat_output2 = loadfile['y_hat_output']
g_hat_output2 = loadfile['g_hat_output']
loadfile.close()

loadfile = np.load('analysis_output' + suffix2 + '.npz')
Psi_xx_output2 = loadfile['Psi_xx_output']
Psi_xy_output2 = loadfile['Psi_xy_output']
Psi_yy_output2 = loadfile['Psi_yy_output']
M_xx_output2 = loadfile['M_xx_output']
M_xy_output2 = loadfile['M_xy_output']
M_yy_output2 = loadfile['M_yy_output']
distance_along_line2 = loadfile['distance_along_line']
loadfile.close()



plt.figure()
plt.plot(q_R_array1,q_Z_array1,'r')
plt.plot(q_R_array2,q_Z_array2,'b')