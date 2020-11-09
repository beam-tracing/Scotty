# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:06:45 2020

@author: VH Chen
"""


import numpy as np
import matplotlib.pyplot as plt

from Scotty_fun_general import find_widths_and_curvatures, find_Psi_3D_lab_Cartesian, find_Psi_3D_lab, contract_special

loadfile = np.load('data_input0.npz')
launch_position = loadfile['launch_position']
launch_K = loadfile['launch_K']
loadfile.close()

loadfile = np.load('data_output0.npz')
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
q_zeta_array = loadfile['q_zeta_array']
q_Z_array = loadfile['q_Z_array']
K_R_array = loadfile['K_R_array']
K_zeta_initial = loadfile['K_zeta_initial']
K_Z_array = loadfile['K_Z_array']
b_hat_output = loadfile['b_hat_output']
g_hat_output = loadfile['g_hat_output']
x_hat_output = loadfile['x_hat_output']
y_hat_output = loadfile['y_hat_output']
Psi_3D_output = loadfile['Psi_3D_output']
Psi_3D_lab_launch = loadfile['Psi_3D_lab_launch']
dH_dKR_output = loadfile['dH_dKR_output']
dH_dKzeta_output = loadfile['dH_dKzeta_output']
dH_dKZ_output = loadfile['dH_dKZ_output']
loadfile.close()

loadfile = np.load('analysis_output0.npz')
Psi_xx_output = loadfile['Psi_xx_output']
Psi_xy_output = loadfile['Psi_xy_output']
Psi_yy_output = loadfile['Psi_yy_output']
K_magnitude_array = loadfile['K_magnitude_array']
loadfile.close()

numberOfDataPoints = np.size(K_magnitude_array)


dH_dKX_output = dH_dKR_output * np.cos(q_zeta_array ) - dH_dKzeta_output * q_R_array  * np.sin(q_zeta_array )
dH_dKY_output = dH_dKR_output * np.sin(q_zeta_array ) + dH_dKzeta_output * q_R_array  * np.cos(q_zeta_array )

g_hat_Cartesian_output = np.zeros([numberOfDataPoints,3])
g_magnitude_output = (dH_dKX_output**2 + dH_dKY_output**2 + dH_dKZ_output**2)**0.5
g_hat_Cartesian_output[:,0] = dH_dKX_output / g_magnitude_output
g_hat_Cartesian_output[:,1] = dH_dKY_output / g_magnitude_output
g_hat_Cartesian_output[:,2] = dH_dKZ_output / g_magnitude_output
    
b_hat_Cartesian = np.zeros([numberOfDataPoints,3])
b_hat_Cartesian[:,0] = b_hat_output[:,0]*np.cos(q_zeta_array ) - b_hat_output[:,1]*np.sin(q_zeta_array )
b_hat_Cartesian[:,1] = b_hat_output[:,0]*np.sin(q_zeta_array ) + b_hat_output[:,1]*np.cos(q_zeta_array )
b_hat_Cartesian[:,2] = b_hat_output[:,2]

y_hat_Cartesian = np.zeros([numberOfDataPoints,3])
x_hat_Cartesian = np.zeros([numberOfDataPoints,3])
y_Cartesian = np.cross(b_hat_Cartesian,g_hat_Cartesian_output) 
x_Cartesian = np.cross(g_hat_Cartesian_output,y_Cartesian) 
y_Cartesian_magnitude = np.linalg.norm(y_Cartesian,axis=1)
x_Cartesian_magnitude = np.linalg.norm(x_Cartesian,axis=1)
y_hat_Cartesian[:,0] = y_Cartesian[:,0] / y_Cartesian_magnitude
y_hat_Cartesian[:,1] = y_Cartesian[:,1] / y_Cartesian_magnitude
y_hat_Cartesian[:,2] = y_Cartesian[:,2] / y_Cartesian_magnitude
x_hat_Cartesian[:,0] = x_Cartesian[:,0] / x_Cartesian_magnitude
x_hat_Cartesian[:,1] = x_Cartesian[:,1] / x_Cartesian_magnitude
x_hat_Cartesian[:,2] = x_Cartesian[:,2] / x_Cartesian_magnitude

y_hat_Cartesian2 = np.zeros([numberOfDataPoints,3])
x_hat_Cartesian2 = np.zeros([numberOfDataPoints,3])
y_hat_Cartesian2[:,0] = y_hat_output[:,0]*np.cos(q_zeta_array ) - y_hat_output[:,1]*np.sin(q_zeta_array )
y_hat_Cartesian2[:,1] = y_hat_output[:,0]*np.sin(q_zeta_array ) + y_hat_output[:,1]*np.cos(q_zeta_array )
y_hat_Cartesian2[:,2] = y_hat_output[:,2]
x_hat_Cartesian2[:,0] = x_hat_output[:,0]*np.cos(q_zeta_array ) - x_hat_output[:,1]*np.sin(q_zeta_array )
x_hat_Cartesian2[:,1] = x_hat_output[:,0]*np.sin(q_zeta_array ) + x_hat_output[:,1]*np.cos(q_zeta_array )
x_hat_Cartesian2[:,2] = x_hat_output[:,2]



Psi_3D_Cartesian = find_Psi_3D_lab_Cartesian(Psi_3D_output, q_R_array, q_zeta_array, K_R_array, K_zeta_initial)
Psi_xx_Cartesian = contract_special(x_hat_Cartesian,contract_special(Psi_3D_Cartesian,x_hat_Cartesian))
Psi_xy_Cartesian = contract_special(x_hat_Cartesian,contract_special(Psi_3D_Cartesian,y_hat_Cartesian))
Psi_yy_Cartesian = contract_special(y_hat_Cartesian,contract_special(Psi_3D_Cartesian,y_hat_Cartesian))

Psi_xx_Cartesian2 = np.zeros(numberOfDataPoints)
Psi_xy_Cartesian2 = np.zeros(numberOfDataPoints)
Psi_yy_Cartesian2 = np.zeros(numberOfDataPoints)
for ii in range(0,numberOfDataPoints):
    Psi_3D_Cartesian2 = find_Psi_3D_lab_Cartesian(Psi_3D_output[ii,:,:], q_R_array[ii], q_zeta_array[ii], K_R_array[ii], K_zeta_initial)
    Psi_xx_Cartesian2[ii] = np.dot(x_hat_Cartesian[ii,:],np.dot(Psi_3D_Cartesian2,x_hat_Cartesian[ii,:]))
    Psi_xy_Cartesian2[ii] = np.dot(x_hat_Cartesian[ii,:],np.dot(Psi_3D_Cartesian2,y_hat_Cartesian[ii,:]))
    Psi_yy_Cartesian2[ii] = np.dot(y_hat_Cartesian[ii,:],np.dot(Psi_3D_Cartesian2,y_hat_Cartesian[ii,:]))

plt.figure()
plt.subplot(1,3,1)
plt.plot(Psi_xx_Cartesian,'r')
plt.plot(Psi_xx_Cartesian2,'k')
plt.subplot(1,3,2)
plt.plot(Psi_xy_Cartesian,'r')
plt.plot(Psi_xy_Cartesian2,'k')
plt.subplot(1,3,3)
plt.plot(Psi_yy_Cartesian,'r')
plt.plot(Psi_yy_Cartesian2,'k')

plt.figure()
plt.subplot(1,3,1)
plt.plot(y_hat_Cartesian[:,0],'r')
plt.plot(y_hat_Cartesian2[:,0],'k')
plt.subplot(1,3,2)
plt.plot(y_hat_Cartesian[:,1],'r')
plt.plot(y_hat_Cartesian2[:,1],'k')
plt.subplot(1,3,3)
plt.plot(y_hat_Cartesian[:,2],'r')
plt.plot(y_hat_Cartesian2[:,2],'k')


plt.figure()
plt.subplot(1,3,1)
plt.plot(x_hat_Cartesian[:,0],'r')
plt.plot(x_hat_Cartesian2[:,0],'k')
plt.subplot(1,3,2)
plt.plot(x_hat_Cartesian[:,1],'r')
plt.plot(x_hat_Cartesian2[:,1],'k')
plt.subplot(1,3,3)
plt.plot(x_hat_Cartesian[:,2],'r')
plt.plot(x_hat_Cartesian2[:,2],'k')

plt.figure()
plt.plot(y_hat_Cartesian[:,0]**2  + y_hat_Cartesian[:,1]**2  + y_hat_Cartesian[:,2]**2,'r')
plt.plot(y_hat_Cartesian2[:,0]**2 + y_hat_Cartesian2[:,1]**2 + y_hat_Cartesian2[:,2]**2,'k')
plt.figure()
plt.plot(x_hat_Cartesian[:,0]**2  + x_hat_Cartesian[:,1]**2  + x_hat_Cartesian[:,2]**2,'r')
plt.plot(x_hat_Cartesian2[:,0]**2 + x_hat_Cartesian2[:,1]**2 + x_hat_Cartesian2[:,2]**2,'k')

#
#Psi_w_Cartesian = np.array([
#        [Psi_xx_Cartesian,Psi_xy_Cartesian],
#        [Psi_xy_Cartesian,Psi_yy_Cartesian]
#        ])
#
#Psi_w = np.array([
#        [Psi_xx_output[0],Psi_xy_output[0]],
#        [Psi_xy_output[0],Psi_yy_output[0]]
#        ])
#
#
#
#Psi_xx2 = np.matmul(x_hat_output[0,:],np.matmul(Psi_3D_output[0,:,:],x_hat_output[0,:]))
#Psi_xy2 = np.matmul(x_hat_output[0,:],np.matmul(Psi_3D_output[0,:,:],y_hat_output[0,:]))
#Psi_yy2 = np.matmul(y_hat_output[0,:],np.matmul(Psi_3D_output[0,:,:],y_hat_output[0,:]))
#
#Psi_w2 = np.array([
#        [Psi_xx2,Psi_xy2],
#        [Psi_xy2,Psi_yy2]
#        ])
## ------------------------------
#
#W_xx_entry = np.sqrt(2/np.imag(Psi_xx_entry))
#W_xy_entry = np.sign(np.imag(Psi_xy_entry))*np.sqrt(2/abs(np.imag(Psi_xy_entry)))
#W_yy_entry = np.sqrt(2/np.imag(Psi_yy_entry))
#R_xx_entry = K_magnitude_entry/np.real(Psi_xx_entry)
#R_xy_entry = K_magnitude_entry/np.real(Psi_xy_entry)
#R_yy_entry = K_magnitude_entry/np.real(Psi_yy_entry)
#
#W_xx_initial = np.sqrt(2/np.imag(Psi_xx_output[0]))
#W_xy_initial = np.sign(np.imag(Psi_xy_output[0]))*np.sqrt(2/abs(np.imag(Psi_xy_output[0])))
#W_yy_initial = np.sqrt(2/np.imag(Psi_yy_output[0]))
#R_xx_initial = K_magnitude_array[0]/np.real(Psi_xx_output[0])
#R_xy_initial = K_magnitude_array[0]/np.real(Psi_xy_output[0])
#R_yy_initial = K_magnitude_array[0]/np.real(Psi_yy_output[0])
#
#numberOfPlotPoints=25
#W_ellipse_points_entry = np.zeros([2,numberOfPlotPoints])
#W_ellipse_points_initial = np.zeros([2,numberOfPlotPoints])
#
#R_ellipse_points_entry = np.zeros([2,numberOfPlotPoints])
#R_ellipse_points_initial = np.zeros([2,numberOfPlotPoints])
#
#W_matrix_entry = np.array([
#            [W_xx_entry,W_xy_entry],
#            [W_xy_entry,W_yy_entry]
#            ])
#W_matrix_initial = np.array([
#            [W_xx_initial,W_xy_initial],
#            [W_xy_initial,W_yy_initial]
#            ])
#
#R_matrix_entry = np.array([
#            [R_xx_entry,R_xy_entry],
#            [R_xy_entry,R_yy_entry]
#            ])
#R_matrix_initial = np.array([
#            [R_xx_initial,R_xy_initial],
#            [R_xy_initial,R_yy_initial]
#            ])
#
#dummy_array = np.array([np.cos(np.linspace(0,2*np.pi,numberOfPlotPoints)), np.sin(np.linspace(0,2*np.pi,numberOfPlotPoints))])
#for ii in range(0,numberOfPlotPoints):
#    W_ellipse_points_entry[:,ii] = np.matmul(W_matrix_entry,dummy_array[:,ii])
#    W_ellipse_points_initial[:,ii] = np.matmul(W_matrix_initial,dummy_array[:,ii])
#
#    R_ellipse_points_entry[:,ii] = np.matmul(R_matrix_entry,dummy_array[:,ii])
#    R_ellipse_points_initial[:,ii] = np.matmul(R_matrix_initial,dummy_array[:,ii])
#
#
#
#
#plt.figure()
#plt.subplot(1,2,1)
#plt.plot(W_ellipse_points_entry[0,:],W_ellipse_points_entry[1,:],'r')
#plt.plot(W_ellipse_points_initial[0,:],W_ellipse_points_initial[1,:],'g')
#plt.plot(widths_entry[0]*widths_entry_eigvec[0,0],widths_entry[0]*widths_entry_eigvec[0,1],'ro') 
#plt.plot(widths_entry[1]*widths_entry_eigvec[1,0],widths_entry[1]*widths_entry_eigvec[1,1],'ro') 
#plt.axhline(y=0, color='k')
#plt.axvline(x=0, color='k')
#
#plt.subplot(1,2,2)
#plt.plot(R_ellipse_points_entry[0,:],R_ellipse_points_entry[1,:],'r')
#plt.plot(R_ellipse_points_initial[0,:],R_ellipse_points_initial[1,:],'g')
#plt.axhline(y=0, color='k')
#plt.axvline(x=0, color='k')