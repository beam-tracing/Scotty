# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from Scotty_fun_general import find_waist, find_distance_from_waist,find_q_lab_Cartesian, find_nearest, contract_special
import tikzplotlib

suffix1 = 'benchmark'

loadfile = np.load('data_output' + suffix1 + '.npz')
tau_array1 = loadfile['tau_array']
q_R_array1 = loadfile['q_R_array']
q_zeta_array1 = loadfile['q_zeta_array']
q_Z_array1 = loadfile['q_Z_array']
K_R_array1 = loadfile['K_R_array']
#K_zeta_array1 = loadfile['K_zeta_array']
K_Z_array1 = loadfile['K_Z_array']
Psi_3D_output1 = loadfile['Psi_3D_output']
dH_dKR_output1 = loadfile['dH_dKR_output']
dH_dKzeta_output1 = loadfile['dH_dKzeta_output']
dH_dKZ_output1 = loadfile['dH_dKZ_output']

x_hat_output1 = loadfile['x_hat_output']
y_hat_output1 = loadfile['y_hat_output']
b_hat_output1 = loadfile['b_hat_output']
g_hat_output1 = loadfile['g_hat_output']
loadfile.close()

loadfile = np.load('analysis_output' + suffix1 + '.npz')
M_xx_output1 = loadfile['M_xx_output']
M_xy_output1 = loadfile['M_xy_output']
M_yy_output1 = loadfile['M_yy_output']
Psi_xx_output1 = loadfile['Psi_xx_output']
Psi_xy_output1 = loadfile['Psi_xy_output']
Psi_yy_output1 = loadfile['Psi_yy_output']
delta_k_perp_21 = loadfile['delta_k_perp_2']
delta_theta_m1 = loadfile['delta_theta_m']
theta_m_output1 = loadfile['theta_m_output']
distance_along_line1 = loadfile['distance_along_line']
xhat_dot_grad_bhat_dot_xhat_output1 = loadfile['xhat_dot_grad_bhat_dot_xhat_output']
xhat_dot_grad_bhat_dot_yhat_output1 = loadfile['xhat_dot_grad_bhat_dot_yhat_output']
xhat_dot_grad_bhat_dot_ghat_output1 = loadfile['xhat_dot_grad_bhat_dot_ghat_output']
yhat_dot_grad_bhat_dot_xhat_output1 = loadfile['yhat_dot_grad_bhat_dot_xhat_output']
yhat_dot_grad_bhat_dot_yhat_output1 = loadfile['yhat_dot_grad_bhat_dot_yhat_output']
yhat_dot_grad_bhat_dot_ghat_output1 = loadfile['yhat_dot_grad_bhat_dot_ghat_output']
localisation_ray1 = loadfile['localisation_ray']
loadfile.close()

suffix2 = ''

loadfile = np.load('solver_output' + suffix2 + '.npz')
tau_array2 = loadfile['tau_array']
q_R_array2 = loadfile['q_R_array']
q_zeta_array2 = loadfile['q_zeta_array']
q_Z_array2 = loadfile['q_Z_array']
K_R_array2 = loadfile['K_R_array']
#K_zeta_array2 = loadfile['K_zeta_array']
K_Z_array2 = loadfile['K_Z_array']
Psi_3D_output2 = loadfile['Psi_3D_output']
loadfile.close()

loadfile = np.load('analysis_output' + suffix2 + '.npz')
# dH_dKR_output2 = loadfile['dH_dKR_output']
# dH_dKzeta_output2 = loadfile['dH_dKzeta_output']
# dH_dKZ_output2 = loadfile['dH_dKZ_output']
M_xx_output2 = loadfile['M_xx_output']
M_xy_output2 = loadfile['M_xy_output']
M_yy_output2 = loadfile['M_yy_output']
Psi_xx_output2 = loadfile['Psi_xx_output']
Psi_xy_output2 = loadfile['Psi_xy_output']
Psi_yy_output2 = loadfile['Psi_yy_output']
delta_k_perp_22 = loadfile['delta_k_perp_2']
delta_theta_m2 = loadfile['delta_theta_m']
theta_m_output2 = loadfile['theta_m_output']
distance_along_line2 = loadfile['distance_along_line']
xhat_dot_grad_bhat_dot_xhat_output2 = loadfile['xhat_dot_grad_bhat_dot_xhat_output']
xhat_dot_grad_bhat_dot_yhat_output2 = loadfile['xhat_dot_grad_bhat_dot_yhat_output']
xhat_dot_grad_bhat_dot_ghat_output2 = loadfile['xhat_dot_grad_bhat_dot_ghat_output']
yhat_dot_grad_bhat_dot_xhat_output2 = loadfile['yhat_dot_grad_bhat_dot_xhat_output']
yhat_dot_grad_bhat_dot_yhat_output2 = loadfile['yhat_dot_grad_bhat_dot_yhat_output']
yhat_dot_grad_bhat_dot_ghat_output2 = loadfile['yhat_dot_grad_bhat_dot_ghat_output']
cutoff_index2 = loadfile['cutoff_index']
localisation_ray2 = loadfile['localisation_ray']

x_hat_output2 = loadfile['x_hat_output']
y_hat_output2 = loadfile['y_hat_output']
b_hat_output2 = loadfile['b_hat_output']
g_hat_output2 = loadfile['g_hat_output']
loadfile.close()

plt.figure()
plt.subplot(2,3,1)
plt.plot(tau_array1,q_R_array1,'k')
plt.plot(tau_array2,q_R_array2,'r')
plt.title('R')

plt.subplot(2,3,2)
plt.plot(tau_array1,q_zeta_array1,'k')
plt.plot(tau_array2,q_zeta_array2,'r')
plt.title('zeta')

plt.subplot(2,3,3)
plt.plot(tau_array1,q_Z_array1,'k')
plt.plot(tau_array2,q_Z_array2,'r')
plt.title('Z')

plt.subplot(2,3,4)
plt.plot(tau_array1,K_R_array1,'k')
plt.plot(tau_array2,K_R_array2,'r')
plt.title('K_R')

plt.subplot(2,3,5)


plt.subplot(2,3,6)
plt.plot(tau_array1,K_Z_array1,'k')
plt.plot(tau_array2,K_Z_array2,'r')
plt.title('K_Z')


plt.figure()
plt.subplot(3,3,1)
plt.plot(tau_array1,np.real(Psi_3D_output1)[:,0,0],'k')
plt.plot(tau_array2,np.real(Psi_3D_output2)[:,0,0],'r')
plt.subplot(3,3,2)
plt.plot(tau_array1,np.real(Psi_3D_output1)[:,0,1],'k')
plt.plot(tau_array2,np.real(Psi_3D_output2)[:,0,1],'r')
plt.subplot(3,3,3)
plt.plot(tau_array1,np.real(Psi_3D_output1)[:,0,2],'k')
plt.plot(tau_array2,np.real(Psi_3D_output2)[:,0,2],'r')
plt.subplot(3,3,4)
plt.subplot(3,3,5)
plt.plot(tau_array1,np.real(Psi_3D_output1)[:,1,1],'k')
plt.plot(tau_array2,np.real(Psi_3D_output2)[:,1,1],'r')
plt.subplot(3,3,6)
plt.plot(tau_array1,np.real(Psi_3D_output1)[:,1,2],'k')
plt.plot(tau_array2,np.real(Psi_3D_output2)[:,1,2],'r')
plt.subplot(3,3,7)
plt.subplot(3,3,8)
plt.subplot(3,3,9)
plt.plot(tau_array1,np.real(Psi_3D_output1)[:,2,2],'k')
plt.plot(tau_array2,np.real(Psi_3D_output2)[:,2,2],'r')

plt.figure()
plt.subplot(3,3,1)
plt.plot(tau_array1,np.imag(Psi_3D_output1)[:,0,0],'k')
plt.plot(tau_array2,np.imag(Psi_3D_output2)[:,0,0],'r')
plt.subplot(3,3,2)
plt.plot(tau_array1,np.imag(Psi_3D_output1)[:,0,1],'k')
plt.plot(tau_array2,np.imag(Psi_3D_output2)[:,0,1],'r')
plt.subplot(3,3,3)
plt.plot(tau_array1,np.imag(Psi_3D_output1)[:,0,2],'k')
plt.plot(tau_array2,np.imag(Psi_3D_output2)[:,0,2],'r')
plt.subplot(3,3,4)
plt.subplot(3,3,5)
plt.plot(tau_array1,np.imag(Psi_3D_output1)[:,1,1],'k')
plt.plot(tau_array2,np.imag(Psi_3D_output2)[:,1,1],'r')
plt.subplot(3,3,6)
plt.plot(tau_array1,np.imag(Psi_3D_output1)[:,1,2],'k')
plt.plot(tau_array2,np.imag(Psi_3D_output2)[:,1,2],'r')
plt.subplot(3,3,7)
plt.subplot(3,3,8)
plt.subplot(3,3,9)
plt.plot(tau_array1,np.imag(Psi_3D_output1)[:,2,2],'k')
plt.plot(tau_array2,np.imag(Psi_3D_output2)[:,2,2],'r')

# plt.figure()
# plt.subplot(1,3,1)
# plt.plot(tau_array1,dH_dKR_output1,'k')
# plt.plot(tau_array2,dH_dKR_output2,'r')
# plt.title('dH_dKR')

# plt.subplot(1,3,2)
# plt.plot(tau_array1,dH_dKzeta_output1,'k')
# plt.plot(tau_array2,dH_dKzeta_output2,'r')
# plt.title('dH_dKzeta')

# plt.subplot(1,3,3)
# plt.plot(tau_array1,dH_dKZ_output1,'k')
# plt.plot(tau_array2,dH_dKZ_output2,'r')
# plt.title('dH_dKZ')







# plt.figure()
# plt.subplot(2,3,1)
# plt.plot(distance_along_line1,xhat_dot_grad_bhat_dot_xhat_output1,'r')
# plt.plot(distance_along_line2,xhat_dot_grad_bhat_dot_xhat_output2,'g')
# plt.subplot(2,3,2)
# plt.plot(distance_along_line1,xhat_dot_grad_bhat_dot_yhat_output1,'r')
# plt.plot(distance_along_line2,xhat_dot_grad_bhat_dot_yhat_output2,'g')
# plt.subplot(2,3,3)
# plt.plot(distance_along_line1,xhat_dot_grad_bhat_dot_ghat_output1,'r')
# plt.plot(distance_along_line2,xhat_dot_grad_bhat_dot_ghat_output2,'g')
# plt.subplot(2,3,4)
# plt.plot(distance_along_line1,yhat_dot_grad_bhat_dot_xhat_output1,'r')
# plt.plot(distance_along_line2,yhat_dot_grad_bhat_dot_xhat_output2,'g')
# plt.subplot(2,3,5)
# plt.plot(distance_along_line1,yhat_dot_grad_bhat_dot_yhat_output1,'r')
# plt.plot(distance_along_line2,yhat_dot_grad_bhat_dot_yhat_output2,'g')
# plt.subplot(2,3,6)
# plt.plot(distance_along_line1,yhat_dot_grad_bhat_dot_ghat_output1,'r')
# plt.plot(distance_along_line2,yhat_dot_grad_bhat_dot_ghat_output2,'g')




# plt.figure()
# plt.subplot(2,3,1)
# plt.plot(distance_along_line1,np.real(M_xx_output1),'r')
# plt.plot(distance_along_line2,np.real(M_xx_output2),'g')
# plt.subplot(2,3,2)
# plt.plot(distance_along_line1,np.real(M_xy_output1),'r')
# plt.plot(distance_along_line2,np.real(M_xy_output2),'g')
# plt.subplot(2,3,3)
# plt.plot(distance_along_line1,np.real(M_yy_output1),'r')
# plt.plot(distance_along_line2,np.real(M_yy_output2),'g')
# plt.subplot(2,3,4)
# plt.plot(distance_along_line1,np.imag(M_xx_output1),'r')
# plt.plot(distance_along_line2,np.imag(M_xx_output2),'g')
# plt.subplot(2,3,5)
# plt.plot(distance_along_line1,np.imag(M_xy_output1),'r')
# plt.plot(distance_along_line2,np.imag(M_xy_output2),'g')
# plt.subplot(2,3,6)
# plt.plot(distance_along_line1,np.imag(M_yy_output1),'r')
# plt.plot(distance_along_line2,np.imag(M_yy_output2),'g')

# plt.figure()
# plt.subplot(2,3,1)
# plt.plot(distance_along_line1,np.real(Psi_xx_output1),'r')
# plt.plot(distance_along_line2,np.real(Psi_xx_output2),'g')
# plt.subplot(2,3,2)
# plt.plot(distance_along_line1,np.real(Psi_xy_output1),'r')
# plt.plot(distance_along_line2,np.real(Psi_xy_output2),'g')
# plt.subplot(2,3,3)
# plt.plot(distance_along_line1,np.real(Psi_yy_output1),'r')
# plt.plot(distance_along_line2,np.real(Psi_yy_output2),'g')
# plt.subplot(2,3,4)
# plt.plot(distance_along_line1,np.imag(Psi_xx_output1),'r')
# plt.plot(distance_along_line2,np.imag(Psi_xx_output2),'g')
# plt.subplot(2,3,5)
# plt.plot(distance_along_line1,np.imag(Psi_xy_output1),'r')
# plt.plot(distance_along_line2,np.imag(Psi_xy_output2),'g')
# plt.subplot(2,3,6)
# plt.plot(distance_along_line1,np.imag(Psi_yy_output1),'r')
# plt.plot(distance_along_line2,np.imag(Psi_yy_output2),'g')

# plt.figure()
# plt.subplot(2,3,1)
# plt.plot(distance_along_line1,x_hat_output1[:,0],'r')
# plt.plot(distance_along_line2,x_hat_output2[:,0],'g')
# plt.subplot(2,3,2)
# plt.plot(distance_along_line1,x_hat_output1[:,1],'r')
# plt.plot(distance_along_line2,x_hat_output2[:,1],'g')
# plt.subplot(2,3,3)
# plt.plot(distance_along_line1,x_hat_output1[:,2],'r')
# plt.plot(distance_along_line2,x_hat_output2[:,2],'g')
# plt.subplot(2,3,4)
# plt.plot(distance_along_line1,y_hat_output1[:,0],'r')
# plt.plot(distance_along_line2,y_hat_output2[:,0],'g')
# plt.subplot(2,3,5)
# plt.plot(distance_along_line1,y_hat_output1[:,1],'r')
# plt.plot(distance_along_line2,y_hat_output2[:,1],'g')
# plt.subplot(2,3,6)
# plt.plot(distance_along_line1,y_hat_output1[:,2],'r')
# plt.plot(distance_along_line2,y_hat_output2[:,2],'g')

# plt.figure()
# plt.subplot(2,3,1)
# plt.plot(distance_along_line1,b_hat_output1[:,0],'r')
# plt.plot(distance_along_line2,b_hat_output2[:,0],'g')
# plt.subplot(2,3,2)
# plt.plot(distance_along_line1,b_hat_output1[:,1],'r')
# plt.plot(distance_along_line2,b_hat_output2[:,1],'g')
# plt.subplot(2,3,3)
# plt.plot(distance_along_line1,b_hat_output1[:,2],'r')
# plt.plot(distance_along_line2,b_hat_output2[:,2],'g')
# plt.subplot(2,3,4)
# plt.plot(distance_along_line1,g_hat_output1[:,0],'r')
# plt.plot(distance_along_line2,g_hat_output2[:,0],'g')
# plt.subplot(2,3,5)
# plt.plot(distance_along_line1,g_hat_output1[:,1],'r')
# plt.plot(distance_along_line2,g_hat_output2[:,1],'g')
# plt.subplot(2,3,6)
# plt.plot(distance_along_line1,g_hat_output1[:,2],'r')
# plt.plot(distance_along_line2,g_hat_output2[:,2],'g')

plt.figure()
plt.subplot(1,3,1)
plt.plot(distance_along_line1,delta_k_perp_21,'r')
plt.plot(distance_along_line2,delta_k_perp_22,'g')
plt.subplot(1,3,2)
plt.plot(distance_along_line1,delta_theta_m1,'r')
plt.plot(distance_along_line2,delta_theta_m2,'g')
plt.subplot(1,3,3)
plt.plot(distance_along_line1,theta_m_output1,'r')
plt.plot(distance_along_line2,theta_m_output2,'g')

plt.figure()
plt.subplot(1,3,1)
plt.plot(distance_along_line1,localisation_ray1,'r')
plt.plot(distance_along_line2,localisation_ray2,'g')

