# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:20:11 2022

@author: tanzy


UNIT TESTING OF CARTESIAN FUNCTION find_Psi_3D_plasma
"""

import numpy as np

from Scotty_fun_general_CART import find_Psi_3D_plasma as find_Psi_3D_plasma_CART
from Scotty_fun_general_cylin_test import find_Psi_3D_plasma as find_Psi_3D_plasma_CYLI
from Scotty_fun_general_cylin_test import find_Psi_3D_lab_Cartesian
from Scotty_fun_general_cylin_test import find_Psi_3D_lab


#=================================== SETUP ===================================
# Conversions from cylindrical to cartesian can be found in Scotty_fun_general
# (original source code)

# Interface coordinates (cartesian)
q_R = 4.2
q_zeta = 7.6
q_Z = 1.3

# Interface coordinates (cylindrical)
q_X = q_R * np.cos(q_zeta)
q_Y = q_R * np.sin(q_zeta)

# Wavevector coordinates (cartesian)
K_R = 2.3
K_zeta = 3.1
K_Z = 8.5

# Wavevector coordinates (cylindrical) --> See eqn (56) to (58) Valerian paper
K_X = K_R*np.cos( q_zeta ) - K_zeta*np.sin( q_zeta ) / q_R 

K_Y = K_R*np.sin( q_zeta ) + K_zeta*np.cos( q_zeta ) / q_R 


# Psi in vacuum (cylindrical)
Psi_v_CYLI = np.zeros((3, 3))

Psi_v_RR= 10.7
Psi_v_Rzeta= 7.6
Psi_v_RZ= 5.3
Psi_v_zetazeta= 9.9
Psi_v_zetaZ= 12.2
Psi_v_ZZ= 17.8

Psi_v_CYLI[0,0] = Psi_v_RR
Psi_v_CYLI[0,1] = Psi_v_Rzeta
Psi_v_CYLI[0,2] = Psi_v_RZ
Psi_v_CYLI[1,0] = Psi_v_CYLI[0,1]
Psi_v_CYLI[1,1] = Psi_v_zetazeta
Psi_v_CYLI[1,2] = Psi_v_zetaZ
Psi_v_CYLI[2,0] = Psi_v_CYLI[0,2]
Psi_v_CYLI[2,1] = Psi_v_CYLI[1,2] 
Psi_v_CYLI[2,2] = Psi_v_ZZ


# Position derivative of H (Cylindrical)
dH_dR = 4.7
dH_dzeta = 0 # cylindrical symmetry
dH_dZ = 6.7

# Position derivatives of H (Cartesian)
dH_dX = ((q_X/np.sqrt(q_X**2 + q_Y**2)) * dH_dR
         - (q_Y/(q_X**2 + q_Y**2)) * dH_dzeta)
dH_dY = ((q_Y/np.sqrt(q_X**2 + q_Y**2)) * dH_dR
         + (q_X/(q_X**2 + q_Y**2)) * dH_dzeta)

# Wavevector derivatives of H (Cylindrical)
dH_dKR = 0.4
dH_dKzeta = 2.9
dH_dKZ = 1.5

# Wavevector derivatives of H (Cartesian)
dH_dKX = np.cos(q_zeta) * dH_dKR - q_R * np.sin(q_zeta) * dH_dKzeta
dH_dKY = np.sin(q_zeta) * dH_dKR + q_R * np.cos(q_zeta) * dH_dKzeta 

# Position derivatives of poloidal flux (Cartesian)
d_poloidal_flux_dR = 8.1
d_poloidal_flux_dzeta= 0
d_poloidal_flux_dZ = 5.5

# Position derivatives of poloidal flux (Cylindrical)
d_poloidal_flux_dX = ((q_X/np.sqrt(q_X**2 + q_Y**2)) * d_poloidal_flux_dR
                      - (q_Y/(q_X**2 + q_Y**2)) * d_poloidal_flux_dzeta)

d_poloidal_flux_dY = ((q_Y/np.sqrt(q_X**2 + q_Y**2)) * d_poloidal_flux_dR
                      + (q_X/(q_X**2 + q_Y**2)) * d_poloidal_flux_dzeta)



#=========================== TESTING (CYLINDRICAL) ===========================

# Find Psi in plasma
Psi_p_CYLI = find_Psi_3D_plasma_CYLI(Psi_v_CYLI,
                       dH_dKR, dH_dKzeta, dH_dKZ,
                        dH_dR, dH_dZ,
                        d_poloidal_flux_dR, d_poloidal_flux_dZ)

print(Psi_p_CYLI)


### Test that conversions are working properly
# Psi_p_CART_test = find_Psi_3D_lab_Cartesian(Psi_p_CYLI, q_R, q_zeta, K_R, K_zeta)
# Psi_p_CYLI_test = find_Psi_3D_lab(Psi_p_CART_test, q_R, q_zeta, K_R, K_zeta)
# print(Psi_p_CYLI_test)

print('===========================================')
#=========================== TESTING (CARTESIAN) ===========================

# Convert Cylindrical Psi_v to Cartesian Psi_v
Psi_v_CART = find_Psi_3D_lab_Cartesian(Psi_v_CYLI, q_R, q_zeta, K_R, K_zeta)

# Find Psi in plasma
Psi_p_CART = find_Psi_3D_plasma_CART(Psi_v_CART,
                       q_zeta, #q_R,
                       #K_X, K_Y, 
                       dH_dKX, dH_dKY, dH_dKZ,
                       dH_dX, dH_dY, dH_dZ,
                       d_poloidal_flux_dX, d_poloidal_flux_dY,
                       d_poloidal_flux_dZ)

# Convert back to Cylindrical
print(find_Psi_3D_lab(Psi_p_CART, q_R, q_zeta, K_R, K_zeta))











