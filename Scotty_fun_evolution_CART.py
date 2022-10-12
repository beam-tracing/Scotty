# -*- coding: utf-8 -*-
"""
Created: 11/08/2022
Last Updated: 08/09/2022
@author: Tan Zheng Yang

Functions for Scotty to evolve the beam or ray. This has been deliberately
separated from Scotty_fun_general_CART to prevent circular importing. This code
is a modified version of the original from Dr. Valerian Hall-Chen
"""


#============================== FUNCTION IMPORTS ==============================
import numpy as np
import sys

# Functions for \grad H
from Scotty_fun_FFD_CART import find_dH_dX, find_dH_dY, find_dH_dZ 

# Functions for \grad_K H
from Scotty_fun_CFD_CART import find_dH_dKX, find_dH_dKY, find_dH_dKZ

# Functions for \grad \grad H
from Scotty_fun_FFD_CART import find_d2H_dX2, find_d2H_dY2, find_d2H_dZ2
from Scotty_fun_FFD_CART import find_d2H_dX_dY, find_d2H_dX_dZ, find_d2H_dY_dZ

# Functions for \grad_K \grad_K H
from Scotty_fun_CFD_CART import find_d2H_dKX2, find_d2H_dKY2, find_d2H_dKZ2
from Scotty_fun_CFD_CART import find_d2H_dKX_dKY, find_d2H_dKX_dKZ, find_d2H_dKY_dKZ

# Functions for \grad_K \grad H
from Scotty_fun_CFD_CART import find_d2H_dKX_dX, find_d2H_dKX_dY, find_d2H_dKX_dZ
from Scotty_fun_CFD_CART import find_d2H_dKY_dX, find_d2H_dKY_dY, find_d2H_dKY_dZ
from Scotty_fun_CFD_CART import find_d2H_dKZ_dX, find_d2H_dKZ_dY, find_d2H_dKZ_dZ
#==============================================================================


########## Gradients of H ##########
'''
Ignore this part for now (not needeeded for checkpoint 1)
'''


########## Functions for the Solver ##########

def ray_evolution_2D_fun(tau, ray_parameters_2D,
                         launch_angular_frequency, mode_flag,
                         delta_X, delta_Y, delta_Z, 
                         delta_K_X, delta_K_Y, delta_K_Z,
                         find_poloidal_flux, find_density_1D, 
                         find_B_X, find_B_Y, find_B_Z):
    """
    DESCRIPTION
    ==============================
    Calculates the RHS of equations (24) and (25) of Valerian's paper, which 
    are the derivatives of K and q with respect to tau, using the derivatives 
    of H (\grad H and \grad_K H)
    
    Note that in reality, we are using it on tau_bar and H_bar. See equations
    (43)

    INPUT
    ==============================
    tau (float):
        Parameter along the ray. Note that we will actually use this function
        for tau_bar instead of tau
        
    ray_parameters_2D (vector of shape (1,6)):
        Numpy array containing the ray parameters q_X, q_Y, q_Z, K_X, K_Y, K_Z
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.
        
    mode_flag (+1 or -1):
        This corresponds to plus-or-minus symbol in the quadratic formula. 
        Indicates the X and the O mode, see equation (51) in Valerian's paper.
        
    delta_X (float):
        spatial step in the X-direction.
        
    delta_Y (float):
        spatial step in the Y-direction.
        
    delta_Z (float):
        spatial step in the Z-direction.
        
    find_poloidal_flux (function):
        calculates the poloidal flux given a location (q_X, q_Y, q_Z)
        
    find_density_1D (function):
        computes the electron density based on the poloidal flux. See page 12
        of Valerian's paper.
        
     find_B_X (function): 
         Computes the X component of the magnetic field. 
         
     find_B_Y (function): 
         Computes the Y component of the magnetic field. 
         
     find_B_Z (function): 
         Computes the Z component of the magnetic field.

    OUTPUT
    ==============================
    d_ray_parameters_2D_d_tau (vector of shape (1,6)):
        numpy array containing the respective derivatives of q and K with
        respect to tau.

    """
    
    q_X = ray_parameters_2D[0]
    q_Y = ray_parameters_2D[1]
    q_Z = ray_parameters_2D[2]
    K_X = ray_parameters_2D[3]
    K_Y = ray_parameters_2D[4]
    K_Z = ray_parameters_2D[5]
    
    # Calculate the positional derivatives of H for \grad H
    dH_dX = find_dH_dX(delta_X,
                       q_X, q_Y, q_Z, 
                       K_X, K_Y, K_Z, 
                       launch_angular_frequency, mode_flag, 
                       find_poloidal_flux, find_density_1D, 
                       find_B_X, find_B_Y, find_B_Z)
    
    dH_dY = find_dH_dY(delta_Y,
                       q_X, q_Y, q_Z, 
                       K_X, K_Y, K_Z, 
                       launch_angular_frequency, mode_flag, 
                       find_poloidal_flux, find_density_1D, 
                       find_B_X, find_B_Y, find_B_Z)
    
    dH_dZ = find_dH_dZ(delta_Z,
                       q_X, q_Y, q_Z, 
                       K_X, K_Y, K_Z, 
                       launch_angular_frequency, mode_flag, 
                       find_poloidal_flux, find_density_1D, 
                       find_B_X, find_B_Y, find_B_Z)
    
    # Calculate the wave vector derivatives of H for \grad_K H
    dH_dKX = find_dH_dKX(delta_K_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    dH_dKY = find_dH_dKY(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    dH_dKZ = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Create array to store the LHS of equations (24) and (25)
    d_ray_parameters_2D_d_tau = np.zeros_like(ray_parameters_2D)
    
    # Calculate the LHS
    d_ray_parameters_2D_d_tau[0] = dH_dKX   # dq_X/dtau
    d_ray_parameters_2D_d_tau[1] = dH_dKY   # dq_Y/dtau
    d_ray_parameters_2D_d_tau[2] = dH_dKZ   # dq_Z/dtau
    d_ray_parameters_2D_d_tau[3] = -dH_dX   # dK_X/dtau
    d_ray_parameters_2D_d_tau[4] = -dH_dY   # dK_Y/dtau
    d_ray_parameters_2D_d_tau[5] = -dH_dZ   # dK_Z/dtau
    
    return d_ray_parameters_2D_d_tau  
    
    
def beam_evolution_fun(tau, beam_parameters,
                       launch_angular_frequency, mode_flag,
                       delta_X, delta_Y, delta_Z, 
                       delta_K_X, delta_K_Y, delta_K_Z,
                       find_poloidal_flux, find_density_1D, 
                       find_B_X, find_B_Y, find_B_Z):
    """
    DESCRIPTION
    ==============================
    Calculates the RHS of equations (24), (25) and (27) of Valerian's paper, 
    which  are the derivatives of K and q with respect to tau, using the 
    derivatives of H (\grad H and \grad_K H)
    
    Note that in reality, we are using it on tau_bar and H_bar. See equations
    (43)
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Added additional definition statements for the real and imaginary
      components of Psi for code readability

    INPUT
    ==============================
    tau (float):
        Parameter along the ray. Note that we will actually use this function
        for tau_bar instead of tau
        
    beam (vector of shape (1,18)):
        Numpy array containing the beam parameters:
            q_X, q_Y, q_Z, K_X, K_Y, K_Z,
            Psi_XX_real, Psi_XY_real, Psi_XZ_real, 
            Psi_YY_real, Psi_YZ_real, Psi_ZZ_real,
            Psi_XX_imaginary, Psi_XY_imaginary, Psi_XZ_imaginary, 
            Psi_YY_imaginary, Psi_YZ_imaginary, Psi_ZZ_imaginary,
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.
        
    mode_flag (+1 or -1):
        This corresponds to plus-or-minus symbol in the quadratic formula. 
        Indicates the X and the O mode, see equation (51) in Valerian's paper.
        
    delta_X (float):
        spatial step in the X-direction.
        
    delta_Y (float):
        spatial step in the Y-direction.
        
    delta_Z (float):
        spatial step in the Z-direction.
        
    find_poloidal_flux (function):
        calculates the poloidal flux at a given position (q_X, q_Y, q_Z)
        
    find_density_1D (function):
        computes the electron density based on the poloidal flux. See page 12
        of Valerian's paper.
        
     find_B_X (function): 
         Computes the X component of the magnetic field. 
         
     find_B_Y (function): 
         Computes the Y component of the magnetic field. 
         
     find_B_Z (function): 
         Computes the Z component of the magnetic field.

    OUTPUT
    ==============================
    d_ray_parameters_2D_d_tau (vector of shape (1,6)):
        numpy array containing the respective derivatives of the RHS of the 
        system (24), (25), (26)

    """  
    
    q_X = beam_parameters[0]
    q_Y = beam_parameters[1]
    q_Z = beam_parameters[2]
    K_X = beam_parameters[3]
    K_Y = beam_parameters[4]
    K_Z = beam_parameters[5]
    Psi_XX_real = beam_parameters[6]
    Psi_XY_real = beam_parameters[7]
    Psi_XZ_real = beam_parameters[8]
    Psi_YY_real = beam_parameters[9]
    Psi_YZ_real = beam_parameters[10]
    Psi_ZZ_real = beam_parameters[11]
    Psi_XX_imag = beam_parameters[12]
    Psi_XY_imag = beam_parameters[13]
    Psi_XZ_imag = beam_parameters[14]
    Psi_YY_imag = beam_parameters[15]
    Psi_YZ_imag = beam_parameters[16]
    Psi_ZZ_imag = beam_parameters[17]
    
    # Initialize Psi_3D (remember, it is symmetric)
    Psi_3D = np.zeros([3,3],dtype='complex128')     
    Psi_3D[0,0] = Psi_XX_real  + 1j*Psi_XX_imag     # Psi_XX
    Psi_3D[1,1] = Psi_YY_real  + 1j*Psi_YY_imag     # Psi_YY
    Psi_3D[2,2] = Psi_ZZ_real  + 1j*Psi_ZZ_imag     # Psi_ZZ
    Psi_3D[0,1] = Psi_XY_real  + 1j*Psi_XY_imag     # Psi_XY
    Psi_3D[0,2] = Psi_XZ_real  + 1j*Psi_XZ_imag     # Psi_XZ
    Psi_3D[1,2] = Psi_YZ_real  + 1j*Psi_YZ_imag     # Psi_YZ
    Psi_3D[1,0] = Psi_3D[0,1] 
    Psi_3D[2,0] = Psi_3D[0,2]
    Psi_3D[2,1] = Psi_3D[1,2]
    
    # Calculate the positional derivatives of H for \grad H
    dH_dX = find_dH_dX(delta_X,
                       q_X, q_Y, q_Z, 
                       K_X, K_Y, K_Z, 
                       launch_angular_frequency, mode_flag, 
                       find_poloidal_flux, find_density_1D, 
                       find_B_X, find_B_Y, find_B_Z)
    
    dH_dY = find_dH_dY(delta_Y,
                       q_X, q_Y, q_Z, 
                       K_X, K_Y, K_Z, 
                       launch_angular_frequency, mode_flag, 
                       find_poloidal_flux, find_density_1D, 
                       find_B_X, find_B_Y, find_B_Z)
    
    dH_dZ = find_dH_dZ(delta_Z,
                       q_X, q_Y, q_Z, 
                       K_X, K_Y, K_Z, 
                       launch_angular_frequency, mode_flag, 
                       find_poloidal_flux, find_density_1D, 
                       find_B_X, find_B_Y, find_B_Z)
    
    # Calculate the wave vector derivatives of H for \grad_K H
    dH_dKX = find_dH_dKX(delta_K_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    dH_dKY = find_dH_dKY(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    dH_dKZ = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Calculate the second order position derivatives of H for \grad \grad H
    d2H_dX2 = find_d2H_dX2(delta_X,
                           q_X, q_Y, q_Z, 
                           K_X, K_Y, K_Z, 
                           launch_angular_frequency, mode_flag, 
                           find_poloidal_flux, find_density_1D, 
                           find_B_X, find_B_Y, find_B_Z)
    
    d2H_dY2 = find_d2H_dY2(delta_Y,
                           q_X, q_Y, q_Z, 
                           K_X, K_Y, K_Z, 
                           launch_angular_frequency, mode_flag, 
                           find_poloidal_flux, find_density_1D, 
                           find_B_X, find_B_Y, find_B_Z)
    
    d2H_dZ2 = find_d2H_dZ2(delta_Z,
                           q_X, q_Y, q_Z, 
                           K_X, K_Y, K_Z, 
                           launch_angular_frequency, mode_flag, 
                           find_poloidal_flux, find_density_1D, 
                           find_B_X, find_B_Y, find_B_Z)
    
    d2H_dX_dY = find_d2H_dX_dY(delta_X, delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    d2H_dX_dZ = find_d2H_dX_dZ(delta_X, delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    d2H_dY_dZ = find_d2H_dY_dZ(delta_Y, delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Calculate the second order wave vector derivatives of H for 
    # \grad_K \grad_K H
    
    d2H_dKX2 = find_d2H_dKX2(delta_K_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKY2 = find_d2H_dKY2(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKZ2 = find_d2H_dKZ2(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKX_dKY = find_d2H_dKX_dKY(delta_K_X, delta_K_Y,
                                   q_X, q_Y, q_Z, 
                                   K_X, K_Y, K_Z, 
                                   launch_angular_frequency, mode_flag,
                                   find_poloidal_flux, find_density_1D, 
                                   find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKX_dKZ = find_d2H_dKX_dKZ(delta_K_X, delta_K_Z,
                                   q_X, q_Y, q_Z, 
                                   K_X, K_Y, K_Z, 
                                   launch_angular_frequency, mode_flag,
                                   find_poloidal_flux, find_density_1D, 
                                   find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKY_dKZ = find_d2H_dKY_dKZ(delta_K_Y, delta_K_Z,
                                   q_X, q_Y, q_Z, 
                                   K_X, K_Y, K_Z, 
                                   launch_angular_frequency, mode_flag,
                                   find_poloidal_flux, find_density_1D, 
                                   find_B_X, find_B_Y, find_B_Z)
    
    # Calculate the second order wave vector derivatives of H for 
    # \grad_K \grad H
    d2H_dKX_dX = find_d2H_dKX_dX(delta_K_X, delta_X,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKX_dY = find_d2H_dKX_dY(delta_K_X, delta_Y,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKX_dZ = find_d2H_dKX_dZ(delta_K_X, delta_Z,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKY_dX = find_d2H_dKY_dX(delta_K_Y, delta_X,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKY_dY = find_d2H_dKY_dY(delta_K_Y, delta_Y,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKY_dZ = find_d2H_dKY_dZ(delta_K_Y, delta_Z,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKZ_dX = find_d2H_dKZ_dX(delta_K_Z, delta_X,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKZ_dY = find_d2H_dKZ_dY(delta_K_Z, delta_Y,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    d2H_dKZ_dZ = find_d2H_dKZ_dZ(delta_K_Z, delta_Z,
                                 q_X, q_Y, q_Z, 
                                 K_X, K_Y, K_Z, 
                                 launch_angular_frequency, mode_flag,
                                 find_poloidal_flux, find_density_1D, 
                                 find_B_X, find_B_Y, find_B_Z)
    
    # Initialize the grad grad H matrix
    #
    # np.squeeze and .item() are there as safety measures. np.squeeze() ensures
    # there are no 1D axes, e.g. converts [[1, 2, 3]] into [1, 2, 3]
    # on the other hand .item() ensures we get floats and not arrays. e.g. 
    # converts [2] into 2
    grad_grad_H = np.squeeze(np.array([
                [d2H_dX2.item()  , d2H_dX_dY.item(), d2H_dX_dZ.item()],
                [d2H_dX_dY.item(), d2H_dY2.item()  , d2H_dY_dZ.item()],
                [d2H_dX_dZ.item(), d2H_dY_dZ.item(), d2H_dZ2.item()] 
                ]))
    
    # Initialize the grad_K grad_K H matrix 
    gradK_gradK_H = np.squeeze(np.array([
                [d2H_dKX2.item()   , d2H_dKX_dKY.item(), d2H_dKX_dKZ.item()],
                [d2H_dKX_dKY.item(), d2H_dKY2.item()   , d2H_dKY_dKZ.item()],
                [d2H_dKX_dKZ.item(), d2H_dKY_dKZ.item(), d2H_dKZ2.item()   ]
                ]))
    
    # Initialize the gradK grad H matrix
    gradK_grad_H = np.squeeze(np.array([
                [d2H_dKX_dX.item(), d2H_dKX_dY.item(), d2H_dKX_dZ.item()],
                [d2H_dKY_dX.item(), d2H_dKY_dY.item(), d2H_dKY_dZ.item()],
                [d2H_dKZ_dX.item(), d2H_dKZ_dY.item(), d2H_dKZ_dZ.item()]
                ]))
    
    # Initialize the grad gradK H matrix
    grad_gradK_H = np.transpose(gradK_grad_H)
    
    # Equation (26)
    d_Psi_d_tau = ( - grad_grad_H
                    - np.matmul(
                                Psi_3D, gradK_grad_H
                                )
                    - np.matmul(
                                grad_gradK_H, Psi_3D
                                )
                    - np.matmul(np.matmul(
                                            Psi_3D, gradK_gradK_H
                                            ),
                                Psi_3D
                                )
                    )
    
    
    # Calculate the LHS of equations (24), (25), (27)
    d_beam_parameters_d_tau = np.zeros_like(beam_parameters)
    
    d_beam_parameters_d_tau[0] = dH_dKX # dq_X/dtau
    d_beam_parameters_d_tau[1] = dH_dKY # dq_Y/dtau
    d_beam_parameters_d_tau[2] = dH_dKZ # dq_Z/dtau
    
    d_beam_parameters_d_tau[3] = - dH_dX # dK_X/dtau
    d_beam_parameters_d_tau[4] = - dH_dY # dK_Y/dtau
    d_beam_parameters_d_tau[5] = - dH_dZ # dK_Z/dtau
    
    d_beam_parameters_d_tau[6]  = np.real(d_Psi_d_tau[0,0]) # real part of d(Psi_XX)/dtau
    d_beam_parameters_d_tau[7]  = np.real(d_Psi_d_tau[0,1]) # real part of d(Psi_XY)/dtau
    d_beam_parameters_d_tau[8]  = np.real(d_Psi_d_tau[0,2]) # real part of d(Psi_XZ)/dtau
    d_beam_parameters_d_tau[9]  = np.real(d_Psi_d_tau[1,1]) # real part of d(Psi_YY)/dtau
    d_beam_parameters_d_tau[10]  = np.real(d_Psi_d_tau[1,2]) # real part of d(Psi_YZ)/dtau
    d_beam_parameters_d_tau[11]  = np.real(d_Psi_d_tau[2,2]) # real part of d(Psi_ZZ)/dtau
    
    d_beam_parameters_d_tau[12]  = np.imag(d_Psi_d_tau[0,0]) # imag part of d(Psi_XX)/dtau
    d_beam_parameters_d_tau[13]  = np.imag(d_Psi_d_tau[0,1]) # imag part of d(Psi_XY)/dtau
    d_beam_parameters_d_tau[14]  = np.imag(d_Psi_d_tau[0,2]) # imag part of d(Psi_XZ)/dtau
    d_beam_parameters_d_tau[15]  = np.imag(d_Psi_d_tau[1,1]) # imag part of d(Psi_YY)/dtau
    d_beam_parameters_d_tau[16]  = np.imag(d_Psi_d_tau[1,2]) # imag part of d(Psi_YZ)/dtau
    d_beam_parameters_d_tau[17]  = np.imag(d_Psi_d_tau[2,2]) # imag part of d(Psi_ZZ)/dtau
    
    ## FOR DEBUGGING
    print('dH_dKX = ' + str(dH_dKX))
    print('dH_dKY = ' + str(dH_dKY))
    print('dH_dKZ = ' + str(dH_dKZ))
    
    print('dH_dX = ' + str(dH_dX))
    print('dH_dY = ' + str(dH_dY))
    print('dH_dZ = ' + str(dH_dZ))
    
    print('d2H_dX2 = ' + str(d2H_dX2))
    print('d2H_dY2 = ' + str(d2H_dY2))
    print('d2H_dZ2 = ' + str(d2H_dZ2))
    
    print('d2H_dX_dY = ' + str(d2H_dX_dY))
    print('d2H_dX_dZ = ' + str(d2H_dX_dZ))
    print('d2H_dY_dZ = ' + str(d2H_dY_dZ))
    
    print('d2H_dKX2 = ' + str(d2H_dKX2))
    print('d2H_dKY2 = ' + str(d2H_dKY2))
    print('d2H_dKZ2 = ' + str(d2H_dKZ2))
    
    print('d2H_dKX_dKY = ' + str(d2H_dKX_dKY))
    print('d2H_dKX_dKZ = ' + str(d2H_dKX_dKZ))
    print('d2H_dKY_dKZ = ' + str(d2H_dKY_dKZ))
    
    print('d2H_dKX_dX = ' + str(d2H_dKX_dX))
    print('d2H_dKX_dY = ' + str(d2H_dKX_dY))
    print('d2H_dKX_dZ = ' + str(d2H_dKX_dZ))
    
    print('d2H_dKY_dX = ' + str(d2H_dKX_dX))
    print('d2H_dKY_dY = ' + str(d2H_dKX_dY))
    print('d2H_dKY_dZ = ' + str(d2H_dKX_dZ))
    
    print('d2H_dKZ_dX = ' + str(d2H_dKX_dX))
    print('d2H_dKZ_dY = ' + str(d2H_dKX_dY))
    print('d2H_dKZ_dZ = ' + str(d2H_dKX_dZ))
    
    
    sys.exit()
    
    return d_beam_parameters_d_tau



















