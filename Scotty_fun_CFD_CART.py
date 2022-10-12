# -*- coding: utf-8 -*-
"""
Created: 27/07/2022
Last Updated: 08/09/2022
@author: Tan Zheng Yang

Functions for finding derivatives of H using center finite difference.
This code modifies the original source code by Dr. Valerian Hall-Chen from a 
cylindrical version to a cartesian one.
"""

from Scotty_fun_general_CART import find_H


########## First Order Positional Derivatives of H ##########


def find_dH_dX(delta_X,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the first order partial derivative of H with respect to X using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_X (float):
        spatial step in the X direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dX (float or vector of shape (n,)):
        the value of the first order partial derivative in X after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X + delta_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X - delta_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dX
    dH_dX = (H_plus - H_minus) / (2 * delta_X)
    return dH_dX


def find_dH_dY(delta_Y,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the first order partial derivative of H with respect to Y using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_Y (float):
        spatial step in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dY (float or vector of shape (n,)):
        the value of the first order partial derivative in Y after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y + delta_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y - delta_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dY
    dH_dY = (H_plus - H_minus) / (2 * delta_Y)
    return dH_dY


def find_dH_dZ(delta_Z,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the first order partial derivative of H with respect to Z using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dZ (float or vector of shape (n,)):
        the value of the first order partial derivative in Z after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z + delta_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z - delta_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dZ
    dH_dZ = (H_plus - H_minus) / (2 * delta_Z)
    
    return dH_dZ




########## First Order Wavevector Derivatives of H ##########


def find_dH_dKX(delta_K_X,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the first order partial derivative of H with respect to K_X using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of K in the X direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dKX (float or vector of shape (n,)):
        the value of the first order partial derivative in K_X after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z, 
                     K_X + delta_K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z, 
                     K_X - delta_K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKX
    dH_dKX = (H_plus - H_minus) / (2 * delta_K_X)
    
    # DEBUGGING
    # print('H_plus = ' + str(H_plus))
    # print('H_minus = ' + str(H_minus))
    # print('K_X = ' + str(H_plus))
    
    return dH_dKX


def find_dH_dKY(delta_K_Y,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the first order partial derivative of H with respect to K_Y using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of K in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dKY (float or vector of shape (n,)):
        the value of the first order partial derivative in K_Y after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y + delta_K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y - delta_K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKY
    dH_dKY = (H_plus - H_minus) / (2 * delta_K_Y)
    return dH_dKY


def find_dH_dKZ(delta_K_Z,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the first order partial derivative of H with respect to K_Z using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of K in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dKZ (float or vector of shape (n,)):
        the value of the first order partial derivative in K_Z after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z + delta_K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z - delta_K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKZ
    dH_dKZ = (H_plus - H_minus) / (2 * delta_K_Z)
    return dH_dKZ



########## Second Order Spatial Derivatives ##########

def find_d2H_dX2(delta_X,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the second order partial derivative of H with respect to X using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_X (float):
        spatial step in the X direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dX2 (float or vector of shape (n,)):
        the value of the second order partial derivative in X after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X + delta_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n step in CFD
    H_current = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X - delta_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d^2/dX^2
    d2H_dX2 = (H_plus - 2*H_current + H_minus) / (delta_X**2)
    return d2H_dX2


def find_d2H_dY2(delta_Y,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the second order partial derivative of H with respect to Y using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_Y (float):
        spatial step in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dY2 (float or vector of shape (n,)):
        the value of the second order partial derivative in Y after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y + delta_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n step in CFD
    H_current = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y - delta_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d^2/dY^2
    d2H_dY2 = (H_plus - 2*H_current + H_minus) / (delta_Y**2)
    return d2H_dY2


def find_d2H_dZ2(delta_Z,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the second order partial derivative of H with respect to Z using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dZ2 (float or vector of shape (n,)):
        the value of the second order partial derivative in Z after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z + delta_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n step in CFD
    H_current = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z - delta_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d^2/dY^2
    d2H_dZ2 = (H_plus - 2*H_current + H_minus) / (delta_Z**2)
    return d2H_dZ2




########## Second Order Mixed Spatial Derivatives ##########

def find_d2H_dX_dY(delta_X, delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dXdY) using second order CFD 
    approximation. Rewrites the derivative as
        d/dX(dH/dY)
    and apply a second order CFD to the above first order derivative (d/dX)
    by utilizing the find_dH_dY function (first applies CFD to dH/dY).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dY when computing the CFD.

    INPUT
    ==============================
    delta_X (float):
        spatial step in the X direction.
        
    delta_Y (float):
        spatial step in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dX_dY (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dX dY) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dY in n+1 step
    dH_dY_plus = find_dH_dY(delta_Y,
                   q_X + delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dY in n-1 step
    dH_dY_minus = find_dH_dY(delta_Y,
                   q_X - delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dX
    d2H_dX_dY = (dH_dY_plus - dH_dY_minus)/ (delta_X**2)
    
    return d2H_dX_dY


def find_d2H_dX_dZ(delta_X, delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dXdZ) using second order CFD 
    approximation. Rewrites the derivative as
        d/dX(dH/dZ)
    and apply a second order CFD to the above first order derivative (d/dX)
    by utilizing the find_dH_dZ function (first applies CFD to dH/dZ).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dZ when computing the CFD.

    INPUT
    ==============================
    delta_X (float):
        spatial step in the X direction.
        
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dX_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dX dZ) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dZ in n+1 step
    dH_dZ_plus = find_dH_dZ(delta_Z,
                   q_X + delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dZ in n-1 step
    dH_dZ_minus = find_dH_dZ(delta_Z,
                   q_X - delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dX
    d2H_dX_dZ = (dH_dZ_plus - dH_dZ_minus)/ (delta_X**2)
    
    return d2H_dX_dZ


def find_d2H_dY_dZ(delta_Y, delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dYdZ) using second order CFD 
    approximation. Rewrites the derivative as
        d/dY(dH/dZ)
    and apply a second order CFD to the above first order derivative (d/dY)
    by utilizing the find_dH_dZ function (first applies CFD to dH/dZ).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dZ when computing the CFD.

    INPUT
    ==============================
    delta_X (float):
        spatial step in the X direction.
        
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dX_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dX dZ) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dZ in n+1 step
    dH_dZ_plus = find_dH_dZ(delta_Z,
                   q_X, q_Y + delta_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dZ in n-1 step
    dH_dZ_minus = find_dH_dZ(delta_Z,
                   q_X, q_Y - delta_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dY
    d2H_dY_dZ = (dH_dZ_plus - dH_dZ_minus)/ (delta_Y**2)
    
    return d2H_dY_dZ




########## Second Order Wavevector Derivatives ##########


def find_d2H_dKX2(delta_K_X,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the second order partial derivative of H with respect to K_X using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX2 (float or vector of shape (n,)):
        the value of the second order partial derivative in K_X after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z, 
                     K_X + delta_K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n step in CFD
    H_current = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z, 
                     K_X - delta_K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d^2/dKX^2
    d2H_dKX2 = (H_plus - 2*H_current + H_minus) / (delta_K_X**2)
    return d2H_dKX2


def find_d2H_dKY2(delta_K_Y,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the second order partial derivative of H with respect to K_Y using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY2 (float or vector of shape (n,)):
        the value of the second order partial derivative in K_Y after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y + delta_K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n step in CFD
    H_current = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y - delta_K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d^2/dKY^2
    d2H_dKY2 = (H_plus - 2*H_current + H_minus) / (delta_K_Y**2)
    return d2H_dKY2


def find_d2H_dKZ2(delta_K_Z,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag,
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the second order partial derivative of H with respect to K_Z using 
    second order center finite difference (CFD) approximation.

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ2 (float or vector of shape (n,)):
        the value of the second order partial derivative in K_Z after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in CFD
    H_plus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z + delta_K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n step in CFD
    H_current = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # n-1 step in CFD
    H_minus = find_H(q_X, q_Y, q_Z, 
                     K_X, K_Y, K_Z - delta_K_Z, 
                     launch_angular_frequency, mode_flag, 
                     find_poloidal_flux, find_density_1D, 
                     find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d^2/dKZ^2
    d2H_dKZ2 = (H_plus - 2*H_current + H_minus) / (delta_K_Z**2)
    return d2H_dKZ2


########## Second Order Mixed Wavevector Derivatives ##########

def find_d2H_dKX_dKY(delta_K_X, delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKX dKY) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKX(dH/dKY)
    and apply a second order CFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dKY function (first applies CFD to dH/dKY).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dKY when computing the CFD.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X direction.
        
    delta_K_Y (float):
        grid spacing of the component of K in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dKY (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKX dKY) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dKY in n+1 step
    dH_dKY_plus = find_dH_dKY(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dKY in n-1 step
    dH_dKY_minus = find_dH_dKY(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X - delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKX
    d2H_dKX_dKY = (dH_dKY_plus - dH_dKY_minus)/ (delta_K_X**2)
    
    return d2H_dKX_dKY


def find_d2H_dKX_dKZ(delta_K_X, delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKX dKZ) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKX(dH/dKZ)
    and apply a second order CFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dKZ function (first applies CFD to dH/dKZ).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dKZ when computing the CFD.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X direction.
        
    delta_K_Z (float):
        grid spacing of the component of K in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dKZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKX dKZ) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dKZ in n+1 step
    dH_dKZ_plus = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dKZ in n-1 step
    dH_dKZ_minus = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X - delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKX
    d2H_dKX_dKZ = (dH_dKZ_plus - dH_dKZ_minus)/ (delta_K_X**2)
    
    return d2H_dKX_dKZ


def find_d2H_dKY_dKZ(delta_K_Y, delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKY dKZ) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKY(dH/dKZ)
    and apply a second order CFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dKZ function (first applies CFD to dH/dKZ).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dKZ when computing the CFD.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y direction.
        
    delta_K_Z (float):
        grid spacing of the component of K in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dKZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKY dKZ) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dKZ in n+1 step
    dH_dKZ_plus = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dKZ in n-1 step
    dH_dKZ_minus = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y - delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKY
    d2H_dKY_dKZ = (dH_dKZ_plus - dH_dKZ_minus)/ (delta_K_Y**2)
    
    return d2H_dKY_dKZ




########## Mixed Wavevector and Position Derivatives ##########

def find_d2H_dKX_dX(delta_K_X, delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKX dX) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKX(dH/dX)
    and apply a second order CFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dX function (first applies CFD to dH/dX).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dX when computing the CFD.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X direction.
        
    delta_X (float):
        spatial step in the X direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dX (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKX dX) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dX in n+1 step
    dH_dX_plus = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dX in n-1 step
    dH_dX_minus = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X - delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKX
    d2H_dKX_dX = (dH_dX_plus - dH_dX_minus)/ (delta_K_X**2)
    
    return d2H_dKX_dX


def find_d2H_dKX_dY(delta_K_X, delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKX dY) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKX(dH/dY)
    and apply a second order CFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dY function (first applies CFD to dH/dY).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dY when computing the CFD.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X direction.
        
    delta_Y (float):
        spatial step in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dY (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKX dY) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dY in n+1 step
    dH_dY_plus = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dY in n-1 step
    dH_dY_minus = find_dH_dX(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X - delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKX
    d2H_dKX_dY = (dH_dY_plus - dH_dY_minus)/ (delta_K_X**2)
    
    return d2H_dKX_dY


def find_d2H_dKX_dZ(delta_K_X, delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKX dZ) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKX(dH/dZ)
    and apply a second order CFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dZ function (first applies CFD to dH/dZ).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dZ when computing the CFD.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X direction.
        
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKX dZ) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dZ in n+1 step
    dH_dZ_plus = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dY in n-1 step
    dH_dZ_minus = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X - delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKX
    d2H_dKX_dZ = (dH_dZ_plus - dH_dZ_minus)/ (delta_K_X**2)
    
    return d2H_dKX_dZ


def find_d2H_dKY_dX(delta_K_Y, delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKY dX) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKY(dH/dX)
    and apply a second order CFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dX function (first applies CFD to dH/dX).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dX when computing the CFD.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y direction.
        
    delta_X (float):
        spatial step in the X direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dX (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKY dX) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dX in n+1 step
    dH_dX_plus = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dX in n-1 step
    dH_dX_minus = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y - delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKY
    d2H_dKY_dX = (dH_dX_plus - dH_dX_minus)/ (delta_K_Y**2)
    
    return d2H_dKY_dX


def find_d2H_dKY_dY(delta_K_Y, delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKY dY) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKY(dH/dY)
    and apply a second order CFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dY function (first applies CFD to dH/dY).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dY when computing the CFD.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y direction.
        
    delta_Y (float):
        spatial step in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dY (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKY dY) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dY in n+1 step
    dH_dY_plus = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dY in n-1 step
    dH_dY_minus = find_dH_dX(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y - delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKY
    d2H_dKY_dY = (dH_dY_plus - dH_dY_minus)/ (delta_K_Y**2)
    
    return d2H_dKY_dY


def find_d2H_dKY_dZ(delta_K_Y, delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKY dZ) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKY(dH/dZ)
    and apply a second order CFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dZ function (first applies CFD to dH/dZ).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dZ when computing the CFD.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y direction.
        
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKY dZ) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dZ in n+1 step
    dH_dZ_plus = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dY in n-1 step
    dH_dZ_minus = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y - delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKY
    d2H_dKY_dZ = (dH_dZ_plus - dH_dZ_minus)/ (delta_K_Y**2)
    
    return d2H_dKY_dZ


def find_d2H_dKZ_dX(delta_K_Z, delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKZ dX) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKZ(dH/dX)
    and apply a second order CFD to the above first order derivative (d/dKZ)
    by utilizing the find_dH_dX function (first applies CFD to dH/dX).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dX when computing the CFD.

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z direction.
        
    delta_X (float):
        spatial step in the X direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ_dX (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKZ dX) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dX in n+1 step
    dH_dX_plus = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + delta_K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dX in n-1 step
    dH_dX_minus = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z - delta_K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKZ
    d2H_dKZ_dX = (dH_dX_plus - dH_dX_minus)/ (delta_K_Z**2)
    
    return d2H_dKZ_dX


def find_d2H_dKZ_dY(delta_K_Z, delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKZ dY) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKZ(dH/dY)
    and apply a second order CFD to the above first order derivative (d/dKZ)
    by utilizing the find_dH_dY function (first applies CFD to dH/dY).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dY when computing the CFD.

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z direction.
        
    delta_Y (float):
        spatial step in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ_dY (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKZ dY) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dY in n+1 step
    dH_dY_plus = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + delta_K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dY in n-1 step
    dH_dY_minus = find_dH_dX(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z - delta_K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKZ
    d2H_dKZ_dY = (dH_dY_plus - dH_dY_minus)/ (delta_K_Z**2)
    
    return d2H_dKZ_dY


def find_d2H_dKZ_dZ(delta_K_Z, delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKZ dZ) using second order CFD 
    approximation. Rewrites the derivative as
        d/dKZ(dH/dZ)
    and apply a second order CFD to the above first order derivative (d/dKZ)
    by utilizing the find_dH_dZ function (first applies CFD to dH/dZ).
    
    MODIFICATIONS TO VALERIAN'S ORIGINAL CODE
    ==============================
    - Uses the function find_dH_dZ when computing the CFD.

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z direction.
        
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/(dKZ dZ) after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # dH_dZ in n+1 step
    dH_dZ_plus = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + delta_K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH_dY in n-1 step
    dH_dZ_minus = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z - delta_K_Z, 
                   launch_angular_frequency, mode_flag,
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Apply CFD to d/dKY
    d2H_dKZ_dZ = (dH_dZ_plus - dH_dZ_minus)/ (delta_K_Z**2)
    
    return d2H_dKZ_dZ





















"""
DESCRIPTION
==============================


MODIFICATIONS FROM VALERIANS ORIGINAL CODE
==============================


INPUT
==============================


OUTPUT
==============================

"""