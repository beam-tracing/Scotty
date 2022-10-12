# -*- coding: utf-8 -*-
"""
Created: 25/07/2022
Last Updated: 08/09/2022
@author: Tan Zheng Yang

Functions for finding derivatives of H using forward finite difference.
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
    second order forward finite difference (FFD) approximation.

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
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X + delta_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X + 2*delta_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    dH_dX = ( (-3/2)*H_0 + (2)*H_1 + (-1/2)*H_2 ) / (delta_X)
    
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
    second order forward finite difference (FFD) approximation.

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
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y + delta_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y + 2*delta_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    dH_dY = ( (-3/2)*H_0 + (2)*H_1 + (-1/2)*H_2 ) / (delta_Y)
    
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
    second order forward finite difference (FFD) approximation.

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
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z + delta_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z + 2*delta_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    dH_dZ = ( (-3/2)*H_0 + (2)*H_1 + (-1/2)*H_2 ) / (delta_Z)
    
    return dH_dZ




########## First Order Wave Vector Derivatives of H ##########
    
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
    second order forward finite difference (FFD) approximation.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the X-component of the wavevector.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dKX (float or vector of shape (n,)):
        the value of the first order partial derivative in K_X after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z, K_X + delta_K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z, K_X + 2*delta_K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    dH_dKX = ( (-3/2)*H_0 + (2)*H_1 + (-1/2)*H_2 ) / (delta_K_X)
    
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
    second order forward finite difference (FFD) approximation.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the Y-component of the wavevector.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dKY (float or vector of shape (n,)):
        the value of the first order partial derivative in K_Y after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z, K_X, K_Y + delta_K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z, K_X, K_Y + 2*delta_K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    dH_dKY = ( (-3/2)*H_0 + (2)*H_1 + (-1/2)*H_2 ) / (delta_K_Y)
    
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
    second order forward finite difference (FFD) approximation.

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the Z-component of the wavevector.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    dH_dKY (float or vector of shape (n,)):
        the value of the first order partial derivative in K_Z after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z + delta_K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z + 2*delta_K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    dH_dKZ = ( (-3/2)*H_0 + (2)*H_1 + (-1/2)*H_2 ) / (delta_K_Z)
    
    return dH_dKZ 




########## Second Order Positional Derivatives of H ##########

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
    second order (delta_X^2) forward finite difference (FFD) approximation.

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
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X + delta_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X + 2*delta_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+3 step in the FFD
    H_3 = find_H(q_X + 3*delta_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    d2H_dX2 = ( (2)*H_0 + (-5)*H_1 + (4)*H_2 + (-1)*H_3 ) / (delta_X**2)
    
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
    second order (delta_Y^2) forward finite difference (FFD) approximation.

    INPUT
    ==============================
    delta_Y (float):
        spatial step in the Y direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dX2 (float or vector of shape (n,)):
        the value of the second order partial derivative in Y after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y + delta_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y + 2*delta_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+3 step in the FFD
    H_3 = find_H(q_X, q_Y + 3*delta_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    d2H_dY2 = ( (2)*H_0 + (-5)*H_1 + (4)*H_2 + (-1)*H_3 ) / (delta_Y**2)
    
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
    second order (delta_Z^2) forward finite difference (FFD) approximation.

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
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z + delta_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z + 2*delta_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # n+3 step in the FFD
    H_3 = find_H(q_X, q_Y, q_Z + 3*delta_Z, K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    d2H_dZ2 = ( (2)*H_0 + (-5)*H_1 + (4)*H_2 + (-1)*H_3 ) / (delta_Z**2)
    
    return d2H_dZ2




########## Second Order Mixed Positional Derivatives of H ##########
#
# Assume commutativity of partial derivatives, ie. dH/dXdY = dH/dYdX

def find_d2H_dX_dY(delta_X, delta_Y,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dXdY) using second order (delta_X^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dX(dH/dY)
    and apply a second order FFD to the above first order derivative (d/dX)
    by utilizing the find_dH_dY function (first applies FFD to dH/dY).

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
        the value of the mixed derivative d^2H/dXdY after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dY in n step in the FFD
    dH_dY_0 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+1 step in the FFD
    dH_dY_1 = find_dH_dY(delta_Y,
                   q_X + delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+2 step in the FFD
    dH_dY_2 = find_dH_dY(delta_Y,
                   q_X + 2*delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dX
    d2H_dX_dY = ( (-3/2)*dH_dY_0 + (2)*dH_dY_1 + (-1/2)*dH_dY_2 ) / (delta_X)
    
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
    Finds the mixed derivative d^2H/(dXdZ) using second order (delta_X^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dX(dH/dZ)
    and apply a second order FFD to the above first order derivative (d/dX)
    by utilizing the find_dH_dZ function (first applies FFD to dH/dZ).

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
        the value of the mixed derivative d^2H/dXdZ after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dZ in n step in the FFD
    dH_dZ_0 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+1 step in the FFD
    dH_dZ_1 = find_dH_dZ(delta_Z,
                   q_X + delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+2 step in the FFD
    dH_dZ_2 = find_dH_dZ(delta_Z,
                   q_X + 2*delta_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dX
    d2H_dX_dZ = ( (-3/2)*dH_dZ_0 + (2)*dH_dZ_1 + (-1/2)*dH_dZ_2 ) / (delta_X)
    
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
    Finds the mixed derivative d^2H/(dYdZ) using second order (delta_Y^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dY(dH/dZ)
    and apply a second order FFD to the above first order derivative (d/dY)
    by utilizing the find_dH_dZ function (first applies FFD to dH/dZ).

    INPUT
    ==============================
    delta_Y (float):
        spatial step in the Y direction.
        
    delta_Z (float):
        spatial step in the Z direction.
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dY_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2H/dXdZ after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dZ in n step in the FFD
    dH_dZ_0 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+1 step in the FFD
    dH_dZ_1 = find_dH_dZ(delta_Z,
                   q_X, q_Y + delta_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+2 step in the FFD
    dH_dZ_2 = find_dH_dZ(delta_Z,
                   q_X, q_Y + 2*delta_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dY
    d2H_dY_dZ = ( (-3/2)*dH_dZ_0 + (2)*dH_dZ_1 + (-1/2)*dH_dZ_2 ) / (delta_Y)
    
    return d2H_dY_dZ




########## Second Order Wavevector Derivatives of H ##########

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
    second order (delta_K_X^2) forward finite difference (FFD) approximation.

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX2 (float or vector of shape (n,)):
        the value of the second order partial derivative in K_X after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z, 
                 K_X + delta_K_X, K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z, 
                 K_X + 2*delta_K_X, K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+3 step in the FFD
    H_3 = find_H(q_X, q_Y, q_Z, 
                 K_X + 3*delta_K_X, K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    d2H_dKX2 = ( (2)*H_0 + (-5)*H_1 + (4)*H_2 + (-1)*H_3 ) / (delta_K_X**2)
    
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
    second order (delta_K_Y^2) forward finite difference (FFD) approximation.

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY2 (float or vector of shape (n,)):
        the value of the second order partial derivative in K_Y after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y + delta_K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y + 2*delta_K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+3 step in the FFD
    H_3 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y + 3*delta_K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    d2H_dKY2 = ( (2)*H_0 + (-5)*H_1 + (4)*H_2 + (-1)*H_3 ) / (delta_K_Y**2)
    
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
    second order (delta_K_Z^2) forward finite difference (FFD) approximation.

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ2 (float or vector of shape (n,)):
        the value of the second order partial derivative in K_Z after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # n step in the FFD
    H_0 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y, K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+1 step in the FFD
    H_1 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y, K_Z + delta_K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+2 step in the FFD
    H_2 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y, K_Z + 2*delta_K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # n+3 step in the FFD
    H_3 = find_H(q_X, q_Y, q_Z, 
                 K_X, K_Y, K_Z + 3*delta_K_Z, 
                 launch_angular_frequency, mode_flag, 
                 find_poloidal_flux, find_density_1D, 
                 find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD
    d2H_dKZ2 = ( (2)*H_0 + (-5)*H_1 + (4)*H_2 + (-1)*H_3 ) / (delta_K_Z**2)
    
    return d2H_dKZ2




########## Second Order Mixed Wavevector Derivatives of H ##########


def find_d2H_dKX_dKY(delta_K_X, delta_K_Y,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKX dKY) using second order (delta_X^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKX(dH/dKY)
    and apply a second order FFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dKY function (first applies FFD to dH/dKY).

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X-direction
        
    delta_K_Y (float):
        grid spacing of the component of K in the Y-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dKY (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_X dK_Y) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dKY in n step in the FFD
    dH_dKY_0 = find_dH_dKY(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dKY in n+1 step in the FFD
    dH_dKY_1 = find_dH_dKY(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dKY in n+2 step in the FFD
    dH_dKY_2 = find_dH_dKY(delta_K_Y,
                   q_X, q_Y, q_Z, 
                   K_X + 2*delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKX
    d2H_dKX_dKY = ((
                    (-3/2)*dH_dKY_0 + (2)*dH_dKY_1 + (-1/2)*dH_dKY_2 
                    )/ (delta_K_X)
                    )
    
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
    Finds the mixed derivative d^2H/(dKX dKZ) using second order (delta_X^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKX(dH/dKZ)
    and apply a second order FFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dKZ function (first applies FFD to dH/dKZ).

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X-direction
        
    delta_K_Z (float):
        grid spacing of the component of K in the Z-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dKZ (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_X dK_Z) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dZ in n step in the FFD
    dH_dKZ_0 = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+1 step in the FFD
    dH_dKZ_1 = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+2 step in the FFD
    dH_dKZ_2 = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X + 2*delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKX
    d2H_dKX_dKZ = ((
                    (-3/2)*dH_dKZ_0 + (2)*dH_dKZ_1 + (-1/2)*dH_dKZ_2 
                    )/ (delta_K_X)
                    )
    
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
    Finds the mixed derivative d^2H/(dKY dKZ) using second order (delta_Y^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKY(dH/dKZ)
    and apply a second order FFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dKZ function (first applies FFD to dH/dKZ).

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y-direction
        
    delta_K_Z (float):
        grid spacing of the component of K in the Z-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dKZ (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_Y dK_Z) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dZ in n step in the FFD
    dH_dKZ_0 = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+1 step in the FFD
    dH_dKZ_1 = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+2 step in the FFD
    dH_dKZ_2 = find_dH_dKZ(delta_K_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + 2*delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKY
    d2H_dKY_dKZ = ((
                    (-3/2)*dH_dKZ_0 + (2)*dH_dKZ_1 + (-1/2)*dH_dKZ_2 
                    )/ (delta_K_Y)
                    )
    
    return d2H_dKY_dKZ


########## Mixed Wavevector and Position Derivatives of H ##########


def find_d2H_dKX_dX(delta_K_X, delta_X,
               q_X, q_Y, q_Z, 
               K_X, K_Y, K_Z, 
               launch_angular_frequency, mode_flag, 
               find_poloidal_flux, find_density_1D, 
               find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative d^2H/(dKX dX) using second order (delta_K_X^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKX(dH/dX)
    and apply a second order FFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dX function (first applies FFD to dH/dX).

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X-direction
        
    delta_X (float):
        spatial step in the X-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dX (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_X dX) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dX in n step in the FFD
    dH_dX_0 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dX in n+1 step in the FFD
    dH_dX_1 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dX in n+2 step in the FFD
    dH_dX_2 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X + 2*delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKX
    d2H_dKX_dX = ((
                    (-3/2)*dH_dX_0 + (2)*dH_dX_1 + (-1/2)*dH_dX_2 
                    )/ (delta_K_X)
                    )
    
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
    Finds the mixed derivative d^2H/(dKX dY) using second order (delta_K_X^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKX(dH/dY)
    and apply a second order FFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dY function (first applies FFD to dH/dY).

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X-direction
        
    delta_Y (float):
        spatial step in the Y-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dY (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_X dY) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dY in n step in the FFD
    dH_dY_0 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+1 step in the FFD
    dH_dY_1 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+2 step in the FFD
    dH_dY_2 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X + 2*delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKX
    d2H_dKX_dY = ((
                    (-3/2)*dH_dY_0 + (2)*dH_dY_1 + (-1/2)*dH_dY_2 
                    )/ (delta_K_X)
                    )
    
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
    Finds the mixed derivative d^2H/(dKX dZ) using second order (delta_K_X^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKX(dH/dZ)
    and apply a second order FFD to the above first order derivative (d/dKX)
    by utilizing the find_dH_dZ function (first applies FFD to dH/dZ).

    INPUT
    ==============================
    delta_K_X (float):
        grid spacing of the component of K in the X-direction
        
    delta_Z (float):
        spatial step in the Z-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKX_dZ (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_X dZ) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dZ in n step in the FFD
    dH_dZ_0 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+1 step in the FFD
    dH_dZ_1 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X + delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+2 step in the FFD
    dH_dZ_2 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X + 2*delta_K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKX
    d2H_dKX_dZ = ((
                    (-3/2)*dH_dZ_0 + (2)*dH_dZ_1 + (-1/2)*dH_dZ_2 
                    )/ (delta_K_X)
                    )
    
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
    Finds the mixed derivative d^2H/(dKY dX) using second order (delta_K_Y^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKY(dH/dX)
    and apply a second order FFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dX function (first applies FFD to dH/dX).

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y-direction
        
    delta_X (float):
        spatial step in the X-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dX (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_Y dX) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dX in n step in the FFD
    dH_dX_0 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dX in n+1 step in the FFD
    dH_dX_1 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dX in n+2 step in the FFD
    dH_dX_2 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + 2*delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKY
    d2H_dKY_dX = ((
                    (-3/2)*dH_dX_0 + (2)*dH_dX_1 + (-1/2)*dH_dX_2 
                    )/ (delta_K_Y)
                    )
    
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
    Finds the mixed derivative d^2H/(dKY dY) using second order (delta_K_Y^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKY(dH/dY)
    and apply a second order FFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dY function (first applies FFD to dH/dY).

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y-direction
        
    delta_Y (float):
        spatial step in the Y-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dY (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_Y dY) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dY in n step in the FFD
    dH_dY_0 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+1 step in the FFD
    dH_dY_1 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+2 step in the FFD
    dH_dY_2 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + 2*delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKY
    d2H_dKY_dY = ((
                    (-3/2)*dH_dY_0 + (2)*dH_dY_1 + (-1/2)*dH_dY_2 
                    )/ (delta_K_Y)
                    )
    
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
    Finds the mixed derivative d^2H/(dKY dZ) using second order (delta_K_Y^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKY(dH/dZ)
    and apply a second order FFD to the above first order derivative (d/dKY)
    by utilizing the find_dH_dZ function (first applies FFD to dH/dZ).

    INPUT
    ==============================
    delta_K_Y (float):
        grid spacing of the component of K in the Y-direction
        
    delta_Z (float):
        spatial step in the Z-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKY_dZ (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_Y dZ) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dZ in n step in the FFD
    dH_dZ_0 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+1 step in the FFD
    dH_dZ_1 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+2 step in the FFD
    dH_dZ_2 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y + 2*delta_K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKY
    d2H_dKY_dZ = ((
                    (-3/2)*dH_dZ_0 + (2)*dH_dZ_1 + (-1/2)*dH_dZ_2 
                    )/ (delta_K_Y)
                    )
    
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
    Finds the mixed derivative d^2H/(dKZ dX) using second order (delta_K_Z^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKZ(dH/dX)
    and apply a second order FFD to the above first order derivative (d/dKZ)
    by utilizing the find_dH_dX function (first applies FFD to dH/dX).

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z-direction
        
    delta_X (float):
        spatial step in the X-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ_dX (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_Z dX) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dX in n step in the FFD
    dH_dX_0 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dX in n+1 step in the FFD
    dH_dX_1 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + delta_K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dX in n+2 step in the FFD
    dH_dX_2 = find_dH_dX(delta_X,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + 2*delta_K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKZ
    d2H_dKZ_dX = ((
                    (-3/2)*dH_dX_0 + (2)*dH_dX_1 + (-1/2)*dH_dX_2 
                    )/ (delta_K_Z)
                    )
    
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
    Finds the mixed derivative d^2H/(dKZ dY) using second order (delta_K_Z^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKZ(dH/dY)
    and apply a second order FFD to the above first order derivative (d/dKZ)
    by utilizing the find_dH_dY function (first applies FFD to dH/dY).

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z-direction
        
    delta_Y (float):
        spatial step in the Y-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ_dY (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_Z dY) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dY in n step in the FFD
    dH_dY_0 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+1 step in the FFD
    dH_dY_1 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + delta_K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dY in n+2 step in the FFD
    dH_dY_2 = find_dH_dY(delta_Y,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + 2*delta_K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKZ
    d2H_dKZ_dY = ((
                    (-3/2)*dH_dY_0 + (2)*dH_dY_1 + (-1/2)*dH_dY_2 
                    )/ (delta_K_Z)
                    )
    
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
    Finds the mixed derivative d^2H/(dKZ dZ) using second order (delta_K_Z^2) 
    forward finite difference (FFD) approximation. Rewrites the derivative as
        d/dKZ(dH/dZ)
    and apply a second order FFD to the above first order derivative (d/dKZ)
    by utilizing the find_dH_dZ function (first applies FFD to dH/dZ).

    INPUT
    ==============================
    delta_K_Z (float):
        grid spacing of the component of K in the Z-direction
        
    delta_Z (float):
        spatial step in the Z-direction
        
    The other inputs are exactly the same as find_H in Scotty_fun_general_CART,
    see their descriptions in the docstring of find_H.

    OUTPUT
    ==============================
    d2H_dKZ_dZ (float or vector of shape (n,)):
        the value of the mixed derivative (d^2H/dK_Z dZ) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dH/dZ in n step in the FFD
    dH_dZ_0 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+1 step in the FFD
    dH_dZ_1 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + delta_K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # dH/dZ in n+2 step in the FFD
    dH_dZ_2 = find_dH_dZ(delta_Z,
                   q_X, q_Y, q_Z, 
                   K_X, K_Y, K_Z + 2*delta_K_Z, 
                   launch_angular_frequency, mode_flag, 
                   find_poloidal_flux, find_density_1D, 
                   find_B_X, find_B_Y, find_B_Z)
    
    # Second Order FFD to d/dKZ
    d2H_dKZ_dZ = ((
                    (-3/2)*dH_dZ_0 + (2)*dH_dZ_1 + (-1/2)*dH_dZ_2 
                    )/ (delta_K_Z)
                    )
    
    return d2H_dKZ_dZ




########## First Order Derivative of Poloidal Flux ##########

def find_dpolflux_dX(delta_X,
                     q_X, q_Y, q_Z, 
                     find_poloidal_flux):
    
    """
    DESCRIPTION
    ==============================
    Computes the first order derivative of the poloidal flux with respect to
    X.

    INPUT
    ==============================
    delta_X (float):
        spatial step in the X direction
        
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    find_poloidal_flux (function):
        calculate the poloidal flux given a position (q_X, q_Y, q_Z)

    OUTPUT
    ==============================
    dpolflux_dX (float or vector of shape (n,)):
        first order derivative of the poloidal flux with respect to X. If it is
        a vector, then it is the value of the derivative at different times.

    """
    # Note that we are using a different interpolator than the original Scotty.
    # We are using RegularGridInterpolator instead of RectBivariateSpline
    
    # n step
    polflux_0 = find_poloidal_flux(q_X, q_Y, q_Z)
    
    # n+1 step
    polflux_1 = find_poloidal_flux(q_X + delta_X, q_Y, q_Z)
    
    # n+2 step
    polflux_2 = find_poloidal_flux(q_X + 2*delta_X, q_Y, q_Z)
    
    # Second order FFD on d/dR
    dpolflux_dX = (
                    (
                        (-3/2)*polflux_0 + (2)*polflux_1 + (-1/2)*polflux_2 
                        ) / (delta_X)
                    )
    
    return dpolflux_dX


def find_dpolflux_dY(delta_Y,
                     q_X, q_Y, q_Z, 
                     find_poloidal_flux):
    
    """
    DESCRIPTION
    ==============================
    Computes the first order derivative of the poloidal flux with respect to
    Y.

    INPUT
    ==============================
    delta_Y (float):
        spatial step in the Y direction
        
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    find_poloidal_flux (function):
        calculates the poloidal flux given a position (q_X, q_Y, q_Z)

    OUTPUT
    ==============================
    dpolflux_dY (float or vector of shape (n,)):
        first order derivative of the poloidal flux with respect to Y. If it is
        a vector, then it is the value of the derivative at different times.
    """
    # n step
    polflux_0 = find_poloidal_flux(q_X, q_Y, q_Z)
    
    # n+1 step
    polflux_1 = find_poloidal_flux(q_X, q_Y + delta_Y, q_Z)
    
    # n+2 step
    polflux_2 = find_poloidal_flux(q_X, q_Y + 2*delta_Y, q_Z)
    
    # Second order FFD on d/dR
    dpolflux_dY = (
                    (
                        (-3/2)*polflux_0 + (2)*polflux_1 + (-1/2)*polflux_2 
                        ) / (delta_Y)
                    )
    
    return dpolflux_dY


def find_dpolflux_dZ(delta_Z,
                     q_X, q_Y, q_Z, 
                     find_poloidal_flux):
    
    """
    DESCRIPTION
    ==============================
    Computes the first order derivative of the poloidal flux with respect to
    Z.

    INPUT
    ==============================
    delta_Z (float):
        spatial step in the Z direction
        
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    find_poloidal_flux (function):
        calculates the poloidal flux given the position (q_X, q_Y, q_Z)

    OUTPUT
    ==============================
    dpolflux_dZ (float or vector of shape (n,)):
        first order derivative of the poloidal flux with respect to Z. If it is
        a vector, then it is the value of the derivative at different times.

    """
    # n step
    polflux_0 = find_poloidal_flux(q_X, q_Y, q_Z)
    
    # n+1 step
    polflux_1 = find_poloidal_flux(q_X, q_Y, q_Z + delta_Z)
    
    # n+2 step
    polflux_2 = find_poloidal_flux(q_X, q_Y, q_Z + 2*delta_Z)
    
    # Second order FFD on d/dR
    dpolflux_dZ = (
                    (
                        (-3/2)*polflux_0 + (2)*polflux_1 + (-1/2)*polflux_2 
                        ) / (delta_Z)
                    )
    
    return dpolflux_dZ