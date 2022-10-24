# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

Functions for Scotty to evolve the beam or ray.
I've separated this from scotty.fun_general to prevent circular importing

@author: chenv
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

Run in Python 3,  does not work in Python 2
"""

import numpy as np

from scotty.fun_FFD import find_dH_dR, find_dH_dZ # \nabla H
from scotty.fun_CFD import find_dH_dKR, find_dH_dKZ, find_dH_dKzeta # \nabla_K H
from scotty.fun_FFD import find_d2H_dR2, find_d2H_dZ2, find_d2H_dR_dZ # \nabla \nabla H
from scotty.fun_CFD import find_d2H_dKR2, find_d2H_dKR_dKzeta, find_d2H_dKR_dKZ, find_d2H_dKzeta2, find_d2H_dKzeta_dKZ, find_d2H_dKZ2 # \nabla_K \nabla_K H
from scotty.fun_mix import find_d2H_dKR_dR, find_d2H_dKR_dZ, find_d2H_dKzeta_dR, find_d2H_dKzeta_dZ, find_d2H_dKZ_dR, find_d2H_dKZ_dZ # \nabla_K \nabla H

from scotty.fun_general import find_H_numba

# Functions (gradients of H, vectorised)
def find_grad_grad_H_vectorised(q_R, q_Z, K_R, K_zeta, K_Z,
                                launch_angular_frequency, mode_flag,
                                delta_R, delta_Z,
                                interp_poloidal_flux, find_density_1D,
                                find_B_R, find_B_T, find_B_Z):
    # \nabla \nabla H
    d2H_dR2   = find_d2H_dR2(q_R, q_Z, K_R, K_zeta, K_Z,
                             launch_angular_frequency, mode_flag,
                             delta_R,
                             interp_poloidal_flux, find_density_1D,
                             find_B_R, find_B_T, find_B_Z
                             )
    d2H_dR_dZ = find_d2H_dR_dZ(q_R ,q_Z, K_R, K_zeta, K_Z,
                               launch_angular_frequency, mode_flag,
                               delta_R, delta_Z,
                               interp_poloidal_flux, find_density_1D,
                               find_B_R, find_B_T, find_B_Z
                               )
    d2H_dZ2   = find_d2H_dZ2(q_R, q_Z, K_R, K_zeta, K_Z,
                             launch_angular_frequency, mode_flag,
                             delta_Z,
                             interp_poloidal_flux, find_density_1D,
                             find_B_R, find_B_T, find_B_Z
                             )
    
    zeros = np.zeros_like(d2H_dR2)
    grad_grad_H = np.moveaxis(
            np.squeeze(
                np.array([
                    [d2H_dR2  , zeros, d2H_dR_dZ],
                    [zeros    , zeros, zeros    ],
                    [d2H_dR_dZ, zeros, d2H_dZ2  ]
                ])
            ),
        2,0) # Such that shape is [points,3,3] instead of [3,3,points]
    
    return grad_grad_H

def find_gradK_grad_H_vectorised(q_R, q_Z, K_R, K_zeta, K_Z,
                                 launch_angular_frequency, mode_flag,
                                 delta_K_R, delta_K_zeta, delta_K_Z, 
                                 delta_R, delta_Z,
                                 interp_poloidal_flux, find_density_1D,
                                 find_B_R, find_B_T, find_B_Z
                                 ):
    # \nabla_K \nabla H
    d2H_dKR_dR    = find_d2H_dKR_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_R, delta_R,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    d2H_dKZ_dZ    = find_d2H_dKZ_dZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_Z, delta_Z,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    d2H_dKR_dZ    = find_d2H_dKR_dZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_R, delta_Z,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    d2H_dKzeta_dZ = find_d2H_dKzeta_dZ(q_R, q_Z, K_R, K_zeta, K_Z, 
                                       launch_angular_frequency, mode_flag,
                                       delta_K_zeta, delta_Z,
                                       interp_poloidal_flux, find_density_1D,
                                       find_B_R, find_B_T, find_B_Z
                                       )
    d2H_dKzeta_dR = find_d2H_dKzeta_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                                       launch_angular_frequency, mode_flag,
                                       delta_K_zeta, delta_R,
                                       interp_poloidal_flux, find_density_1D,
                                       find_B_R, find_B_T, find_B_Z
                                       )
    d2H_dKZ_dR    = find_d2H_dKZ_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_Z, delta_R,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    
    zeros = np.zeros_like(d2H_dKR_dR)
    gradK_grad_H = np.moveaxis(
            np.squeeze(
                np.array([
                    [d2H_dKR_dR,    zeros, d2H_dKR_dZ   ],
                    [d2H_dKzeta_dR, zeros, d2H_dKzeta_dZ],
                    [d2H_dKZ_dR,    zeros, d2H_dKZ_dZ   ]
                ])
            ),
        2,0) # Such that shape is [points,3,3] instead of [3,3,points]

    return gradK_grad_H

def find_gradK_gradK_H_vectorised(q_R, q_Z, K_R, K_zeta, K_Z,
                                  launch_angular_frequency, mode_flag,
                                  delta_K_R, delta_K_zeta, delta_K_Z,
                                  interp_poloidal_flux, find_density_1D,
                                  find_B_R, find_B_T, find_B_Z
                                  ):
    # \nabla_K \nabla_K H
    d2H_dKR2       = find_d2H_dKR2(q_R, q_Z, K_R, K_zeta, K_Z,
                                   launch_angular_frequency, mode_flag,
                                   delta_K_R,
                                   interp_poloidal_flux, find_density_1D,
                                   find_B_R, find_B_T, find_B_Z
                                   )
    d2H_dKzeta2    = find_d2H_dKzeta2(q_R, q_Z, K_R, K_zeta, K_Z,
                                      launch_angular_frequency, mode_flag,
                                      delta_K_zeta,
                                      interp_poloidal_flux, find_density_1D,
                                      find_B_R, find_B_T, find_B_Z
                                      )
    d2H_dKZ2       = find_d2H_dKZ2(q_R, q_Z, K_R, K_zeta, K_Z,
                                   launch_angular_frequency, mode_flag,
                                   delta_K_Z,
                                   interp_poloidal_flux, find_density_1D,
                                   find_B_R, find_B_T, find_B_Z
                                   )
    d2H_dKR_dKzeta = find_d2H_dKR_dKzeta(q_R, q_Z, K_R, K_zeta, K_Z,
                                         launch_angular_frequency, mode_flag,
                                         delta_K_R, delta_K_zeta,
                                         interp_poloidal_flux, find_density_1D,
                                         find_B_R, find_B_T, find_B_Z
                                         )
    d2H_dKR_dKZ    = find_d2H_dKR_dKZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                      launch_angular_frequency, mode_flag,
                                      delta_K_R, delta_K_Z,
                                      interp_poloidal_flux, find_density_1D,
                                      find_B_R, find_B_T, find_B_Z
                                      )
    d2H_dKzeta_dKZ = find_d2H_dKzeta_dKZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                         launch_angular_frequency, mode_flag,
                                         delta_K_zeta, delta_K_Z,
                                         interp_poloidal_flux, find_density_1D,
                                         find_B_R, find_B_T, find_B_Z
                                         )
    gradK_gradK_H = np.moveaxis(
            np.squeeze(
                np.array([
                    [d2H_dKR2      , d2H_dKR_dKzeta, d2H_dKR_dKZ   ],
                    [d2H_dKR_dKzeta, d2H_dKzeta2   , d2H_dKzeta_dKZ],
                    [d2H_dKR_dKZ   , d2H_dKzeta_dKZ, d2H_dKZ2      ]
                ])
            ),
        2,0) # Such that shape is [points,3,3] instead of [3,3,points]
    
    return gradK_gradK_H

# ---------------------------------


    # Functions (solver)
# Defines the ray evolution function
# Not necessary for what I want to do, but it does make life easier
def ray_evolution_2D_fun(tau, ray_parameters_2D, K_zeta, 
                         launch_angular_frequency, mode_flag,
                         delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z,
                         interp_poloidal_flux, find_density_1D, 
                         find_B_R, find_B_T, find_B_Z):
    """
    Parameters
    ----------
    tau : float
        Parameter along the ray.
    ray_parameters_2D : complex128
        q_R, q_Z, K_R, K_Z

    Returns
    -------
    d_beam_parameters_d_tau
        d (beam_parameters) / d tau
        
    Notes
    -------
 
    """

    ## Clean input up. Not necessary, but aids readability        
    q_R = ray_parameters_2D[0]
    q_Z = ray_parameters_2D[1]
    K_R = ray_parameters_2D[2]
    K_Z = ray_parameters_2D[3]
    
    ## Find derivatives of H
    # \nabla H
    dH_dR = find_dH_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                       launch_angular_frequency, mode_flag,
                       delta_R,
                       interp_poloidal_flux, find_density_1D, 
                       find_B_R, find_B_T, find_B_Z
                       )
    dH_dZ = find_dH_dZ(q_R, q_Z, K_R, K_zeta, K_Z,
                       launch_angular_frequency, mode_flag,
                       delta_Z,
                       interp_poloidal_flux, find_density_1D,
                       find_B_R, find_B_T, find_B_Z
                       )

    # \nabla_K H
    dH_dKR    = find_dH_dKR(q_R, q_Z, K_R, K_zeta, K_Z,
                            launch_angular_frequency, mode_flag,
                            delta_K_R,
                            interp_poloidal_flux, find_density_1D,
                            find_B_R, find_B_T, find_B_Z
                            )
    dH_dKZ    = find_dH_dKZ(q_R, q_Z, K_R, K_zeta, K_Z,
                            launch_angular_frequency, mode_flag,
                            delta_K_Z,
                            interp_poloidal_flux, find_density_1D,
                            find_B_R, find_B_T, find_B_Z
                            )

    d_ray_parameters_2D_d_tau = np.zeros_like(ray_parameters_2D)
    
    d_ray_parameters_2D_d_tau[0]  = dH_dKR # d (q_R) / d tau
    d_ray_parameters_2D_d_tau[1]  = dH_dKZ # d (q_Z) / d tau
    d_ray_parameters_2D_d_tau[2]  = -dH_dR # d (K_R) / d tau
    d_ray_parameters_2D_d_tau[3]  = -dH_dZ # d (K_Z) / d tau

    return d_ray_parameters_2D_d_tau    
# -------------------


# Defines the beam evolution function
def beam_evolution_fun(tau, beam_parameters, K_zeta, 
                       launch_angular_frequency, mode_flag,
                       delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z,
                       interp_poloidal_flux, find_density_1D, 
                       find_B_R, find_B_T, find_B_Z):
    """
    Parameters
    ----------
    tau : float
        Parameter along the ray.
    beam_parameters : complex128
        q_R, q_zeta, q_Z, K_R, K_Z, Psi_RR, Psi_zetazeta, Psi_ZZ, Psi_Rzeta, Psi_RZ, Psi_zetaZ.

    Returns
    -------
    d_beam_parameters_d_tau
        d (beam_parameters) / d tau
    """

    ## Clean input up. Not necessary, but aids readability        
    q_R          = beam_parameters[0]
    q_zeta       = beam_parameters[1]
    q_Z          = beam_parameters[2]
    K_R          = beam_parameters[3]
    K_Z          = beam_parameters[4]
    
    Psi_3D = np.zeros([3,3],dtype='complex128')     
    Psi_3D[0,0] = beam_parameters[5]  + 1j*beam_parameters[11] # Psi_RR
    Psi_3D[1,1] = beam_parameters[6]  + 1j*beam_parameters[12] # Psi_zetazeta
    Psi_3D[2,2] = beam_parameters[7]  + 1j*beam_parameters[13] # Psi_ZZ
    Psi_3D[0,1] = beam_parameters[8]  + 1j*beam_parameters[14] # Psi_Rzeta
    Psi_3D[0,2] = beam_parameters[9]  + 1j*beam_parameters[15] # Psi_RZ
    Psi_3D[1,2] = beam_parameters[10] + 1j*beam_parameters[16] # Psi_zetaZ
    Psi_3D[1,0] = Psi_3D[0,1] # Psi_3D is symmetric
    Psi_3D[2,0] = Psi_3D[0,2]
    Psi_3D[2,1] = Psi_3D[1,2]
    
    ## Find derivatives of H
    # \nabla H
    dH_dR = find_dH_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                       launch_angular_frequency, mode_flag,
                       delta_R,
                       interp_poloidal_flux, find_density_1D, 
                       find_B_R, find_B_T, find_B_Z
                       )
    dH_dZ = find_dH_dZ(q_R, q_Z, K_R, K_zeta, K_Z,
                       launch_angular_frequency, mode_flag,
                       delta_Z,
                       interp_poloidal_flux, find_density_1D,
                       find_B_R, find_B_T, find_B_Z
                       )

    # \nabla_K H
    dH_dKR    = find_dH_dKR(q_R, q_Z, K_R, K_zeta, K_Z,
                            launch_angular_frequency, mode_flag,
                            delta_K_R,
                            interp_poloidal_flux, find_density_1D,
                            find_B_R, find_B_T, find_B_Z
                            )
    dH_dKzeta = find_dH_dKzeta(q_R, q_Z, K_R, K_zeta, K_Z,
                               launch_angular_frequency, mode_flag, 
                               delta_K_zeta,
                               interp_poloidal_flux, find_density_1D,
                               find_B_R, find_B_T, find_B_Z
                               )
    dH_dKZ    = find_dH_dKZ(q_R, q_Z, K_R, K_zeta, K_Z,
                            launch_angular_frequency, mode_flag,
                            delta_K_Z,
                            interp_poloidal_flux, find_density_1D,
                            find_B_R, find_B_T, find_B_Z
                            )

    # \nabla \nabla H
    d2H_dR2   = find_d2H_dR2(q_R, q_Z, K_R, K_zeta, K_Z,
                             launch_angular_frequency, mode_flag,
                             delta_R,
                             interp_poloidal_flux, find_density_1D,
                             find_B_R, find_B_T, find_B_Z
                             )
    d2H_dR_dZ = find_d2H_dR_dZ(q_R ,q_Z, K_R, K_zeta, K_Z,
                               launch_angular_frequency, mode_flag,
                               delta_R, delta_Z,
                               interp_poloidal_flux, find_density_1D,
                               find_B_R, find_B_T, find_B_Z
                               )
    d2H_dZ2   = find_d2H_dZ2(q_R, q_Z, K_R, K_zeta, K_Z,
                             launch_angular_frequency, mode_flag,
                             delta_Z,
                             interp_poloidal_flux, find_density_1D,
                             find_B_R, find_B_T, find_B_Z
                             )
    grad_grad_H = np.squeeze(np.array([
        [d2H_dR2.item()  , 0.0, d2H_dR_dZ.item()],
        [0.0             , 0.0, 0.0             ],
        [d2H_dR_dZ.item(), 0.0, d2H_dZ2.item()  ] #. item() to convert variable from type ndarray to float, such that the array elements all have the same type
        ]))

    # \nabla_K \nabla H
    d2H_dKR_dR    = find_d2H_dKR_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_R, delta_R,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    d2H_dKZ_dZ    = find_d2H_dKZ_dZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_Z, delta_Z,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    d2H_dKR_dZ    = find_d2H_dKR_dZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_R, delta_Z,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    d2H_dKzeta_dZ = find_d2H_dKzeta_dZ(q_R, q_Z, K_R, K_zeta, K_Z, 
                                       launch_angular_frequency, mode_flag,
                                       delta_K_zeta, delta_Z,
                                       interp_poloidal_flux, find_density_1D,
                                       find_B_R, find_B_T, find_B_Z
                                       )
    d2H_dKzeta_dR = find_d2H_dKzeta_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                                       launch_angular_frequency, mode_flag,
                                       delta_K_zeta, delta_R,
                                       interp_poloidal_flux, find_density_1D,
                                       find_B_R, find_B_T, find_B_Z
                                       )
    d2H_dKZ_dR    = find_d2H_dKZ_dR(q_R, q_Z, K_R, K_zeta, K_Z,
                                    launch_angular_frequency, mode_flag,
                                    delta_K_Z, delta_R,
                                    interp_poloidal_flux, find_density_1D,
                                    find_B_R, find_B_T, find_B_Z
                                    )
    gradK_grad_H = np.squeeze(np.array([
        [d2H_dKR_dR.item(),    0.0, d2H_dKR_dZ.item()   ],
        [d2H_dKzeta_dR.item(), 0.0, d2H_dKzeta_dZ.item()],
        [d2H_dKZ_dR.item(),    0.0, d2H_dKZ_dZ.item()   ]
        ]))
    grad_gradK_H = np.transpose(gradK_grad_H)

    # \nabla_K \nabla_K H
    d2H_dKR2       = find_d2H_dKR2(q_R, q_Z, K_R, K_zeta, K_Z,
                                   launch_angular_frequency, mode_flag,
                                   delta_K_R,
                                   interp_poloidal_flux, find_density_1D,
                                   find_B_R, find_B_T, find_B_Z
                                   )
    d2H_dKzeta2    = find_d2H_dKzeta2(q_R, q_Z, K_R, K_zeta, K_Z,
                                      launch_angular_frequency, mode_flag,
                                      delta_K_zeta,
                                      interp_poloidal_flux, find_density_1D,
                                      find_B_R, find_B_T, find_B_Z
                                      )
    d2H_dKZ2       = find_d2H_dKZ2(q_R, q_Z, K_R, K_zeta, K_Z,
                                   launch_angular_frequency, mode_flag,
                                   delta_K_Z,
                                   interp_poloidal_flux, find_density_1D,
                                   find_B_R, find_B_T, find_B_Z
                                   )
    d2H_dKR_dKzeta = find_d2H_dKR_dKzeta(q_R, q_Z, K_R, K_zeta, K_Z,
                                         launch_angular_frequency, mode_flag,
                                         delta_K_R, delta_K_zeta,
                                         interp_poloidal_flux, find_density_1D,
                                         find_B_R, find_B_T, find_B_Z
                                         )
    d2H_dKR_dKZ    = find_d2H_dKR_dKZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                      launch_angular_frequency, mode_flag,
                                      delta_K_R, delta_K_Z,
                                      interp_poloidal_flux, find_density_1D,
                                      find_B_R, find_B_T, find_B_Z
                                      )
    d2H_dKzeta_dKZ = find_d2H_dKzeta_dKZ(q_R, q_Z, K_R, K_zeta, K_Z,
                                         launch_angular_frequency, mode_flag,
                                         delta_K_zeta, delta_K_Z,
                                         interp_poloidal_flux, find_density_1D,
                                         find_B_R, find_B_T, find_B_Z
                                         )
    gradK_gradK_H = np.squeeze(np.array([
        [d2H_dKR2.item()      , d2H_dKR_dKzeta.item(), d2H_dKR_dKZ.item()   ],
        [d2H_dKR_dKzeta.item(), d2H_dKzeta2.item()   , d2H_dKzeta_dKZ.item()],
        [d2H_dKR_dKZ.item()   , d2H_dKzeta_dKZ.item(), d2H_dKZ2.item()      ]
        ]))

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

    d_beam_parameters_d_tau = np.zeros_like(beam_parameters)
    
    d_beam_parameters_d_tau[0]  = dH_dKR # d (q_R) / d tau
    d_beam_parameters_d_tau[1]  = dH_dKzeta # d (q_zeta) / d tau
    
    d_beam_parameters_d_tau[2]  = dH_dKZ # d (q_Z) / d tau
    d_beam_parameters_d_tau[3]  = -dH_dR # d (K_R) / d tau
    d_beam_parameters_d_tau[4]  = -dH_dZ # d (K_Z) / d tau
    
    d_beam_parameters_d_tau[5]  = np.real(d_Psi_d_tau[0,0]) # d (Psi_RR) / d tau
    d_beam_parameters_d_tau[6]  = np.real(d_Psi_d_tau[1,1]) # d (Psi_zetazeta) / d tau
    d_beam_parameters_d_tau[7]  = np.real(d_Psi_d_tau[2,2]) # d (Psi_ZZ) / d tau
    d_beam_parameters_d_tau[8]  = np.real(d_Psi_d_tau[0,1]) # d (Psi_Rzeta) / d tau
    d_beam_parameters_d_tau[9]  = np.real(d_Psi_d_tau[0,2]) # d (Psi_RZ) / d tau
    d_beam_parameters_d_tau[10] = np.real(d_Psi_d_tau[1,2]) # d (Psi_zetaZ) / d tau
    
    d_beam_parameters_d_tau[11] = np.imag(d_Psi_d_tau[0,0]) # d (Psi_RR) / d tau
    d_beam_parameters_d_tau[12] = np.imag(d_Psi_d_tau[1,1]) # d (Psi_zetazeta) / d tau
    d_beam_parameters_d_tau[13] = np.imag(d_Psi_d_tau[2,2]) # d (Psi_ZZ) / d tau
    d_beam_parameters_d_tau[14] = np.imag(d_Psi_d_tau[0,1]) # d (Psi_Rzeta) / d tau
    d_beam_parameters_d_tau[15] = np.imag(d_Psi_d_tau[0,2]) # d (Psi_RZ) / d tau
    d_beam_parameters_d_tau[16] = np.imag(d_Psi_d_tau[1,2]) # d (Psi_zetaZ) / d tau 
    
    return d_beam_parameters_d_tau

#----------------------------------


# def find_second_derivatives(K_magnitude, 
#                             K_magnitude_p_dKZ_1, K_magnitude_m_dKZ_1, 
#                             sin_theta_m_sq, 
#                             electron_density, 
#                             B_Total, 
#                             launch_angular_frequency, mode_flag,
#                             delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z):
    
#     H_0 = find_H_numba(K_magnitude, electron_density, B_Total, sin_theta_m_sq, launch_angular_frequency, mode_flag)

#     H_p_dZ = find_H_numba(K_magnitude, electron_density, B_Total, sin_theta_m_sq, launch_angular_frequency, mode_flag)


#     return grad_grad_H, grad_gradK_H, gradK_gradK_H

