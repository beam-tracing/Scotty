from scotty.fun_general import angular_frequency_to_wavenumber
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D
from scotty.typing import FloatArray
from typing import Union
import numpy as np
import warnings

def launch_beam_3D(
    toroidal_launch_angle_Torbeam: float,
    poloidal_launch_angle_Torbeam: float,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position_cartesian: FloatArray,
    launch_angular_frequency: float,
    mode_flag: int,
    field: MagneticField_3D_Cartesian,
    hamiltonian: Hamiltonian_3D,
    vacuumLaunch_flag: bool = True,
    vacuum_propagation_flag: bool = True,
    Psi_BC_flag: Union[bool, str, None] = True,
    poloidal_flux_enter: float = 1.0,
    delta_X: float = -1e-4,
    delta_Y: float = 1e-4,
    delta_Z: float = 1e-4,
    temperature=None,
):
    
    if Psi_BC_flag is True:
        warnings.warn(
            "Boolean `Psi_BC_flag` is deprecated, please use None, 'continuous', or 'discontinuous'",
            DeprecationWarning,
        )
        print("Setting Psi_BC_flag = 'continuous' for backward compatibility")
        Psi_BC_flag = "continuous"
    elif Psi_BC_flag is False:
        warnings.warn(
            "Boolean `Psi_BC_flag` is deprecated, please use None, 'continuous', or 'discontinuous'",
            DeprecationWarning,
        )
        print("Setting Psi_BC_flag = None for backward compatibility ")
        Psi_BC_flag = None
    elif (
        (Psi_BC_flag is not None)
        and (Psi_BC_flag != "continuous")
        and (Psi_BC_flag != "discontinuous")
    ):
        raise ValueError(
            f"Unexpected value for `Psi_BC_flag` ({Psi_BC_flag}), expected one of None, 'continuous, or 'discontinuous'"
        )
    
    q_X_launch, q_Y_launch, q_Z_launch = launch_position_cartesian
    q_R_launch = np.sqrt(q_X_launch**2 + q_Y_launch**2)

    toroidal_launch_angle = np.deg2rad(toroidal_launch_angle_Torbeam)
    poloidal_launch_angle = np.deg2rad(poloidal_launch_angle_Torbeam + np.pi)

    # Finding K_launch
    wavenumber_K0 = angular_frequency_to_wavenumber(launch_angular_frequency)
    K_R_launch    = wavenumber_K0 * np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    K_zeta_launch = wavenumber_K0 * np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle) * q_R_launch
    K_X_launch = K_R_launch*(q_X_launch / q_R_launch) - K_zeta_launch*(q_Y_launch/q_R_launch**2)
    K_Y_launch = K_R_launch*(q_Y_launch / q_R_launch) + K_zeta_launch*(q_X_launch/q_R_launch**2)
    K_Z_launch    = wavenumber_K0 * np.sin(poloidal_launch_angle)
    K_launch_cartesian = np.array([K_X_launch, K_Y_launch, K_Z_launch])

    # Finding Psi_3D_launch_labframe_cartesian
    diag = wavenumber_K0*launch_beam_curvature + 2j/launch_beam_width**2 # K_0/R + 2i/W^2
    Psi_3D_launch_beamframe = np.array([[diag, 0,    0],
                                        [0,    diag, 0],
                                        [0,    0,    0]])
    
    toroidal_rotation_angle = toroidal_launch_angle
    sin_tor = np.sin(toroidal_rotation_angle)
    cos_tor = np.cos(toroidal_rotation_angle)
    poloidal_rotation_angle = np.deg2rad(poloidal_launch_angle_Torbeam + np.pi/2)
    sin_pol = np.sin(poloidal_rotation_angle)
    cos_pol = np.cos(poloidal_rotation_angle)

    toroidal_rotation_matrix = np.array([[ cos_tor, sin_tor,       0],
                                         [-sin_tor, cos_tor,       0],
                                         [       0,       0,       0]])
    poloidal_rotation_matrix = np.array([[ cos_pol,       0, sin_pol],
                                         [       0,       1,       0],
                                         [-sin_pol,       0, cos_pol]])
    rotation_matrix = np.matmul(poloidal_rotation_matrix, toroidal_rotation_matrix)
    rotation_matrix_inverse = np.transpose(rotation_matrix)

    # Psi_labframe = R^-1 * Psi_beamframe * R
    Psi_3D_launch_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_launch_beamframe, rotation_matrix))

    if vacuum_propagation_flag: print("Not done yet! Only vacuum_propagation_flag = False is supported.")
    else: return launch_position_cartesian, K_launch_cartesian, Psi_3D_launch_labframe_cartesian