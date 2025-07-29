import numpy as np
from scotty.fun_general import find_normalised_gyro_freq, find_normalised_plasma_freq, find_mode_flag_sign

def find_Psi_3D_plasma_with_continuous_BC(
        Psi_3D_vacuum_labframe_cartesian,
        dH_dX,
        dH_dY,
        dH_dZ,
        dH_dKx,
        dH_dKy,
        dH_dKz,
        dp_dX,  # d_polflux_dX
        dp_dY,  # d_polflux_dY
        dp_dZ): # d_polflux_dZ
    
    # For continuous n_e across the plasma-vacuum boundary,
    # but discontinuous grad(n_e).
    #
    # Comments from the original function code:
        # Potential future improvement: write wrapper function
        # to wrap find_Psi_3D_plasma_continuous and
        # find_Psi_3D_plasma_discontinuous

    # Gradients at the plasma-vacuum boundary can be
    # finnicky, so it's good to check
    if np.isnan(dH_dX):
        raise ValueError("Error, dH_dX is NaN when applying continuous boundary conditions in \n `apply_continuous_BC_to_Psi_3D_cartesian`")
    elif np.isnan(dH_dY):
        raise ValueError("Error, dH_dY is NaN when applying continuous boundary conditions in \n `apply_continuous_BC_to_Psi_3D_cartesian`")
    elif np.isnan(dH_dZ):
        raise ValueError("Error, dH_dZ is NaN when applying continuous boundary conditions in \n `apply_continuous_BC_to_Psi_3D_cartesian`")
    elif np.isnan(dH_dKx):
        raise ValueError("Error, dH_dKx is NaN when applying continuous boundary conditions in \n `apply_continuous_BC_to_Psi_3D_cartesian`")
    elif np.isnan(dH_dKy):
        raise ValueError("Error, dH_dKy is NaN when applying continuous boundary conditions in \n `apply_continuous_BC_to_Psi_3D_cartesian`")
    elif np.isnan(dH_dKz):
        raise ValueError("Error, dH_dKz is NaN when applying continuous boundary conditions in \n `apply_continuous_BC_to_Psi_3D_cartesian`")
    
    # At the plasma-vacuum boundary, we have two Psi matrices:
    # one corresponding to Psi in the plasma, and the other
    # corresponding to Psi in the vacuum. We denote these by
    # the subscripts 'p' and 'v' respectively
    Psi_XX_v = Psi_3D_vacuum_labframe_cartesian[0, 0]
    Psi_XY_v = Psi_3D_vacuum_labframe_cartesian[0, 1]
    Psi_XZ_v = Psi_3D_vacuum_labframe_cartesian[0, 2]
    Psi_YY_v = Psi_3D_vacuum_labframe_cartesian[1, 1]
    Psi_YZ_v = Psi_3D_vacuum_labframe_cartesian[1, 2]
    Psi_ZZ_v = Psi_3D_vacuum_labframe_cartesian[2, 2]

    # Now we set up the interface matrix using 6 linearly
    # independent equations to obtain a relation between the
    # entries of "Psi_v" and "Psi_p"
    interface_matrix = np.zeros([6, 6])
    interface_matrix[0, 0] = dp_dY**2
    interface_matrix[0, 1] = -2 * dp_dX * dp_dY
    interface_matrix[0, 3] = dp_dX**2
    interface_matrix[1, 0] = dp_dZ**2
    interface_matrix[1, 2] = -2 * dp_dX * dp_dZ
    interface_matrix[1, 5] = dp_dX**2
    interface_matrix[2, 3] = dp_dZ**2
    interface_matrix[2, 4] = -2 * dp_dY * dp_dZ
    interface_matrix[2, 5] = dp_dY**2
    interface_matrix[3, 0] = dH_dKx
    interface_matrix[3, 1] = dH_dKy
    interface_matrix[3, 2] = dH_dKz
    interface_matrix[4, 1] = dH_dKx
    interface_matrix[4, 3] = dH_dKy
    interface_matrix[4, 4] = dH_dKz
    interface_matrix[5, 2] = dH_dKx
    interface_matrix[5, 4] = dH_dKy
    interface_matrix[5, 5] = dH_dKz
    
    # Comment from the original function code:
        # interface_matrix will be singular if one tries to
        # transition while still in vacuum (and there's no
        # plasma at all); at least that's what happens in
        # my experience
    interface_matrix_inverse = np.linalg.inv(interface_matrix)

    RHS_vector = [(Psi_XX_v * dp_dY**2) + (Psi_YY_v * dp_dX**2) - (2 * Psi_XY_v * dp_dX * dp_dY),
                  (Psi_XX_v * dp_dZ**2) + (Psi_ZZ_v * dp_dX**2) - (2 * Psi_XZ_v * dp_dX * dp_dZ),
                  (Psi_YY_v * dp_dZ**2) + (Psi_ZZ_v * dp_dY**2) - (2 * Psi_YZ_v * dp_dY * dp_dZ),
                  -dH_dX,
                  -dH_dY,
                  -dH_dZ]

    [Psi_XX_p,
     Psi_XY_p,
     Psi_XZ_p,
     Psi_YY_p,
     Psi_YZ_p,
     Psi_ZZ_p] = np.matmul(interface_matrix_inverse, RHS_vector)

    # Forming back up to get Psi in the plasma
    Psi_3D_plasma_labframe_cartesian = np.zeros([3, 3], dtype="complex128")
    Psi_3D_plasma_labframe_cartesian[0, 0] = Psi_XX_p
    Psi_3D_plasma_labframe_cartesian[1, 1] = Psi_YY_p
    Psi_3D_plasma_labframe_cartesian[2, 2] = Psi_ZZ_p
    Psi_3D_plasma_labframe_cartesian[0, 1] = Psi_XY_p
    Psi_3D_plasma_labframe_cartesian[1, 0] = Psi_3D_plasma_labframe_cartesian[0, 1]
    Psi_3D_plasma_labframe_cartesian[0, 2] = Psi_XZ_p
    Psi_3D_plasma_labframe_cartesian[2, 0] = Psi_3D_plasma_labframe_cartesian[0, 2]
    Psi_3D_plasma_labframe_cartesian[1, 2] = Psi_YZ_p
    Psi_3D_plasma_labframe_cartesian[2, 1] = Psi_3D_plasma_labframe_cartesian[1, 2]

    return Psi_3D_plasma_labframe_cartesian



def apply_continuous_BC_3D(
        q_X, q_Y, q_Z,
        K_X, K_Y, K_Z,
        Psi_3D_vacuum_labframe_cartesian,
        field,
        hamiltonian,
        delta_X, delta_Y, delta_Z):
    
    # In the continuous boundary condition case,
    # the continuity of the electron density means
    # that the K_plasma = K_vacuum. Hence we only
    # concern ourselves with finding Psi_3D_plasma
    
    dH = hamiltonian.derivatives(q_X, q_Y, q_Z, K_X, K_Y, K_Z)

    Psi_3D_plasma_labframe_cartesian = find_Psi_3D_plasma_with_continuous_BC(
        Psi_3D_vacuum_labframe_cartesian,
        dH["dH_dX"],
        dH["dH_dY"],
        dH["dH_dZ"],
        dH["dH_dKx"],
        dH["dH_dKy"],
        dH["dH_dKz"],
        field.dpolflux_dX(q_X, q_Y, q_Z, delta_X),
        field.dpolflux_dY(q_X, q_Y, q_Z, delta_Y),
        field.dpolflux_dZ(q_X, q_Y, q_Z, delta_Z),
    )

    return [K_X, K_Y, K_Z], Psi_3D_plasma_labframe_cartesian











def find_K_plasma_with_discontinuous_BC(
    q_X, q_Y, q_Z,
    K_X, K_Y, K_Z,
    B_X, B_Y, B_Z,
    electron_density_p, launch_angular_frequency, temperature, mode_flag,
    dp_dX, dp_dY, dp_dZ
    ):

    # For discontinuous n_e across the plasma-vacuum boundary
    #
    # Comments from the original function code:
        # I'm not sure if this works when the mismatch angle is close to 90deg
        # In my experience, it's difficult to reach such a situation
        # Seems to work for a mismatch angle up to 50ish deg
    
    # Checks the plasma density
    B_magnitude = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)
    plasma_freq = find_normalised_plasma_freq(electron_density_p, launch_angular_frequency, temperature)
    gyro_freq = find_normalised_gyro_freq(B_magnitude, launch_angular_frequency, temperature)
    omega_L = 0.5 * (-gyro_freq + np.sqrt(gyro_freq**2 + 4 * plasma_freq**2))
    omega_R = 0.5 * ( gyro_freq + np.sqrt(gyro_freq**2 + 4 * plasma_freq**2))
    omega_UH = np.sqrt(plasma_freq**2 + gyro_freq**2)

    mode_flag_sign = find_mode_flag_sign(electron_density_p, B_magnitude, launch_angular_frequency, temperature)

    if ((mode_flag_sign * mode_flag == 1  and plasma_freq >= 1) or
        (mode_flag_sign * mode_flag == -1 and omega_L >= 1) or
        (mode_flag_sign * mode_flag == -1 and omega_R >= 1 and omega_UH <= 1)):
        raise ValueError("Error: cut-off freq higher than beam freq on plasma side of plasma-vac boundary")
    
    # In our derivations, we find three vectors which are parallel to
    # the flux surface by considering three displacements in X, Y, and Z
    # to be zero, respectively. Selecting two of these vectors yields a
    # linearly independent (but not necessarily orthogonal) basis which
    # locally parametrises the surface. Thus, what we do is to calculate
    # the vector normal to these two vectors, and then calculate the
    # binormal vector (by using one of the original vectors and the
    # normal vector). This allows us to calculate K_plasma from K_vacuum
    # by projecting K_vacuum onto the two parallel vectors. The last
    # component (normal to the surface) can then be found by solving
    # the dispersion relation H = 0

    # parallel_vector1 corresponds to delta_X = 0
    # parallel_vector2 corresponds to delta_Y = 0
    # normal vector = parallel_vector1 x parallel_vector2
    parallel_vector1 = np.array([0, -dp_dZ, dp_dY]) / np.sqrt(dp_dY**2 + dp_dZ**2)
    parallel_vector2 = np.array([-dp_dZ, 0, dp_dX]) / np.sqrt(dp_dX**2 + dp_dZ**2)
    normal_vector = np.array([-dp_dX*dp_dZ, dp_dY*dp_dZ, -dp_dZ**2]) / np.sqrt((dp_dX*dp_dZ)**2 + (dp_dY*dp_dZ)**2 + dp_dZ**4)
    binormal_vector = None # finish on Tuesday








    find_normalised_gyro_freq
    
    
    
    
    
    
    find_normalised_plasma_freq, find_mode_flag_sign