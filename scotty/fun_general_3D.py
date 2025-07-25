import numpy as np

def apply_continuous_BC_to_Psi_3D_cartesian(
        Psi_3D_vacuum_labframe_cartesian,
        dH_dX,
        dH_dY,
        dH_dZ,
        dH_dKx,
        dH_dKy,
        dH_dKz,
        dp_dX, # d_polflux_dX
        dp_dY, # d_polflux_dY
        dp_dZ, # d_polflux_dZ
):
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

    RHS_vector = [
        # to do on Monday
    ]

    [Psi_XX_p,
     Psi_XY_p,
     Psi_XZ_p,
     Psi_YY_p,
     Psi_YZ_p,
     Psi_ZZ_p] = np.matmul(interface_matrix, RHS_vector)

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




def apply_continuous_BC_3D():
    return