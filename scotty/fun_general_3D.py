from scipy.optimize import newton
import numpy as np
from scotty.fun_general import (
    angular_frequency_to_wavenumber,
    find_Booker_alpha,
    find_Booker_beta,
    find_Booker_gamma,
    find_mode_flag_sign,
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
)
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D
from scotty.typing import FloatArray

def check_vector_pointing_into_plasma(q_X: float, q_Y: float, q_Z: float, vector: FloatArray, field: MagneticField_3D_Cartesian):
    if np.size(vector) != 3: raise ValueError(f"The vector provided must only be 3-D!")
    else: unitvector = vector / np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

    q_X_minus, q_Y_minus, q_Z_minus = (q_X, q_Y, q_Z) - 0.01*unitvector
    q_X_plus,  q_Y_plus,  q_Z_plus  = (q_X, q_Y, q_Z) + 0.01*unitvector

    polflux_at_q_minus = field.polflux(q_X_minus, q_Y_minus, q_Z_minus)
    polflux_at_q = field.polflux(q_X, q_Y, q_Z)
    polflux_at_q_plus = field.polflux(q_X_plus, q_Y_plus, q_Z_plus)

    if polflux_at_q_minus < polflux_at_q < polflux_at_q_plus: raise ValueError(f"K_plasma is pointing out of the plasma!")
    elif polflux_at_q_plus < polflux_at_q < polflux_at_q_minus: pass
    else: raise ValueError(f"Unable to check if K_plasma is pointing in or out of the plasma!")



def apply_continuous_BC_3D(
    q_X: float, q_Y: float, q_Z: float,
    K_X: float, K_Y: float, K_Z: float,
    Psi_3D_vacuum_labframe_cartesian: FloatArray,
    field: MagneticField_3D_Cartesian,
    hamiltonian: Hamiltonian_3D):
    
    # For continuous n_e across the plasma-vacuum boundary,
    # but discontinuous grad(n_e).
    #
    # In the continuous boundary condition case, the
    # continuity of the electron density means that the
    # K_plasma = K_vacuum. Hence we only concern ourselves
    # with finding Psi_3D_plasma
    #
    # Comments from the original function code:
        # Potential future improvement: write wrapper function
        # to wrap find_Psi_3D_plasma_continuous and
        # find_Psi_3D_plasma_discontinuous
    
    # Getting important quantities
    delta_X, delta_Y, delta_Z = hamiltonian.spacings["X"], hamiltonian.spacings["Y"], hamiltonian.spacings["Z"]
    dH = hamiltonian.derivatives(q_X, q_Y, q_Z, K_X, K_Y, K_Z)
    dH_dX = dH["dH_dX"]
    dH_dY = dH["dH_dY"]
    dH_dZ = dH["dH_dZ"]
    dH_dKx = dH["dH_dKx"]
    dH_dKy = dH["dH_dKy"]
    dH_dKz = dH["dH_dKz"]
    dp_dX = field.d_polflux_dX(q_X, q_Y, q_Z, delta_X)
    dp_dY = field.d_polflux_dY(q_X, q_Y, q_Z, delta_Y)
    dp_dZ = field.d_polflux_dZ(q_X, q_Y, q_Z, delta_Z)

    # TO REMOVE
    print("dH_dX", dH_dX)
    print("dH_dY", dH_dY)
    print("dH_dZ", dH_dZ)

    # Gradients at the plasma-vacuum boundary can be
    # finnicky, so it's good to check
    for gradient in dH.keys():
        if np.isnan(dH[gradient]):
            raise ValueError(f"Error: {gradient} is NaN when applying continuous boundary conditions in \n `apply_continuous_BC_3D`")
    
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

    # TO REMOVE: Testing another derivation instead
    dx0 = dy0 = -dp_dZ
    dz0 = dp_dX + dp_dY
    interface_matrix[2, 0] = dx0**2
    interface_matrix[2, 1] = 2*dx0*dy0
    interface_matrix[2, 2] = 2*dx0*dz0
    interface_matrix[2, 3] = dy0**2
    interface_matrix[2, 4] = 2*dy0*dz0
    interface_matrix[2, 5] = dz0**2
    
    # Comment from the original function code:
        # interface_matrix will be singular if one tries to
        # transition while still in vacuum (and there's no
        # plasma at all); at least that's what happens in
        # my experience
    interface_matrix_inverse = np.linalg.inv(interface_matrix)

    # RHS_vector = [(Psi_XX_v * dp_dY**2) + (Psi_YY_v * dp_dX**2) - (2 * Psi_XY_v * dp_dX * dp_dY),
    #               (Psi_XX_v * dp_dZ**2) + (Psi_ZZ_v * dp_dX**2) - (2 * Psi_XZ_v * dp_dX * dp_dZ),
    #               (Psi_YY_v * dp_dZ**2) + (Psi_ZZ_v * dp_dY**2) - (2 * Psi_YZ_v * dp_dY * dp_dZ),
    #               -dH_dX,
    #               -dH_dY,
    #               -dH_dZ]
    # TO REMOVE: the below RHS_vector is for the new derivation; above for the old
    RHS_vector = [(Psi_XX_v * dp_dY**2) + (Psi_YY_v * dp_dX**2) - (2 * Psi_XY_v * dp_dX * dp_dY),
                  (Psi_XX_v * dp_dZ**2) + (Psi_ZZ_v * dp_dX**2) - (2 * Psi_XZ_v * dp_dX * dp_dZ),
    Psi_XX_v*dx0**2 + Psi_YY_v*dy0**2 + Psi_ZZ_v*dz0**2 + 2*Psi_XY_v*dx0*dy0 + 2*Psi_XZ_v*dx0*dz0 + 2*Psi_YZ_v*dy0*dz0,
                  -dH_dX,
                  -dH_dY,
                  -dH_dZ]

    print()
    print("interface matrix")
    for i in range(6): print(interface_matrix[i])
    print()
    print("RHS_vector")
    for i in range(6): print(RHS_vector[i])
    print()

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

    return [K_X, K_Y, K_Z], Psi_3D_plasma_labframe_cartesian



def find_H_bar_3D(
    K_parallel, K_binormal, K_normal,
    parallel_unitvector, binormal_unitvector, normal_unitvector,
    B_X, B_Y, B_Z,
    electron_density, launch_angular_frequency, temperature, mode_flag, mode_flag_sign):
    
    K_0 = angular_frequency_to_wavenumber(launch_angular_frequency)
    K_cartesian = K_parallel*parallel_unitvector + K_binormal*binormal_unitvector + K_normal*normal_unitvector
    K_X, K_Y, K_Z = K_cartesian
    print("K_cartesian", K_cartesian)
    K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
    B_magnitude = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)
    sin_theta_m = (K_X*B_X + K_Y*B_Y + K_Z*B_Z) / (K_magnitude*B_magnitude)
    sin_theta_m_sq = sin_theta_m**2

    # TO REMOVE
    print("sin_theta_m", sin_theta_m)

    Booker_alpha = find_Booker_alpha(electron_density, B_magnitude, sin_theta_m_sq, launch_angular_frequency, temperature)
    Booker_beta  = find_Booker_beta(electron_density, B_magnitude, sin_theta_m_sq, launch_angular_frequency, temperature)
    Booker_gamma = find_Booker_gamma(electron_density, B_magnitude, launch_angular_frequency, temperature)

    H_bar = K_magnitude**2 + K_0**2 * (
        (Booker_beta + mode_flag*mode_flag_sign*np.sqrt(max(0, Booker_beta**2 - 4*Booker_alpha*Booker_gamma)))
        / (2*Booker_alpha)
    )

    return H_bar



def find_K_plasma_with_discontinuous_BC(
    q_X, q_Y, q_Z,
    K_X, K_Y, K_Z,
    field,
    hamiltonian,
    electron_density_p, launch_angular_frequency, temperature, mode_flag):

    # For discontinuous n_e across the plasma-vacuum boundary
    #
    # Comments from the original function code:
        # I'm not sure if this works when the mismatch angle is close to 90deg
        # In my experience, it's difficult to reach such a situation
        # Seems to work for a mismatch angle up to 50ish deg
    
    # Getting important quantities
    delta_X, delta_Y, delta_Z = hamiltonian.spacings["X"], hamiltonian.spacings["Y"], hamiltonian.spacings["Z"]
    B_X = field.B_X(q_X, q_Y, q_Z)
    B_Y = field.B_Y(q_X, q_Y, q_Z)
    B_Z = field.B_Z(q_X, q_Y, q_Z)
    dp_dX = field.d_polflux_dX(q_X, q_Y, q_Z, delta_X)
    dp_dY = field.d_polflux_dY(q_X, q_Y, q_Z, delta_Y)
    dp_dZ = field.d_polflux_dZ(q_X, q_Y, q_Z, delta_Z)
    
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

    # parallel_vector1 corresponds to delta_X = 0 and is equal to
        # np.array([0, -dp_dZ, dp_dY]) / np.sqrt( dp_dY**2 + dp_dZ**2 )
    # parallel_vector2 corresponds to delta_Y = 0 and is equal to
        # np.array([-dp_dZ, 0, dp_dX]) / np.sqrt( dp_dX**2 + dp_dZ**2 )
    # normal_vector = parallel_vector1 x parallel_vector2 and is equal to
        # np.array([-dp_dX*dp_dZ, -dp_dY*dp_dZ, -dp_dZ**2]) / np.sqrt( (dp_dX*dp_dZ)**2 + (dp_dY*dp_dZ)**2 + (dp_dZ**2)**2 )
    # binormal_vector = parallel_vector1 x normal_vector
    parallel_vector1 = np.array([0, -dp_dZ, dp_dY])
    normal_vector    = np.array([-dp_dX*dp_dZ, -dp_dY*dp_dZ, -dp_dZ**2])
    binormal_vector  = np.array([dp_dZ**3+dp_dY**2*dp_dZ, -dp_dX*dp_dY*dp_dZ, -dp_dX*dp_dZ**2])

    parallel_unitvector1 = parallel_vector1 / np.sqrt(np.dot(parallel_vector1, parallel_vector1))
    normal_unitvector = normal_vector / np.sqrt(np.dot(normal_vector, normal_vector))
    binormal_unitvector = binormal_vector / np.sqrt(np.dot(binormal_vector, binormal_vector))

    # TO REMOVE
    print()
    print("parallel_unitvector1", parallel_unitvector1)
    print("normal_unitvector", normal_unitvector)
    print("binormal_unitvector", binormal_unitvector)
    print()

    # Checking to see if the normal vector points into or out of
    # the plasma by comparing the poloidal flux value at (q_X, q_Y, q_Z)
    # and at a point in the direction of the normal vector away from
    # (q_X, q_Y, q_Z). This makes sure that the normal vector points
    # into the plasma
    polflux_at_boundary = field.polflux(q_X, q_Y, q_Z)
    new_q_X, new_q_Y, new_q_Z = (q_X, q_Y, q_Z) + 0.01*normal_unitvector
    polflux_at_new_point = field.polflux(new_q_X, new_q_Y, new_q_Z)
    if polflux_at_boundary < polflux_at_new_point: normal_unitvector = -normal_unitvector

    # We project K_vacuum onto parallel_vector1 and binormal_vector
    # to find K_parallel and K_binormal
    K_vacuum = np.array([K_X, K_Y, K_Z])
    K_parallel = np.dot(K_vacuum, parallel_unitvector1)
    K_binormal = np.dot(K_vacuum, binormal_unitvector)

    # TO REMOVE
    K_normal = np.dot(K_vacuum, normal_unitvector)
    print()
    print("K_parallel", K_parallel)
    print("K_normal", K_normal)
    print("K_binormal", K_binormal)
    print()

    # Now we find the component of K_plasma normal to the flux surface
    # by using the dispersion relation H = 0. We first guess what it
    # could be, and then use that to numerically compute what it
    # actually is. This guess will be exact if theta_m = 0
    K_0 = angular_frequency_to_wavenumber(launch_angular_frequency)
    K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
    sin_theta_m = (K_X*B_X + K_Y*B_Y + K_Z*B_Z) / (K_magnitude*B_magnitude)
    sin_theta_m_sq = sin_theta_m**2
    Booker_alpha = find_Booker_alpha(electron_density_p, B_magnitude, sin_theta_m_sq, launch_angular_frequency, temperature)
    Booker_beta  = find_Booker_beta(electron_density_p, B_magnitude, sin_theta_m_sq, launch_angular_frequency, temperature)
    Booker_gamma = find_Booker_gamma(electron_density_p, B_magnitude, launch_angular_frequency, temperature)
    print("TO REMOVE: Booker discriminant for K_normal_plasma_initial_guess", Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
    print("TO REMOVE: H_bar", Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
    print()
    K_normal_plasma_initial_guess = np.sqrt(
        abs(K_parallel**2 + K_binormal**2 + K_0**2 * (
                (Booker_beta + mode_flag*mode_flag_sign*np.sqrt(max(0, Booker_beta**2 - 4*Booker_alpha*Booker_gamma)))
                / (2*Booker_alpha)
                )
            )
        )
    
    def find_H_bar_3D_wrapper(K_normal_plasma_guess):
        # TO REMOVE
        print("K_normal_plasma_guessing", K_normal_plasma_guess)
        print("H_bar_guess", find_H_bar_3D(
            K_parallel, K_binormal, K_normal_plasma_guess,
            parallel_unitvector1, binormal_unitvector, normal_unitvector,
            B_X, B_Y, B_Z,
            electron_density_p, launch_angular_frequency, temperature, mode_flag, mode_flag_sign))
        print()
        return find_H_bar_3D(
            K_parallel, K_binormal, K_normal_plasma_guess,
            parallel_unitvector1, binormal_unitvector, normal_unitvector,
            B_X, B_Y, B_Z,
            electron_density_p, launch_angular_frequency, temperature, mode_flag, mode_flag_sign)
    
    # Comments from the original function code:
        # This will fail if the beam is too glancing such that there
        # is no possible K_normal_plasma that satisfies H = 0
    K_normal_plasma = newton(find_H_bar_3D_wrapper, K_normal_plasma_initial_guess, tol=1e-6, maxiter=5000)

    # After finding K_normal_plasma, we now find K_plasma
    K_plasma = K_parallel*parallel_unitvector1 + K_binormal*binormal_unitvector + K_normal_plasma*normal_unitvector

    # TO REMOVE
    print("FINAL K_normal_plasma_initial_guess", K_normal_plasma_initial_guess)
    print("FINAL K_normal_plasma", K_normal_plasma)
    print("FINAL K_vacuum", K_vacuum)
    print("FINAL K_plasma", K_plasma)
    print("FINAL H_bar")

    # To make sure K_plasma is valid, we
        # i) check if it points into the plasma; and
        # ii) check if it satisfies H = 0
    check_vector_pointing_into_plasma(q_X, q_Y, q_Z, K_plasma, field)

    H_bar_check = find_H_bar_3D(
        K_plasma[0],       K_plasma[1],       K_plasma[2],
        np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
        B_X,               B_Y,               B_Z,
        electron_density_p, launch_angular_frequency, temperature, mode_flag, mode_flag_sign)
    
    # TO REMOVE
    print("TO REMOVE: K_plasma", K_plasma)

    if abs(H_bar_check) > 1e-3:
        # raise ValueError(f"Unable to find K_plasma with discontinuous boundary conditions! \n H_bar_check = {H_bar_check}")
        # TO REMOVE
        print(f"Unable to find K_plasma with discontinuous boundary conditions! \n H_bar_check = {H_bar_check}")
    
    return K_plasma



def find_Psi_3D_plasma_with_discontinuous_BC(
    q_X, q_Y, q_Z,
    K_X_v, K_Y_v, K_Z_v,
    K_X_p, K_Y_p, K_Z_p,
    Psi_3D_vacuum_labframe_cartesian,
    field,
    hamiltonian):
    
    # For discontinuous n_e across the plasma-vacuum boundary
    
    # Getting important quantities
    delta_X, delta_Y, delta_Z = hamiltonian.spacings["X"], hamiltonian.spacings["Y"], hamiltonian.spacings["Z"]
    dH = hamiltonian.derivatives(q_X, q_Y, q_Z, K_X_v, K_Y_v, K_Z_v)
    dH_dX = dH["dH_dX"]
    dH_dY = dH["dH_dY"]
    dH_dZ = dH["dH_dZ"]
    dH_dKx = dH["dH_dKx"]
    dH_dKy = dH["dH_dKy"]
    dH_dKz = dH["dH_dKz"]
    dp_dX = field.d_polflux_dX(q_X, q_Y, q_Z, delta_X)
    dp_dY = field.d_polflux_dY(q_X, q_Y, q_Z, delta_Y)
    dp_dZ = field.d_polflux_dZ(q_X, q_Y, q_Z, delta_Z)
    d2p_dX2  = field.d2_polflux_dX2(q_X, q_Y, q_Z, delta_X)
    d2p_dY2  = field.d2_polflux_dY2(q_X, q_Y, q_Z, delta_Y)
    d2p_dZ2  = field.d2_polflux_dZ2(q_X, q_Y, q_Z, delta_Z)
    d2p_dXdY = field.d2_polflux_dXdY(q_X, q_Y, q_Z, delta_X, delta_Y)
    d2p_dXdZ = field.d2_polflux_dXdZ(q_X, q_Y, q_Z, delta_X, delta_Z)
    d2p_dYdZ = field.d2_polflux_dYdZ(q_X, q_Y, q_Z, delta_Y, delta_Z)

    # TO REMOVE -- just forcing our Psi to match the one from cyl Scotty, for debugging
    # Psi_3D_vacuum_labframe_cartesian = np.array(
    #     [[-6.62804095e-30+7.05949757e-31j, -2.24962293e-16-2.82933013e-31j, -1.08282224e-13+1.15290345e-14j],
    #      [-2.24962293e-16-2.82933013e-31j, -1.74669066e+03+1.88283421e+02j, -1.75162308e-46+0.00000000e+00j],
    #      [-1.08282224e-13+1.15290345e-14j, -1.75162308e-46+0.00000000e+00j, -1.76838292e+03+1.88283421e+02j]]
    # )

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
    # interface_matrix[2, 3] = dp_dZ**2
    # interface_matrix[2, 4] = -2 * dp_dY * dp_dZ
    # interface_matrix[2, 5] = dp_dY**2
    interface_matrix[3, 0] = dH_dKx
    interface_matrix[3, 1] = dH_dKy
    interface_matrix[3, 2] = dH_dKz
    interface_matrix[4, 1] = dH_dKx
    interface_matrix[4, 3] = dH_dKy
    interface_matrix[4, 4] = dH_dKz
    interface_matrix[5, 2] = dH_dKx
    interface_matrix[5, 4] = dH_dKy
    interface_matrix[5, 5] = dH_dKz

    # TO REMOVE
    print("TO REMOVE: interface matrix", interface_matrix)

    # For the discontinuous boundary conditions,
    # K_vacuum =/= K_plasma in general, and this introduces
    # an `eta` term for each case delta_X, delta_Y, delta_Z = 0
    # corresponding to displacements in the YZ, XZ, and XY-planes,
    # respectively, which also corresponds to eta_YZ, eta_XZ,
    # and eta_XY respectively
    eta_YZ = -0.5 * (d2p_dY2*dp_dZ**2 - 2*d2p_dYdZ*dp_dY*dp_dZ + d2p_dZ2*dp_dY**2) / (dp_dY**2 + dp_dZ**2)
    eta_XZ = -0.5 * (d2p_dX2*dp_dZ**2 - 2*d2p_dXdZ*dp_dX*dp_dZ + d2p_dZ2*dp_dX**2) / (dp_dX**2 + dp_dZ**2)
    eta_XY = -0.5 * (d2p_dX2*dp_dY**2 - 2*d2p_dXdY*dp_dX*dp_dY + d2p_dY2*dp_dX**2) / (dp_dX**2 + dp_dY**2)

    # TO REMOVE: Testing another derivation instead
    dx0 = dy0 = -dp_dZ
    dz0 = dp_dX + dp_dY
    interface_matrix[2, 0] = dx0**2
    interface_matrix[2, 1] = 2*dx0*dy0
    interface_matrix[2, 2] = 2*dx0*dz0
    interface_matrix[2, 3] = dy0**2
    interface_matrix[2, 4] = 2*dy0*dz0
    interface_matrix[2, 5] = dz0**2

    eta_XYZ = -0.5 * (d2p_dX2*dx0**2 + d2p_dY2*dy0**2 + d2p_dZ2*dz0**2 + 2*d2p_dXdY*dx0*dy0 + 2*d2p_dXdZ*dx0*dz0 + 2*d2p_dYdZ*dy0*dz0) / ( dp_dX**2 + dp_dY**2 + dp_dZ**2 )
    RHS_vector_2 = Psi_XX_v*dx0**2 + Psi_YY_v*dy0**2 + Psi_ZZ_v*dz0**2 + 2*Psi_XY_v*dx0*dy0 + 2*Psi_XZ_v*dx0*dz0 + 2*Psi_YZ_v*dy0*dz0 + 2*(K_X_v-K_X_p)*dp_dX*eta_XYZ + 2*(K_Y_v-K_Y_p)*dp_dY*eta_XYZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_XYZ

    # TO REMOVE
    print("eta_YZ", eta_YZ)
    print("eta_XZ", eta_XZ)
    print("eta_XY", eta_XY)
    print("eta_XYZ", eta_XYZ)
    print()
    parameter = 0.01
    print("polflux:", field.polflux(q_X,
                                        q_Y,
                                        q_Z))

    print("polflux, YZ:", field.polflux(q_X,
                                        q_Y - dp_dZ*parameter + eta_YZ*dp_dY*parameter**2,
                                        q_Z + dp_dY*parameter + eta_YZ*dp_dZ*parameter**2))
    
    print("polflux, XZ:", field.polflux(q_X - dp_dZ*parameter + eta_XZ*dp_dX*parameter**2,
                                        q_Y,
                                        q_Z + dp_dX*parameter + eta_XZ*dp_dZ*parameter**2))
    
    print("polflux, XY:", field.polflux(q_X - dp_dY*parameter + eta_XY*dp_dX*parameter**2,
                                        q_Y + dp_dX*parameter + eta_XY*dp_dY*parameter**2,
                                        q_Z))
    
    print("polflux, XYZ:", field.polflux(q_X - dp_dZ*parameter + eta_XYZ*dp_dX*parameter**2,
                                         q_Y - dp_dZ*parameter + eta_XYZ*dp_dY*parameter**2,
                                         q_Z + (dp_dX+dp_dY)*parameter + eta_XYZ*dp_dZ*parameter**2))
    print()





    # # TO REMOVE: Testing an even more contrived derivation method
    # dx0 = -2*dp_dZ
    # dy0 = -  dp_dZ
    # dz0 = 2*dp_dX + dp_dY
    # interface_matrix[4, 0] = dx0**2
    # interface_matrix[4, 1] = 2*dx0*dy0
    # interface_matrix[4, 2] = 2*dx0*dz0
    # interface_matrix[4, 3] = dy0**2
    # interface_matrix[4, 4] = 2*dy0*dz0
    # interface_matrix[4, 5] = dz0**2

    # eta_2XYZ = -0.5 * (d2p_dX2*dx0**2 + d2p_dY2*dy0**2 + d2p_dZ2*dz0**2 + 2*d2p_dXdY*dx0*dy0 + 2*d2p_dXdZ*dx0*dz0 + 2*d2p_dYdZ*dy0*dz0) / ( dp_dX**2 + dp_dY**2 + dp_dZ**2 )
    # RHS_vector_4 = Psi_XX_v*dx0**2 + Psi_YY_v*dy0**2 + Psi_ZZ_v*dz0**2 + 2*Psi_XY_v*dx0*dy0 + 2*Psi_XZ_v*dx0*dz0 + 2*Psi_YZ_v*dy0*dz0 + 2*(K_X_v-K_X_p)*dp_dX*eta_2XYZ + 2*(K_Y_v-K_Y_p)*dp_dY*eta_2XYZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_2XYZ

    # dx0 = -3*dp_dZ
    # dy0 = -  dp_dZ
    # dz0 = 3*dp_dX + dp_dY
    # interface_matrix[5, 0] = dx0**2
    # interface_matrix[5, 1] = 2*dx0*dy0
    # interface_matrix[5, 2] = 2*dx0*dz0
    # interface_matrix[5, 3] = dy0**2
    # interface_matrix[5, 4] = 2*dy0*dz0
    # interface_matrix[5, 5] = dz0**2

    # eta_3XYZ = -0.5 * (d2p_dX2*dx0**2 + d2p_dY2*dy0**2 + d2p_dZ2*dz0**2 + 2*d2p_dXdY*dx0*dy0 + 2*d2p_dXdZ*dx0*dz0 + 2*d2p_dYdZ*dy0*dz0) / ( dp_dX**2 + dp_dY**2 + dp_dZ**2 )
    # RHS_vector_5 = Psi_XX_v*dx0**2 + Psi_YY_v*dy0**2 + Psi_ZZ_v*dz0**2 + 2*Psi_XY_v*dx0*dy0 + 2*Psi_XZ_v*dx0*dz0 + 2*Psi_YZ_v*dy0*dz0 + 2*(K_X_v-K_X_p)*dp_dX*eta_3XYZ + 2*(K_Y_v-K_Y_p)*dp_dY*eta_3XYZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_3XYZ




    
    # TO REMOVE
    print("TO REMOVE: interface matrix")
    for i in range(6):
        print(interface_matrix[i][0], interface_matrix[i][1], interface_matrix[i][2], interface_matrix[i][3], interface_matrix[i][4], interface_matrix[i][5])
    
    # Comment from the original function code:
        # interface_matrix will be singular if one tries to
        # transition while still in vacuum (and there's no
        # plasma at all); at least that's what happens in
        # my experience
    interface_matrix_inverse = np.linalg.inv(interface_matrix)

    # RHS_vector = [(Psi_XX_v*dp_dY**2) + (Psi_YY_v*dp_dX**2) - (2*Psi_XY_v*dp_dX*dp_dY) + 2*(K_Y_v-K_Y_p)*dp_dY*eta_YZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_YZ,
    #               (Psi_XX_v*dp_dZ**2) + (Psi_ZZ_v*dp_dX**2) - (2*Psi_XZ_v*dp_dX*dp_dZ) + 2*(K_X_v-K_X_p)*dp_dX*eta_XZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_XZ,
    #               (Psi_YY_v*dp_dZ**2) + (Psi_ZZ_v*dp_dY**2) - (2*Psi_YZ_v*dp_dY*dp_dZ) + 2*(K_X_v-K_X_p)*dp_dX*eta_XY + 2*(K_Y_v-K_Y_p)*dp_dY*eta_XY,
    #               -dH_dX,
    #               -dH_dY,
    #               -dH_dZ]
    # TO REMOVE: the below RHS_vector is for the new derivation; above for the old
    RHS_vector = [(Psi_XX_v*dp_dY**2) + (Psi_YY_v*dp_dX**2) - (2*Psi_XY_v*dp_dX*dp_dY) + 2*(K_X_v-K_X_p)*dp_dX*eta_XY + 2*(K_Y_v-K_Y_p)*dp_dY*eta_XY,
                  (Psi_XX_v*dp_dZ**2) + (Psi_ZZ_v*dp_dX**2) - (2*Psi_XZ_v*dp_dX*dp_dZ) + 2*(K_X_v-K_X_p)*dp_dX*eta_XZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_XZ,
    Psi_XX_v*dx0**2 + Psi_YY_v*dy0**2 + Psi_ZZ_v*dz0**2 + 2*Psi_XY_v*dx0*dy0 + 2*Psi_XZ_v*dx0*dz0 + 2*Psi_YZ_v*dy0*dz0 + 2*(K_X_v-K_X_p)*dp_dX*eta_XYZ + 2*(K_Y_v-K_Y_p)*dp_dY*eta_XYZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_XYZ,
                  -dH_dX,
                  -dH_dY,
                  -dH_dZ]
    
    # TO REMOVE
    print("this should be 0 for continuous ne?", 2*(K_Y_v-K_Y_p)*dp_dY*eta_YZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_YZ)
    print("dp_dY", dp_dY)
    print("dp_dZ", dp_dZ)
    print("K_Y_v-K_Y_p", K_Y_v-K_Y_p)
    print("K_Z_v-K_Z_p", K_Z_v-K_Z_p)
    print("eta_YZ", eta_YZ)
    print()

    print("this should also be 0 for continuous ne?", 2*(K_X_v-K_X_p)*dp_dX*eta_XZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_XZ)
    print("dp_dX", dp_dX)
    print("dp_dZ", dp_dZ)
    print("K_X_v-K_X_p", K_X_v-K_X_p)
    print("K_Z_v-K_Z_p", K_Z_v-K_Z_p)
    print("eta_XZ", eta_XZ)
    print()

    print("this should also also be 0 for continuous ne?", 2*(K_X_v-K_X_p)*dp_dX*eta_XY + 2*(K_Y_v-K_Y_p)*dp_dY*eta_XY)
    print("dp_dX", dp_dX)
    print("dp_dY", dp_dY)
    print("K_X_v-K_X_p", K_X_v-K_X_p)
    print("K_Y_v-K_Y_p", K_Y_v-K_Y_p)
    print("eta_XY", eta_XY)
    print()

    print("interface matrix")
    for i in range(6): print(interface_matrix[i])
    print()
    print("RHS_vector")
    for i in range(6): print(RHS_vector[i])
    print()

    [Psi_XX_p,
    Psi_XY_p,
    Psi_XZ_p,
    Psi_YY_p,
    Psi_YZ_p,
    Psi_ZZ_p] = np.matmul(interface_matrix_inverse, RHS_vector)







    # TO REMOVE; just matching with cyl Scotty for debugging
    # test_interface_matrix = np.zeros([6, 6])
    # test_interface_matrix[0][3] = 1
    # test_interface_matrix[1][0] = dp_dZ**2
    # test_interface_matrix[1][2] = -2 * dp_dX * dp_dZ
    # test_interface_matrix[1][5] = dp_dX**2
    # test_interface_matrix[2][1] = -dp_dZ
    # test_interface_matrix[2][4] = dp_dX
    # test_interface_matrix[3][0] = dH_dKx
    # test_interface_matrix[3][1] = dH_dKy
    # test_interface_matrix[3][2] = dH_dKz
    # test_interface_matrix[4][1] = dH_dKx
    # test_interface_matrix[4][3] = dH_dKy
    # test_interface_matrix[4][4] = dH_dKz
    # test_interface_matrix[5][2] = dH_dKx
    # test_interface_matrix[5][4] = dH_dKy
    # test_interface_matrix[5][5] = dH_dKz

    # test_interface_matrix_inverse = np.linalg.inv(test_interface_matrix)

    # test_RHS_vector = [Psi_YY_v,
    #                    Psi_XX_v*dp_dZ**2 - 2*Psi_XZ_v*dp_dX*dp_dZ + Psi_ZZ_v*dp_dX**2 + 2*(K_X_v-K_X_p)*dp_dX*eta_XZ + 2*(K_Z_v-K_Z_p)*dp_dZ*eta_XZ,
    #                    -Psi_XY_v*dp_dZ + Psi_YZ_v*dp_dX,
    #                    -dH_dX,
    #                    -dH_dY,
    #                    -dH_dZ]
    
    # [Psi_XX_p,
    # Psi_XY_p,
    # Psi_XZ_p,
    # Psi_YY_p,
    # Psi_YZ_p,
    # Psi_ZZ_p] = np.matmul(test_interface_matrix_inverse, test_RHS_vector)
    








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

    # TO REMOVE
    print("Checking Psi \cdot gradK(H) + grad(H) = 0")
    print(np.dot(Psi_3D_plasma_labframe_cartesian, np.array(([dH_dKx],[dH_dKy],[dH_dKz]))))
    print(np.array(([dH_dX],[dH_dY],[dH_dZ])))
    print()
    print(np.dot(Psi_3D_plasma_labframe_cartesian, np.array(([dH_dKx],[dH_dKy],[dH_dKz]))) + np.array(([dH_dX],[dH_dY],[dH_dZ])))
    print()
    print("dH_dX", dH_dX)
    print("dH_dY", dH_dY)
    print("dH_dZ", dH_dZ)
    print("dH_dKx", dH_dKx)
    print("dH_dKy", dH_dKy)
    print("dH_dKz", dH_dKz)
    print()

    return Psi_3D_plasma_labframe_cartesian



def apply_discontinuous_BC_3D(
    q_X: float,   q_Y: float,   q_Z: float,
    K_X_v: float, K_Y_v: float, K_Z_v: float,
    Psi_3D_vacuum_labframe_cartesian: FloatArray,
    field: MagneticField_3D_Cartesian,
    hamiltonian: Hamiltonian_3D):
    
    polflux_at_boundary = field.polflux(q_X, q_Y, q_Z)
    electron_density_p = hamiltonian.density(polflux_at_boundary)
    launch_angular_frequency = hamiltonian.angular_frequency
    temperature = hamiltonian.temperature
    mode_flag = hamiltonian.mode_flag

    K_plasma = find_K_plasma_with_discontinuous_BC(
        q_X, q_Y, q_Z,
        K_X_v, K_Y_v, K_Z_v,
        field,
        hamiltonian,
        electron_density_p, launch_angular_frequency, temperature, mode_flag)
    
    Psi_3D_plasma_labframe_cartesian = find_Psi_3D_plasma_with_discontinuous_BC(
        q_X, q_Y, q_Z,
        K_X_v, K_Y_v, K_Z_v,
        K_plasma[0], K_plasma[1], K_plasma[2],
        Psi_3D_vacuum_labframe_cartesian,
        field,
        hamiltonian)
    
    return K_plasma, Psi_3D_plasma_labframe_cartesian



def apply_BC_3D(
    q_X: float, q_Y: float, q_Z: float,
    K_X: float, K_Y: float, K_Z: float,
    Psi_3D_vacuum_labframe_cartesian: FloatArray,
    field: MagneticField_3D_Cartesian,
    hamiltonian: Hamiltonian_3D,
    Psi_BC_flag: str):

    if Psi_BC_flag == "continuous":
        K_plasma, Psi_3D_plasma_labframe_cartesian = apply_continuous_BC_3D(
            q_X, q_Y, q_Z,
            K_X, K_Y, K_Z,
            Psi_3D_vacuum_labframe_cartesian,
            field,
            hamiltonian)
    elif Psi_BC_flag == "discontinuous":
        K_plasma, Psi_3D_plasma_labframe_cartesian = apply_discontinuous_BC_3D(
            q_X, q_Y, q_Z,
            K_X, K_Y, K_Z,
            Psi_3D_vacuum_labframe_cartesian,
            field,
            hamiltonian)
    else: raise ValueError(f"Psi_BC_flag is invalid! Only 'discontinuous', 'continuous' or None is accepted")
    
    return K_plasma, Psi_3D_plasma_labframe_cartesian