from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, root_scalar
from scotty.fun_general import (
    angular_frequency_to_wavenumber,
    find_inverse_2D,
    make_array_3x3,
    toroidal_to_cartesian,
)
from scotty.fun_general_3D import apply_BC_3D
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D
from scotty.typing import FloatArray
from typing import Union
import numpy as np
import warnings

def launch_beam_3D(
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    launch_angular_frequency: float,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position_cartesian: FloatArray,
    mode_flag: int,
    field: MagneticField_3D_Cartesian,
    hamiltonian: Hamiltonian_3D,
    vacuumLaunch_flag: bool = True,
    vacuum_propagation_flag: bool = True,
    Psi_BC_flag: Union[bool, str, None] = True,
    poloidal_flux_enter: float = 1.0,
    delta_X: float = -1e-3,
    delta_Y: float = 1e-3,
    delta_Z: float = 1e-3,
    temperature=None,
    from_cyl_scotty___q_entry_cartesian = None,
    from_cyl_scotty___K_entry_cartesian = None,
    from_cyl_scotty___Psi_3D_entry_labframe_cartesian = None,
):
    
    if Psi_BC_flag is True:
        warnings.warn("Boolean `Psi_BC_flag` is deprecated, please use None, 'continuous', or 'discontinuous'", DeprecationWarning)
        print("Setting Psi_BC_flag = 'continuous' for backward compatibility")
        Psi_BC_flag = "continuous"
    elif Psi_BC_flag is False:
        warnings.warn("Boolean `Psi_BC_flag` is deprecated, please use None, 'continuous', or 'discontinuous'", DeprecationWarning)
        print("Setting Psi_BC_flag = None for backward compatibility")
        Psi_BC_flag = None
    elif ( (Psi_BC_flag is not None) and
           (Psi_BC_flag != "continuous") and
           (Psi_BC_flag != "discontinuous") ):
        raise ValueError(f"Unexpected value for `Psi_BC_flag` ({Psi_BC_flag}), expected one of None, 'continuous, or 'discontinuous'")
    
    q_launch_cartesian = launch_position_cartesian
    # q_X_launch, q_Y_launch, q_Z_launch = q_launch_cartesian
    # q_R_launch = np.sqrt(q_X_launch**2 + q_Y_launch**2)

    toroidal_launch_angle = np.deg2rad(toroidal_launch_angle_Torbeam)
    poloidal_launch_angle = np.deg2rad(poloidal_launch_angle_Torbeam)

    # Finding K_launch
    wavenumber_K0 = hamiltonian.wavenumber_K0
    K_X_launch = -wavenumber_K0 * np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    K_Y_launch = -wavenumber_K0 * np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    K_Z_launch = -wavenumber_K0 * np.sin(poloidal_launch_angle)
    K_launch_cartesian = np.array([K_X_launch, K_Y_launch, K_Z_launch])
    # K_R_launch    = wavenumber_K0 * np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    # K_zeta_launch = wavenumber_K0 * np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle) * q_R_launch
    # K_X_launch = K_R_launch*(q_X_launch / q_R_launch) - K_zeta_launch*(q_Y_launch/q_R_launch**2)
    # K_Y_launch = K_R_launch*(q_Y_launch / q_R_launch) + K_zeta_launch*(q_X_launch/q_R_launch**2)
    # K_Z_launch = wavenumber_K0 * np.sin(poloidal_launch_angle)
    # K_launch_cartesian = np.array([K_X_launch, K_Y_launch, K_Z_launch])

    # Finding Psi_w_launch_beamframe_cartesian and Psi_3D_launch_beamframe_cartesian
    # Entries on the off-diagonal = 0, because beamframe
    # Entries on the diagonal = K_0/R + 2i/W^2, where:
    #    R is beam radius of curvature (in metres); and
    #    W is beam width (in metres)
    # First row/column is y-direction; second is x-direction; third is g-direction (beamframe)
    # Not to be confused with X-, Y-, Z-directions (labframe)
    diag = wavenumber_K0*launch_beam_curvature + 2j/launch_beam_width**2
    Psi_w_launch_beamframe_cartesian = diag * np.eye(2)
    Psi_3D_launch_beamframe_cartesian = make_array_3x3(Psi_w_launch_beamframe_cartesian)
    
    # Setting up the rotation matrices, so that we can convert
    # Psi_3D_launch_beamframe_cartesian into Psi_3D_launch_labframe_cartesian
    toroidal_rotation_angle = toroidal_launch_angle
    sin_tor = np.sin(toroidal_rotation_angle)
    cos_tor = np.cos(toroidal_rotation_angle)
    poloidal_rotation_angle = np.deg2rad(poloidal_launch_angle_Torbeam) + np.pi/2
    sin_pol = np.sin(poloidal_rotation_angle)
    cos_pol = np.cos(poloidal_rotation_angle)

    toroidal_rotation_matrix = np.array([[ cos_tor, sin_tor,       0],
                                         [-sin_tor, cos_tor,       0],
                                         [       0,       0,       1]])
    poloidal_rotation_matrix = np.array([[ cos_pol,       0, sin_pol],
                                         [       0,       1,       0],
                                         [-sin_pol,       0, cos_pol]])
    rotation_matrix = np.matmul(poloidal_rotation_matrix, toroidal_rotation_matrix)
    rotation_matrix_inverse = np.transpose(rotation_matrix)

    # Finding Psi_3D_launch_labframe_cartesian using:
    # Psi_labframe = R^-1 * Psi_beamframe * R, where
    #    R is the rotation matrix to convert a vector from beamframe to labframe
    Psi_3D_launch_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_launch_beamframe_cartesian, rotation_matrix))

    # If vacuum_propagation_flag is False, that most likely means
    # we want to start the propagation from just outside the plasma
    if not vacuum_propagation_flag:
        return (q_launch_cartesian, # q_launch_cartesian,
                None,               # q_initial_cartesian,
                None, # distance_from_launch_to_entry,
                K_launch_cartesian, # K_launch_cartesian,
                K_launch_cartesian, # K_initial_cartesian,
                Psi_3D_launch_labframe_cartesian, # Psi_3D_launch_labframe_cartesian,
                None,                             # Psi_3D_entry_labframe_cartesian,
                Psi_3D_launch_labframe_cartesian, # Psi_3D_initial_labframe_cartesian,
                )

    # If vacuum_propagation_flag is True, then the beam is launched
    # from outside the plasma, and now we have to propagate the beam
    # until it reaches the plasma boundary, and then apply either the
    # continuous or discontinuous boundary conditions to find K_entry
    # and Psi_entry when the beam enters the plasma
    Psi_w_inverse_launch_beamframe_cartesian = find_inverse_2D(Psi_w_launch_beamframe_cartesian)
    q_initial_cartesian = find_entry_point_3D(
        q_launch_cartesian,
        poloidal_launch_angle_Torbeam,
        toroidal_launch_angle_Torbeam,
        poloidal_flux_enter,
        field,
    )
    distance_from_launch_to_entry = np.sqrt(
        (q_launch_cartesian[0] - q_initial_cartesian[0])**2 
        + (q_launch_cartesian[1] - q_initial_cartesian[1])**2
        + (q_launch_cartesian[2] - q_initial_cartesian[2])**2
    )
    Psi_w_inverse_entry_beamframe_cartesian = (
        distance_from_launch_to_entry / wavenumber_K0 * np.eye(2)
        + Psi_w_inverse_launch_beamframe_cartesian)
    
    # 'Psi_3D_entry' is still in vacuum, so the components of Psi in the
    # beam frame along g are all zero (since grad_H = 0)
    Psi_3D_entry_beamframe_cartesian = make_array_3x3(find_inverse_2D(Psi_w_inverse_entry_beamframe_cartesian))
    Psi_3D_entry_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_entry_beamframe_cartesian, rotation_matrix))

    # Now we have the Psi at the plasma boundary, so we apply the
    # continuous or discontinuous boundary conditions to find K_entry
    # and Psi_entry when the beam enters the plasma. Here, we use
    # K_entry and Psi_entry (without boundary conditions) and get
    # K_initial and Psi_initial (with boundary conditions) which are
    # later fed into the `solve_ivp` as the initial values

    # TO REMOVE -- this triple if statement only used for temporary debugging
    if from_cyl_scotty___q_entry_cartesian is not None:
        q_initial_cartesian = from_cyl_scotty___q_entry_cartesian
    if from_cyl_scotty___K_entry_cartesian is not None:
        K_X_launch, K_Y_launch, K_Z_launch = from_cyl_scotty___K_entry_cartesian
    if from_cyl_scotty___Psi_3D_entry_labframe_cartesian is not None:
        Psi_3D_entry_labframe_cartesian = from_cyl_scotty___Psi_3D_entry_labframe_cartesian

    if (Psi_BC_flag == "discontinuous") or (Psi_BC_flag == "continuous"):
        # TO REMOVE
        print()
        print("seeing what K is passed into BC")
        print(K_X_launch, K_Y_launch, K_Z_launch)
        print()
        K_initial_cartesian, Psi_3D_initial_labframe_cartesian = apply_BC_3D(
            q_initial_cartesian[0], q_initial_cartesian[1], q_initial_cartesian[2], # q_X, q_Y, q_Z
            K_X_launch, K_Y_launch, K_Z_launch,
            Psi_3D_entry_labframe_cartesian,
            field,
            hamiltonian,
            Psi_BC_flag,
        )
    else: # No BC case
        K_initial_cartesian = K_launch_cartesian
        Psi_3D_initial_labframe_cartesian = Psi_3D_entry_labframe_cartesian

    return (q_launch_cartesian,  # q_launch_cartesian,
            q_initial_cartesian, # q_initial_cartesian,
            distance_from_launch_to_entry, # distance_from_launch_to_entry,
            K_launch_cartesian,  # K_launch_cartesian,
            K_initial_cartesian, # K_initial_cartesian,
            Psi_3D_launch_labframe_cartesian,  # Psi_3D_launch_labframe_cartesian,
            Psi_3D_entry_labframe_cartesian,   # Psi_3D_entry_labframe_cartesian,
            Psi_3D_initial_labframe_cartesian, # Psi_3D_initial_labframe_cartesian,
            )



def find_entry_point_3D(
    q_launch_cartesian: FloatArray,
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    poloidal_flux_enter: float,
    field: MagneticField_3D_Cartesian,
    boundary_adjust: float = 1e-6,
) -> FloatArray:
    
    # The plasma is contained entirely in the boundaries of ``field``,
    # i.e. the (X,Y,Z) mesh, so the maximum distance the ray could
    # possibly travel before hitting the plasma is when it is aimed
    # at the top/bottom corner of the grid, on the far side of the
    # torus. This is an overestimate, but it is just used to parametrise
    # and estimate the location at which the ray barely just enters the
    # poloidal flux surface with coordinate `poloidal_flux_enter`.
    X_start, Y_start, Z_start = q_launch_cartesian
    X_length = abs(X_start) + field.X_coord.max()
    Y_length = abs(Y_start) + field.Y_coord.max()
    Z_length = abs(Z_start) + field.Z_coord.max()
    max_length = np.sqrt(X_length**2 + Y_length**2 + Z_length**2)

    # TORBEAM antenna angles are anti-clockwise from negative X-axis,
    # so we need to rotate the toroidal angle by pi. This will take
    # care of the direction of the beam. The poloidal angle is also
    # reversed from its usual sense, so we can just flip it by adding
    # a minus sign in front of the X, Y, Z steps (just like for K_launch)

    """ Wtf there has to be something wrong with cyl Scotty right? """
    poloidal_launch_angle = -np.deg2rad(poloidal_launch_angle_Torbeam)
    toroidal_launch_angle =  np.deg2rad(toroidal_launch_angle_Torbeam) + np.pi

    # zeta = np.arctan2(Y_start, X_start)
    # X_step = np.cos(np.pi - zeta)
    # Y_step = np.sin(np.pi - zeta)
    # Z_step = 0

    # poloidal_launch_angle = -np.deg2rad(poloidal_launch_angle_Torbeam)
    # toroidal_launch_angle =  np.deg2rad(toroidal_launch_angle_Torbeam) + np.pi # - np.arctan2(Y_start, X_start)
    # R_step = np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    # zeta_step = np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle)

    # zeta_angle = np.arctan2(Y_start, X_start)

    # X_step =  R_step * np.cos(zeta_angle + zeta_step)
    # Y_step = -R_step * np.sin(zeta_angle + zeta_step)
    # Z_step = np.sin(poloidal_launch_angle)

    X_step = -np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    Y_step = -np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle)
    Z_step = -np.sin(poloidal_launch_angle)

    """ # TO REMOVE?
    This is the old code
    toroidal_launch_angle = toroidal_launch_angle + np.pi
    poloidal_launch_angle = -poloidal_launch_angle

    # This parametrises the ray in a line normal to the antenna,
    # up to a distance of `max_length`, and we can be sure that
    # the ray will either hit the plasma or miss it entirely.
    X_step, Y_step, Z_step = toroidal_to_cartesian(max_length, poloidal_launch_angle, toroidal_launch_angle)
    """
    XYZ_array = np.array((X_step, Y_step, Z_step))

    # TO REMOVE -- just seeing what step_array looks like
    print()
    print("step array")
    print(XYZ_array / np.sqrt( np.dot(XYZ_array, XYZ_array) ))
    print()

    def ray_line(tau):
        """Parameterised line in beam direction"""
        return q_launch_cartesian + tau * XYZ_array

    def poloidal_flux_boundary_along_ray_line(tau):
        """Signed poloidal flux distance to plasma boundary"""
        X, Y, Z = ray_line(tau)
        return field.polflux(X, Y, Z) - poloidal_flux_enter
    
    # Now, we want to create a cubic spline of the poloidal flux
    # coordinates encountered by our ray line. However, if `max_length`
    # is extremely large, then our parameterised ray line might not have
    # enough points to actually capture the plasma at all. Here, we make
    # sure that we have at least 10 points in the plasma and a minimum
    # of 100 points in total. With this, we then find the roots of the
    # spline, i.e. where points of the ray line have a poloidal flux
    # coordinate equal to the what the user wants.
    Nx_steps = int(10 * max_length / (field.X_coord.max() - field.X_coord.min()))
    Ny_steps = int(10 * max_length / (field.Y_coord.max() - field.Y_coord.min()))
    Nz_steps = int(10 * max_length / (field.Z_coord.max() - field.Z_coord.min()))
    tau = np.linspace(-1, 1, max(100, Nx_steps, Ny_steps, Nz_steps)) # TO REMOVE -- original is linspace(0,1), but this was set to -1 temporarily because the ray was going in the opposite direction of the plasma (possibly because I didnt define my launch angles correctly? Like I'm missing a minus sign or a pi somewhere)
    # spline = CubicSpline(tau, [poloidal_flux_boundary_along_ray_line(t) for t in tau], extrapolate=False)
    # spline_roots = spline.roots()
    import math
    print("TO REMOVE: launch position", q_launch_cartesian) # TO REMOVE
    print()
    print("TO REMOVE: XYZ_array (steps)", XYZ_array)
    print()
    print("TO REMOVE: tau", tau) # TO REMOVE
    print()
    temp_poloidal_flux_boundary_along_ray_line = [poloidal_flux_boundary_along_ray_line(t) for t in tau]
    print("TO REMOVE: pol_flux_bd_along_ray_line", temp_poloidal_flux_boundary_along_ray_line)
    print()
    temp_temp_poloidal_flux_boundary_along_ray_line = [1 if math.isnan(x) else x for x in temp_poloidal_flux_boundary_along_ray_line]
    print("TO REMOVE: fixed? pol_flux_bd_along_ray_line", temp_temp_poloidal_flux_boundary_along_ray_line)
    print()
    spline = CubicSpline(tau, temp_temp_poloidal_flux_boundary_along_ray_line, extrapolate=False)
    spline_roots = spline.roots()
    
    # If there are no roots, then the beam never actually enters the
    # plasma, and we should abort. We also get an idea of where the
    # closest encounter the ray line makes with the plasma is.
    if len(spline_roots) == 0:
        minimum = minimize_scalar(poloidal_flux_boundary_along_ray_line)
        X_closest, Y_closest, Z_closest = ray_line(minimum.x)
        closest_coords = f"(X={X_closest}, Y={Y_closest}, Z={Z_closest})"
        raise RuntimeError(
            f"Beam does not hit plasma. Closest point is at {closest_coords}, "
            f"distance in poloidal flux to boundary={minimum.fun}"
        )

    # The spline roots are a pretty good guess for the boundary
    # location, which we now try to refine.
    boundary = root_scalar(poloidal_flux_boundary_along_ray_line, x0=spline_roots[0], x1=spline_roots[0] + 1e-3)
    if not boundary.converged: raise RuntimeError(f"Could not find plasma boundary, root finding failed with '{boundary.flag}'")

    # The root might be just outside the plasma due to floating point
    # errors. If so, take small steps `boundary_adjust` until the ray
    # is definitely inside.
    boundary_tau = boundary.root
    X_boundary, Y_boundary, Z_boundary = ray_line(boundary_tau)
    print("TO REMOVE: boundary coordinates", X_boundary, Y_boundary, Z_boundary)
    print("TO REMOVE: boundary polflux", field.polflux(X_boundary, Y_boundary, Z_boundary))

    # TO REMOVE: change 1.0 back to poloidal_flux_enter
    if field.polflux(X_boundary, Y_boundary, Z_boundary) > 1.0:
        boundary_tau += boundary_adjust
        X_boundary, Y_boundary, Z_boundary = ray_line(boundary_tau)
        print("TO REMOVE: adjusted boundary coordinates", X_boundary, Y_boundary, Z_boundary)
        print("TO REMOVE: adjusted boundary polflux", field.polflux(X_boundary, Y_boundary, Z_boundary))

    # TO REMOVE: this is to prevent issues with the boundary derivative
    # for iteration in range(10000):
    #     if ((field.polflux(X_boundary-1e-3, Y_boundary, Z_boundary) > poloidal_flux_enter) or
    #         (field.polflux(X_boundary, Y_boundary+1e-3, Z_boundary) > poloidal_flux_enter) or
    #         (field.polflux(X_boundary, Y_boundary, Z_boundary+1e-3) > poloidal_flux_enter)):
    #         boundary_tau += boundary_adjust
    #         X_boundary, Y_boundary, Z_boundary = ray_line(boundary_tau)
    #     else:
    #         print()
    #         print("NEW BOUNDARY:", X_boundary, Y_boundary, Z_boundary)
    #         print("polflux at NEW BOUNDARY", field.polflux(X_boundary, Y_boundary, Z_boundary))
    #         print("polflux at NEW BOUNDARY+deltas", field.polflux(X_boundary-1e-3, Y_boundary+1e-3, Z_boundary+1e-3))
    #         print("iteration", iteration)
    #         print()
    #         return np.array(ray_line(boundary_tau))
    
    return np.array(ray_line(boundary_tau))