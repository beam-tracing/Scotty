from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, root_scalar
from scotty.fun_general import angular_frequency_to_wavenumber, toroidal_to_cartesian
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
    poloidal_launch_angle = np.deg2rad(poloidal_launch_angle_Torbeam) + np.pi

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

    # Psi_labframe = R^-1 * Psi_beamframe * R
    Psi_3D_launch_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_launch_beamframe, rotation_matrix))

    if vacuum_propagation_flag: print("Not done yet! Only vacuum_propagation_flag = False is supported.")
    else: return launch_position_cartesian, K_launch_cartesian, Psi_3D_launch_labframe_cartesian



def find_entry_point_3D(
    launch_position: FloatArray,
    poloidal_launch_angle: float,
    toroidal_launch_angle: float,
    poloidal_flux_enter: float,
    field: MagneticField_3D_Cartesian,
    boundary_adjust: float = 1e-8,
) -> FloatArray:
    
    # The plasma is contained entirely in the boundaries of ``field``,
    # i.e. the (X,Y,Z) mesh, so the maximum distance the ray could
    # possibly travel before hitting the plasma is when it is aimed
    # at the top/bottom corner of the grid, on the far side of the
    # torus. This is an overestimate, but it is just used to parametrise
    # and estimate the location at which the ray barely just enters the
    # poloidal flux surface with coordinate `poloidal_flux_enter`.
    X_start, Y_start, Z_start = launch_position
    X_length = abs(X_start) + field.X_coord.max()
    Y_length = abs(Y_start) + field.Y_coord.max()
    Z_length = abs(Z_start) + field.Z_coord.max()
    max_length = np.sqrt(X_length**2 + Y_length**2 + Z_length**2)

    # TORBEAM antenna angles are anti-clockwise from negative X-axis,
    # so we need to rotate the toroidal angle by pi. This will take
    # care of the direction of the beam. The poloidal angle is also
    # reversed from its usual sense, so we can just flip it
    toroidal_launch_angle = toroidal_launch_angle + np.pi
    poloidal_launch_angle = -poloidal_launch_angle

    # This parametrises the ray in a line normal to the antenna,
    # up to a distance of `max_length`, and we can be sure that
    # the ray will either hit the plasma or miss it entirely.
    X_step, Y_step, Z_step = toroidal_to_cartesian(max_length, poloidal_launch_angle, toroidal_launch_angle)
    XYZ_array = np.array((X_step, Y_step, Z_step))

    def ray_line(tau):
        """Parameterised line in beam direction"""
        return launch_position + tau * XYZ_array

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
    tau = np.linspace(0, 1, max(100, Nx_steps, Ny_steps, Nz_steps))
    spline = CubicSpline(tau, [poloidal_flux_boundary_along_ray_line(t) for t in tau], extrapolate=False)
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
    if field.polflux(X_boundary, Y_boundary, Z_boundary) > poloidal_flux_enter: boundary_tau += boundary_adjust

    return np.array(ray_line(boundary_tau))