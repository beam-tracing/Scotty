from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, root_scalar
from scotty.boundary_conditions_3D import apply_BC_3D
from scotty.fun_general import find_inverse_2D, make_array_3x3
from scotty.fun_general_3D import find_H_Cardano_eig, ray_line, poloidal_flux_difference_along_ray_line
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import DielectricTensor_3D, Hamiltonian_3D
from scotty.typing import FloatArray
from typing import Literal, Optional
import numpy as np



def find_plasma_entry_position(poloidal_launch_angle_deg_Torbeam: float,
                               toroidal_launch_angle_deg_Torbeam: float,
                               q_launch_cartesian: FloatArray,
                               field: MagneticField_3D_Cartesian,
                               poloidal_flux_enter: float,
                               boundary_adjust: float = 1e-6):
    
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

    # This parametrises the ray in a line normal to the antenna,
    # up to a distance of `max_length`, and we can be sure that
    # the ray will either hit the plasma or miss it entirely.
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

    # TO REMOVE -- need to find a way to automatically get the correct
    # start and stop for the tau linspace. When doing benchmarking, sometimes
    # the ray line actually exits the plasma from the other (inboard) side,
    # and thus my solver begins the calculation at the wrong place
    tau = np.linspace(0, 0.5, max(100, Nx_steps, Ny_steps, Nz_steps))
    polflux_difference_along_ray_line = poloidal_flux_difference_along_ray_line(X_start,
                                                                                Y_start,
                                                                                Z_start,
                                                                                tau,
                                                                                poloidal_launch_angle_deg_Torbeam,
                                                                                toroidal_launch_angle_deg_Torbeam,
                                                                                field,
                                                                                poloidal_flux_enter)

    spline = CubicSpline(tau, polflux_difference_along_ray_line, extrapolate=False)
    spline_roots = np.array(spline.roots())
    root = spline_roots[(np.abs(spline_roots - poloidal_flux_enter)).argmin()]

    # TO REMOVE -- to put this in logger next time
    # print("TO REMOVE find_plasma_entry_position")
    # print(poloidal_flux_boundary_along_ray_line_list)
    # print(spline_roots)
    # print([field.polflux(*ray_line(root)) for root in spline_roots])    
    # print("TO REMOVE find_plasma_entry_position")
    # print(root)
    # print(field.polflux(*ray_line(root)))

    # TO REMOVE -- should I change cubicspline to something else?
    # print(poloidal_flux_boundary_along_ray_line_list)
    # print(spline_roots)
    # print("Roots")
    # for root in spline_roots:
    #     print( field.polflux(*ray_line(root)) )
    # print()

    # Defining some wrappers for ease later
    def _ray_line_scalar_wrapper(tau): return ray_line(X_start, Y_start, Z_start, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)
    def _poloidal_flux_difference_along_ray_line_scalar_wrapper(tau): return poloidal_flux_difference_along_ray_line(X_start, Y_start, Z_start, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, field, poloidal_flux_enter)

    # If there are no roots, then the beam never actually enters the
    # plasma, and we should abort. We also get an idea of where the
    # closest encounter the ray line makes with the plasma is.
    if len(spline_roots) == 0:
        minimum = minimize_scalar(_poloidal_flux_difference_along_ray_line_scalar_wrapper)
        X_closest, Y_closest, Z_closest = _ray_line_scalar_wrapper(minimum.x)
        raise RuntimeError(f"Beam does not hit plasma. Closest point is at (X={X_closest}, Y={Y_closest}, Z={Z_closest}),"
                           f"distance in poloidal flux to boundary={minimum.fun}")
    
    # The spline roots are a pretty good guess for the boundary
    # location, which we now try to refine.
    boundary = root_scalar(_poloidal_flux_difference_along_ray_line_scalar_wrapper, x0=root, x1=root + 1e-3)
    if not boundary.converged: raise RuntimeError(f"Could not find plasma boundary, root finding failed with '{boundary.flag}'")
    else: boundary_tau = boundary.root

    # The root might be just outside the plasma due to floating point
    # errors. If so, take a small step of size `boundary_adjust` to
    # ensure the ray is definitely inside.
    q_initial_cartesian = _ray_line_scalar_wrapper(boundary_tau)
    if field.polflux(*q_initial_cartesian) > poloidal_flux_enter: q_initial_cartesian = _ray_line_scalar_wrapper(boundary_tau + boundary_adjust)

    return q_initial_cartesian



def find_auto_delta_signs(auto_delta_sign: bool,
                          q_initial_cartesian: FloatArray,
                          delta_X: float, delta_Y: float, delta_Z: float,
                          field: MagneticField_3D_Cartesian):

    # Now, we also perform auto_delta_sign checks (if the user specifies)
    if auto_delta_sign:
        if field.d_polflux_dX(*q_initial_cartesian, delta_X) > 0: delta_X = -1*delta_X
        if field.d_polflux_dY(*q_initial_cartesian, delta_Y) > 0: delta_Y = -1*delta_Y
        if field.d_polflux_dZ(*q_initial_cartesian, delta_Z) > 0: delta_Z = -1*delta_Z
    
    return delta_X, delta_Y, delta_Z



def find_plasma_entry_parameters(vacuumLaunch_flag: bool,
                                 vacuum_propagation_flag: bool,
                                 Psi_BC_flag: Optional[str],
                                 mode_flag_launch: Literal["O", "X", -1, 1],
                                 poloidal_launch_angle_deg_Torbeam: float,
                                 toroidal_launch_angle_deg_Torbeam: float,
                                 q_launch_cartesian: FloatArray,
                                 q_initial_cartesian: FloatArray,
                                 launch_wavenumber: float,
                                 launch_beam_width: float,
                                 launch_beam_curvature: float,
                                 field: MagneticField_3D_Cartesian,
                                 K_plasmaLaunch_cartesian: Optional[FloatArray] = np.zeros(3),
                                 Psi_3D_plasmaLaunch_labframe_cartesian: Optional[FloatArray] = np.zeros((3,3)),
                                 hamiltonian_pos1: Optional[Hamiltonian_3D] = None,
                                 hamiltonian_neg1: Optional[Hamiltonian_3D] = None):
    
    # If vacuumLaunch_flag is False, that most likely means
    # we want to start the propagation from inside the plasma
    if not vacuumLaunch_flag: return (None,                     # K_launch_cartesian,
                                      K_plasmaLaunch_cartesian, # K_initial_cartesian,
                                      None,                                   # Psi_3D_launch_labframe_cartesian,
                                      None,                                   # Psi_3D_entry_labframe_cartesian,
                                      Psi_3D_plasmaLaunch_labframe_cartesian, # Psi_3D_initial_labframe_cartesian,
                                      None, # distance_from_launch_to_entry,
                                      None, # e_hat_initial
                                      mode_flag_launch, # mode_flag_initial
                                      None, # mode_index
                                      )
    
    # If vacuumLaunch_flag is True, then continue as usual
    # TO REMOVE -- not supposed to be _Torbeam???
    poloidal_launch_angle = np.deg2rad(poloidal_launch_angle_deg_Torbeam)
    toroidal_launch_angle = np.deg2rad(toroidal_launch_angle_deg_Torbeam)
    q_launch_cartesian = q_launch_cartesian
    q_initial_cartesian = q_initial_cartesian
    launch_angular_frequency = hamiltonian_pos1.angular_frequency
    K0 = launch_wavenumber

    # Finding K_launch
    K_launch_cartesian = np.array([-K0 * np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle),
                                   -K0 * np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle),
                                   -K0 * np.sin(poloidal_launch_angle)])
    
    # Finding Psi_w_launch_beamframe_cartesian and Psi_3D_launch_beamframe_cartesian
    # Entries on the off-diagonal = 0, because beamframe
    # Entries on the diagonal = K_0/R + 2i/W^2, where:
    #    R is beam radius of curvature (in metres); and
    #    W is beam width (in metres)
    # First row/column is y-direction; second is x-direction; third is g-direction (beamframe)
    # Not to be confused with X-, Y-, Z-directions (labframe)
    diag = K0*launch_beam_curvature + 2j/launch_beam_width**2
    Psi_w_launch_beamframe_cartesian = diag * np.eye(2)
    Psi_3D_launch_beamframe_cartesian = make_array_3x3(Psi_w_launch_beamframe_cartesian)
    
    # Setting up the rotation matrices, so that we can convert
    # Psi_3D_launch_beamframe_cartesian into Psi_3D_launch_labframe_cartesian
    poloidal_rotation_angle = poloidal_launch_angle + np.pi/2
    toroidal_rotation_angle = toroidal_launch_angle
    sin_pol, cos_pol = np.sin(poloidal_rotation_angle), np.cos(poloidal_rotation_angle)
    sin_tor, cos_tor = np.sin(toroidal_rotation_angle), np.cos(toroidal_rotation_angle)
    poloidal_rotation_matrix = np.array([[ cos_pol,       0, sin_pol],
                                         [       0,       1,       0],
                                         [-sin_pol,       0, cos_pol]])
    toroidal_rotation_matrix = np.array([[ cos_tor, sin_tor,       0],
                                         [-sin_tor, cos_tor,       0],
                                         [       0,       0,       1]])
    rotation_matrix = np.matmul(poloidal_rotation_matrix, toroidal_rotation_matrix)
    rotation_matrix_inverse = np.transpose(rotation_matrix)

    # Finding Psi_3D_launch_labframe_cartesian using:
    # Psi_labframe = R^-1 * Psi_beamframe * R, where
    #    R is the rotation matrix to convert a vector from beamframe to labframe
    Psi_3D_launch_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_launch_beamframe_cartesian, rotation_matrix))

    # TO REMOVE -- does this actually make sense though? Consider:
    #       suppose vacuumLaunch_flag = True but vacuum_propagation_flag = False.
    #       Then we launch from very close to the plasma, but according to this code
    #       we don't apply the BCs even if Psi_BC_flag is True? (we return the output
    #       at the top, but Psi_BC_flag is only checked after it)

    # If vacuum_propagation_flag is False, that most likely means
    # we want to start the propagation from just outside the plasma
    if not vacuum_propagation_flag: return (K_launch_cartesian, # K_launch_cartesian,
                                            K_launch_cartesian, # K_initial_cartesian,
                                            Psi_3D_launch_labframe_cartesian, # Psi_3D_launch_labframe_cartesian,
                                            None,                             # Psi_3D_entry_labframe_cartesian,
                                            Psi_3D_launch_labframe_cartesian, # Psi_3D_initial_labframe_cartesian,
                                            None, # distance_from_launch_to_entry,
                                            None, # e_hat_initial
                                            mode_flag_launch, # mode_flag_initial
                                            None, # mode_index
                                            )
    
    # If vacuum_propagation_flag is True, then the beam is launched
    # from outside the plasma, and now we have to propagate the beam
    # until it reaches the plasma boundary, and then apply either the
    # continuous or discontinuous boundary conditions to find K_entry
    # and Psi_entry when the beam enters the plasma
    Psi_w_inverse_launch_beamframe_cartesian = find_inverse_2D(Psi_w_launch_beamframe_cartesian)
    distance_from_launch_to_entry = np.linalg.norm(q_launch_cartesian - q_initial_cartesian)
    Psi_w_inverse_entry_beamframe_cartesian = distance_from_launch_to_entry / K0 * np.eye(2) + Psi_w_inverse_launch_beamframe_cartesian

    # 'Psi_3D_entry' is still in vacuum, so the components of Psi in the
    # beam frame along g are all zero (since grad_H = 0)
    Psi_3D_entry_beamframe_cartesian = make_array_3x3(find_inverse_2D(Psi_w_inverse_entry_beamframe_cartesian))
    Psi_3D_entry_labframe_cartesian = np.matmul(rotation_matrix_inverse, np.matmul(Psi_3D_entry_beamframe_cartesian, rotation_matrix))

    # Now we have the Psi at the plasma boundary, so we apply the
    # continuous or discontinuous boundary conditions to find K_entry
    # and Psi_entry when the beam enters the plasma. Here, we use
    # K_entry and Psi_entry (without boundary conditions) and get
    # K_initial and Psi_initial (with boundary conditions) which are
    # later fed into the `solve_ivp` as the initial values. We solve
    # this for mode_flags = 1 and -1, then later we check to see which
    # set of K_initial and Psi_initial corresponds to O or X mode
    Psi_BC_flag = Psi_BC_flag
    mode_flag = mode_flag_launch
    if (Psi_BC_flag == "discontinuous") or (Psi_BC_flag == "continuous"):
        if hamiltonian_pos1:
            electron_density = hamiltonian_pos1.density(field.polflux(*q_initial_cartesian))
            electron_temperature = hamiltonian_pos1.temperature(field.polflux(*q_initial_cartesian)) if hamiltonian_pos1.temperature else None
        else:
            electron_density = hamiltonian_neg1.density(field.polflux(*q_initial_cartesian))
            electron_temperature = hamiltonian_neg1.temperature(field.polflux(*q_initial_cartesian)) if hamiltonian_neg1.temperature else None
        B_magnitude = field.magnitude(*q_initial_cartesian)
        b_hat = field.unitvector(*q_initial_cartesian)
        epsilon = DielectricTensor_3D(electron_density, launch_angular_frequency, B_magnitude, electron_temperature)
        
        O_mode, X_mode = [], []
        mode_index_tol = 1e-2
        O_mode_polarisation_tol = 5e-2

        # If mode_flag is 1 or -1, then we just calculate the
        # quantities for those. If mode_flag is "O" or "X", then
        # we calculate the quantities for both 1 and -1. After
        # obtaining the quantities corresponding to H_booker = 0,
        # we calculate H_Cardano and find the polarisation vector
        # corresponding to H_booker = H_Cardano = 0 to see if it's
        # O- or X-mode

        if mode_flag in [1, "O", "X"]:
            (K_initial_cartesian_pos1, 
             Psi_3D_initial_labframe_cartesian_pos1) = apply_BC_3D(
                *q_initial_cartesian,
                *K_launch_cartesian,
                Psi_3D_entry_labframe_cartesian,
                field,
                hamiltonian_pos1,
                Psi_BC_flag)
            
            K_hat_initial_cartesian_pos1 = K_initial_cartesian_pos1 / np.linalg.norm(K_initial_cartesian_pos1)

            (H_Cardano_pos1,
             e_hats_pos1) = find_H_Cardano_eig(
                 np.linalg.norm(K_initial_cartesian_pos1),
                 launch_angular_frequency,
                 epsilon.e_bb,
                 epsilon.e_11,
                 epsilon.e_12,
                 np.arcsin(np.dot(b_hat, K_hat_initial_cartesian_pos1)))
            
            mode_index_pos1 = np.where(np.abs(H_Cardano_pos1) <= mode_index_tol)[0]

            # If more than one H_Cardano is close to 0, then we're
            # unable to check which corresponds to O- or X-mode
            if mode_index_pos1.size != 1: raise ValueError(f"Unable to check which mode index corresponds to O- and X-mode. Found {mode_index_pos1.size} solutions!")
            else: e_hat_pos1 = e_hats_pos1[:, mode_index_pos1[0]]

            # TO REMOVE -- 8 Oct
            print()
            print("mode_index_pos1", mode_index_pos1)
            print()
            print("H 0", H_Cardano_pos1[0])
            print("e hat 0", e_hats_pos1[:, 0])
            print()
            print("H 1", H_Cardano_pos1[1])
            print("e hat 1", e_hats_pos1[:, 1])
            print()
            print("H 2", H_Cardano_pos1[2])
            print("e hat 2", e_hats_pos1[:, 2])

            # If e_hat ~ e_hat for O-mode = [0,0,1], then it's O-mode,
            # otherwise it's X-mode
            temp_pos1 = (K_initial_cartesian_pos1, Psi_3D_initial_labframe_cartesian_pos1, e_hat_pos1, +1, mode_index_pos1)
            if 1-abs(e_hat_pos1[-1]) < O_mode_polarisation_tol: O_mode.append(temp_pos1)
            else: X_mode.append(temp_pos1)
        
        if mode_flag in [-1, "O", "X"]:
            (K_initial_cartesian_neg1,
             Psi_3D_initial_labframe_cartesian_neg1) = apply_BC_3D(
                *q_initial_cartesian,
                *K_launch_cartesian,
                Psi_3D_entry_labframe_cartesian,
                field,
                hamiltonian_neg1,
                Psi_BC_flag)
            
            K_hat_initial_cartesian_neg1 = K_initial_cartesian_neg1 / np.linalg.norm(K_initial_cartesian_neg1)
            
            (H_Cardano_neg1,
             e_hats_neg1) = find_H_Cardano_eig(
                np.linalg.norm(K_initial_cartesian_neg1),
                launch_angular_frequency,
                epsilon.e_bb,
                epsilon.e_11,
                epsilon.e_12,
                np.arcsin(np.dot(b_hat, K_hat_initial_cartesian_neg1)))
            
            mode_index_neg1 = np.where(np.abs(H_Cardano_neg1) <= mode_index_tol)[0]

            # If more than one H_Cardano is close to 0, then we're
            # unable to check which corresponds to O- or X-mode
            if mode_index_neg1.size != 1: raise ValueError(f"Unable to check which mode index corresponds to O- and X-mode. Found {mode_index_neg1.size} solutions!")
            else: e_hat_neg1 = e_hats_neg1[:, mode_index_neg1[0]]

            # TO REMOVE -- 8 Oct
            print()
            print("mode_index_neg1", mode_index_neg1)
            print()
            print("H 0", H_Cardano_neg1[0])
            print("e hat 0", e_hats_neg1[:, 0])
            print()
            print("H 1", H_Cardano_neg1[1])
            print("e hat 1", e_hats_neg1[:, 1])
            print()
            print("H 2", H_Cardano_neg1[2])
            print("e hat 2", e_hats_neg1[:, 2])

            # If e_hat ~ e_hat for O-mode = [0,0,1], then it's O-mode,
            # otherwise it's X-mode
            temp_neg1 = (K_initial_cartesian_neg1, Psi_3D_initial_labframe_cartesian_neg1, e_hat_neg1, -1, mode_index_neg1)
            if 1-abs(e_hat_neg1[-1]) < O_mode_polarisation_tol: O_mode.append(temp_neg1)
            else: X_mode.append(temp_neg1)
        
        # A bunch of final checks
        if   len(O_mode) > 1: raise ValueError((f"Found {len(O_mode)} solutions for O-mode!"))
        elif len(X_mode) > 1: raise ValueError((f"Found {len(X_mode)} solutions for X-mode!"))
        elif len(O_mode) == 0 and mode_flag == "O": raise ValueError((f"Found no solutions for O-mode instead of 1!"))
        elif len(X_mode) == 0 and mode_flag == "X": raise ValueError((f"Found no solutions for O-mode instead of 1!"))

        # If the user specified 1 or -1, then we must have only calculated
        # 1 set of quantities in the previous code, so just return that.
        # If the user specified "O" or "X", then we calculated 2 sets of
        # quantities in the previous code, so select the set corresponding
        # to the mode that the user wants
        if mode_flag in [1, -1]:
            if   len(O_mode) == 1 and len(X_mode) == 0: temp = O_mode[0]
            elif len(X_mode) == 1 and len(O_mode) == 0: temp = X_mode[0]
            else: raise ValueError(f"Unable to unpack corresponding mode solution!")
        elif mode_flag == "O": temp = O_mode[0]
        elif mode_flag == "X": temp = X_mode[0]
        
        (K_initial_cartesian,
         Psi_3D_initial_labframe_cartesian,
         e_hat_initial,
         mode_flag_initial,
         mode_index) = temp
        
        # TO REMOVE
        print()
        print("O mode")
        print(O_mode)
        print()
        print("X mode")
        print(X_mode)
        print()
    
    else: # No BC case
        K_initial_cartesian = K_launch_cartesian
        Psi_3D_initial_labframe_cartesian = Psi_3D_entry_labframe_cartesian
        e_hat_initial = None
        mode_flag_initial = mode_flag_launch
        mode_index = None

    return (K_launch_cartesian,  # K_launch_cartesian,
            K_initial_cartesian, # K_initial_cartesian,
            Psi_3D_launch_labframe_cartesian,  # Psi_3D_launch_labframe_cartesian,
            Psi_3D_entry_labframe_cartesian,   # Psi_3D_entry_labframe_cartesian,
            Psi_3D_initial_labframe_cartesian, # Psi_3D_initial_labframe_cartesian,
            distance_from_launch_to_entry, # distance_from_launch_to_entry,
            e_hat_initial,
            mode_flag_initial,
            mode_index[0]
            )