import logging
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, root_scalar
from scotty.boundary_conditions_3D import apply_BC_3D
from scotty.fun_general import find_inverse_2D, make_array_3x3
from scotty.fun_general_3D import find_H_Cardano_eig, ray_line, poloidal_flux_difference_along_ray_line
from scotty.geometry_3D import MagneticField_Cartesian
from scotty.hamiltonian_3D import DielectricTensor_Cartesian, Hamiltonian_3D
from scotty.logger_3D import mln, arr2str
from scotty.typing import FloatArray
from typing import Literal, Optional
import numpy as np

log = logging.getLogger(__name__)

def find_plasma_entry_position(poloidal_launch_angle_deg_Torbeam: float,
                               toroidal_launch_angle_deg_Torbeam: float,
                               q_launch_cartesian: FloatArray,
                               field: MagneticField_Cartesian,
                               poloidal_flux_enter: float,
                               boundary_adjust: float = 1e-6):
    
    log.info(f"Finding plasma entry position")
    
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

    # Defining some wrappers for ease
    def _ray_line_scalar_wrapper(tau):
        return ray_line(X_start, Y_start, Z_start, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)
    def _poloidal_flux_difference_along_ray_line_scalar_wrapper(tau):
        return poloidal_flux_difference_along_ray_line(X_start, Y_start, Z_start, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, field, poloidal_flux_enter)

    # TO REMOVE -- need to find a way to automatically get the correct
    # start and stop for the tau linspace. When doing benchmarking, sometimes
    # the ray line actually exits the plasma from the other (inboard) side,
    # and thus my solver begins the calculation at the wrong place.
    # ***Sometimes tau = 0.5 is not enough -- need to adjust to 1.5
    # TO REMOVE -- should I change cubicspline to something else?
    # print(poloidal_flux_boundary_along_ray_line_list)
    # print(spline_roots)
    # print("Roots")
    # for root in spline_roots:
    #     print( field.polflux(*ray_line(root)) )
    # print()
    def _find_plasma_root(tau_max):
        taus = np.linspace(0, tau_max, max(int(100*tau_max), Nx_steps, Ny_steps, Nz_steps))
        polflux_difference_along_ray_line = poloidal_flux_difference_along_ray_line(X_start,
                                                                                    Y_start,
                                                                                    Z_start,
                                                                                    taus,
                                                                                    poloidal_launch_angle_deg_Torbeam,
                                                                                    toroidal_launch_angle_deg_Torbeam,
                                                                                    field,
                                                                                    poloidal_flux_enter)

        # Try finding a root
        # If no root is found (ValueError or ZeroDivisionError) then log the
        # fail and set `root` = None (in except) and log the try (in finally)
        # If a root is found (no error), then return the root (in else)
        #    and log the try (in finally)
        try:
            spline = CubicSpline(taus, polflux_difference_along_ray_line, extrapolate=False)
            spline_roots = np.array(spline.roots())
            root = spline_roots[(np.abs(spline_roots - poloidal_flux_enter)).argmin()]
        except (ValueError, ZeroDivisionError) as e:
            log.debug(f"Unable to find any roots with `tau_max` = {tau_max} with error: {e}")
            spline_roots = None
            root = None
        else: return root, spline_roots
        finally:
            _arr = polflux_difference_along_ray_line + poloidal_flux_enter if polflux_difference_along_ray_line is not None else None
            log.debug(f"""
        ##################################################
        #
        # Finding plasma entry position. Trying:
        #   - Launch position [X,Y,Z] = {arr2str(q_launch_cartesian)}
        #   - tau = 0 to {tau_max}
        #   - poloidal_flux_enter = {poloidal_flux_enter}
        #   - poloidal flux along the ray line =
        #        {mln(_arr, "        #        ") if _arr is not None else None}
        #   - (Cubic) spline roots = {mln(spline_roots, "        #        ") if spline_roots is not None else None}
        #   - Selected root = {root}
        #   - Position of selected root [X,Y,Z] = {arr2str(_ray_line_scalar_wrapper(root)) if root else None}
        #   - Poloidal flux position of selected root = {_poloidal_flux_difference_along_ray_line_scalar_wrapper(root) + poloidal_flux_enter if root else None}
        #
        ##################################################
        """)
    
    _tau_max_list = [0.5, 1.0, 1.5, 2.0]
    for tau_max in _tau_max_list:
        result = _find_plasma_root(tau_max)
        if result is not None:
            root, spline_roots = _find_plasma_root(tau_max)
            break
        else:
            if tau_max == max(_tau_max_list): raise ValueError(f"No root found for `tau_max` = {tau_max}")
            else: log.info(f"No root found for `tau_max` = {tau_max}")

    # If there are no roots, then the beam never actually enters the
    # plasma, and we should abort. We also get an idea of where the
    # closest encounter the ray line makes with the plasma is.
    if len(spline_roots) == 0:
        minimum = minimize_scalar(_poloidal_flux_difference_along_ray_line_scalar_wrapper)
        q_closest_approach = _ray_line_scalar_wrapper(minimum.x)
        raise RuntimeError(f"Beam does not hit plasma. Closest point is at [X,Y,Z] = {arr2str(q_closest_approach)},"
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
    if field.polflux(*q_initial_cartesian) > poloidal_flux_enter:
        q_initial_cartesian = _ray_line_scalar_wrapper(boundary_tau + boundary_adjust)
        log.debug(f"""
        Poloidal flux at plasma entry position is greater than `poloidal_flux_enter.`
        
        Adjusting by tau = {boundary_adjust} to [X,Y,Z] = {arr2str(q_initial_cartesian)}
        """)
    
    log.debug(f"Plasma entry position, `q_initial_cartesian`, is [X,Y,Z] = {arr2str(q_initial_cartesian)}")

    return q_initial_cartesian



def find_auto_delta_signs(auto_delta_sign: bool,
                          q_initial_cartesian: FloatArray,
                          delta_X: float, delta_Y: float, delta_Z: float,
                          field: MagneticField_Cartesian):
    
    # Now, we also perform auto_delta_sign checks (if the user specifies)
    if auto_delta_sign:
        log.info(f"Setting delta signs")

        if field.d_polflux_dX(*q_initial_cartesian, delta_X) > 0:
            delta_X = -1*delta_X
            log.debug(f"Switched delta_X from {-1*delta_X} to {delta_X}")
        
        if field.d_polflux_dY(*q_initial_cartesian, delta_Y) > 0:
            delta_Y = -1*delta_Y
            log.debug(f"Switched delta_Y from {-1*delta_Y} to {delta_Y}")
        
        if field.d_polflux_dZ(*q_initial_cartesian, delta_Z) > 0:
            delta_Z = -1*delta_Z
            log.debug(f"Switched delta_Z from {-1*delta_Z} to {delta_Z}")
    
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
                                 field: MagneticField_Cartesian,
                                 K_plasmaLaunch_cartesian: Optional[FloatArray] = np.zeros(3),
                                 Psi_3D_plasmaLaunch_labframe_cartesian: Optional[FloatArray] = np.zeros((3,3)),
                                 hamiltonian_pos1: Optional[Hamiltonian_3D] = None,
                                 hamiltonian_neg1: Optional[Hamiltonian_3D] = None,
                                 tol_H: float = 1e-5,
                                 tol_O_mode_polarisation: float = 0.25):
    
    log.info(f"Finding plasma entry parameters")
    
    # TO REMOVE -- 5 Nov
    # need to double confirm this
    #
    # hey Valerian, so for the launch_beam code in cyl Scotty, this is what I'm thinking the flow is right now:
    #
    # vacuumLaunch and vacuum_propagation are True => normal launches like irl, and proceed as per normal
    #
    # vacuumLaunch True but propagation False => we launch from just outside the plasma. Initialise a Gaussian beam right outside the plasma and then immediately apply BCs? But the problem is that the code doesn't actually apply BCs
    #
    # vacuumLaunch False but propagation True => doesn't make sense? and we should stop the user from passing this at check_input.py?
    #
    # vacuumLaunch and vacuum_propagation are False => we launch from wherever in the plasma, and we should enforce that the user passes plasmaLaunch_K and plasmaLaunch_Psi at check_input.py?
    
    log.debug(f"""
        ##################################################
        #
        # Finding plasma entry parameters with:
        #   - vacuumLaunch_flag = {vacuumLaunch_flag}
        #   - vacuum_propagation_flag = {vacuum_propagation_flag}
        #   - Psi_BC_flag = {Psi_BC_flag}
        #   - mode_flag (at launch, from user) = {mode_flag_launch}
        #
        ##################################################
        """)
    
    # If vacuumLaunch_flag is False, that most likely means
    # we want to start the propagation from inside the plasma
    if not vacuumLaunch_flag:
        K_launch_cartesian = None
        K_initial_cartesian = K_plasmaLaunch_cartesian
        Psi_3D_launch_labframe_cartesian = None
        Psi_3D_entry_labframe_cartesian = None
        Psi_3D_initial_labframe_cartesian = Psi_3D_plasmaLaunch_labframe_cartesian
        distance_from_launch_to_entry = None
        e_hat_initial = None
        mode_flag_initial = mode_flag_launch
        mode_index = None
        log.debug(f"""
        `vacuumLaunch_flag` is {not vacuumLaunch_flag}. Launching directly from inside plasma
        ##################################################
        #
        # Calculated values at {q_initial_cartesian}:
        #   - K_launch_cartesian = {K_launch_cartesian}
        #   - K_initial_cartesian = {K_initial_cartesian}
        #   - Psi_3D_launch_labframe_cartesian =
        #        {Psi_3D_launch_labframe_cartesian}
        #   - Psi_3D_entry_labframe_cartesian =
        #        {Psi_3D_entry_labframe_cartesian}
        #   - Psi_3D_initial_labframe_cartesian =
        #        {Psi_3D_initial_labframe_cartesian}
        #   - distance_from_launch_to_entry = {distance_from_launch_to_entry}
        #   - e_hat_initial = {e_hat_initial}
        #   - mode_flag at plasma boundary (corresponding to mode_flag at launch) = {mode_flag_initial}
        #   - mode_index = {mode_index}
        #
        ##################################################
        """)
        return (K_launch_cartesian,
                K_initial_cartesian,
                Psi_3D_launch_labframe_cartesian,
                Psi_3D_entry_labframe_cartesian,
                Psi_3D_initial_labframe_cartesian,
                distance_from_launch_to_entry,
                e_hat_initial,
                mode_flag_initial,
                mode_index)
    
    # If vacuumLaunch_flag is True, then continue as usual
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
    if not vacuum_propagation_flag:
        K_initial_cartesian = K_launch_cartesian
        Psi_3D_entry_labframe_cartesian = None
        Psi_3D_initial_labframe_cartesian = Psi_3D_launch_labframe_cartesian
        distance_from_launch_to_entry = None
        e_hat_initial = None
        mode_flag_initial = mode_flag_launch
        mode_index = None
        log.debug(f"""
        `vacuum_propagation_flag` is {not vacuum_propagation_flag}. Launching directly from right outside plasma
        ##################################################
        #
        # Calculated values:
        #   - K_launch_cartesian = {K_launch_cartesian}
        #   - K_initial_cartesian = {K_initial_cartesian}
        #   - Psi_3D_launch_labframe_cartesian =
        #        {Psi_3D_launch_labframe_cartesian}
        #   - Psi_3D_entry_labframe_cartesian =
        #        {Psi_3D_entry_labframe_cartesian}
        #   - Psi_3D_initial_labframe_cartesian =
        #        {Psi_3D_initial_labframe_cartesian}
        #   - distance_from_launch_to_entry = {distance_from_launch_to_entry}
        #   - e_hat_initial = {e_hat_initial}
        #   - mode_flag at plasma boundary (corresponding to mode_flag at launch) = {mode_flag_initial}
        #   - mode_index = {mode_index}
        #
        ##################################################
        """)
        return (K_launch_cartesian,
                K_initial_cartesian,
                Psi_3D_launch_labframe_cartesian,
                Psi_3D_entry_labframe_cartesian,
                Psi_3D_initial_labframe_cartesian,
                distance_from_launch_to_entry,
                e_hat_initial,
                mode_flag_initial,
                mode_index)
    
    # If vacuum_propagation_flag is True, then the beam is launched
    # from outside the plasma, and now we have to propagate the beam
    # until it reaches the plasma boundary, and then apply either the
    # continuous or discontinuous boundary conditions to find K_entry
    # and Psi_entry when the beam enters the plasma
    log.debug(f"`vacuumLaunch_flag` and `vacuum_propagation_flag` are True. Launching from outside the plasma")
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
        log.debug(f"`Psi_BC_flag` is {Psi_BC_flag}. Applying boundary conditions to find K and Psi")
        try:               density_fit, temperature_fit = hamiltonian_pos1.density, hamiltonian_pos1.temperature
        except NameError:  density_fit, temperature_fit = hamiltonian_neg1.density, hamiltonian_neg1.temperature

        polflux = field.polflux(*q_initial_cartesian)
        electron_density = density_fit(polflux)
        electron_temperature = temperature_fit(polflux) if temperature_fit else None

        B_magnitude = field.magnitude(*q_initial_cartesian)
        b_hat = field.unitvector(*q_initial_cartesian)
        epsilon = DielectricTensor_Cartesian(electron_density, launch_angular_frequency, B_magnitude, electron_temperature)

        log.debug(f"""
        ##################################################
        #
        # Calculated values at [X,Y,Z] = {arr2str(q_initial_cartesian)}:
        #   - poloidal flux = {polflux}
        #   - n_e = {electron_density} (e19)
        #   - T_e = {electron_temperature}
        #   - |B| = {B_magnitude}
        #   - b_hat = {arr2str(b_hat)}
        #   - epsilon_11 = {epsilon.e_11}
        #   - epsilon_12 = {epsilon.e_12}
        #   - epsilon_bb = {epsilon.e_bb}
        #
        ##################################################
        """)

        # If mode_flag is 1 or -1, then we just calculate the
        # quantities for those. If mode_flag is "O" or "X", then
        # we calculate the quantities for both 1 and -1. After
        # obtaining the quantities corresponding to H_booker = 0,
        # we calculate H_Cardano and find the polarisation vector
        # corresponding to H_booker = H_Cardano = 0 to see if it's
        # O- or X-mode

        O_mode, X_mode = [], []

        if mode_flag in [1, "O", "X"]:
            log.debug(f"`mode_flag` is {mode_flag}. Applying boundary conditions for Hamiltonian with `mode_flag` = 1")
            (K_initial_cartesian_pos1, 
             Psi_3D_initial_labframe_cartesian_pos1) = apply_BC_3D(
                *q_initial_cartesian,
                *K_launch_cartesian,
                Psi_3D_entry_labframe_cartesian,
                field,
                hamiltonian_pos1,
                Psi_BC_flag)
            
            K_hat_initial_cartesian_pos1 = K_initial_cartesian_pos1 / np.linalg.norm(K_initial_cartesian_pos1)
            theta_m_pos1 = np.arcsin(np.dot(b_hat, K_hat_initial_cartesian_pos1))

            (H_Cardano_pos1,
             e_hats_pos1) = find_H_Cardano_eig(
                np.linalg.norm(K_initial_cartesian_pos1),
                launch_angular_frequency,
                epsilon.e_bb,
                epsilon.e_11,
                epsilon.e_12,
                theta_m_pos1)
            
            mode_index_pos1 = np.squeeze(np.where(np.abs(H_Cardano_pos1) <= tol_H)[0])

            # If more than one H_Cardano is close to 0, then we're
            # unable to check which corresponds to O- or X-mode
            if mode_index_pos1.size != 1: raise ValueError(f"Unable to check which mode index corresponds to O- and X-mode. Found {mode_index_pos1.size} solutions!")
            else: e_hat_pos1 = e_hats_pos1[:, mode_index_pos1]

            log.debug(f"""
        ##################################################
        #
        # Calculated values for Hamiltonian with `+1`:
        #   - K_initial_cartesian = {arr2str(K_initial_cartesian_pos1)}
        #
        #   - Psi_3D_initial_labframe_cartesian =
        #        [{arr2str(Psi_3D_initial_labframe_cartesian_pos1[0])},
        #         {arr2str(Psi_3D_initial_labframe_cartesian_pos1[1])},
        #         {arr2str(Psi_3D_initial_labframe_cartesian_pos1[2])}]
        #
        #   - theta_m (in radians) = {theta_m_pos1}
        #   - theta_m (in degrees) = {np.rad2deg(theta_m_pos1)}
        #
        #   - H[0] = {H_Cardano_pos1[0]}
        #   - e_hat[0] = {arr2str(e_hats_pos1[:, 0])}
        #
        #   - H[1] = {H_Cardano_pos1[1]}
        #   - e_hat[1] = {arr2str(e_hats_pos1[:, 1])}
        #
        #   - H[2] = {H_Cardano_pos1[2]}
        #   - e_hat[2] = {arr2str(e_hats_pos1[:, 2])}
        #
        #   - mode_index = {mode_index_pos1}
        #   - selected H = {H_Cardano_pos1[mode_index_pos1]}
        #   - selected e_hat = {arr2str(e_hat_pos1)}
        #
        ##################################################
        """)

            temp_pos1 = {"K": K_initial_cartesian_pos1,
                         "Psi": Psi_3D_initial_labframe_cartesian_pos1,
                         "mode_flag": hamiltonian_pos1.mode_flag,
                         "mode_index": mode_index_pos1,
                         "H": H_Cardano_pos1[mode_index_pos1],
                         "e_hat": e_hat_pos1}
        
        if mode_flag in [-1, "O", "X"]:
            log.debug(f"`mode_flag` is {mode_flag}. Applying boundary conditions for Hamiltonian with `mode_flag` = -1")
            (K_initial_cartesian_neg1,
             Psi_3D_initial_labframe_cartesian_neg1) = apply_BC_3D(
                *q_initial_cartesian,
                *K_launch_cartesian,
                Psi_3D_entry_labframe_cartesian,
                field,
                hamiltonian_neg1,
                Psi_BC_flag)
            
            K_hat_initial_cartesian_neg1 = K_initial_cartesian_neg1 / np.linalg.norm(K_initial_cartesian_neg1)
            theta_m_neg1 = np.arcsin(np.dot(b_hat, K_hat_initial_cartesian_neg1))
            
            (H_Cardano_neg1,
             e_hats_neg1) = find_H_Cardano_eig(
                np.linalg.norm(K_initial_cartesian_neg1),
                launch_angular_frequency,
                epsilon.e_bb,
                epsilon.e_11,
                epsilon.e_12,
                np.arcsin(np.dot(b_hat, K_hat_initial_cartesian_neg1)))
            
            mode_index_neg1 = np.squeeze(np.where(np.abs(H_Cardano_neg1) <= tol_H)[0])

            # If more than one H_Cardano is close to 0, then we're
            # unable to check which corresponds to O- or X-mode
            if mode_index_neg1.size != 1: raise ValueError(f"Unable to check which mode index corresponds to O- and X-mode. Found {mode_index_neg1.size} solutions!")
            else: e_hat_neg1 = e_hats_neg1[:, mode_index_neg1]

            log.debug(f"""
        ##################################################
        #
        # Calculated values for Hamiltonian with `-1`:
        #   - K_initial_cartesian = {arr2str(K_initial_cartesian_neg1)}
        #
        #   - Psi_3D_initial_labframe_cartesian =
        #        [{arr2str(Psi_3D_initial_labframe_cartesian_neg1[0])},
        #         {arr2str(Psi_3D_initial_labframe_cartesian_neg1[1])},
        #         {arr2str(Psi_3D_initial_labframe_cartesian_neg1[2])}]
        #
        #   - theta_m (in radians) = {theta_m_neg1}
        #   - theta_m (in degrees) = {np.rad2deg(theta_m_neg1)}
        #
        #   - H[0] = {H_Cardano_neg1[0]}
        #   - e_hat[0] = {arr2str(e_hats_neg1[:, 0])}
        #
        #   - H[1] = {H_Cardano_neg1[1]}
        #   - e_hat[1] = {arr2str(e_hats_neg1[:, 1])}
        #
        #   - H[2] = {H_Cardano_neg1[2]}
        #   - e_hat[2] = {arr2str(e_hats_neg1[:, 2])}
        #
        #   - mode_index = {mode_index_neg1}
        #   - selected H = {H_Cardano_neg1[mode_index_neg1]}
        #   - selected e_hat = {arr2str(e_hat_neg1)}
        #
        ##################################################
        """)

            temp_neg1 = {"K": K_initial_cartesian_neg1,
                         "Psi": Psi_3D_initial_labframe_cartesian_neg1,
                         "mode_flag": hamiltonian_neg1.mode_flag,
                         "mode_index": mode_index_neg1,
                         "H": H_Cardano_neg1[mode_index_neg1],
                         "e_hat": e_hat_neg1}
        
        # If the user has specified mode_flag = "O" or "X", then
        # we compare the polarisations to determine which is the
        # correct mode, and also check with tol_O_mode_polarisation
        # to double confirm. If the user has specified 
        # mode_flag = 1 or -1, then just use tol_O_mode_polarisation to check
        if mode_flag in ["O", "X"]:
            Re_e_hat_z_pos1 = abs(np.real(e_hat_pos1[-1]))
            Re_e_hat_z_neg1 = abs(np.real(e_hat_neg1[-1]))
            if Re_e_hat_z_pos1 < 1-tol_O_mode_polarisation and Re_e_hat_z_neg1 < 1-tol_O_mode_polarisation:
                log.warning(f"Both computed polarisation vectors ({Re_e_hat_z_pos1} and {Re_e_hat_z_neg1} for `mode_flag` = +1 and -1 respectively) are not within the O-mode polarisation tolerance (tol = {tol_O_mode_polarisation})!")
                log.warning(f"This may or may not cause issues.")
                log.warning(f"Setting the vector with the larger z-component to be O-mode")
            if Re_e_hat_z_pos1 > Re_e_hat_z_neg1:
                O_mode.append(temp_pos1)
                X_mode.append(temp_neg1)
            else:
                O_mode.append(temp_neg1)
                X_mode.append(temp_pos1)
        elif mode_flag == 1:
            Re_e_hat_z_pos1 = np.real(e_hat_pos1[-1])
            if 1-Re_e_hat_z_pos1 < tol_O_mode_polarisation: X_mode.append(temp_pos1)
            else: O_mode.append(temp_pos1)
        elif mode_flag == -1:
            Re_e_hat_z_neg1 = np.real(e_hat_neg1[-1])
            if 1-Re_e_hat_z_neg1 < tol_O_mode_polarisation: X_mode.append(temp_neg1)
            else: O_mode.append(temp_neg1)
        
        # A bunch of final checks
        if   len(O_mode) > 1: raise ValueError((f"Found {len(O_mode)} solutions for O-mode!"))
        elif len(X_mode) > 1: raise ValueError((f"Found {len(X_mode)} solutions for X-mode!"))
        elif len(O_mode) == 0 and mode_flag == "O": raise ValueError((f"Found no solutions for O-mode instead of 1!"))
        elif len(X_mode) == 0 and mode_flag == "X": raise ValueError((f"Found no solutions for X-mode instead of 1!"))

        # If the user specified 1 or -1, then we must have only calculated
        # 1 set of quantities in the previous code, so just return that.
        # If the user specified "O" or "X", then we calculated 2 sets of
        # quantities in the previous code, so select the set corresponding
        # to the mode that the user wants
        if mode_flag in [1, -1]:
            if   len(O_mode) == 1 and len(X_mode) == 0:
                sel,    soln  = "O", O_mode[0]
                notsel, other = "X", None
            elif len(X_mode) == 1 and len(O_mode) == 0:
                sel,    soln  = "X", X_mode[0]
                notsel, other = "O", None
            else: raise ValueError(f"Unable to unpack corresponding mode solution!")
        elif mode_flag == "O":
            sel,    soln  = "O", O_mode[0]
            notsel, other = "X", X_mode[0]
        elif mode_flag == "X":
            sel,    soln  = "X", X_mode[0]
            notsel, other = "O", O_mode[0]
        
        log.debug(f"""
        ##################################################
        #
        # Calculated plasma entry parameters for:
        #   - {sel}-mode (selected):
        #        - K = {arr2str(soln["K"])}
        #
        #        - Psi_3D =
        #             [{arr2str(soln["Psi"][0])},
        #              {arr2str(soln["Psi"][1])},
        #              {arr2str(soln["Psi"][2])}]
        #
        #        - mode_flag = {soln["mode_flag"]}
        #
        #        - mode_index = {soln["mode_index"]}
        #
        #        - H = {soln["H"]}
        #
        #        - e_hat = {arr2str(soln["e_hat"])}
        #
        #   - {notsel}-mode (not selected):
        #        - K = {arr2str(other["K"]) if other is not None else None}
        #
        #        - Psi_3D =
        #             [{arr2str(other["Psi"][0]) + "," if other is not None else None}
        #              {arr2str(other["Psi"][1]) + "," if other is not None else ""}
        #              {arr2str(other["Psi"][2]) + "]" if other is not None else ""}
        #
        #        - mode_flag = {other["mode_flag"] if other is not None else None}
        #
        #        - mode_index = {other["mode_index"] if other is not None else None}
        #
        #        - H = {other["H"] if other is not None else None}
        #
        #        - e_hat = {arr2str(other["e_hat"]) if other is not None else None}
        #
        ##################################################
        """)
    
        # Unpacking the solution
        K_initial_cartesian = soln["K"]
        Psi_3D_initial_labframe_cartesian = soln["Psi"]
        mode_flag_initial = soln["mode_flag"]
        mode_index = soln["mode_index"]
        H_initial = soln["H"]
        e_hat_initial = soln["e_hat"]
    
    # No BC case
    # TO REMOVE TODO: Not implemented properly yet? Esp. mode_index
    else:
        K_initial_cartesian = K_launch_cartesian
        Psi_3D_initial_labframe_cartesian = Psi_3D_entry_labframe_cartesian
        e_hat_initial = None
        mode_flag_initial = mode_flag_launch
        mode_index = None

        log.debug(f"""
        `Psi_BC_flag` is {Psi_BC_flag}. Boundary conditions are not applied
        ##################################################
        #
        # Calculated plasma entry parameters for:
        #   - K = {K_initial_cartesian}
        #   - Psi_3D =
        #        {Psi_3D_initial_labframe_cartesian}
        #   - e_hat = {e_hat_initial}
        #   - mode_flag = {mode_flag_initial}
        #   - mode_index = {mode_index}
        #
        ##################################################
        """)

    return (K_launch_cartesian,  # K_launch_cartesian,
            K_initial_cartesian, # K_initial_cartesian,
            Psi_3D_launch_labframe_cartesian,  # Psi_3D_launch_labframe_cartesian,
            Psi_3D_entry_labframe_cartesian,   # Psi_3D_entry_labframe_cartesian,
            Psi_3D_initial_labframe_cartesian, # Psi_3D_initial_labframe_cartesian,
            distance_from_launch_to_entry, # distance_from_launch_to_entry,
            e_hat_initial,
            mode_flag_initial,
            mode_index)