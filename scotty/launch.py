import logging
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, root_scalar
from scotty.boundary_conditions import apply_boundary_conditions
from scotty.checks import (
    VALID_FIELDS,
    VALID_LAUNCH_FLAGS,
    VALID_LAUNCH_MODE_FLAGS,
    VALID_BOUNDARY_FLAGS,
)
from scotty.fun_general import (
    find_K_labframe_cart_to_cyl,
    find_K_magnitude,
    find_q_labframe_cart_to_cyl,
    find_q_labframe_cyl_to_cart,
    find_Psi_3D_labframe_cart_to_cyl,
    ray_line,
    poloidal_flux_difference_along_ray_line,
    make_array_3x3,
    find_inverse_2D,
    find_epsilon_terms,
    find_H_Cardano_eigh,
)
from scotty.geometry import MagneticField_Cylindrical
from scotty.hamiltonian import DielectricTensor, Hamiltonian
from scotty.logger import arr2str
from scotty.typing import FloatArray, ComplexFloatArray
from typing import cast, Optional, Tuple, Union, List, Literal

log = logging.getLogger(__name__)


def find_plasma_entry_position(
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    q_launch: FloatArray,
    launch_flag: VALID_LAUNCH_FLAGS,
    field: VALID_FIELDS,
    poloidal_flux_enter: float,
    boundary_adjust: float = 1e-6,
) -> FloatArray:
    r"""Returns `q_initial` in cartesian or cylindrical"""

    log.info(f"Finding plasma entry position")

    # If launch_flag is "plasma", we are already in the plasma
    # so take this as the entry position
    if launch_flag == "plasma":
        log.debug(
            f"`launch_flag` = {launch_flag}, so skipping plasma entry position calculations"
        )
        return q_launch

    # The following calculations will be performed in cartesian and
    # then depending on the geometry we convert accordingly. The following
    # code is equivalent to finding the part of the ray that first enters
    # the plasma and is going deeper. We do this entirely in cartesian
    # coordinates, and then depending on the geometry we convert accordingly.

    if isinstance(field, MagneticField_Cylindrical):
        cart = False
        q_launch_cartesian = find_q_labframe_cyl_to_cart(q_launch)
        max_dist = max(q_launch_cartesian)
    else:  # isinstance(field, MagneticField_Cartesian):
        cart = True
        q_launch_cartesian = q_launch
        max_dist = max(q_launch_cartesian)

    # Defining some wrapper functions for ease
    def _ray_line_wrapper(tau):
        return ray_line(*q_launch_cartesian, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)  # type: ignore

    def _poloidal_flux_difference_along_ray_line_wrapper(tau):
        return poloidal_flux_difference_along_ray_line(*q_launch_cartesian, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, field.polflux_cartesian, poloidal_flux_enter)  # type: ignore

    # Given a particular field configuration, we find the maximum
    # distance that can be travelled by a ray before 'striking the
    # centre column', then parametrise the path into sections of
    # 0.01m each, and find how many are needed. We also insist that
    # there are at least 100 steps

    num_tau = int(max(max_dist / 0.01, 100))
    tau_arr = np.linspace(0, max_dist, num_tau)

    # We then launch a ray along this line, which is parametrised by tau,
    # calculate the poloidal flux coordinates along this line, filter out
    # the `NaN`s, find the index of the minimum poloidal flux value, and
    # then create a refined array by re-selecting the start and end taus.
    # If the entire array is `NaN`s, then raise error

    X_arr, Y_arr, Z_arr = ray_line(*q_launch_cartesian, tau_arr, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)  # type: ignore
    polflux_arr = field.polflux_cartesian(X_arr, Y_arr, Z_arr)

    if np.isnan(polflux_arr).all():
        raise RuntimeError(
            f"The ray does not intersect the plasma. Check that the launch position is from q_zeta = 0 with acute launch angles"
        )

    start_idx = np.where(~np.isnan(polflux_arr))[0][0]
    stop_idx = np.nanargmin(polflux_arr)

    start_tau = tau_arr[start_idx]
    stop_tau = tau_arr[stop_idx]

    # With the `tau`s of  the start and endpoints, we now create a
    # refined array, get the corresponding poloidal flux values, and
    # feed it into a cubic spline to find the root (i.e. `tau` value
    # where poloidal flux is close to `poloidal_flux_enter`

    tau_arr_refined = np.linspace(start_tau, stop_tau, num_tau)
    polflux_arr_refined = field.polflux_cartesian(*ray_line(*q_launch_cartesian, tau_arr_refined, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam))  # type: ignore

    if np.isnan(polflux_arr_refined).all():
        log.warning(f"NaNs should not occur in the refined `tau` array search")

    spline = CubicSpline(
        tau_arr_refined, polflux_arr_refined - poloidal_flux_enter, extrapolate=False
    )
    spline_roots = np.array(spline.roots())

    # If there are no roots, then the ray never actually enters the
    # plasma, and we should abort. We also get an idea of where the
    # closest encounter the ray line makes with the plasma is

    if len(spline_roots) == 0:
        minimum = minimize_scalar(_poloidal_flux_difference_along_ray_line_wrapper)
        q_closest_approach_cartesian = _ray_line_wrapper(minimum.x)
        q_closest_approach_cylindrical = find_q_labframe_cart_to_cyl(
            q_closest_approach_cartesian
        )
        _str = (
            f"[X,Y,Z] = {arr2str(q_closest_approach_cartesian)}"
            if cart
            else f"[R,zeta,Z] = {arr2str(q_closest_approach_cylindrical)}"
        )
        raise RuntimeError(
            f"The ray does not intersect the plasma. Closest point is at {_str},"
            f"distance in poloidal flux to boundary = {minimum.fun}"
        )  # type: ignore

    # The spline roots are a pretty good guess for the boundary
    # location, which we now try to refine

    root = spline_roots[(np.abs(spline_roots - poloidal_flux_enter)).argmin()]
    boundary = root_scalar(
        _poloidal_flux_difference_along_ray_line_wrapper, x0=root, x1=root + 1e-3
    )
    if not boundary.converged:
        raise RuntimeError(
            f"Unable to find plasma boundary. Root finding failed with `{boundary.flag}`"
        )
    else:
        boundary_tau = boundary.root

    # The root might be just outside the plasma due to floating point
    # errors. If so, take a small step of size `boundary_adjust` to
    # ensure the ray is definitely inside

    q_initial_cartesian = _ray_line_wrapper(boundary_tau)
    if field.polflux_cartesian(*q_initial_cartesian) > poloidal_flux_enter:
        q_initial_cartesian = _ray_line_wrapper(boundary_tau + boundary_adjust)
        log.debug(
            f"""
        Poloidal flux at plasma entry position is greater than `poloidal_flux_enter.`
        Adjusting by a small tau = {boundary_adjust} to obtain adjusted entry point [X,Y,Z] = {arr2str(q_initial_cartesian)}"""
        )

    # Final conversions back to the proper geometry
    log.debug(
        f"Plasma entry position, `q_initial_cartesian`, is [X,Y,Z] = {arr2str(q_initial_cartesian)}"
    )
    return (
        q_initial_cartesian
        if cart
        else find_q_labframe_cart_to_cyl(q_initial_cartesian)
    )


def find_auto_delta_signs(
    auto_delta_sign: bool,
    q_initial: FloatArray,
    deltas: FloatArray,
    field: VALID_FIELDS,
) -> FloatArray:

    # Now, we also perform auto_delta_sign checks (if the user specifies)
    # If the user does not want to flip the signs, just return them
    # Otherwise, check which geometry is being used and check the signs
    if not auto_delta_sign:
        return deltas

    log.info(f"Setting delta signs")
    deltas = np.asarray(deltas, dtype=float)

    def _check_delta_and_log(derivative, delta_name, delta_value):
        if derivative(*q_initial, delta_value) > 0:
            log.debug(
                f"Switching `{delta_name}` from `{delta_value}` to `{-delta_value}`"
            )
            delta_value = -delta_value
        return delta_value

    if isinstance(field, MagneticField_Cylindrical):
        delta_R, delta_Z = deltas
        delta_R = _check_delta_and_log(field.d_polflux_dR, "delta_R", delta_R)
        delta_Z = _check_delta_and_log(field.d_polflux_dZ, "delta_Z", delta_Z)
        deltas = np.array([delta_R, delta_Z])

    else:  # isinstance(field, MagneticField_Cartesian):
        delta_X, delta_Y, delta_Z = deltas
        delta_X = _check_delta_and_log(field.d_polflux_dX, "delta_X", delta_X)
        delta_Y = _check_delta_and_log(field.d_polflux_dY, "delta_Y", delta_Y)
        delta_Z = _check_delta_and_log(field.d_polflux_dZ, "delta_Z", delta_Z)
        deltas = np.array([delta_X, delta_Y, delta_Z])

    return deltas


def find_mode_index_and_ehat(
    mode_flag: VALID_LAUNCH_MODE_FLAGS,
    H_Cardanos: FloatArray,
    ehats: ComplexFloatArray,
    tol_H: float = 1e-3,
    tol_O_mode_polarisation: float = 0.25,
) -> Tuple[int, float, ComplexFloatArray]:
    r"""`H_Cardanos` and `ehats` must be arranged such that `H_Cardano[i]` corresponds
    to `ehats[:,i]`. If more than one `mode_flag` is provided, then `mode_flag[i]` must
    also correspond to `H_cardano[i]`"""
    soln_idxs = np.where(np.abs(H_Cardanos) <= tol_H)[0]
    soln_H = H_Cardanos[soln_idxs]
    soln_ehats = ehats[
        :, soln_idxs
    ]  # note that soln_H still corresponds to soln_ehats[i,:]

    # We have three cases to check for:
    #   (i)   If there are none or three solutions, raise an error
    #   (ii)  If there is one solution, just return that
    #   (iii) If there is more than 1 solution and the user specifies `mode_flag` = 1 or -1,
    # then we have no way of knowing what exactly the user wants (i.e. we don't know
    # if the O-mode or X-mode solution is wanted), so warn the user and choose the smallest solution
    #   Otherwise, check which mode index yields the desired mode

    if soln_idxs.shape in [(0,), (3,)]:
        log.warning(
            f"Possible wrong choice in mode index corresponding to O- and X-mode. Found {soln_idxs.shape} solutions (H = {H_Cardanos})! Check that `poloidal_flux_enter` is not too deep in the plasma"
        )
        log.warning(f"Selecting min(H) as the solution")
        mode_idx = np.argmin(np.abs(H_Cardanos))
        H = H_Cardanos[mode_idx]
        ehat = ehats[mode_idx]

    elif soln_idxs.shape == (1,):
        mode_idx = soln_idxs[0]
        H = soln_H[0]
        ehat = soln_ehats

    elif soln_idxs.shape == (2,) and mode_flag in [1, -1]:
        log.warning(
            f"Specific mode to be used not specified (`mode_flag` = {mode_flag}) and unable to determine which mode indices correspond to O- and X-mode. Perhaps `tol_H` may be too large, so try selecting a deeper `poloidal_flux_enter` or use `boundary_flag` = `continuous` instead"
        )
        log.warning(f"Selecting min(H) as the solution")
        mode_idx = np.argmin(np.abs(H_Cardanos))
        H = H_Cardanos[mode_idx]
        ehat = ehats[mode_idx]

    else:  # soln_idxs.shape == (2,) and mode_flag in ["O", "X"]:
        H0, H1 = H_Cardanos[soln_idxs]
        e0, e1 = soln_ehats.T  # because soln_ehats.shape == (3,2)
        abs_Re_e0_b = abs(np.real(e0[-1]))
        abs_Re_e1_b = abs(np.real(e1[-1]))

        if (
            abs_Re_e0_b < 1 - tol_O_mode_polarisation
            and abs_Re_e1_b < 1 - tol_O_mode_polarisation
        ):
            log.warning(
                f"Both computed vectors ({e0} and {e1}) are not within the O-mode polarisation tolerance (tol = {tol_O_mode_polarisation})"
            )
            log.warning(
                f"Setting the vector with the largets b-component to be the O-mode. This may or may not cause issues"
            )

        idx = 0 if abs_Re_e0_b > abs_Re_e1_b else 1
        mode_idx = soln_idxs[idx]
        H = soln_H[idx]
        ehat = soln_ehats.T[idx]  # because soln_ehats.shape == (3,2)

    return cast(int, np.squeeze(mode_idx)), np.squeeze(H), np.squeeze(ehat)


def find_plasma_entry_parameters(
    ray_tracing_flag: bool,
    launch_flag: VALID_LAUNCH_FLAGS,
    boundary_flag: VALID_BOUNDARY_FLAGS,
    mode_flag_launch: VALID_LAUNCH_MODE_FLAGS,
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    q_launch: FloatArray,
    q_initial: FloatArray,
    launch_beam_width: float,
    launch_beam_curvature: float,
    field: VALID_FIELDS,
    K_plasmaLaunch_cartesian: Optional[FloatArray] = None,
    Psi_3D_plasmaLaunch_labframe_cartesian: Optional[FloatArray] = None,
    hamiltonian_pos1: Optional[Hamiltonian] = None,
    hamiltonian_neg1: Optional[Hamiltonian] = None,
    tol_H: float = 1e-3,
    tol_O_mode_polarisation: float = 0.25,
) -> Tuple[
    FloatArray,
    FloatArray,
    Optional[ComplexFloatArray],
    Optional[ComplexFloatArray],
    ComplexFloatArray,
    Optional[float],
    ComplexFloatArray,
    Literal[1, -1],
    int,
]:
    r"""Returns `(K_launch, K_initial, Psi_3D_launch_labframe,
    Psi_3D_entry_labframe, Psi_3D_initial_labframe,
    distance_from_launch_to_entry, e_hat_initial,
    mode_flag_initial, mode_index)`"""

    log.info(f"Finding plasma entry parameters")

    log.debug(f"""
        ##################################################
        #
        # Finding plasma entry parameters with:
        #   - launch_flag = {launch_flag}
        #   - boundary_flag = {boundary_flag}
        #   - mode_flag (at launch, from user) = {mode_flag_launch}
        #
        ##################################################
        """)

    # The following calculations will be performed in cartesian and
    # then depending on the geometry we convert accordingly. The following
    # code is equivalent to finding the part of the ray that first enters
    # the plasma and is going deeper. We do this entirely in cartesian
    # coordinates, and then depending on the geometry we convert accordingly.
    if isinstance(field, MagneticField_Cylindrical):
        cart = False
        q_launch_cartesian = find_q_labframe_cyl_to_cart(q_launch)
        q_initial_cartesian = find_q_labframe_cyl_to_cart(q_initial)
    else:  # isinstance(field, MagneticField_Cartesian):
        cart = True
        q_launch_cartesian = q_launch
        q_initial_cartesian = q_initial

    distance_from_launch_to_entry = cast(
        float, np.linalg.norm(q_launch_cartesian - q_initial_cartesian)
    )

    # If `launch_flag` = "plasma", that means we start propagation
    # from inside the plasma, so we can skip the plasma entry calculations
    # TODO -- TO REMOVE -- this isnt fully implemented, because we dont
    # select the mode_flag_initial based mode_flag_launch, and we also
    # dont calculate the corresponding e_hat, mode_index, etc
    if launch_flag == "plasma":
        log.debug(
            f"`launch_flag` = {launch_flag}. Launching directly from inside the plasma"
        )
        raise NotImplementedError(f"Not implemented yet")

    # Otherwise, we start propagation from vacuum, so we need to
    # calculate the K and Psi, and plus other stuff
    # This, however, is fully implemented -- TO REMOVE this msg
    else:
        log.debug(
            f"`launch_flag` = {launch_flag}. Launching from outside the plasma boundary"
        )
        poloidal_launch_angle = np.deg2rad(poloidal_launch_angle_deg_Torbeam)
        toroidal_launch_angle = np.deg2rad(toroidal_launch_angle_deg_Torbeam)

        if hamiltonian_pos1 is not None:
            launch_angular_freq = hamiltonian_pos1.angular_frequency
            K0 = hamiltonian_pos1.wavenumber
            density_fit = hamiltonian_pos1.density
            temperature_fit = hamiltonian_pos1.temperature

        elif hamiltonian_neg1 is not None:
            launch_angular_freq = hamiltonian_neg1.angular_frequency
            K0 = hamiltonian_neg1.wavenumber
            density_fit = hamiltonian_neg1.density
            temperature_fit = hamiltonian_neg1.temperature

        else:
            raise ValueError(
                f"At least one of `hamiltonian_pos1` and `hamiltonian_neg1` must be provided to find the plasma entry parameters"
            )

        K0 = cast(float, K0)  # to stop the typechecker from complaining

        # Find K_launch
        # In truth, K_launch in cylindrical is the same as in cartesian, but
        # note that K in Scotty does not represent the true toroidal component
        # but the toroidal mode number i.e. K_zeta_true = K_launch[1] / q_R
        K_launch_cartesian = -K0 * np.array(
            [
                np.cos(poloidal_launch_angle) * np.cos(toroidal_launch_angle),
                np.cos(poloidal_launch_angle) * np.sin(toroidal_launch_angle),
                np.sin(poloidal_launch_angle),
            ]
        )
        K_launch = K_launch_cartesian

        if ray_tracing_flag:
            Psi_3D_launch_labframe = None
            Psi_3D_entry_labframe = None
            Psi_3D_entry_labframe_cartesian = None
        else:
            # Finding Psi_w_launch_beamframe_cartesian and Psi_3D_launch_beamframe_cartesian
            # Entries on the off-diagonal = 0, because beamframe
            # Entries on the diagonal = K_0/R + 2i/W^2, where:
            #    R is beam radius of curvature (in metres); and
            #    W is beam width (in metres)
            # First row/column is y-direction; second is x-direction; third is g-direction (beamframe)
            # Not to be confused with X-, Y-, Z-directions (labframe)
            diag = K0 * launch_beam_curvature + 2j / launch_beam_width**2
            Psi_w_launch_beamframe_cartesian = diag * np.eye(2)
            Psi_3D_launch_beamframe_cartesian = make_array_3x3(
                Psi_w_launch_beamframe_cartesian
            )

            # Setting up the rotation matrices, so that we can convert
            # Psi_3D_launch_beamframe_cartesian into Psi_3D_launch_labframe_cartesian
            poloidal_rotation_angle = poloidal_launch_angle + np.pi / 2
            toroidal_rotation_angle = toroidal_launch_angle
            sin_pol, cos_pol = np.sin(poloidal_rotation_angle), np.cos(
                poloidal_rotation_angle
            )
            sin_tor, cos_tor = np.sin(toroidal_rotation_angle), np.cos(
                toroidal_rotation_angle
            )
            poloidal_rotation_matrix = np.array(
                [[cos_pol, 0, sin_pol], [0, 1, 0], [-sin_pol, 0, cos_pol]]
            )
            toroidal_rotation_matrix = np.array(
                [[cos_tor, sin_tor, 0], [-sin_tor, cos_tor, 0], [0, 0, 1]]
            )
            rotation_matrix = np.matmul(
                poloidal_rotation_matrix, toroidal_rotation_matrix
            )
            rotation_matrix_inverse = np.transpose(rotation_matrix)

            # Finding Psi_3D_launch_labframe_cartesian using:
            # Psi_labframe = R^-1 * Psi_beamframe * R, where
            #    R is the rotation matrix to convert a vector from beamframe to labframe
            Psi_3D_launch_labframe_cartesian = np.matmul(
                rotation_matrix_inverse,
                np.matmul(Psi_3D_launch_beamframe_cartesian, rotation_matrix),
            )
            Psi_3D_launch_labframe = (
                Psi_3D_launch_labframe_cartesian
                if cart
                else find_Psi_3D_labframe_cart_to_cyl(
                    Psi_3D_launch_labframe_cartesian,
                    K_launch_cartesian,
                    q_launch_cartesian,
                )
            )

            # Now we propagate the beam until it reaches the plasma boundary,
            # and then apply either the continuous or discontinuous or no
            # boundary conditions to find K_entry and Psi_entry when the
            # beam enters the plasma
            Psi_w_inverse_launch_beamframe_cartesian = find_inverse_2D(
                Psi_w_launch_beamframe_cartesian
            )
            Psi_w_inverse_entry_beamframe_cartesian = (
                distance_from_launch_to_entry / K0 * np.eye(2)
                + Psi_w_inverse_launch_beamframe_cartesian
            )

            # 'Psi_3D_entry' is still in vacuum, so the components of Psi in the
            # beam frame along g are all zero (since grad_H = 0)
            Psi_3D_entry_beamframe_cartesian = make_array_3x3(
                find_inverse_2D(Psi_w_inverse_entry_beamframe_cartesian)
            )
            Psi_3D_entry_labframe_cartesian = np.matmul(
                rotation_matrix_inverse,
                np.matmul(Psi_3D_entry_beamframe_cartesian, rotation_matrix),
            )
            Psi_3D_entry_labframe = (
                Psi_3D_entry_labframe_cartesian
                if cart
                else find_Psi_3D_labframe_cart_to_cyl(
                    Psi_3D_entry_labframe_cartesian,
                    K_launch_cartesian,
                    q_initial_cartesian,
                )
            )

        # If `boundary_flag` is None, then we assume that the electron density
        # profile is both continuous and differentiable at the plasma boundary,
        # so we can just take the values at the plasma launch position as the
        # values to start the solvers at
        # TODO -- TO REMOVE -- not fully implemented, because we dont select
        # the mode_flag_initial, the mode_index, and the e_hat_initial
        if boundary_flag is None:
            log.debug(
                f"`boundary_flag` = {boundary_flag}. No boundary conditions applied"
            )
            raise NotImplementedError(f"Not implemented yet")

        # If `boundary_flag` = "continuous" or "discontinuous", then we assume
        # that the electron density is not differentiable at the plasma boundary,
        # so we apply the appropriate boundary condition calculations. Here, we use
        # K_entry and Psi_entry (without boundary conditions) and get K_initial and
        # Psi_initial (with boundary conditions) which are later fed into the
        # `solve_ivp` as the initial values. We solve this for mode_flags = +1 and -1,
        # then later we check to see which set of K_initial and Psi_initial
        # corresponds to O or X mode
        else:
            log.debug(
                f"`boundary_flag` = {boundary_flag}. Applying {boundary_flag} boundary conditions to find K and Psi"
            )
            polflux = field.polflux(*q_initial)
            electron_density = density_fit(polflux)
            electron_temperature = temperature_fit(polflux) if temperature_fit else None

            B_magnitude = field.magnitude(*q_initial)
            b_hat = field.unitvector(*q_initial)
            epsilon = DielectricTensor(
                launch_angular_freq, B_magnitude, electron_density, electron_temperature
            )

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
            # we calculate H_Cardano and find the polarisation vectorl
            # corresponding to H_booker = H_Cardano = 0 to see if it's
            # O- or X-mode. We do this, literally, by comparing the
            # e_hats with the O-mode polarisation vector; specifically,
            # since the O-mode polarisation vector ~ [0,0,1] (in the u1,
            # u2, b) basis -- see the paper), we compare the last
            # component of both vectors. If the last component of
            # the vector is not close to 1 (i.e. |v[-1] - 1| < tol),
            # then we say it's X-mode; otherwise, it's O-mode

            # We store the corresponding solutions in these lists
            solns = []

            # Pre-calculating some quantities
            epsilon_para, epsilon_perp, epsilon_g = find_epsilon_terms(
                launch_angular_freq=launch_angular_freq,
                B_total=B_magnitude,
                electron_density=electron_density,
                temperature=electron_temperature,
            )

            if mode_flag_launch in [1, "O", "X"] and hamiltonian_pos1 is not None:
                log.debug(
                    f"`mode_flag_launch` is {mode_flag_launch}. Applying boundary conditions for Hamiltonian with `mode_flag` = 1"
                )
                K_initial_pos1, Psi_3D_initial_labframe_pos1 = (
                    apply_boundary_conditions(
                        ray_tracing_flag=ray_tracing_flag,
                        boundary_flag=boundary_flag,
                        q_vacuum_entry_cartesian=q_initial_cartesian,
                        K_vacuum_entry_cartesian=K_launch_cartesian,
                        Psi_3D_vacuum_entry_labframe=Psi_3D_entry_labframe,
                        field=field,
                        hamiltonian=hamiltonian_pos1,
                    )
                )

                K_magnitude_initial_pos1 = find_K_magnitude(cart, *K_initial_pos1, q_initial[0])  # type: ignore
                K_hat_initial_pos1 = K_initial_pos1 / (
                    K_magnitude_initial_pos1
                    * np.array([1, 1 if cart else q_initial[0], 1])
                )
                theta_m_pos1 = np.arcsin(np.dot(b_hat, K_hat_initial_pos1))

                H_Cardanos_pos1, e_hats_pos1 = find_H_Cardano_eigh(
                    launch_angular_freq=launch_angular_freq,
                    K_magnitude=K_magnitude_initial_pos1,
                    epsilon_para=epsilon_para,
                    epsilon_perp=epsilon_perp,
                    epsilon_g=epsilon_g,
                    theta_m=theta_m_pos1,
                )

                mode_index_pos1, H_Cardano_pos1, e_hat_pos1 = find_mode_index_and_ehat(
                    mode_flag=mode_flag_launch,
                    H_Cardanos=H_Cardanos_pos1,
                    ehats=e_hats_pos1,
                    tol_H=tol_H,
                    tol_O_mode_polarisation=tol_O_mode_polarisation,
                )

                log.debug(f"""
        ##################################################
        #
        # Calculated values for Hamiltonian with `+1`:
        #   - K_initial_{"cartesian" if cart else "cylindrical"} = {arr2str(K_initial_pos1)}
        #
        #   - Psi_3D_initial_labframe_{"cartesian" if cart else "cylindrical"} =
        #        [{Psi_3D_initial_labframe_pos1 if Psi_3D_initial_labframe_pos1 is None else arr2str(Psi_3D_initial_labframe_pos1[0])},
        #         {Psi_3D_initial_labframe_pos1 if Psi_3D_initial_labframe_pos1 is None else arr2str(Psi_3D_initial_labframe_pos1[1])},
        #         {Psi_3D_initial_labframe_pos1 if Psi_3D_initial_labframe_pos1 is None else arr2str(Psi_3D_initial_labframe_pos1[2])}]
        #
        #   - theta_m (in radians) = {theta_m_pos1}
        #   - theta_m (in degrees) = {np.rad2deg(theta_m_pos1)}
        #
        #   - H[0] = {H_Cardanos_pos1[0]}
        #   - e_hat[0] = {arr2str(e_hats_pos1[:, 0])}
        #
        #   - H[1] = {H_Cardanos_pos1[1]}
        #   - e_hat[1] = {arr2str(e_hats_pos1[:, 1])}
        #
        #   - H[2] = {H_Cardanos_pos1[2]}
        #   - e_hat[2] = {arr2str(e_hats_pos1[:, 2])}
        #
        #   - mode_index = {mode_index_pos1}
        #   - selected H = {H_Cardano_pos1}
        #   - selected e_hat = {arr2str(e_hat_pos1)}
        #
        ##################################################
        """)

                solns.append(
                    {
                        "K_initial": K_initial_pos1,
                        "Psi_3D_initial": Psi_3D_initial_labframe_pos1,
                        "H_Cardano_initial": H_Cardano_pos1,
                        "e_hat_initial": e_hat_pos1,
                        "mode_flag": hamiltonian_pos1.mode_flag,  # sanity check: this should be `1`
                        "mode_index": mode_index_pos1,
                    }
                )

            if mode_flag_launch in [-1, "O", "X"] and hamiltonian_neg1 is not None:
                log.debug(
                    f"`mode_flag_launch` is {mode_flag_launch}. Applying boundary conditions for Hamiltonian with `mode_flag` = -1"
                )
                K_initial_neg1, Psi_3D_initial_labframe_neg1 = (
                    apply_boundary_conditions(
                        ray_tracing_flag=ray_tracing_flag,
                        boundary_flag=boundary_flag,
                        q_vacuum_entry_cartesian=q_initial_cartesian,
                        K_vacuum_entry_cartesian=K_launch_cartesian,
                        Psi_3D_vacuum_entry_labframe=Psi_3D_entry_labframe,
                        field=field,
                        hamiltonian=hamiltonian_neg1,
                    )
                )

                K_magnitude_initial_neg1 = find_K_magnitude(cart, *K_initial_neg1, q_initial[0])  # type: ignore
                K_hat_initial_neg1 = K_initial_neg1 / (
                    K_magnitude_initial_neg1
                    * np.array([1, 1 if cart else q_initial[0], 1])
                )
                theta_m_neg1 = np.arcsin(np.dot(b_hat, K_hat_initial_neg1))

                H_Cardanos_neg1, e_hats_neg1 = find_H_Cardano_eigh(
                    launch_angular_freq=launch_angular_freq,
                    K_magnitude=K_magnitude_initial_neg1,
                    epsilon_para=epsilon_para,
                    epsilon_perp=epsilon_perp,
                    epsilon_g=epsilon_g,
                    theta_m=theta_m_neg1,
                )

                mode_index_neg1, H_Cardano_neg1, e_hat_neg1 = find_mode_index_and_ehat(
                    mode_flag=mode_flag_launch,
                    H_Cardanos=H_Cardanos_neg1,
                    ehats=e_hats_neg1,
                    tol_H=tol_H,
                    tol_O_mode_polarisation=tol_O_mode_polarisation,
                )

                log.debug(f"""
        ##################################################
        #
        # Calculated values for Hamiltonian with `-1`:
        #   - K_initial_{"cartesian" if cart else "cylindrical"} = {arr2str(K_initial_neg1)}
        #
        #   - Psi_3D_initial_labframe_{"cartesian" if cart else "cylindrical"} =
        #        [{Psi_3D_initial_labframe_neg1 if Psi_3D_initial_labframe_neg1 is None else arr2str(Psi_3D_initial_labframe_neg1[0])},
        #         {Psi_3D_initial_labframe_neg1 if Psi_3D_initial_labframe_neg1 is None else arr2str(Psi_3D_initial_labframe_neg1[1])},
        #         {Psi_3D_initial_labframe_neg1 if Psi_3D_initial_labframe_neg1 is None else arr2str(Psi_3D_initial_labframe_neg1[2])}]
        #
        #   - theta_m (in radians) = {theta_m_neg1}
        #   - theta_m (in degrees) = {np.rad2deg(theta_m_neg1)}
        #
        #   - H[0] = {H_Cardanos_neg1[0]}
        #   - e_hat[0] = {arr2str(e_hats_neg1[:, 0])}
        #
        #   - H[1] = {H_Cardanos_neg1[1]}
        #   - e_hat[1] = {arr2str(e_hats_neg1[:, 1])}
        #
        #   - H[2] = {H_Cardanos_neg1[2]}
        #   - e_hat[2] = {arr2str(e_hats_neg1[:, 2])}
        #
        #   - mode_index = {mode_index_neg1}
        #   - selected H = {H_Cardano_neg1}
        #   - selected e_hat = {arr2str(e_hat_neg1)}
        #
        ##################################################
        """)

                solns.append(
                    {
                        "K_initial": K_initial_neg1,
                        "Psi_3D_initial": Psi_3D_initial_labframe_neg1,
                        "H_Cardano_initial": H_Cardano_neg1,
                        "e_hat_initial": e_hat_neg1,
                        "mode_flag": hamiltonian_neg1.mode_flag,  # sanity check: this should be `-1`
                        "mode_index": mode_index_neg1,
                    }
                )

            # If the user has specified `mode_flag` = "O" or "X", then we
            # compare the polarisations to determine which is the correct
            # mode, and also check with `tol_O_mode_polarisation` to
            # double confirm. If the user has specified `mode_flag` = `1`
            # or `-1`, then just check with `tol_O_mode_polarisation`.

            if mode_flag_launch in ["O", "X"]:
                soln_idx, _, _ = find_mode_index_and_ehat(
                    mode_flag=mode_flag_launch,
                    H_Cardanos=np.array([H_Cardano_pos1, H_Cardano_neg1]),  # type: ignore
                    ehats=np.array([e_hat_pos1, e_hat_neg1]).T,  # type: ignore
                    tol_H=tol_H,
                    tol_O_mode_polarisation=tol_O_mode_polarisation,
                )

                O_mode = solns[soln_idx if mode_flag_launch == "O" else not soln_idx]
                X_mode = solns[not soln_idx if mode_flag_launch == "O" else soln_idx]

            else:  # elif mode_flag_launch in [1, -1]:
                soln = solns[0]  # since there is only one solution
                soln_is_O_mode = (
                    abs(soln["e_hat_initial"][-1]) > 1 - tol_O_mode_polarisation
                )

                O_mode = soln if soln_is_O_mode else {}
                X_mode = {} if soln_is_O_mode else soln

            if mode_flag_launch in [1, -1, "O"] and O_mode != {}:
                sel, toreturn = "O", O_mode
            elif mode_flag_launch in [1, -1, "X"] and X_mode != {}:
                sel, toreturn = "X", X_mode
            else:
                raise RuntimeError(
                    f"Unable to determine correct mode solution for `mode_flag` = {mode_flag_launch}"
                )

            log.debug(f"""
        ##################################################
        #
        # Calculated plasma entry parameters for:
        #   - O-mode {"(selected)" if sel == "O" else ""}:
        #        - K_cartesian = {O_mode.get("K_initial")}
        #
        #        - Psi_3D_cartesian =
        #              {O_mode.get("Psi_3D_initial")}
        #
        #        - H_Cardano = {O_mode.get("H_Cardano")}
        #
        #        - e_hat = {O_mode.get("e_hat")}
        #
        #        - mode_flag = {O_mode.get("mode_flag")}
        #
        #        - mode_index = {O_mode.get("mode_index")}
        #
        #   - X-mode {"(selected)" if sel == "X" else ""}:
        #        - K_cartesian = {X_mode.get("K_initial")}
        #
        #        - Psi_3D_cartesian =
        #              {X_mode.get("Psi_3D_initial")}
        #
        #        - H_Cardano = {X_mode.get("H_Cardano")}
        #
        #        - e_hat = {X_mode.get("e_hat")}
        #
        #        - mode_flag = {X_mode.get("mode_flag")}
        #
        #        - mode_index = {X_mode.get("mode_index")}
        #
        ##################################################
        """)

            (
                K_initial,
                Psi_3D_initial_labframe,
                H_Cardano_initial,
                e_hat_initial,
                mode_flag_initial,
                mode_index,
            ) = toreturn.values()

    return (
        K_launch,
        K_initial,
        Psi_3D_launch_labframe,
        Psi_3D_entry_labframe,
        Psi_3D_initial_labframe,
        distance_from_launch_to_entry,
        e_hat_initial,
        mode_flag_initial,
        mode_index,
    )
