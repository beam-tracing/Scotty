# Copyright 2023 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

import logging
import numpy as np
from scipy import constants
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, root_scalar
from scotty.boundary_conditions import (
    apply_boundary_conditions,
    check_density_smooth_at_edge,
)
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
    find_Booker_discriminant,
    find_Booker_terms,
    find_epsilon_terms,
    find_H_Cardano_eigh,
)
from scotty.geometry import MagneticField_Cylindrical
from scotty.hamiltonian import DielectricTensor, Hamiltonian
from scotty.logger import arr2str
from scotty.typing import FloatArray, ComplexFloatArray
from typing import cast, Optional, Tuple, Union, List, Literal

log = logging.getLogger(__name__)


def max_search_distance(field: VALID_FIELDS, q_launch_cartesian: FloatArray) -> float:
    r"""Length of launch line that reaches every point of the field's domain.

    For a cylindrical field with grid R <= R_max, Z_min <= Z <= Z_max, any point
    of the domain is within :math:`R_l + R_{max}` of the launch point
    horizontally (triangle inequality through the axis, whatever the toroidal
    angles), and within :math:`\max(|Z_{min} - Z_l|, |Z_{max} - Z_l|)`
    vertically. For a Cartesian field it is the distance to the farthest
    corner of the grid's box. This holds for inboard, outboard, off-midplane
    and toroidally steered launches alike. v3 bounded the search the same
    way; v4 had used max(q_launch_cartesian), which is not a length.
    """
    X_l, Y_l, Z_l = q_launch_cartesian
    if isinstance(field, MagneticField_Cylindrical):
        R_coord = getattr(field, "R_coord", None)
        Z_coord = getattr(field, "Z_coord", None)
        if R_coord is not None and Z_coord is not None:
            horizontal = np.hypot(X_l, Y_l) + np.max(R_coord)
            vertical = max(abs(np.min(Z_coord) - Z_l), abs(np.max(Z_coord) - Z_l))
            return float(np.hypot(horizontal, vertical))
    else:
        coords = [getattr(field, f"{k}_coord", None) for k in "XYZ"]
        if all(c is not None for c in coords):
            extents = [(np.min(c), np.max(c)) for c in coords]
            corners = np.array(np.meshgrid(*extents)).reshape(3, -1).T
            distances = np.linalg.norm(corners - q_launch_cartesian, axis=1)
            return float(np.max(distances))

    distance = max(2 * float(np.linalg.norm(q_launch_cartesian)), 1.0)
    log.warning(
        f"{type(field).__name__} has no grid coordinates, so the plasma entry "
        f"search along the launch line uses {distance:.3f} m"
    )
    return distance


def find_plasma_entry_position(
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    q_launch: FloatArray,
    launch_flag: VALID_LAUNCH_FLAGS,
    field: VALID_FIELDS,
    poloidal_flux_enter: float,
    boundary_adjust: float = 1e-6,
) -> FloatArray:
    r"""Returns `q_initial` in cartesian or cylindrical.

    As in v3, the launch angles are taken in the TORBEAM convention for an
    antenna at zeta = 0: the launch line's direction, like the launch K, is
    measured from the X axis, so `q_launch` should have zeta = 0 (Y = 0)."""

    log.info(f"Finding plasma entry position")

    # If launch_flag is "plasma", we are already in the plasma
    # so take this as the entry position, after checking it is inside
    if launch_flag == "plasma":
        log.debug(
            f"`launch_flag` = {launch_flag}, so skipping plasma entry position calculations"
        )
        polflux_launch = float(field.polflux(*q_launch))
        if not polflux_launch <= poloidal_flux_enter:
            raise ValueError(
                f"`launch_flag` = 'plasma', but `launch_position` = {q_launch} is "
                f"outside the plasma (poloidal flux {polflux_launch:.6g} > "
                f"`poloidal_flux_enter` = {poloidal_flux_enter}). Use "
                "`launch_flag` = 'vacuum' to launch from outside the plasma"
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
    else:  # isinstance(field, MagneticField_Cartesian):
        cart = True
        q_launch_cartesian = q_launch
    max_dist = max_search_distance(field, np.asarray(q_launch_cartesian, dtype=float))

    # Defining some wrapper functions for ease
    def _ray_line_wrapper(tau):
        return ray_line(*q_launch_cartesian, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam)  # type: ignore

    def _poloidal_flux_difference_along_ray_line_wrapper(tau):
        return poloidal_flux_difference_along_ray_line(*q_launch_cartesian, tau, poloidal_launch_angle_deg_Torbeam, toroidal_launch_angle_deg_Torbeam, field.polflux_cartesian, poloidal_flux_enter)  # type: ignore

    # Search the launch line up to the far side of the field's domain
    # (max_search_distance), in sections of 0.01 m, with at least 100 steps

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

    # The entry is the first crossing along the line, i.e. the smallest spline
    # root. Refine it on the actual flux, bracketed by the neighbouring
    # samples, so that the result does not depend on the sampling grid

    root = np.min(spline_roots)
    i = int(np.clip(np.searchsorted(tau_arr_refined, root), 1, num_tau - 1))
    bracket = (tau_arr_refined[i - 1], tau_arr_refined[i])
    f_bracket = [_poloidal_flux_difference_along_ray_line_wrapper(t) for t in bracket]
    if np.all(np.isfinite(f_bracket)) and f_bracket[0] * f_bracket[1] < 0:
        boundary = root_scalar(
            _poloidal_flux_difference_along_ray_line_wrapper,
            bracket=bracket,
            method="brentq",
            xtol=1e-12,
        )
    else:  # the spline's root lies between two samples on the same side
        boundary = root_scalar(
            _poloidal_flux_difference_along_ray_line_wrapper,
            x0=root,
            x1=root + 1e-3,
        )
    if not boundary.converged:
        raise RuntimeError(
            f"Unable to find plasma boundary. Root finding failed with `{boundary.flag}`"
        )
    boundary_tau = boundary.root

    # Rounding can leave the root just outside the plasma. Step in by the
    # smallest amount that lands inside (doubling from 1e-12 m, at most
    # `boundary_adjust`), so the beam starts where the density is still
    # zero to rounding and the vacuum K satisfies H = 0

    step = 0.0
    q_initial_cartesian = _ray_line_wrapper(boundary_tau)
    while field.polflux_cartesian(*q_initial_cartesian) > poloidal_flux_enter:
        step = 1e-12 if step == 0.0 else 2 * step
        if step > boundary_adjust:
            raise RuntimeError(
                f"Plasma entry point still outside the plasma after stepping "
                f"`boundary_adjust` = {boundary_adjust} m along the launch line"
            )
        q_initial_cartesian = _ray_line_wrapper(boundary_tau + step)
    if step > 0:
        log.debug(
            f"Stepped {step:.1e} m into the plasma to the entry point [X,Y,Z] = {arr2str(q_initial_cartesian)}"
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
        # Point the step down the poloidal flux gradient, i.e. into the plasma,
        # whatever the sign the user gave: the step's sign alone must not decide
        # it, or an inward (negative) step would be flipped outward
        gradient = derivative(*q_initial, delta_value)
        if gradient != 0:
            new_value = -np.sign(gradient) * abs(delta_value)
            if new_value != delta_value:
                log.debug(
                    f"Switching `{delta_name}` from `{delta_value}` to `{new_value}`"
                )
            delta_value = new_value
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
        ehat = ehats[:, mode_idx]

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
        ehat = ehats[:, mode_idx]

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


def _check_dispersion_relation(
    hamiltonians: List[Hamiltonian],
    q_initial: FloatArray,
    K_initial: FloatArray,
    context: str,
    tol_H_error: float = 1e-3,
    tol_H_warning: float = 1e-5,
) -> None:
    r"""Check that `K_initial` at `q_initial` satisfies the dispersion relation
    of at least one of `hamiltonians` (|H| close to 0). Raises above
    `tol_H_error`, warns above `tol_H_warning`."""
    H_values = {
        hamiltonian.mode_flag: float(np.abs(hamiltonian(q_initial, K_initial)))
        for hamiltonian in hamiltonians
    }
    closest_mode_flag = min(H_values, key=H_values.get)
    H_min = H_values[closest_mode_flag]
    _values = ", ".join(
        f"|H| = {H:.3e} for `mode_flag` = {m}" for m, H in H_values.items()
    )

    if H_min > tol_H_error:
        raise ValueError(
            f"{context}: K does not satisfy the dispersion relation ({_values}; "
            f"closest is `mode_flag` = {closest_mode_flag}, tolerance {tol_H_error}). "
            f"Check the magnitude and direction of K"
        )
    if H_min > tol_H_warning:
        log.warning(
            f"{context}: K only approximately satisfies the dispersion relation ({_values})"
        )


# Density (10^19 m^-3) below which a mode's polarisation is taken as its
# zero-density limit, since at exactly zero density the O- and X-modes are
# degenerate (any polarisation transverse to K is allowed)
POLARISATION_PROBE_DENSITY = 1e-6


def _mode_polarisations(
    launch_angular_freq: float,
    B_magnitude: float,
    theta_m: float,
    electron_density: float,
    temperature: Optional[float] = None,
    n_probe: float = POLARISATION_PROBE_DENSITY,
) -> dict:
    r"""Polarisation of each Booker root, `mode_flag` = +1 and -1, for a K
    along `theta_m`. Returns `{mode_flag: e_hat}`, `e_hat` in the (u1, u2, b)
    basis.

    For each root, N^2 solves its Booker quadratic, and e_hat is the
    eigenvector of D whose eigenvalue vanishes at that N^2. As n_e -> 0 both
    roots tend to N = 1, where D is degenerate in the plane transverse to K.
    Their polarisations still tend to distinct limits, the eigenvectors of
    the susceptibility projected onto that plane (degenerate perturbation
    theory), which depend on |B| and theta_m but not n_e. So below `n_probe`
    the density is replaced by `n_probe`, which gives those limits.

    If either root does not propagate at this density (N^2 <= 0, e.g. past
    its cut-off), the two are compared in their zero-density limits instead.
    """
    K0 = launch_angular_freq / constants.c
    sin_theta_m_sq = np.sin(theta_m) ** 2

    def polarisations_at(n_e: float) -> Optional[dict]:
        alpha, beta, _ = find_Booker_terms(
            launch_angular_freq, B_magnitude, sin_theta_m_sq, n_e, temperature
        )
        discriminant = find_Booker_discriminant(
            launch_angular_freq, B_magnitude, sin_theta_m_sq, n_e, temperature
        )
        epsilon_terms = find_epsilon_terms(
            launch_angular_freq, B_magnitude, n_e, temperature
        )
        result = {}
        for mode_flag in (1, -1):
            # The root of H_Booker = N^2 + (beta - mode_flag sqrt(disc)) / (2 alpha)
            N_sq = -(beta - mode_flag * np.sqrt(discriminant)) / (2 * alpha)
            if not (np.isfinite(N_sq) and N_sq > 0):
                return None
            eigvals, eigvecs = find_H_Cardano_eigh(
                launch_angular_freq, np.sqrt(N_sq) * K0, *epsilon_terms, theta_m
            )
            result[mode_flag] = eigvecs[:, int(np.argmin(np.abs(eigvals)))]
        return result

    polarisations = polarisations_at(max(float(electron_density), n_probe))
    if polarisations is None:
        log.debug(
            f"A root does not propagate at n_e = {electron_density} (e19), so "
            "the O- and X-mode polarisations are compared at zero density"
        )
        polarisations = cast(dict, polarisations_at(n_probe))
    return polarisations


def _O_mode_flag(polarisations: dict, min_difference: float = 0.05) -> int:
    r"""The `mode_flag` (+1 or -1) of the O-mode: the root more polarised
    along b (for perpendicular propagation, exactly E parallel to B). Raises
    if the two are too alike to tell apart, as for K nearly along B, where
    both are circularly polarised transverse to b."""
    e_b = {mode_flag: abs(e_hat[-1]) for mode_flag, e_hat in polarisations.items()}
    if abs(e_b[1] - e_b[-1]) < min_difference:
        raise ValueError(
            f"The O- and X-mode polarisations are too alike to tell apart "
            f"(|e_b| = {e_b[1]:.3f} for `mode_flag` = 1, {e_b[-1]:.3f} for -1), as "
            "when K is nearly along B. Give `mode_flag` = 1 or -1"
        )
    return 1 if e_b[1] > e_b[-1] else -1


def _theta_m(q: FloatArray, K: FloatArray, field: VALID_FIELDS, cart: bool) -> float:
    r"""Angle between K and the plane perpendicular to B: sin(theta_m) =
    b_hat . K_hat (in cylindrical, K_zeta is the toroidal mode number)"""
    K_magnitude = find_K_magnitude(cart, *K, q[0])  # type: ignore
    K_hat = K / (K_magnitude * np.array([1, 1 if cart else q[0], 1]))
    return float(np.arcsin(np.dot(field.unitvector(*q), K_hat)))


def _mode_name(mode_flag: int, O_mode_flag: Optional[int]) -> str:
    """"O" or "X", or "O or X" if the two polarisations are too alike to label."""
    if O_mode_flag is None:
        return "O or X"
    return "O" if mode_flag == O_mode_flag else "X"


def find_O_mode_flag(
    q: FloatArray,
    K: FloatArray,
    field: VALID_FIELDS,
    launch_angular_freq: float,
    density_fit,
    temperature_fit=None,
) -> int:
    r"""The `mode_flag` (+1 or -1) whose Booker root is the O-mode for a K
    along `K` at `q` (in the field's coordinates), judged by polarisation.
    Works at zero density too, from the polarisations' zero-density limits."""
    cart = not isinstance(field, MagneticField_Cylindrical)
    polflux = field.polflux(*q)
    polarisations = _mode_polarisations(
        launch_angular_freq,
        float(field.magnitude(*q)),
        _theta_m(q, K, field, cart),
        float(density_fit(polflux)),
        float(temperature_fit(polflux)) if temperature_fit else None,
    )
    return _O_mode_flag(polarisations)


def _plasma_terms_at(
    q_initial: FloatArray,
    q_initial_cartesian: FloatArray,
    field: VALID_FIELDS,
    launch_angular_freq: float,
    density_fit,
    temperature_fit,
) -> Tuple[FloatArray, Tuple[float, float, float], float, float, Optional[float]]:
    r"""Returns `b_hat`, the dielectric terms `(epsilon_para, epsilon_perp,
    epsilon_g)` (as used by `_mode_solution`), |B|, n_e and T_e at
    `q_initial`"""
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

    epsilon_terms = find_epsilon_terms(
        launch_angular_freq=launch_angular_freq,
        B_total=B_magnitude,
        electron_density=electron_density,
        temperature=electron_temperature,
    )
    return (
        b_hat,
        epsilon_terms,
        float(B_magnitude),
        float(electron_density),
        None if electron_temperature is None else float(electron_temperature),
    )


def _mode_solution(
    hamiltonian: Hamiltonian,
    K_initial: FloatArray,
    Psi_3D_initial_labframe: Optional[ComplexFloatArray],
    q_initial: FloatArray,
    cart: bool,
    b_hat: FloatArray,
    epsilon_terms: Tuple[float, float, float],
    launch_angular_freq: float,
    mode_flag_launch: VALID_LAUNCH_MODE_FLAGS,
    tol_H: float,
    tol_O_mode_polarisation: float,
) -> dict:
    r"""For one Hamiltonian, given K and Psi where the solvers start, find the
    Cardano root (mode index) that vanishes there, and its polarisation e_hat"""
    epsilon_para, epsilon_perp, epsilon_g = epsilon_terms

    K_magnitude_initial = find_K_magnitude(cart, *K_initial, q_initial[0])  # type: ignore
    K_hat_initial = K_initial / (
        K_magnitude_initial * np.array([1, 1 if cart else q_initial[0], 1])
    )
    theta_m = np.arcsin(np.dot(b_hat, K_hat_initial))

    H_Cardanos, e_hats = find_H_Cardano_eigh(
        launch_angular_freq=launch_angular_freq,
        K_magnitude=K_magnitude_initial,
        epsilon_para=epsilon_para,
        epsilon_perp=epsilon_perp,
        epsilon_g=epsilon_g,
        theta_m=theta_m,
    )

    mode_index, H_Cardano, e_hat = find_mode_index_and_ehat(
        mode_flag=mode_flag_launch,
        H_Cardanos=H_Cardanos,
        ehats=e_hats,
        tol_H=tol_H,
        tol_O_mode_polarisation=tol_O_mode_polarisation,
    )

    log.debug(f"""
        ##################################################
        #
        # Calculated values for Hamiltonian with `{hamiltonian.mode_flag:+d}`:
        #   - K_initial_{"cartesian" if cart else "cylindrical"} = {arr2str(K_initial)}
        #
        #   - Psi_3D_initial_labframe_{"cartesian" if cart else "cylindrical"} =
        #        [{Psi_3D_initial_labframe if Psi_3D_initial_labframe is None else arr2str(Psi_3D_initial_labframe[0])},
        #         {Psi_3D_initial_labframe if Psi_3D_initial_labframe is None else arr2str(Psi_3D_initial_labframe[1])},
        #         {Psi_3D_initial_labframe if Psi_3D_initial_labframe is None else arr2str(Psi_3D_initial_labframe[2])}]
        #
        #   - theta_m (in radians) = {theta_m}
        #   - theta_m (in degrees) = {np.rad2deg(theta_m)}
        #
        #   - H[0] = {H_Cardanos[0]}
        #   - e_hat[0] = {arr2str(e_hats[:, 0])}
        #
        #   - H[1] = {H_Cardanos[1]}
        #   - e_hat[1] = {arr2str(e_hats[:, 1])}
        #
        #   - H[2] = {H_Cardanos[2]}
        #   - e_hat[2] = {arr2str(e_hats[:, 2])}
        #
        #   - mode_index = {mode_index}
        #   - selected H = {H_Cardano}
        #   - selected e_hat = {arr2str(e_hat)}
        #
        ##################################################
        """)

    return {
        "K_initial": K_initial,
        "Psi_3D_initial": Psi_3D_initial_labframe,
        "H_Cardano_initial": H_Cardano,
        "e_hat_initial": e_hat,
        "mode_flag": hamiltonian.mode_flag,
        "mode_index": mode_index,
    }


def _mode_from_K(
    hamiltonian_pos1: Optional[Hamiltonian],
    hamiltonian_neg1: Optional[Hamiltonian],
    K_initial: FloatArray,
    Psi_3D_initial_labframe: Optional[ComplexFloatArray],
    mode_flag_launch: VALID_LAUNCH_MODE_FLAGS,
    q_initial: FloatArray,
    O_mode_flag: Optional[int],
    tol_H_error: float = 1e-3,
    **mode_kwargs,
) -> dict:
    r"""The mode solution when K is given rather than solved for (a plasma
    launch, or `boundary_flag` = None): the mode is the Hamiltonian whose
    dispersion relation K satisfies, and `O_mode_flag` (from the two roots'
    polarisations, see `_O_mode_flag`) says whether it is O or X. Raises if
    `mode_flag_launch` asks for a different mode.

    Where K satisfies both (zero density, e.g. at the plasma edge), K cannot
    tell the modes apart, but their polarisations still can: "O" or "X" then
    picks `O_mode_flag` or the other root, and 1 or -1 is used as given."""
    hamiltonians = {
        mode_flag: hamiltonian
        for mode_flag, hamiltonian in ((1, hamiltonian_pos1), (-1, hamiltonian_neg1))
        if hamiltonian is not None
    }
    H_values = {
        mode_flag: float(np.abs(hamiltonian(q_initial, K_initial)))
        for mode_flag, hamiltonian in hamiltonians.items()
    }
    _values = ", ".join(
        f"|H| = {H:.3e} for `mode_flag` = {m}" for m, H in H_values.items()
    )
    satisfied = [m for m, H in H_values.items() if H <= tol_H_error]

    if len(satisfied) == 2:
        if mode_flag_launch in [1, -1]:
            mode_flag_found = mode_flag_launch
        else:
            mode_flag_found = O_mode_flag if mode_flag_launch == "O" else -O_mode_flag
    else:
        mode_flag_found = min(H_values, key=H_values.get)
        if mode_flag_launch in [1, -1] and mode_flag_launch != mode_flag_found:
            raise ValueError(
                f"`mode_flag` = {mode_flag_launch}, but K satisfies the dispersion "
                f"relation of `mode_flag` = {mode_flag_found} ({_values})"
            )

    soln = _mode_solution(
        hamiltonian=hamiltonians[mode_flag_found],
        K_initial=K_initial,
        Psi_3D_initial_labframe=Psi_3D_initial_labframe,
        q_initial=q_initial,
        mode_flag_launch=mode_flag_found,
        **mode_kwargs,
    )

    mode_name = _mode_name(mode_flag_found, O_mode_flag)
    if mode_flag_launch in ["O", "X"] and mode_flag_launch != mode_name:
        raise ValueError(
            f"`mode_flag` = {mode_flag_launch!r}, but K is the {mode_name}-mode "
            f"(`mode_flag` = {mode_flag_found}, from the polarisations of the "
            "two roots)"
        )

    log.debug(
        f"K satisfies `mode_flag` = {mode_flag_found} ({_values}), the {mode_name}-mode"
    )
    return soln


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
    poloidal_flux_zero_density: Optional[float] = None,
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

    # Launch frequency, wavenumber and profiles, shared by both Hamiltonians
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

    hamiltonians = [h for h in (hamiltonian_pos1, hamiltonian_neg1) if h is not None]

    # If `launch_flag` = "plasma", we start propagation from inside the
    # plasma, with the user's K and Psi (given in Cartesian coordinates):
    # there is no vacuum propagation and no boundary condition, and the
    # launch, entry and initial values are the same
    if launch_flag == "plasma":
        log.debug(
            f"`launch_flag` = {launch_flag}. Launching directly from inside the plasma"
        )
        if K_plasmaLaunch_cartesian is None:
            raise ValueError(
                "`plasmaLaunch_K_cartesian` must be given if `launch_flag` = 'plasma'"
            )
        K_plasma_cartesian = np.asarray(K_plasmaLaunch_cartesian, dtype=float)
        K_launch = (
            K_plasma_cartesian
            if cart
            else find_K_labframe_cart_to_cyl(K_plasma_cartesian, q_launch_cartesian)
        )

        if ray_tracing_flag:
            Psi_3D_launch_labframe = None
        else:
            if Psi_3D_plasmaLaunch_labframe_cartesian is None:
                raise ValueError(
                    "`plasmaLaunch_Psi_3D_lab_cartesian` must be given if "
                    "`launch_flag` = 'plasma'"
                )
            Psi_plasma_cartesian = np.asarray(
                Psi_3D_plasmaLaunch_labframe_cartesian, dtype=complex
            )
            Psi_3D_launch_labframe = (
                Psi_plasma_cartesian
                if cart
                else find_Psi_3D_labframe_cart_to_cyl(
                    Psi_plasma_cartesian, K_plasma_cartesian, q_launch_cartesian
                )
            )
        Psi_3D_entry_labframe = Psi_3D_launch_labframe

        _check_dispersion_relation(
            hamiltonians,
            q_initial,
            K_launch,
            "Plasma launch (`launch_flag` = 'plasma')",
        )

        def initial_K_and_Psi(hamiltonian):
            return K_launch, Psi_3D_launch_labframe

    # Otherwise, we start propagation from vacuum, so we need to
    # calculate the K and Psi at launch, propagate them to the plasma
    # boundary, and apply the boundary conditions there
    else:
        log.debug(
            f"`launch_flag` = {launch_flag}. Launching from outside the plasma boundary"
        )
        poloidal_launch_angle = np.deg2rad(poloidal_launch_angle_deg_Torbeam)
        toroidal_launch_angle = np.deg2rad(toroidal_launch_angle_deg_Torbeam)

        # Find K_launch, returned in the field's coordinates like K_initial.
        # The two only coincide with no toroidal launch angle: in cylindrical,
        # K_zeta is the toroidal mode number, q_R times the Cartesian K_Y
        # at the launch position (which is at zeta = 0)
        K_launch_cartesian = -K0 * np.array(
            [
                np.cos(poloidal_launch_angle) * np.cos(toroidal_launch_angle),
                np.cos(poloidal_launch_angle) * np.sin(toroidal_launch_angle),
                np.sin(poloidal_launch_angle),
            ]
        )
        K_launch = (
            K_launch_cartesian
            if cart
            else find_K_labframe_cart_to_cyl(K_launch_cartesian, q_launch_cartesian)
        )

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
        # so K and Psi carry straight over from vacuum: K_initial is the vacuum
        # K (in the field's coordinates at the entry point), and Psi_initial is
        # Psi_entry. Check that assumption holds for the density profile
        if boundary_flag is None:
            log.debug(
                f"`boundary_flag` = {boundary_flag}. No boundary conditions applied"
            )
            check_density_smooth_at_edge(density_fit, poloidal_flux_zero_density)

            K_entry = (
                K_launch_cartesian
                if cart
                else find_K_labframe_cart_to_cyl(
                    K_launch_cartesian, q_initial_cartesian
                )
            )
            _check_dispersion_relation(
                hamiltonians,
                q_initial,
                K_entry,
                "Vacuum launch with `boundary_flag` = None",
            )

            def initial_K_and_Psi(hamiltonian):
                return K_entry, Psi_3D_entry_labframe

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

            def initial_K_and_Psi(hamiltonian):
                return apply_boundary_conditions(
                    ray_tracing_flag=ray_tracing_flag,
                    boundary_flag=boundary_flag,
                    q_vacuum_entry_cartesian=q_initial_cartesian,
                    K_vacuum_entry_cartesian=K_launch_cartesian,
                    Psi_3D_vacuum_entry_labframe=Psi_3D_entry_labframe,
                    field=field,
                    hamiltonian=hamiltonian,
                )

    # If mode_flag is 1 or -1, then we just calculate the
    # quantities for those. If mode_flag is "O" or "X", then
    # we calculate the quantities for both 1 and -1. After
    # obtaining the quantities corresponding to H_booker = 0,
    # we calculate H_Cardano and find the polarisation vector
    # corresponding to H_booker = H_Cardano = 0 to see if it's
    # O- or X-mode. We do this, literally, by comparing the
    # e_hats with the O-mode polarisation vector; specifically,
    # since the O-mode polarisation vector ~ [0,0,1] (in the u1,
    # u2, b) basis -- see the paper), we compare the last
    # component of both vectors. If the last component of
    # the vector is not close to 1 (i.e. |v[-1] - 1| < tol),
    # then we say it's X-mode; otherwise, it's O-mode
    b_hat, epsilon_terms, B_magnitude, electron_density, electron_temperature = (
        _plasma_terms_at(
            q_initial,
            q_initial_cartesian,
            field,
            launch_angular_freq,
            density_fit,
            temperature_fit,
        )
    )

    mode_kwargs = dict(
        q_initial=q_initial,
        cart=cart,
        b_hat=b_hat,
        epsilon_terms=epsilon_terms,
        launch_angular_freq=launch_angular_freq,
        tol_H=tol_H,
        tol_O_mode_polarisation=tol_O_mode_polarisation,
    )

    # Which root (`mode_flag` +1 or -1) is the O-mode, from the two roots'
    # polarisations along K where the solvers start. This works at zero
    # density too, where K cannot tell the modes apart, from the
    # polarisations' zero-density limits. For a vacuum launch, the vacuum K
    # at the entry is used (the boundary conditions change it little)
    no_boundary_conditions = launch_flag == "plasma" or boundary_flag is None
    if no_boundary_conditions:
        K_given, Psi_given = initial_K_and_Psi(None)
        K_for_polarisation = K_given
    else:
        K_for_polarisation = (
            K_launch_cartesian
            if cart
            else find_K_labframe_cart_to_cyl(K_launch_cartesian, q_initial_cartesian)
        )
    polarisations = _mode_polarisations(
        launch_angular_freq,
        B_magnitude,
        _theta_m(q_initial, K_for_polarisation, field, cart),
        electron_density,
        electron_temperature,
    )
    try:
        O_mode_flag = _O_mode_flag(polarisations)
    except ValueError:
        # Only "O" or "X" needs the label: 1 or -1 says which root to trace
        if mode_flag_launch not in [1, -1]:
            raise
        O_mode_flag = None
    log.debug(f"The O-mode is `mode_flag` = {O_mode_flag}")

    # Without boundary conditions (a plasma launch, or `boundary_flag` =
    # None), K is given rather than solved for each mode, so the mode is the
    # one whose dispersion relation K satisfies
    if no_boundary_conditions:
        toreturn = _mode_from_K(
            hamiltonian_pos1=hamiltonian_pos1,
            hamiltonian_neg1=hamiltonian_neg1,
            K_initial=K_given,
            Psi_3D_initial_labframe=Psi_given,
            mode_flag_launch=mode_flag_launch,
            O_mode_flag=O_mode_flag,
            **mode_kwargs,
        )

    # With boundary conditions, "O" or "X" is first turned into its root, and
    # the boundary conditions are applied for that root only
    else:
        if mode_flag_launch in [1, -1]:
            mode_flag = mode_flag_launch
        else:
            mode_flag = O_mode_flag if mode_flag_launch == "O" else -O_mode_flag
        hamiltonian = hamiltonian_pos1 if mode_flag == 1 else hamiltonian_neg1
        if hamiltonian is None:
            raise ValueError(
                f"`mode_flag` = {mode_flag_launch!r} needs the Hamiltonian with "
                f"`mode_flag` = {mode_flag}, which was not given"
            )
        log.debug(
            f"`mode_flag_launch` is {mode_flag_launch}. Finding the initial K and Psi for the Hamiltonian with `mode_flag` = {mode_flag}"
        )
        K_initial_mode, Psi_3D_initial_labframe_mode = initial_K_and_Psi(hamiltonian)
        toreturn = _mode_solution(
            hamiltonian=hamiltonian,
            K_initial=K_initial_mode,
            Psi_3D_initial_labframe=Psi_3D_initial_labframe_mode,
            mode_flag_launch=mode_flag,
            **mode_kwargs,
        )

    # The traced root's own polarisation. At (near-)zero density, the eigenvector
    # of D is any transverse vector, and at low density the two roots can be too
    # close for `H` to choose between them
    toreturn["e_hat_initial"] = polarisations[toreturn["mode_flag"]]

    log.debug(
        f"Starting the solvers in the {_mode_name(toreturn['mode_flag'], O_mode_flag)}-mode, `mode_flag` = {toreturn['mode_flag']}"
    )

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
