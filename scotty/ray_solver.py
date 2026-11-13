# Copyright 2023, Valerian Hall-Chen and Scotty contributors
# SPDX-License-Identifier: GPL-3.0

import logging
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeSolution
from scipy.optimize import brentq, minimize_scalar
from scotty.checks import VALID_RESONANCE_FLAGS, VALID_SOLVER_STATUS
from scotty.fun_general import (
    find_K_magnitude,
    find_normalised_gyro_ang_freq,
    find_normalised_plasma_ang_freq,
)
from scotty.geometry import MagneticField_Cartesian, MagneticField_Cylindrical
from scotty.hamiltonian import FIRST_ORDER_KEYS, Hamiltonian
from scotty.logger import timer
from scotty.typing import FloatArray

# from sklearn.utils import Bunch
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Protocol,
    Union,
    Tuple,
    cast,
)

log = logging.getLogger(__name__)


class _Event(Protocol):
    """Protocol describing a `scipy.integrate.solve_ivp` event callback"""

    terminal: bool = False
    direction: float = 0.0

    def __call__(self, *args):
        pass


def _event(terminal: bool, direction: float):
    """Decorator to add the attributes required for
    `scipy.integrate.solve_ivp while keeping the `mypy` type checker
    happy
    """

    def decorator_event(func: Any) -> _Event:
        func.terminal = terminal  # Terminate solver when beam leaves plasma
        func.direction = (
            direction  # +ve value when function result goes from -ve to +ve
        )
        return func

    return decorator_event


# How close to a resonance layer, in omega_ce / omega (or omega_UH / omega),
# counts as starting on it
_ON_LAYER_TOL = 1e-10
# How far from the first harmonic (the fundamental), in omega_ce / omega, the output
# stops. Closer, K is not resolved: it can jump off the dispersion relation in one
# output step
_FIRST_HARMONIC_MARGIN = 1e-3
# How far from the upper hybrid resonance, in omega_UH / omega, the X-mode stops.
# Its refractive index goes to infinity at the layer, so closer, the output points
# are off the dispersion relation, and with K perpendicular to B the ray never
# reaches the layer (it approaches it asymptotically) and the solver stalls
_UPPER_HYBRID_MARGIN = 1e-2

# The resonance events, in `make_solver_events`, and the layers they are at
RESONANCE_EVENTS = {
    "first_harmonic": "the first harmonic of the electron cyclotron resonance (the fundamental)",
    "second_harmonic": "the second harmonic of the electron cyclotron resonance",
    "upper_hybrid": "the upper hybrid resonance",
}


def _tau_short_of_first_harmonic(
    tau_steps: FloatArray, tau_end: float, distance_past_margin: Callable
) -> float:
    """Where the output stops before the first harmonic: the last point before
    `tau_end` where `distance_past_margin` is zero. The search starts from the last
    solver step that is still outside the margin, so a ray that came within the
    margin earlier and turned away isn't cut there. `tau_end` is returned unchanged
    if it is outside the margin, or if the ray is within it at every step"""
    if distance_past_margin(tau_end) >= 0:
        return tau_end
    for tau in reversed(tau_steps):
        if tau < tau_end and distance_past_margin(tau) > 0:
            return brentq(distance_past_margin, tau, tau_end)
    return tau_end


def _find_UH_ratio(q, launch_angular_frequency, field, hamiltonian) -> float:
    """The upper hybrid frequency over the beam frequency at `q`, with the
    relativistic mass correction if there is a temperature profile"""
    q0, q1, q2 = q
    polflux = field.polflux(q0, q1, q2)
    temperature = hamiltonian.temperature(polflux) if hamiltonian.temperature else None
    gyro_ratio = _find_gyro_ratio(q, launch_angular_frequency, field, hamiltonian)
    plasma_ratio = find_normalised_plasma_ang_freq(
        launch_angular_frequency, hamiltonian.density(polflux), temperature
    )
    return np.sqrt(gyro_ratio**2 + plasma_ratio**2)


def _find_gyro_ratio(q, launch_angular_frequency, field, hamiltonian) -> float:
    """The electron cyclotron frequency over the beam frequency at `q`, with the
    relativistic mass correction if there is a temperature profile"""
    q0, q1, q2 = q
    temperature = (
        hamiltonian.temperature(field.polflux(q0, q1, q2))
        if hamiltonian.temperature
        else None
    )
    return find_normalised_gyro_ang_freq(
        launch_angular_frequency, field.magnitude(q0, q1, q2), temperature
    )


def make_solver_events(
    poloidal_flux_enter: float,
    launch_angular_frequency: float,
    field: Union[MagneticField_Cartesian, MagneticField_Cylindrical],
    *,
    X_mode: bool,
    first_harmonic_flag: VALID_RESONANCE_FLAGS = "stop",
    second_harmonic_flag: VALID_RESONANCE_FLAGS = "stop",
    upper_hybrid_flag: VALID_RESONANCE_FLAGS = "stop",
    upper_hybrid_side: Optional[float] = None,
) -> Dict[str, Callable]:
    r"""The ray's solver events. `X_mode` says whether the ray is the X-mode,
    for the upper-hybrid resonance; which `mode_flag` that is depends on the
    plasma and K (see `scotty.launch.find_O_mode_flag`), so it is given
    explicitly rather than assumed from the Hamiltonian's `mode_flag`.

    There is one terminal event per resonance layer (see `RESONANCE_EVENTS`):
    ``"first_harmonic"`` (the fundamental) always, ``"second_harmonic"``
    unless `second_harmonic_flag` is ``"pass"``, and ``"upper_hybrid"`` for
    the X-mode. `first_harmonic_flag` and `upper_hybrid_flag` can't be
    ``"pass"``; with ``"stop"`` and ``"raise"`` the events are the same, and
    `ray_tracing` acts on them.

    The upper hybrid event fires `_UPPER_HYBRID_MARGIN` short of the layer, in
    omega_UH / omega, on the side the ray starts on: `upper_hybrid_side` is +1
    if it starts where omega_UH > omega, and -1 where omega_UH < omega (needed
    for the X-mode). The event is then monotonic in omega_UH / omega, so a
    solver step that crosses the whole band around the layer is seen too."""
    if first_harmonic_flag == "pass":
        raise ValueError(
            "`first_harmonic_flag` can't be 'pass': the ray can't be traced past "
            "the first harmonic (the fundamental)"
        )
    if upper_hybrid_flag == "pass":
        raise ValueError(
            "`upper_hybrid_flag` can't be 'pass': the X-mode's refractive index goes "
            "to infinity at the upper hybrid resonance, so the ray can't be traced "
            "past it"
        )
    if X_mode and upper_hybrid_side not in (1, -1):
        raise ValueError(
            "`upper_hybrid_side` must be 1 or -1 for the X-mode, the side of the "
            f"upper hybrid resonance the ray starts on, but got {upper_hybrid_side!r}"
        )

    cart = True if isinstance(field, MagneticField_Cartesian) else False

    # Triggers when the beam leaves the same poloidal flux value it
    # entered the plasma at
    #   -ve -> +ve when leaving the plasma
    @_event(terminal=True, direction=1.0)
    def event_leave_plasma(tau, ray_parameters, hamiltonian):
        q0, q1, q2, _, _, _ = ray_parameters
        polflux = field.polflux(q0, q1, q2)
        return polflux - poloidal_flux_enter

    # Triggers when the beam leaves the LCFS
    #   -ve -> +ve when leaving LCFS
    @_event(terminal=False, direction=1.0)
    def event_leave_LCFS(tau, ray_parameters, hamiltonian):
        q0, q1, q2, _, _, _ = ray_parameters
        polflux = field.polflux(q0, q1, q2)
        polflux_LCFS = 1.0
        return polflux - polflux_LCFS

    # Capture the bounding box of the magnetic field to check if
    # the beam has left the plasma
    def _minmax(arr) -> Tuple[float, float]:
        return min(arr), max(arr)

    q0_min, q0_max = _minmax(field.X_coord if cart else field.R_coord)  # type: ignore
    q1_min, q1_max = _minmax(field.Y_coord if cart else [-np.inf, np.inf])  # type: ignore
    q2_min, q2_max = _minmax(field.Z_coord)

    # Triggers when the beam leaves the magnetic field region
    #   +ve -> -ve when leaving the simulation region
    @_event(terminal=True, direction=-1.0)
    def event_leave_simulation(tau, ray_parameters, hamiltonian):
        q0, q1, q2, _, _, _ = ray_parameters
        is_inside = (
            (q0_min < q0 < q0_max) and (q1_min < q1 < q1_max) and (q2_min < q2 < q2_max)
        )
        return +1 if is_inside else -1

    # Trigger when the beam frequency crosses a resonance (beam-tracing/Scotty#151):
    # the first harmonic (the fundamental) or the second harmonic of the electron
    # cyclotron frequency, or, for the X-mode, the upper hybrid frequency. With a
    # temperature profile, the cyclotron and plasma frequencies include the
    # relativistic mass correction. Each layer has its own event, so that each
    # can be stopped at, raised at or passed (beam-tracing/Scotty#169), and two
    # layers crossed in one solver step are both seen
    @_event(terminal=True, direction=0.0)
    def event_first_harmonic(tau, ray_parameters, hamiltonian: Hamiltonian):
        gyro_ratio = _find_gyro_ratio(
            ray_parameters[:3], launch_angular_frequency, field, hamiltonian
        )
        return gyro_ratio - 1

    @_event(terminal=True, direction=0.0)
    def event_second_harmonic(tau, ray_parameters, hamiltonian: Hamiltonian):
        gyro_ratio = _find_gyro_ratio(
            ray_parameters[:3], launch_angular_frequency, field, hamiltonian
        )
        return gyro_ratio - 0.5

    # The upper hybrid resonance is reached by the X-mode only, whichever
    # `mode_flag` (or "X") it was launched as. Positive on the side the ray starts
    # on, until it comes within `_UPPER_HYBRID_MARGIN` of the layer
    @_event(terminal=True, direction=0.0)
    def event_upper_hybrid(tau, ray_parameters, hamiltonian: Hamiltonian):
        UH_ratio = _find_UH_ratio(
            ray_parameters[:3], launch_angular_frequency, field, hamiltonian
        )
        return upper_hybrid_side * (UH_ratio - 1) - _UPPER_HYBRID_MARGIN

    # Triggers when the cut-off location is reached (i.e. when the
    # wavenumber K is minimised)
    @_event(terminal=False, direction=1.0)
    def event_reach_K_min(tau, ray_parameters, hamiltonian: Hamiltonian):
        q0, q1, q2, K0, K1, K2 = ray_parameters
        q = np.array([q0, q1, q2])
        K = np.array([K0, K1, K2])
        K_magnitude = find_K_magnitude(cart, K0, K1, K2, q0)
        dH = list(hamiltonian.derivatives(q, K).values())
        spatial_dH = np.array([deriv for deriv in dH[:3]])

        dK_d_tau = -1 * np.sum(spatial_dH * q) / K_magnitude

        # This event does not work properly when the ray reaches resonance
        # The following if statement introduces a trick to avoid this problem
        # When ray reaches resonance, dK_dtau goes to infinity. Just set to 0
        # to tell scotty that we are heading to infinity if we get a NaN.
        #
        # Note to developers:
        # Cannot set condition where K_magnitude > some value. K_magnitude
        # does not actually blow up. Only dK_dtau blows up
        # Matthew Liang, Peter Hill, and Valerian Hall-Chen (01 August 2024)
        if np.isnan(dK_d_tau) == True:
            dK_d_tau = 0

        return dK_d_tau

    return {
        "leave_plasma": event_leave_plasma,
        "leave_LCFS": event_leave_LCFS,
        "leave_simulation": event_leave_simulation,
        "first_harmonic": event_first_harmonic,
        **(
            {"second_harmonic": event_second_harmonic}
            if second_harmonic_flag != "pass"
            else {}
        ),
        **({"upper_hybrid": event_upper_hybrid} if X_mode else {}),
        "reach_K_min": event_reach_K_min,
    }


def _first_resonance(tau_events: Dict[str, FloatArray]) -> Optional[str]:
    """The resonance event that fired first, or None if none did"""
    fired = [
        name
        for name in RESONANCE_EVENTS
        if name in tau_events and len(tau_events[name]) != 0
    ]
    return min(fired, key=lambda name: tau_events[name][0]) if fired else None


def handle_terminating_event(
    tau_events: Dict[str, FloatArray],
    ray_parameters_events: FloatArray,
) -> float:
    """Handle events detected by `scipy.integrate.solve_ivp`. This
    only handles events due to the ray leaving the plasma or
    simulation:

    TODO # TO REMOVE does it actually *only* handle ray leaving plasma or sim?
    doesnt seem like it

    - ``"leave_plasma"``: the ray has left the plasma
    - ``"leave_LCFS"``: the ray has left the last-closed flux
      surface. For most simulations, this is likely identical to
      leaving the plasma
    - ``"leave_simulation"``: the ray has left the simulated area
      (essentially the bounding box of the plasma)
    - ``"first_harmonic"``, ``"second_harmonic"``, ``"upper_hybrid"``: the ray
      has reached a resonance layer (the first one reached, if several fired)

    Parameters
    ----------
    tau_events : Dict[str, FloatArray]
        A mapping between event names and the solver ``t_events``
    ray_parameters_events :

    Returns
    -------
    FloatArray
        The value of ``tau`` when the detected event first occurred
    """

    def detected(event):
        """True if ``event`` was detected"""
        return len(tau_events[event]) != 0

    # Event names here must match those in the `solver_ray_events`
    # dict defined outside this function
    # if detected("leave_plasma") and not detected("leave_LCFS"):
    #     log.info(f"Ray has left plasma")
    #     return tau_events["leave_plasma"][0]

    resonance = _first_resonance(tau_events)
    if resonance is not None:
        log.info(f"Ray has reached {RESONANCE_EVENTS[resonance]}")
        return tau_events[resonance][0]

    if detected("leave_plasma") and detected("leave_LCFS"):
        log.info(f"Ray has left plasma")
        return tau_events["leave_plasma"][0]

    if not detected("leave_plasma") and detected("leave_LCFS"):
        log.info(f"Ray has left LCFS")
        return tau_events["leave_LCFS"][0]

    if detected("leave_plasma") and not detected("leave_LCFS"):
        log.info(f"Ray has left plasma")
        return tau_events["leave_plasma"][0]

    # if detected("leave_plasma") and detected("leave_LCFS"):
    #     K0_at_LCFS = ray_parameters_events[0][2]
    #     if K0_at_LCFS < 0:
    #         log.info(f"Ray has gone through the plasma and LCFS, and exited from the outboard side. Terminating the propagation at the LCFS")
    #         return tau_events["leave_LCFS"][0]

    #     log.info(f"Ray deflection is sufficiently large, so the ray has gone through the plasma and LCFS, but did not exit from the outboard side. Terminating at entry poloidal flux")
    #     return tau_events["leave_plasma"][0]

    log.warning(f"""Ray has left the simulation region without leaving the LCFS

    If one ends up here, things aren't going well. I can think of two possible reasons:
    1) The launch conditions are really weird (hasn't happened yet, in my experience)
    2) The max_step setting of the solver is too large, such that the ray leaves the
        LCFS and enters a region where `poloidal_flux < 1` in a single step. The solver
        thus doesn't log the event when it really should
    """)
    return tau_events["leave_simulation"][0]


def d_ray_parameters_d_tau(
    tau: FloatArray, ray_parameters: FloatArray, hamiltonian: Hamiltonian
) -> FloatArray:
    # tau is unused here, but solve_ivp is strict about the function signatures and requires something like
    # f(t, y, y0), so we put the tau here so that it stops complaining
    dH = hamiltonian.derivatives(ray_parameters[:3], ray_parameters[-3:])
    q_keys, K_keys = FIRST_ORDER_KEYS[
        (
            "cartesian"
            if isinstance(hamiltonian.field, MagneticField_Cartesian)
            else "cylindrical"
        )
    ]
    # dq/dtau = dH/dK, dK/dtau = -dH/dq, picked by name so that the order of
    # the derivative dictionary cannot change the equations of motion
    return np.array([dH[key] for key in K_keys] + [-dH[key] for key in q_keys])


def ray_tracing(
    q_initial: FloatArray,
    K_initial: FloatArray,
    poloidal_flux_enter: float,
    hamiltonian: Hamiltonian,
    ray_tracing_flag: bool,
    rtol: float,
    atol: float,
    len_tau: int,
    tau_max: float = 1e5,
    first_harmonic_flag: VALID_RESONANCE_FLAGS = "stop",
    second_harmonic_flag: VALID_RESONANCE_FLAGS = "stop",
    upper_hybrid_flag: VALID_RESONANCE_FLAGS = "stop",
    *,
    X_mode: bool,
) -> Tuple[VALID_SOLVER_STATUS, int, float, FloatArray, Union[float, FloatArray]]:
    """Propagates a ray, given an initial position `q` and wavevector `K`, using
    `scipy.integrate.solve_ivp` until any of these terminating conditions are
    encountered:
        (i) ray leaves the plasma (at the same `poloidal_flux_enter`);
        (ii) ray leaves the simulation region (i.e. exits the bounds of the magnetic profile); or
        (iii) ray reaches a resonance layer: the first harmonic of the electron cyclotron
              frequency (the fundamental), its second harmonic, or, for the X-mode, the
              upper hybrid frequency.

    `first_harmonic_flag`, `second_harmonic_flag` and `upper_hybrid_flag` (each "stop",
    "raise" or "pass", see `VALID_RESONANCE_FLAGS`) decide what the ray does at each
    layer: "stop" ends the ray there and returns the output, "raise" raises a
    `RuntimeError`, and "pass" carries on through the layer. Stopping at the first
    harmonic ends the output 1e-3 short of it, in omega_ce / omega. The first harmonic
    can't be passed, as the cold-plasma Hamiltonian is singular there.
    `upper_hybrid_flag` applies to the X-mode only (`X_mode`), and is ignored for the
    O-mode. At the upper hybrid layer the X-mode's refractive index goes to infinity,
    so the ray stops (or raises) 1e-2 short of it, in omega_UH / omega, and it can't
    be passed. If several layers are reached in one solver step, the first one
    reached decides.

    Starting on the first harmonic raises whatever the flag, as the ray can't be
    integrated from there, and so does starting on the second harmonic unless its
    flag is "pass", or (X-mode) within 1e-2 of the upper hybrid layer: the ray would
    stop at once.

    In addition to the conditions above, the solver will also return the instances when
    any of the following non-terminating conditions occur:
        (a) ray leaves the LCFS (i.e. when the ray reaches `poloidal_flux` == 1.0); or
        (b) dK/dtau = 0 (i.e. a cut-off is reached).

    If `ray_tracing_flag == True`, then this returns a 2-tuple where the first item is an
    array of equally-spaced `tau` points, and the second is an array of shape `(6, len(tau_points))`
    corresponding to `q0`, `q1`, `q2`, `K0`, `K1`, `K2` in the same order as the first.

    If `ray_tracing_flag == False`, it returns a 2-tuple with the same first item, but the second
    item is the `tau` value when the ray encounters a terminating condition.
    """

    log.trace(f"Packing ray parameters: q = {q_initial}, K = {K_initial}")
    ray_parameters_initial = [*q_initial, *K_initial]

    # In a uniform field, the Hamiltonian's 1 - Y^2 is then zero or rounding along
    # the whole ray, and the solver never stops retrying its steps
    gyro_ratio = _find_gyro_ratio(
        q_initial, hamiltonian.angular_frequency, hamiltonian.field, hamiltonian
    )
    if abs(gyro_ratio - 1) < _ON_LAYER_TOL:
        raise RuntimeError(
            f"The ray starts on {RESONANCE_EVENTS['first_harmonic']} "
            f"(omega_ce / omega = {gyro_ratio}) at q = {q_initial}, where the "
            "cold-plasma Hamiltonian is singular"
        )
    # The resonance event would be zero at the first point, so the ray would stop
    # there, and the beam solver would have no output points
    if second_harmonic_flag != "pass" and abs(gyro_ratio - 0.5) < _ON_LAYER_TOL:
        raise RuntimeError(
            f"The ray starts on {RESONANCE_EVENTS['second_harmonic']} "
            f"(omega_ce / omega = {gyro_ratio}) at q = {q_initial}, where "
            f'`second_harmonic_flag="{second_harmonic_flag}"` stops it at once. '
            '`second_harmonic_flag="pass"` passes the second harmonic'
        )
    upper_hybrid_side = None
    if X_mode:
        UH_ratio = _find_UH_ratio(
            q_initial, hamiltonian.angular_frequency, hamiltonian.field, hamiltonian
        )
        if abs(UH_ratio - 1) < _UPPER_HYBRID_MARGIN:
            raise RuntimeError(
                f"The ray starts within {_UPPER_HYBRID_MARGIN} of "
                f"{RESONANCE_EVENTS['upper_hybrid']}, in omega_UH / omega "
                f"(omega_UH / omega = {UH_ratio}) at q = {q_initial}, where the "
                "X-mode stops at once, whatever `upper_hybrid_flag`"
            )
        upper_hybrid_side = 1.0 if UH_ratio > 1 else -1.0

    # Creating arguments and events to be passed to solver
    solver_ray_events = make_solver_events(
        poloidal_flux_enter,
        hamiltonian.angular_frequency,
        hamiltonian.field,
        X_mode=X_mode,
        first_harmonic_flag=first_harmonic_flag,
        second_harmonic_flag=second_harmonic_flag,
        upper_hybrid_flag=upper_hybrid_flag,
        upper_hybrid_side=upper_hybrid_side,
    )
    solver_arguments = (hamiltonian,)

    # Propagating q, K by solving the ray-tracing equatoins
    log.info(f"Starting the ray solver")

    solver_ray_output, duration_ray_tracing = timer(solve_ivp)(
        fun=d_ray_parameters_d_tau,
        t_span=[0, tau_max],
        y0=ray_parameters_initial,
        method="RK45",
        t_eval=None,
        dense_output=True,
        events=solver_ray_events.values(),
        vectorized=False,
        args=solver_arguments,
        rtol=rtol,
        atol=atol,
        max_step=500,
    )

    log.info(f"""\n
        Ray solver status: {solver_ray_output.status}
        Ray solver took {duration_ray_tracing} s
        Number of ray evolution evaluations: {solver_ray_output.nfev}
        Time per ray evolution evaluation: {duration_ray_tracing / solver_ray_output.nfev}
    """)

    if solver_ray_output.status == 0:
        raise RuntimeError(
            "Ray has not left plasma/simulation region; increase `tau_max` or choose different initial conditions"
        )
    elif solver_ray_output.status == -1:
        raise RuntimeError(
            "Integration step failed. Check that (i) ray does not leave interpolation region; and (ii) density interpolation is not negative"
        )

    # tau_events is a list with the same order as the values of
    # solver_ray_events, so we can use the names from that dict
    # instead of raw indices

    soln_interp = cast(OdeSolution, solver_ray_output.sol)
    tau_events = dict(zip(solver_ray_events.keys(), solver_ray_output.t_events))
    ray_parameters_events = dict(
        zip(solver_ray_events.keys(), solver_ray_output.y_events)
    )

    # The resonance layer that stopped the ray, if any: the first one reached
    resonance = _first_resonance(tau_events)
    flags = {
        "first_harmonic": first_harmonic_flag,
        "second_harmonic": second_harmonic_flag,
        "upper_hybrid": upper_hybrid_flag,
    }
    if resonance is not None and flags[resonance] == "raise":
        where = (
            f"The ray reached {RESONANCE_EVENTS[resonance]} at "
            f"tau = {tau_events[resonance][0]}, "
            f"q = {ray_parameters_events[resonance][0][:3]}"
        )
        # Past the first harmonic, the root that `mode_flag` selects swaps, so the
        # ray would land on the wrong surface or jump to the other mode (#169)
        if resonance == "first_harmonic":
            raise RuntimeError(
                f"{where}, where the cold-plasma Hamiltonian is singular, so it can't "
                'be traced past it. `first_harmonic_flag="stop"` returns the ray up '
                "to 1e-3 short of the layer, in omega_ce / omega"
            )
        if resonance == "upper_hybrid":
            raise RuntimeError(
                f"The ray came within {_UPPER_HYBRID_MARGIN} of "
                f"{RESONANCE_EVENTS[resonance]}, in omega_UH / omega, at "
                f"tau = {tau_events[resonance][0]}, "
                f"q = {ray_parameters_events[resonance][0][:3]}, and "
                '`upper_hybrid_flag="raise"`. `upper_hybrid_flag="stop"` returns the '
                "output up to there. The X-mode can't be traced through the layer"
            )
        raise RuntimeError(
            f'{where}, and `{resonance}_flag="raise"`. `{resonance}_flag="stop"` '
            f'returns the output up to the layer, and `{resonance}_flag="pass"` '
            "carries on through it"
        )

    tau_terminating_event = handle_terminating_event(
        tau_events=tau_events, ray_parameters_events=ray_parameters_events["leave_LCFS"]
    )

    # Stop short of the first harmonic, as the Hamiltonian is singular there
    if resonance == "first_harmonic":

        def distance_past_margin(tau):
            gyro_ratio = _find_gyro_ratio(
                soln_interp(tau)[:3],
                hamiltonian.angular_frequency,
                hamiltonian.field,
                hamiltonian,
            )
            return abs(gyro_ratio - 1) - _FIRST_HARMONIC_MARGIN

        tau_terminating_event = _tau_short_of_first_harmonic(
            solver_ray_output.t, tau_terminating_event, distance_past_margin
        )

    tau_arr_resampled = np.linspace(
        start=0, stop=tau_terminating_event, num=len_tau - 1, endpoint=False
    )

    # Find the cut-off (i.e. `tau` where K_magnitude is minimised) if no resonance is reached
    if resonance is None:

        def K_magnitude(tau):
            q0, _, _, K0, K1, K2 = soln_interp(tau)
            return (
                np.sqrt(K0**2 + K1**2 + K2**2)
                if isinstance(hamiltonian.field, MagneticField_Cartesian)
                else np.sqrt(K0**2 + (K1 / q0) ** 2 + K2**2)
            )

        tau_cutoff = minimize_scalar(
            fun=K_magnitude, bounds=[0, tau_arr_resampled[-1]], options={"xatol": atol}
        )
        tau_arr_resampled = np.sort(
            np.append(
                np.linspace(
                    start=0, stop=tau_terminating_event, num=len_tau - 1, endpoint=False
                ),
                tau_cutoff.x,
            )
        )

    else:
        tau_arr_resampled = np.linspace(
            start=0, stop=tau_terminating_event, num=len_tau - 1, endpoint=False
        )

    # For ray-tracing runs, return `tau` and `result` (containing `q`, `K`)
    # Otherwise return `tau_terminating_event` and `tau_arr_resampled` (since the beam solver calculates everything again)
    if ray_tracing_flag:
        return (
            solver_ray_output.status,
            solver_ray_output.nfev,
            duration_ray_tracing,
            tau_arr_resampled,
            soln_interp(tau_arr_resampled),
        )
    else:
        return (
            solver_ray_output.status,
            solver_ray_output.nfev,
            duration_ray_tracing,
            tau_arr_resampled,
            tau_terminating_event,
        )
