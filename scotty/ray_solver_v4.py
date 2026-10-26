import logging
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeSolution
from scipy.optimize import minimize_scalar
from scotty.checks_v4 import VALID_SOLVER_STATUS
from scotty.fun_general_v4 import find_normalised_gyro_ang_freq, find_K_magnitude
from scotty.geometry_v4 import MagneticField_Cartesian, MagneticField_Cylindrical
from scotty.hamiltonian_v4 import Hamiltonian
from scotty.logger_v4 import timer
from scotty.typing import FloatArray
# from sklearn.utils import Bunch
from typing import Any, Callable, Dict, Literal, Protocol, Union, Tuple, cast

log = logging.getLogger(__name__)

class _Event(Protocol):
    """Protocol describing a `scipy.integrate.solve_ivp` event callback"""
    terminal: bool = False
    direction: float = 0.0
    def __call__(self, *args): pass

def _event(terminal: bool, direction: float):
    """Decorator to add the attributes required for
    `scipy.integrate.solve_ivp while keeping the `mypy` type checker
    happy
    """

    def decorator_event(func: Any) -> _Event:
        func.terminal = terminal # Terminate solver when beam leaves plasma
        func.direction = direction # +ve value when function result goes from -ve to +ve
        return func
    
    return decorator_event



def make_solver_events(
    poloidal_flux_enter: float,
    launch_angular_frequency: float,
    field: Union[MagneticField_Cartesian, MagneticField_Cylindrical],
) -> Dict[str, Callable]:
    
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
    def _minmax(arr) -> Tuple[float, float]: return min(arr), max(arr)
    q0_min, q0_max = _minmax(field.X_coord if cart else field.R_coord) # type: ignore
    q1_min, q1_max = _minmax(field.Y_coord if cart else [-np.inf, np.inf]) # type: ignore
    q2_min, q2_max = _minmax(field.Z_coord)

    # Triggers when the beam leaves the magnetic field region
    #   +ve -> -ve when leaving the simulation region
    @_event(terminal=True, direction=-1.0)
    def event_leave_simulation(tau, ray_parameters, hamiltonian):
        q0, q1, q2, _, _, _ = ray_parameters
        is_inside = ((q0_min < q0 < q0_max)
                 and (q1_min < q1 < q1_max)
                 and (q2_min < q2 < q2_max))
        return +1 if is_inside else -1
    
    # Triggers when the beam frequency is equal to the fundamental
    # electron cyclotron frequency.
    # ***Not implemented for relativistic temperatures
    # ***Used to include a `delta_gyro_freq` where the event triggers
    #    when the beam frequency is close to this frequency, but this
    #    was removed because it worked unreliably
    @_event(terminal=True, direction=0.0)
    def event_cross_resonance(tau, ray_parameters, hamiltonian):
        q0, q1, q2, _, _, _ = ray_parameters
        B_magnitude = field.magnitude(q0, q1, q2)

        # Find the ratio of beam freq to electron cyclotron freq
        gyro_freq = find_normalised_gyro_ang_freq(launch_angular_frequency, B_magnitude)

        # Find the difference. If the sign changes, it means the resonance
        # frequency has been crossed
        return gyro_freq - 1
    
    # Triggers when the beam frequency is equal to the second harmonic of
    # the fundamental electron cyclotron frequency.
    # ***Not implemented for relativistic temperatures
    # ***Used to include a `delta_gyro_freq` where the event triggers
    #    when the beam frequency is close to this frequency, but this
    #    was removed because it worked unreliably
    @_event(terminal=True, direction=0.0)
    def event_cross_resonance2(tau, ray_parameters, hamiltonian):
        q0, q1, q2, _, _, _ = ray_parameters
        B_magnitude = field.magnitude(q0, q1, q2)

        # Find the ratio of beam freq to electron cyclotron freq
        gyro_freq = find_normalised_gyro_ang_freq(launch_angular_frequency, B_magnitude)

        # Find the difference. If the sign changes, it means the resonance
        # frequency has been crossed. We want 0.5, because the second
        # harmonic is 2*\omega_c = \omega such that the normalised
        # gyro frequency is \omega_c / \omega = 0.5
        difference = gyro_freq - 0.5
        return difference
    
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
        if np.isnan(dK_d_tau) == True: dK_d_tau = 0
        
        return dK_d_tau

    return {
        "leave_plasma": event_leave_plasma,
        "leave_LCFS": event_leave_LCFS,
        "leave_simulation": event_leave_simulation,
        "cross_resonance": event_cross_resonance,
        "cross_resonance2": event_cross_resonance2,
        "reach_K_min": event_reach_K_min,
    }



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
    - ``"cross_resonance"``: the ray has crossed a resonance

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

    if detected("cross_resonance"):
        log.info(f"Ray has crossed resonance")
        return tau_events["cross_resonance"][0]

    if detected("cross_resonance2"):
        log.info(f"Ray has crossed resonance 2")
        return tau_events["cross_resonance2"][0]
    
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



def d_ray_parameters_d_tau(tau: FloatArray, ray_parameters: FloatArray, hamiltonian: Hamiltonian) -> FloatArray:
    # tau is unused here, but solve_ivp is strict about the function signatures and requires something like
    # f(t, y, y0), so we put the tau here so that it stops complaining
    dH = hamiltonian.derivatives(ray_parameters[:3], ray_parameters[-3:])
    # d_ray_parameters_d_tau = np.zeros_like(ray_parameters, dtype=np.float64)
    d_ray_parameters_d_tau = np.zeros(6, dtype=np.float64)

    for i, v in enumerate(dH.values()):
        # i < 3:  wavevector derivatives
        # i == 4: dH/dK_zeta = 0
        # i >= 3: spatial derivatives
        d_ray_parameters_d_tau[(i+3)%6] = -v if i < 3 else v
    
    return d_ray_parameters_d_tau



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
) -> Tuple[VALID_SOLVER_STATUS, int, float, FloatArray, Union[float, FloatArray]]:
    
    """Propagates a ray, given an initial position `q` and wavevector `K`, using
    `scipy.integrate.solve_ivp` until any of these terminating conditions are
    encountered:
        (i) ray leaves the plasma (at the same `poloidal_flux_enter`);
        (ii) ray leaves the simulation region (i.e. exits the bounds of the magnetic profile); or
        (iii) ray reaches a resonance layer (i.e. when the beam frequency is equal to the
              fundamental or second harmonic of the electron gyrotron frequency).
    
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

    # Creating arguments and events to be passed to solver
    solver_ray_events = make_solver_events(poloidal_flux_enter, hamiltonian.angular_frequency, hamiltonian.field)
    solver_arguments = (hamiltonian,)

    # Propagating q, K by solving the ray-tracing equatoins
    log.info(f"Starting the ray solver")

    (solver_ray_output,
     duration_ray_tracing) = timer(solve_ivp)(
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

    if   solver_ray_output.status ==  0: raise RuntimeError("Ray has not left plasma/simulation region; increase `tau_max` or choose different initial conditions")
    elif solver_ray_output.status == -1: raise RuntimeError("Integration step failed. Check that (i) ray does not leave interpolation region; and (ii) density interpolation is not negative")

    # tau_events is a list with the same order as the values of
    # solver_ray_events, so we can use the names from that dict
    # instead of raw indices

    soln_interp = cast(OdeSolution, solver_ray_output.sol)
    tau_events            = dict(zip(solver_ray_events.keys(), solver_ray_output.t_events))
    ray_parameters_events = dict(zip(solver_ray_events.keys(), solver_ray_output.y_events))
    tau_terminating_event = handle_terminating_event(tau_events=tau_events, ray_parameters_events=ray_parameters_events["leave_LCFS"])
    tau_arr_resampled = np.linspace(start=0, stop=tau_terminating_event, num=len_tau-1, endpoint=False)

    # Find the cut-off (i.e. `tau` where K_magnitude is minimised) if no resonance is reached
    if (len(tau_events["cross_resonance"]) == 0 and len(tau_events["cross_resonance2"]) == 0):
        def K_magnitude(tau):
            q0, _, _, K0, K1, K2 = soln_interp(tau)
            return np.sqrt(K0**2 + K1**2 + K2**2) if isinstance(hamiltonian.field, MagneticField_Cartesian) else np.sqrt(K0**2 + (K1/q0)**2 + K2**2)
        
        tau_cutoff = minimize_scalar(fun=K_magnitude, bounds=[0, tau_arr_resampled[-1]], options={"xatol": atol})
        tau_arr_resampled = np.sort(np.append(np.linspace(start=0, stop=tau_terminating_event, num=len_tau-1, endpoint=False), tau_cutoff.x))
    
    else:
        tau_arr_resampled = np.linspace(start=0, stop=tau_terminating_event, num=len_tau-1, endpoint=False)
    
    # For ray-tracing runs, return `tau` and `result` (containing `q`, `K`)
    # Otherwise return `tau_terminating_event` and `tau_arr_resampled` (since the beam solver calculates everything again) 
    if ray_tracing_flag: return solver_ray_output.status, solver_ray_output.nfev, duration_ray_tracing, tau_arr_resampled, soln_interp(tau_arr_resampled)
    else:                return solver_ray_output.status, solver_ray_output.nfev, duration_ray_tracing, tau_arr_resampled, tau_terminating_event