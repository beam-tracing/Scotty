from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scotty.fun_general import find_normalised_gyro_freq
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D
from scotty.typing import FloatArray
from time import time
from typing import Any, Callable, Dict, Protocol, Tuple, Union

### Declaring class _Event and def _event
# Decorator event used in make_solver_events()

class _Event(Protocol):
    """Protocol describing a `scipy.integrate.solve_ivp` event callback"""

    terminal: bool = False
    direction: float = 0.0

    def __call__(self, *args):
        pass

def _event(terminal: bool, direction: float):
    """Decorator to add the attributes required for `scipy.integrate.solve_ivp` while
    keeping mypy happy
    """

    def decorator_event(func: Any) -> _Event:
        # Stop the solver when the beam leaves the plasma
        func.terminal = terminal
        # positive value, when function result goes from negative to positive
        func.direction = direction
        return func

    return decorator_event



### Declaring def make_solver_events

def make_solver_events(
    poloidal_flux_enter: float,
    launch_angular_frequency: float,
    field: MagneticField_3D_Cartesian
) -> Dict[str, Callable]:
    
    # This event triggers when the beam leaves the same poloidal flux value it entered at
    # Goes from negative to positive when leaving the plasma
    @_event(terminal = True, direction = 1.0)
    def event_leave_plasma(tau, ray_parameters_3D, hamiltonian: Hamiltonian_3D):
        q_X, q_Y, q_Z, K_X, K_Y, K_Z = ray_parameters_3D
        polflux = field.polflux(q_X, q_Y, q_Z)
        return polflux - poloidal_flux_enter
    
    # This event triggers when the beam leaves the LCFS
    # Goes from negative to positive when leaving the LCFS
    @_event(terminal = False, direction = 1.0)
    def event_leave_LCFS(tau, ray_parameters_3D, hamiltonian: Hamiltonian_3D):
        q_X, q_Y, q_Z, K_X, K_Y, K_Z = ray_parameters_3D
        polflux = field.polflux(q_X, q_Y, q_Z)
        polflux_LCFS = 1.0
        return polflux - polflux_LCFS
    
    # Capture the bounding box of the magnetic field
    # To see if the ray has left the plasma
    X_coord_min, X_coord_max = field.X_coord.min(), field.X_coord.max()
    Y_coord_min, Y_coord_max = field.Y_coord.min(), field.Y_coord.max()
    Z_coord_min, Z_coord_max = field.Z_coord.min(), field.Z_coord.max()

    # This event triggers when the beam leaves the magnetic field region
    # Goes from positive (True) to negative (False) when leaving the simulation region 
    @_event(terminal = True, direction = -1.0)
    def event_leave_simulation(tau, ray_parameters_3D, hamiltonian: Hamiltonian_3D):
        q_X, q_Y, q_Z, K_X, K_Y, K_Z = ray_parameters_3D
        is_inside = ((X_coord_min < q_X < X_coord_max)
                 and (Y_coord_min < q_Y < Y_coord_max)
                 and (Z_coord_min < q_Z < Z_coord_max))
        return +1 if is_inside else -1
    
    # This event triggers when the beam's frequency is equal
    # to the fundamental election cyclotron frequency
    # Not implemented for relativistic temperatures
    # Used to include a delta_gyro_freq where the event triggers when
    # beam frequency is close to the electron frequency, but this is
    # removed because it works unreliably
    @_event(terminal = True, direction = 0.0)
    def event_cross_resonance(tau, ray_parameters_3D, hamiltonian: Hamiltonian_3D):
        q_X, q_Y, q_Z, K_X, K_Y, K_Z = ray_parameters_3D
        B_magnitude = field.magnitude(q_X, q_Y, q_Z)

        # Find the ratio of beam freq to electron cyclotron freq
        gryo_freq = find_normalised_gyro_freq(B_magnitude, launch_angular_frequency)

        # Find the difference. If the sign changes, it means the resonance frequency has been crossed
        difference = gryo_freq - 1
        return difference
    
    # This event triggers when the beam's frequency is equal
    # to the second harmonic of the electron cyclotron frequency
    # Not implemented for relativistic temperatures
    # Used to include a delta_gyro_freq where the event triggers when
    # beam frequency is close to the second harmonic electron frequency,
    # but this is removed because it works unreliably
    @_event(terminal = True, direction = 0.0)
    def event_cross_resonance2(tau, ray_parameters_3D: float, hamiltonian: Hamiltonian_3D):
        q_X, q_Y, q_Z, K_X, K_Y, K_Z = ray_parameters_3D
        B_magnitude = field.magnitude(q_X, q_Y, q_Z)

        # Find the ratio of beam freq to electron cyclotron freq
        gyro_freq = find_normalised_gyro_freq(B_magnitude, launch_angular_frequency)

        # Find the difference. If the sign changes, it means the resonance frequency has been crossed
        # We want 0.5, because the second harmonic is 2*\omega_c = \omega
        # Normalised gyrofreq = \omega_c / \omega = 0.5
        difference = gyro_freq - 0.5
        return difference
    
    # This event triggers when the cut-off location is reached (when the wavenumber K is minimised)
    @_event(terminal = False, direction = 1.0)
    def event_reach_K_min(tau, ray_parameters_3D, hamiltonian: Hamiltonian_3D):
        q_X, q_Y, q_Z, K_X, K_Y, K_Z = ray_parameters_3D
        K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)

        dH = hamiltonian.derivatives(q_X, q_Y, q_Z, K_X, K_Y, K_Z)

        dK_d_tau = -(2 * K_magnitude) * (
            dH["dH_dX"] + dH["dH_dY"] + dH["dH_dZ"]
        )

        # This event does not work properly when the ray reaches resonance
        # The following if statement introduces a trick to avoid this problem
        # When ray reaches resonance, dK_dtau goes to infinity. Just set to 0
        # to tell scotty that we are heading to infinity if we get a NaN.
        ##
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
        "cross_resonance": event_cross_resonance,
        "cross_resonance2": event_cross_resonance2,
        "reach_K_min": event_reach_K_min,
    }



def handle_leaving_plasma_events(
    tau_events: Dict[str, FloatArray], ray_parameters_3D_events: FloatArray
) -> float:
    """Handle events detected by `scipy.integrate.solve_ivp`. This
    only handles events due to the ray leaving the plasma or
    simulation:

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
    K_R_LCFS : float
        The radial wavevector in the case where the beam has crossed
        the LCFS

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
    if detected("leave_plasma") and not detected("leave_LCFS"):
        return tau_events["leave_plasma"][0]

    if detected("cross_resonance"):
        return tau_events["cross_resonance"][0]

    if detected("cross_resonance2"):
        return tau_events["cross_resonance2"][0]
    
    """TO DO, for now we just cut it when Psi = 1.0"""
    if detected("leave_LCFS"):
        return tau_events["leave_LCFS"][0]

    # If one ends up here, things aren't going well. I can think of
    # two possible reasons:
    # - The launch conditions are really weird (hasn't happened yet,
    #   in my experience)
    # - The max_step setting of the solver is too large, such that the
    #   ray leaves the LCFS and enters a region where `poloidal_flux <
    #   1` in a single step. The solver thus doesn't log the event
    #   when it really should

    print("Warning: Ray has left the simulation region without leaving the LCFS.")
    return tau_events["leave_simulation"][0]



def handle_no_resonance(
    solver_ray_output,
    tau_leave: float,
    tau_points: FloatArray,
    solver_arguments,
    event_leave_plasma: Callable,
) -> FloatArray:
    """Add an additional tau point at the cut-off (minimum K) if the
    beam does NOT reach a resonance

    Propagates another ray to find the cut-off location

    TO-DO
    ----
    Check if using ``dense_output`` in the initial ray solver can get
    the same information better/faster

    """
    ray_parameters_3D = solver_ray_output.y
    tau_ray = solver_ray_output.t

    index_max_tau = int(np.argmax(tau_ray[tau_ray <= tau_leave]))

    K_X, K_Y, K_Z = ray_parameters_3D[3, :index_max_tau], ray_parameters_3D[4, :index_max_tau], ray_parameters_3D[5, :index_max_tau]
    ray_K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)

    index_cutoff_estimate = int(np.argmin(ray_K_magnitude))
    start = max(0, index_cutoff_estimate - 1)
    stop = min(len(tau_ray) - 1, index_cutoff_estimate + 1)

    ray_parameters_3D_initial_fine = ray_parameters_3D[:, start]
    tau_start_fine = tau_ray[start]
    tau_end_fine = tau_ray[stop]
    tau_points_fine = np.linspace(tau_start_fine, tau_end_fine, 1001)

    solver_start_time = time()

    solver_ray_output_fine = solve_ivp(
        ray_evolution_3D_fun,
        [tau_start_fine, tau_end_fine],
        ray_parameters_3D_initial_fine,
        method="RK45",
        t_eval=tau_points_fine,
        dense_output=False,
        events=event_leave_plasma,
        vectorized=False,
        args=solver_arguments,
    )

    solver_end_time = time()
    print("Time taken (cut-off finder)", solver_end_time - solver_start_time, "s")

    ray_K_magnitude_fine = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)

    index_cutoff_fine = np.argmin(ray_K_magnitude_fine)
    tau_cutoff_fine = float(solver_ray_output_fine.t[index_cutoff_fine])
    return np.sort(np.append(tau_points, tau_cutoff_fine))



@dataclass
class K_cutoff_data:
    """Properties of :math:`K`-cutoff"""
    q_X: float
    q_Y: float
    q_Z: float
    K_norm_min: float
    polflux: float
    theta_m: float


""" TO-DO
def quick_K_cutoff(
    ray_parameters_turning_pt: FloatArray, K_zeta: float, field: MagneticField
) -> K_cutoff_data:"""



### Declaring class K_cutoff_data_cartesian

class K_cutoff_data_cartesian:
    """Properties of :math:`K`-cutoff"""
    q_X: float
    q_Y: float
    q_Z: float
    K_norm_min: float
    poloidal_flux: float
    theta_m: float



### Declaring def ray_evolution_3D_fun

def ray_evolution_3D_fun(tau, ray_parameters_3D, hamiltonian: Hamiltonian_3D):
    
    # Saving the coordinates
    q_X, q_Y, q_Z, K_X, K_Y, K_Z = ray_parameters_3D

    # Find the derivatives of H
    dH = hamiltonian.derivatives(q_X, q_Y, q_Z, K_X, K_Y, K_Z)

    # d(parameter) / d(tau)
    # indexes 0, 1, 2 correspond to d(q_X)/d(tau), d(q_Y)/d(tau), d(q_Z)/d(tau)
    # indexes 3, 4, 5 correspond to d(K_X)/d(tau), d(K_Y)/d(tau), d(K_Z)/d(tau)
    d_ray_parameters_3D_d_tau = np.zeros_like(ray_parameters_3D)
    d_ray_parameters_3D_d_tau = dH["dH_dX"], dH["dH_dY"], dH["dH_dZ"], dH["dH_dKx"], dH["dH_dKy"], dH["dH_dKz"]

    return d_ray_parameters_3D_d_tau



### Declaring def propagate_ray

def propagate_ray(
    poloidal_flux_enter: float,
    launch_angular_frequency: float,
    field: MagneticField_3D_Cartesian,
    initial_position: FloatArray,
    K_initial: FloatArray,
    hamiltonian: Hamiltonian_3D,
    rtol: float,
    atol: float,
    quick_run: bool,
    len_tau: int,
    tau_max: float = 1e5,
    verbose: bool = True,
) -> Union[Tuple[float, FloatArray], K_cutoff_data_cartesian]:
    
    solver_ray_events = make_solver_events(
        poloidal_flux_enter, launch_angular_frequency, field
    )

    K_X_initial, K_Y_initial, K_Z_initial = K_initial

    # Ray evolves q_X, q_Y, q_Z, K_X, K_Y, K_Z
    ray_parameters_3D_initial = [
        initial_position[0],
        initial_position[1],
        initial_position[2],
        K_X_initial,
        K_Y_initial,
        K_Z_initial,
    ]

    # Additional arguments for solver
    solver_arguments = (hamiltonian,)

    solver_start_time = time()
    solver_ray_output = solve_ivp(
        ray_evolution_3D_fun,
        [0, tau_max],
        ray_parameters_3D_initial,
        method="RK45",
        t_eval=None,
        dense_output=False,
        events=solver_ray_events.values(),
        vectorized=False,
        args=solver_arguments,
        rtol=rtol,
        atol=atol,
        max_step=50,
    )
    
    # to remove, just for debugging
    print(solver_ray_output.t_events)
    print()
    print(solver_ray_output.y_events)

    solver_end_time = time()
    if verbose:
        print("Time taken (ray solver)", solver_end_time - solver_start_time, "s")

    if solver_ray_output.status == 0:
        raise RuntimeError(
            "Ray has not left plasma/simulation region. "
            "Increase tau_max or choose different initial conditions."
        )
    if solver_ray_output.status == -1:
        raise RuntimeError(
            "Integration step failed. Check density interpolation is not negative"
        )

    # tau_events is a list with the same order as the values of
    # solver_ray_events, so we can use the names from that dict
    # instead of raw indices
    tau_events = dict(zip(solver_ray_events.keys(), solver_ray_output.t_events))
    ray_parameters_3D_events = dict(zip(solver_ray_events.keys(), solver_ray_output.y_events))

    tau_leave = handle_leaving_plasma_events(tau_events, ray_parameters_3D_events["leave_LCFS"])

    """
    TO-DO

    if quick_run:
        return quick_K_cutoff(
            ray_parameters_2D_events["reach_K_min"], K_zeta_initial, field
        )
    """

    # The beam solver outputs data at these values of tau
    # Don't include `tau_leave` itself so that last point is inside the plasma
    tau_points = np.linspace(0, tau_leave, len_tau - 1, endpoint=False)

    # you want no resonance at all, so both must be 0
    if (
        len(tau_events["cross_resonance"]) == 0
        and len(tau_events["cross_resonance2"]) == 0
    ):
        tau_points = handle_no_resonance(
            solver_ray_output,
            tau_leave,
            tau_points,
            solver_arguments,
            solver_ray_events["leave_plasma"],
        )

    return tau_leave, tau_points