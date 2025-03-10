import datetime
import datatree
import json
import numpy as np
import pathlib
from scipy.integrate import solve_ivp
from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.profile_fit import ProfileFitLike, profile_fit
from scotty.typing import ArrayLike, FloatArray, PathLike
from scotty._version import __version__
import time
from typing import cast, Optional, Sequence, Union
import uuid
import xarray as xr

# new -- for stellarator case
# from scotty.fun_evolution_stellarator import pack_beam_parameters_3D, unpack_beam_parameters_3D, beam_evolution_fun_stellarator
from scotty.hamiltonian_stellarator import Hamiltonian_stellarator
from scotty.ray_solver_stellarator import propagate_ray_stellarator

def beam_me_up_stellarator(
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    launch_freq_GHz: float,
    mode_flag: int,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position_cartesian: FloatArray,

    # keyword arguments begin
    vacuumLaunch_flag: bool = True,
    relativistic_flag: bool = False,  # includes relativistic corrections to electron mass when set to True
    find_B_method: Union[str, MagneticField_3D_Cartesian] = "stellarator",
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    shot=None,
    equil_time=None,
    vacuum_propagation_flag: bool = False,
    Psi_BC_flag: Union[bool, str, None] = None,
    poloidal_flux_enter: float = 1.0,
    poloidal_flux_zero_density: float = 1.0,  ## When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: float = 1.0,  ## temperature analogue of poloidal_flux_zero_density

    # Finite-difference and solver parameters
    auto_delta_sign = True,  # For flipping signs to maintain forward difference. Applies to delta_R and delta_Z
    delta_X: float = -0.0001, # in the same units as data_X_coord
    delta_Y: float = 0.0001,  # in the same units as data_Y_coord
    delta_Z: float = 0.0001,  # in the same units as data_Z_coord
    delta_K_X: float = 0.1,  # in the same units as K_X
    delta_K_Y: float = 0.1,  # in the same units as K_Y
    delta_K_Z: float = 0.1,  # in the same units as K_Z
    interp_order = 5,     # For the 3D interpolation functions
    interp_smoothing = 5, # For the 3D interpolation functions (specifically, density_fit)
    len_tau: int = 102,
    rtol: float = 1e-3,  # for solve_ivp of the beam solver
    atol: float = 1e-6,  # for solve_ivp of the beam solver
    tau_eval: type = None, # To remove
    points_from_2d_scotty_to_eval: ArrayLike = None, # To remove

    # Input and output settings
    ne_data_path = pathlib.Path("."),
    magnetic_data_path = pathlib.Path("."),
    Te_data_path = pathlib.Path("."),
    output_path = pathlib.Path("."),
    input_filename_suffix = "",
    output_filename_suffix = "",
    figure_flag = True,
    detailed_analysis_flag = True,

    # For quick runs (only ray tracing)
    quick_run: bool = False,

    # For launching within the plasma
    plasmaLaunch_K_cartesian = np.zeros(3),
    plasmaLaunch_Psi_3D_lab_cartesian = np.zeros([3, 3]),
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,

    # For stellarator equilibrium
    equilibrium = None,
) -> datatree.DataTree:
    
    print("Beam trace me up, Scotty -- for stellarators!")
    print(f"scotty version {__version__}")
    run_id = uuid.uuid4()
    print(f"Run ID: {run_id}")

    # ------------------------------
    # Input data #
    # ------------------------------

    # Tidying up the input data
    launch_angular_frequency = freq_GHz_to_angular_frequency(launch_freq_GHz)

    # Ensure paths are `pathlib.Path`
    ne_data_path = pathlib.Path(ne_data_path)
    magnetic_data_path = pathlib.Path(magnetic_data_path)
    Te_data_path = pathlib.Path(Te_data_path)
    output_path = pathlib.Path(output_path)

    # Initialises the Hamiltonian H
    hamiltonian = Hamiltonian_stellarator(
        equilibrium,
        launch_angular_frequency,
        mode_flag,
        delta_X,
        delta_Y,
        delta_Z,
        delta_K_X,
        delta_K_Y,
        delta_K_Z,
        None # temperature_fit
    )

    # ------------------------------
    # Launch parameters #
    # ------------------------------
    if vacuumLaunch_flag: pass
    else:
        print("Beam launched from inside the plasma")
        Psi_3D_lab_initial_cartesian = plasmaLaunch_Psi_3D_lab_cartesian
        K_initial_cartesian = plasmaLaunch_K_cartesian
        initial_position = launch_position_cartesian
        launch_K = None
        Psi_3D_lab_launch_cartesian = None
        Psi_3D_lab_entry_cartersian = None
        distance_from_launch_to_entry = None
    
    # -------------------
    # Propagate the ray

    if points_from_2d_scotty_to_eval is None: # TO REMOVE this entire if-else block
        print("Starting the solvers")
        ray_solver_output = propagate_ray_stellarator(
            poloidal_flux_enter,
            launch_angular_frequency,
            equilibrium,
            initial_position,
            K_initial_cartesian,
            hamiltonian,
            rtol,
            atol,
            quick_run,
            len_tau,
            tau_eval, # TO REMOVE
        )
        if quick_run:
            return ray_solver_output

        tau_leave, tau_points = cast(tuple, ray_solver_output)

        # return ray_solver_output # TO REMOVE -- for debugging purposes only

        # -------------------
        # Propagate the beam

        # Initial conditions for the solver
        beam_parameters_initial = pack_beam_parameters_3D(
            initial_position[0],
            initial_position[1],
            initial_position[2],
            K_initial_cartesian[0],
            K_initial_cartesian[1],
            K_initial_cartesian[2],
            Psi_3D_lab_initial_cartesian,
        )

        print("Psi: ") # TO REMOVE
        print("Psi_xx: ", beam_parameters_initial[6], beam_parameters_initial[12])
        print("Psi_yy: ", beam_parameters_initial[7], beam_parameters_initial[13])
        print("Psi_zz: ", beam_parameters_initial[8], beam_parameters_initial[14])

        print("Psi_xy: ", beam_parameters_initial[9], beam_parameters_initial[15])
        print("Psi_xz: ", beam_parameters_initial[10], beam_parameters_initial[16])
        print("Psi_yz: ", beam_parameters_initial[11], beam_parameters_initial[17])

        solver_start_time = time.time()

        # TO REMOVE
        tau_points = list(tau_points)
        tau_points = dict.fromkeys(tau_points)
        tau_points = list(tau_points)
        tau_points = np.array(tau_points)
        print(tau_points)

        solver_beam_output = solve_ivp(
            beam_evolution_fun_3D,
            [0, tau_leave],
            beam_parameters_initial,
            method="RK45",
            t_eval=tau_points,
            dense_output=False,
            events=None,
            vectorized=False,
            args=(hamiltonian,),
            rtol=rtol,
            atol=atol,
        )

        solver_end_time = time.time()
        solver_time = solver_end_time - solver_start_time
        print(f"Time taken (beam solver) {solver_time}s")
        print(f"Number of beam evolution evaluations: {solver_beam_output.nfev}")
        print(f"Time per beam evolution evaluation: {solver_time / solver_beam_output.nfev}")

        tau_array = solver_beam_output.t
        beam_parameters_final = solver_beam_output.y
        solver_status = solver_beam_output.status

        q_X_array, q_Y_array, q_Z_array, K_X_array, K_Y_array, K_Z_array, Psi_3D_output = unpack_beam_parameters_3D(beam_parameters_final)

        print("Main loop complete")
        # -------------------
    else:
        solver_status = -1
        print("points_from_2d_scotty_to_eval is supposed to be None for stellarators!")



    # TO REMOVE
    if solver_status == -1: print("Solver did not reach completion.")
    else:
        pointwise_data = [[q_X_array[i],
                           q_Y_array[i],
                           q_Z_array[i],
                           K_X_array[i],
                           K_Y_array[i],
                           K_Z_array[i],
                           Psi_3D_output[i],
                           tau_array[i]]
                           for i in range(len(tau_array))]
        
    # I did not include a bunch of code here -- hopefully we don't need it
    
    return pointwise_data # , _numerical_H_values_for_heatmap # TO REMOVE
