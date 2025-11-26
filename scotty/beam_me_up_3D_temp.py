import datetime
import datatree
import logging
import numpy as np
import pathlib
from scipy.integrate import solve_ivp
from scotty.analysis_3D import immediate_analysis_3D, further_analysis_3D, main_analysis_3D
from scotty.checks_3D import Parameters, check_input_before_ray_tracing, VALID_LAUNCH_MODE_FLAGS, VALID_PSI_BC_FLAGS
from scotty.fun_evolution_3D import evolve_beam
from scotty.geometry_3D import MagneticField_3D_Cartesian, create_magnetic_geometry_3D
from scotty.hamiltonian_3D import initialise_hamiltonians, assign_hamiltonians
from scotty.launch_3D import find_plasma_entry_position, find_auto_delta_signs, find_plasma_entry_parameters
from scotty.logger_3D import config_logger, arr2str
from scotty.plotting_3D import (
    plot_delta_theta_m,
    plot_dispersion_relation,
    plot_localisations,
    plot_theta_m,
    plot_trajectory,
    plot_trajectories_individual,
    plot_wavevector,
    plot_widths,

    plot_trajectories_poloidal
)
from scotty.profile_fit import ProfileFitLike, profile_fit
from scotty.ray_solver_3D import propagate_ray
from scotty.typing import ArrayLike, FloatArray, PathLike
from scotty._version import __version__
from typing import cast, Optional, Sequence, Union
import uuid
import xarray as xr

def beam_me_up_3D(
    # General launch parameters
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    launch_freq_GHz: float,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position_cartesian: FloatArray,
    mode_flag: VALID_LAUNCH_MODE_FLAGS,
    vacuumLaunch_flag: bool = True,
    vacuum_propagation_flag: bool = False,
    Psi_BC_flag: VALID_PSI_BC_FLAGS = None,
    relativistic_flag: bool = False, # includes relativistic corrections to electron mass when set to True

    # Data input and output arguments
    magnetic_data_path = pathlib.Path("."),
    ne_data_path = pathlib.Path("."),
    Te_data_path = pathlib.Path("."),
    input_filename_suffix: str = "",
    output_path = pathlib.Path("."),
    output_filename_suffix: str = "",
    shot = None,
    equil_time = None,

    # Solver settings and finite-difference parameters
    auto_delta_sign: bool = True,  # For flipping signs to maintain forward difference. Applies to delta_X, delta_Y, delta_Z
    delta_X: float = 1e-4, # in the same units as data_X_coord
    delta_Y: float = 1e-4, # in the same units as data_Y_coord
    delta_Z: float = 1e-4, # in the same units as data_Z_coord
    delta_K_X: float = 1e-1,  # in the same units as K_X
    delta_K_Y: float = 1e-1,  # in the same units as K_Y
    delta_K_Z: float = 1e-1,  # in the same units as K_Z
    len_tau: int = 102,
    rtol: float = 1e-3, # for solve_ivp of the ray/beam solvers
    atol: float = 1e-6, # for solve_ivp of the ray/beam solvers
    poloidal_flux_enter: float = 1.0,
    poloidal_flux_zero_density: float = 1.0, # When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: float = 1.0, # Temperature analogue of poloidal_flux_zero_density

    # Interpolation settings
    find_B_method: Union[str, MagneticField_3D_Cartesian] = "eduard_3D",
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    interp_order: Union[str, int] = 5, # For the 3D interpolation functions
    interp_smoothing: int = 5, # For the 3D interpolation functions (specifically, make_fit)

    # Logging flags
    console_log_level: Union[str, int] = "INFO", # For returning log messages on console
    file_log_level: Optional[Union[str, int]] = None, # For returning log messages on a log file
    # TO REMOVE -- need to put one more argument for log name

    # Plotting flags
    figure_flag: bool = True,
    further_analysis_flag: bool = False,
    detailed_analysis_flag: bool = False,
    # TO REMOVE -- need to put individual flags for each plot

    # Additional flags
    quick_run: bool = False,       # For quick runs (only ray tracing)
    return_dt_field: bool = False, # For returning the datatree, field class, and Hamiltonians

    # Keeping the extra kwargs for parsing later
    **extras,

    # These are/have been put in checks_3D

    # # Only used for circular flux surfaces
    # B_T_axis = None,
    # B_p_a = None,
    # R_axis = None,
    # minor_radius_a = None,

    # # Launch parameters (only used if ray propagation begins in the plasma)
    # plasmaLaunch_K_cartesian: FloatArray = np.zeros(3),
    # plasmaLaunch_Psi_3D_lab_cartesian: FloatArray = np.zeros([3, 3]),

) -> datatree.DataTree:
    
    ##################################################
    #
    # INITIALISATION ROUTINE
    #
    ##################################################
    q_launch_cartesian = launch_position_cartesian # This is left here for backward compatibility
    params = Parameters(poloidal_launch_angle_Torbeam = poloidal_launch_angle_Torbeam,
                        toroidal_launch_angle_Torbeam = toroidal_launch_angle_Torbeam,
                        launch_freq_GHz = launch_freq_GHz,
                        launch_beam_width = launch_beam_width,
                        launch_beam_curvature = launch_beam_curvature,
                        q_launch_cartesian = q_launch_cartesian,
                        mode_flag_launch = mode_flag,
                        vacuumLaunch_flag = vacuumLaunch_flag,
                        vacuum_propagation_flag = vacuum_propagation_flag,
                        Psi_BC_flag = Psi_BC_flag,
                        relativistic_flag = relativistic_flag,

                        magnetic_data_path = magnetic_data_path,
                        ne_data_path = ne_data_path,
                        Te_data_path = Te_data_path,
                        input_filename_suffix = input_filename_suffix,
                        output_path = output_path,
                        output_filename_suffix = output_filename_suffix,
                        shot = shot,
                        equil_time = equil_time,

                        auto_delta_sign = auto_delta_sign,
                        delta_X = delta_X,
                        delta_Y = delta_Y,
                        delta_Z = delta_Z,
                        delta_K_X = delta_K_X,
                        delta_K_Y = delta_K_Y,
                        delta_K_Z = delta_K_Z,
                        len_tau = len_tau,
                        rtol = rtol,
                        atol = atol,
                        poloidal_flux_enter = poloidal_flux_enter,
                        poloidal_flux_zero_density = poloidal_flux_zero_density,
                        poloidal_flux_zero_temperature = poloidal_flux_zero_temperature,

                        density_fit_parameters = density_fit_parameters,
                        density_fit_method = density_fit_method,
                        temperature_fit_parameters = temperature_fit_parameters,
                        temperature_fit_method = temperature_fit_method,
                        interp_order = interp_order,
                        interp_smoothing = interp_smoothing,

                        # TO REMOVE? still unused
                        figure_flag = figure_flag,
                        further_analysis_flag = further_analysis_flag,
                        detailed_analysis_flag = detailed_analysis_flag,

                        # TO REMOVE? still unused
                        quick_run = quick_run,
                        return_dt_field = return_dt_field,

                        **extras
                        )

    # Setting up the logger
    config_logger(console_log_level, file_log_level, params.output_path, output_filename_suffix)
    log = logging.getLogger(__name__)
    log.debug(f"Saved and validated launch parameters")
    log.debug(f"Initialised logger")
    
    log.info(f"""\n
    ##################################################
    #
    # STARTING ROUTINE
    #
    ##################################################
    #
    # Beam trace me up, Scotty!
    # scotty version {__version__}
    # Run ID: {uuid.uuid4()}
    #
    ##################################################
    #
    # Starting run for:
    #   - Pol. launch angle = {params.poloidal_launch_angle_deg_Torbeam} deg
    #   - Tor. launch angle = {params.toroidal_launch_angle_deg_Torbeam} deg
    #   - Launch frequency = {params.launch_frequency_GHz} GHz
    #   - Mode flag = {params.mode_flag_launch}
    #   - Launch beam width = {params.launch_beam_width} m
    #   - Launch beam curvature = {params.launch_beam_curvature} m^-1
    #   - Launch position [X,Y,Z] = {arr2str(params.q_launch_cartesian)}
    #
    ##################################################
    """)
    
    log.debug(f"Setting experimental profiles")
    params.set_experimental_profiles()

    # Using the electron density data, create a spline fit as a
    # function of poloidal flux, i.e.:
    # density_fit(poloidal_flux) = electron_density
    density_fit = make_fit(params.density_fit_method,
                           params.poloidal_flux_zero_density,
                           params.density_fit_parameters,
                           params.ne_filename)

    # TODO THIS SOON
    # Using the temperature data, create a spline fit as a
    # function of poloidal flux, i.e.:
    # temperature_fit(poloidal_flux) = temperature
    log.debug(f"Making temperature fit profile")
    log.warning(f"Skipping temperature fit profile -- code not done yet")
    temperature_fit = None

    # If the user has already pre-loaded a field, use it
    # Otherwise, create a new one from the specified data pathway
    field = create_magnetic_geometry_3D(find_B_method,
                                        params.magnetic_data_path,
                                        params.input_filename_suffix,
                                        params.interp_order_magnetic_data,
                                        shot,
                                        equil_time)
    
    log.info(f"""\n
    ##################################################
    #
    # LAUNCH ROUTINE
    #
    ##################################################
    """)

    # Calculating the plasma entry position and auto_delta_sign
    params.q_initial_cartesian = find_plasma_entry_position(params.poloidal_launch_angle_deg_Torbeam,
                                                            params.toroidal_launch_angle_deg_Torbeam,
                                                            params.q_launch_cartesian,
                                                            field,
                                                            params.poloidal_flux_enter,
                                                            boundary_adjust = 1e-6)

    # Setting the delta_signs
    (params.delta_X,
     params.delta_Y,
     params.delta_Z) = find_auto_delta_signs(params.auto_delta_sign,
                                             params.q_initial_cartesian,
                                             params.delta_X,
                                             params.delta_Y,
                                             params.delta_Z,
                                             field)

    # Initialises the Hamiltonian H for mode_flags +1 and -1
    (hamiltonian_pos1,
     hamiltonian_neg1) = initialise_hamiltonians(params.launch_angular_frequency,
                                                 params.delta_X,
                                                 params.delta_Y,
                                                 params.delta_Z,
                                                 params.delta_K_X,
                                                 params.delta_K_Y,
                                                 params.delta_K_Z,
                                                 field,
                                                 density_fit,
                                                 temperature_fit)
    
    # Calculating the plasma entry parameters
    (params.K_launch_cartesian,
     params.K_initial_cartesian,
     params.Psi_3D_launch_labframe_cartesian,
     params.Psi_3D_entry_labframe_cartesian,
     params.Psi_3D_initial_labframe_cartesian,
     params.distance_from_launch_to_entry,
     params.e_hat_initial,
     params.mode_flag_initial,
     params.mode_index) = find_plasma_entry_parameters(params.vacuumLaunch_flag,
                                                       params.vacuum_propagation_flag,
                                                       params.Psi_BC_flag,
                                                       params.mode_flag_launch,
                                                       params.poloidal_launch_angle_deg_Torbeam,
                                                       params.toroidal_launch_angle_deg_Torbeam,
                                                       params.q_launch_cartesian,
                                                       params.q_initial_cartesian,
                                                       params.launch_wavenumber,
                                                       params.launch_beam_width,
                                                       params.launch_beam_curvature,
                                                       field,
                                                       params.K_plasmaLaunch_cartesian,
                                                       params.Psi_3D_plasmaLaunch_labframe_cartesian,
                                                       hamiltonian_pos1,
                                                       hamiltonian_neg1,
                                                       tol_H = 1e-5,
                                                       tol_O_mode_polarisation = 0.5)
    
    # Assigning the correct Hamiltonian
    (hamiltonian,
     hamiltonian_other,
     params.mode_flag_initial) = assign_hamiltonians(params.mode_flag_launch,
                                                     params.mode_flag_initial,
                                                     hamiltonian_pos1,
                                                     hamiltonian_neg1,
                                                     params.q_initial_cartesian,
                                                     params.K_initial_cartesian,
                                                     tol_H = 1e-5)

    # Checking validity of user-specified arguments
    # one last time before ray tracing
    check_input_before_ray_tracing(params)
    
    log.info(f"""\n
    ##################################################
    #
    # RAY TRACING ROUTINE
    #
    ##################################################
    """)

    ray_solver_output = propagate_ray(params.poloidal_flux_enter,
                                      params.launch_angular_frequency,
                                      field,
                                      params.q_initial_cartesian,
                                      params.K_initial_cartesian,
                                      hamiltonian,
                                      params.rtol,
                                      params.atol,
                                      quick_run,
                                      params.len_tau)
    
    # TO REMOVE -- need to put one line here to check final value of H_Booker and log it

    # TO REMOVE -- should change quick_run flag into ray_tracing only or something flag
    if quick_run: return ray_solver_output

    tau_leave, tau_points = cast(tuple, ray_solver_output)

    log.info(f"""\n
    ##################################################
    #
    # BEAM TRACING ROUTINE
    #
    ##################################################
    """)

    (solver_status, tau_array,
     (q_X_array, q_Y_array, q_Z_array,
      K_X_array, K_Y_array, K_Z_array,
      Psi_3D_output_labframe_cartesian)) = evolve_beam(params.q_initial_cartesian,
                                                       params.K_initial_cartesian,
                                                       params.Psi_3D_initial_labframe_cartesian,
                                                       tau_leave,
                                                       tau_points,
                                                       hamiltonian,
                                                       params.rtol,
                                                       params.atol)

    log.info(f"""\n
    ##################################################
    #
    # DATA SAVING ROUTINE
    #
    ##################################################
    """)

    # inputs = xr.Dataset(
    #     {
    #         # Data pathways
    #         "ne_data_path": str(params.ne_data_path),
    #         "magnetic_data_path": str(params.magnetic_data_path),
    #         "Te_data_path": str(params.Te_data_path),
    #         "output_path": str(params.output_path),
    #         "input_filename_suffix": str(params.input_filename_suffix),
    #         "output_filename_suffix": str(params.output_filename_suffix),
            
    #         # Important ray/beam stuff
    #         "poloidal_launch_angle_Torbeam": params.poloidal_launch_angle_deg_Torbeam,
    #         "toroidal_launch_angle_Torbeam": params.toroidal_launch_angle_deg_Torbeam,
    #         "launch_position_cartesian": (["col"], params.q_launch_cartesian),
    #         "launch_freq_GHz": params.launch_frequency_GHz,
    #         "launch_angular_frequency": params.launch_angular_frequency,
    #         "launch_beam_width": params.launch_beam_width,
    #         "launch_beam_curvature": params.launch_beam_curvature,
    #         "K_launch_cartesian": params.K_launch_cartesian,
    #         "mode_flag": mode_flag, # TO REMOVE?
    #         "mode_flag_launch": params.mode_flag_launch, # This must be the same as "mode_flag"
    #         "mode_flag_initial": params.mode_flag_initial,
    #         "mode_index": params.mode_index,
    #         "initial_position_cartesian": (["col"], params.q_initial_cartesian),
    #         "initial_K_cartesian": (["col"], params.K_initial_cartesian),
    #         "initial_Psi_3D_lab_cartesian": (["row","col"], params.Psi_3D_initial_labframe_cartesian),
    #         "delta_X": params.delta_X,
    #         "delta_Y": params.delta_Y,
    #         "delta_Z": params.delta_Z,
    #         "delta_K_X": params.delta_K_X,
    #         "delta_K_Y": params.delta_K_Y,
    #         "delta_K_Z": params.delta_K_Z,
    #         "interp_order": interp_order,
    #         "interp_order_magnetic_data": params.interp_order_magnetic_data,
    #         "interp_order_ne_data": params.interp_order_ne_data,
    #         "interp_smoothing": params.interp_smoothing,
    #         "len_tau": params.len_tau,
    #         "rtol": params.rtol,
    #         "atol": params.atol,

    #         # Poloidal flux parameters
    #         "poloidal_flux_enter": poloidal_flux_enter,
    #         "poloidal_flux_zero_density": poloidal_flux_zero_density,
    #         "poloidal_flux_zero_temperature": poloidal_flux_zero_temperature,

    #         # Flags
    #         "vacuumLaunch_flag": vacuumLaunch_flag,
    #         "vacuum_propagation_flag": vacuum_propagation_flag,
    #         "relativistic_flag": relativistic_flag,
    #         "Psi_BC_flag": Psi_BC_flag,
    #         "quick_run": quick_run,
    #         "figure_flag": figure_flag,
    #         "detailed_analysis_flag": detailed_analysis_flag,

    #         # Miscellaneous
    #         "find_B_method": str(find_B_method),
    #         "density_fit_parameters": str(density_fit_parameters),
    #         "temperature_fit_parameters": str(temperature_fit_parameters),
    #         "shot": shot,
    #         "equil_time": equil_time,
    #     },
    #     coords = {
    #         "X": field.X_coord,
    #         "Y": field.Y_coord,
    #         "Z": field.Z_coord,
    #         "row": ["X","Y","Z"],
    #         "col": ["X","Y","Z"],
    #     },
    # )

    # solver_output = xr.Dataset(
    #     {
    #         # Solver output
    #         "solver_status": solver_status,
    #         "q_X": (["tau"], q_X_array, {"long_name": "X", "units": "m"}),
    #         "q_Y": (["tau"], q_Y_array, {"long_name": "Y", "units": "m"}),
    #         "q_Z": (["tau"], q_Z_array, {"long_name": "Z", "units": "m"}),
    #         "K_X": (["tau"], K_X_array),
    #         "K_Y": (["tau"], K_Y_array),
    #         "K_Z": (["tau"], K_Z_array),
    #         "Psi_3D_launch_labframe_cartesian": (["row","col"], params.Psi_3D_launch_labframe_cartesian),
    #         "Psi_3D_entry_labframe_cartesian": (["row","col"], params.Psi_3D_entry_labframe_cartesian),
    #         "initial_Psi_3D_lab_cartesian": (["row","col"], params.Psi_3D_initial_labframe_cartesian), # TO REMOVE -- change key name?
    #         "Psi_3D_labframe_cartesian": (["tau","row","col"], Psi_3D_output_labframe_cartesian), # TO REMOVE -- change key name?
    #     },
    #     coords = {
    #         "tau": tau_array,
    #         "row": ["X","Y","Z"],
    #         "col": ["X","Y","Z"],
    #     },
    # )

    # dt = datatree.DataTree.from_dict({"inputs": inputs, "solver_output": solver_output})
    # dt.attrs = {
    #     "title": output_filename_suffix,
    #     "software_name": "scotty-beam-tracing",
    #     "software_version": __version__,
    #     "date_created": str(datetime.datetime.now()),
    #     "id": str(uuid.uuid4())
    # }

    inputs = xr.Dataset(
        {
            # Data pathways
            "magnetic_data_path": str(params.magnetic_data_path),
            "ne_data_path": str(params.ne_data_path),
            "Te_data_path": str(params.Te_data_path),
            "input_filename_suffix": str(params.input_filename_suffix),
            "output_path": str(params.output_path),
            "output_filename_suffix": str(params.output_filename_suffix),
            
            # Launch settings
            "poloidal_launch_angle_Torbeam": params.poloidal_launch_angle_deg_Torbeam,
            "toroidal_launch_angle_Torbeam": params.toroidal_launch_angle_deg_Torbeam,
            "launch_position_cartesian": (["col"], params.q_launch_cartesian),
            "launch_freq_GHz": params.launch_frequency_GHz,
            "launch_angular_frequency": params.launch_angular_frequency,
            "launch_beam_width": params.launch_beam_width,
            "launch_beam_curvature": params.launch_beam_curvature,
            "mode_flag": mode_flag, # TO REMOVE?
            "mode_flag_launch": params.mode_flag_launch, # This must be the same as "mode_flag"
            "poloidal_flux_enter": poloidal_flux_enter,
            "poloidal_flux_zero_density": poloidal_flux_zero_density,
            "poloidal_flux_zero_temperature": poloidal_flux_zero_temperature,
            "delta_X": params.delta_X,
            "delta_Y": params.delta_Y,
            "delta_Z": params.delta_Z,
            "delta_K_X": params.delta_K_X,
            "delta_K_Y": params.delta_K_Y,
            "delta_K_Z": params.delta_K_Z,
            "interp_order": interp_order,
            "interp_order_magnetic_data": params.interp_order_magnetic_data,
            "interp_order_ne_data": params.interp_order_ne_data,
            "interp_smoothing": params.interp_smoothing,
            "len_tau": params.len_tau,
            "rtol": params.rtol,
            "atol": params.atol,

            # Flags
            "vacuumLaunch_flag": vacuumLaunch_flag,
            "vacuum_propagation_flag": vacuum_propagation_flag,
            "relativistic_flag": relativistic_flag,
            "Psi_BC_flag": Psi_BC_flag,
            "quick_run_flag": quick_run,
            "figure_flag": figure_flag,
            "detailed_analysis_flag": detailed_analysis_flag,

            # Miscellaneous
            "find_B_method": str(find_B_method),
            "density_fit_parameters": str(density_fit_parameters),
            "temperature_fit_parameters": str(temperature_fit_parameters),
            "shot": shot,
            "equil_time": equil_time,
        },
        coords = {
            "X": field.X_coord,
            "Y": field.Y_coord,
            "Z": field.Z_coord,
            "row": ["X","Y","Z"],
            "col": ["X","Y","Z"],
        },
    )

    solver_output = xr.Dataset(
        {
            # Position
            "q_launch_cartesian": (["col"], params.q_launch_cartesian),
            "q_initial_cartesian": (["col"], params.q_initial_cartesian),
            "initial_position_cartesian": (["col"], params.q_initial_cartesian),
            "q_X": (["tau"], q_X_array, {"long_name": "X", "units": "m"}),
            "q_Y": (["tau"], q_Y_array, {"long_name": "Y", "units": "m"}),
            "q_Z": (["tau"], q_Z_array, {"long_name": "Z", "units": "m"}),
            
            # Wavevector
            "K_launch_cartesian":  (["col"], params.K_launch_cartesian),
            "K_initial_cartesian": (["col"], params.K_initial_cartesian),
            "initial_K_cartesian": (["col"], params.K_initial_cartesian),
            "K_X": (["tau"], K_X_array),
            "K_Y": (["tau"], K_Y_array),
            "K_Z": (["tau"], K_Z_array),
            
            # Psi
            "Psi_3D_launch_labframe_cartesian":  (["row","col"], params.Psi_3D_launch_labframe_cartesian),
            "Psi_3D_entry_labframe_cartesian":   (["row","col"], params.Psi_3D_entry_labframe_cartesian),
            "Psi_3D_initial_labframe_cartesian": (["row","col"], params.Psi_3D_initial_labframe_cartesian),
            "initial_Psi_3D_lab_cartesian": (["row","col"], params.Psi_3D_initial_labframe_cartesian),
            "Psi_3D_labframe_cartesian": (["tau","row","col"], Psi_3D_output_labframe_cartesian),

            # Additional stuff
            "solver_status": solver_status,
            "e_hat_initial": (["col"], params.e_hat_initial),
            "mode_flag_initial": params.mode_flag_initial,
            "mode_index": params.mode_index,
        },
        coords = {
            "tau": tau_array,
            "row": ["X","Y","Z"],
            "col": ["X","Y","Z"],
        },
    )

    dt = datatree.DataTree.from_dict({"inputs": inputs, "solver_output": solver_output})
    dt.attrs = {
        "title": output_filename_suffix,
        "software_name": "scotty-beam-tracing",
        "software_version": __version__,
        "date_created": str(datetime.datetime.now()),
        "id": str(uuid.uuid4())
    }

    # If the solver doesn't finish, end the function here
    if solver_status == -1:
        log.warning(f"Beam solver did not reach completion")
        if return_dt_field: return dt, field, hamiltonian, hamiltonian_other
    


    ##################################################
    #
    # DATA ANALYSIS ROUTINE
    #
    # Process the data from the main loop to give a
    # bunch of useful stuff
    #
    ##################################################

    print(f"Performing data analysis")
    dH = hamiltonian.derivatives(q_X_array, q_Y_array, q_Z_array,
                                 K_X_array, K_Y_array, K_Z_array,
                                 second_order = True)
    
    df = main_analysis_3D(solver_output = solver_output,
                          hamiltonian = hamiltonian,
                          hamiltonian_other = hamiltonian_other,
                          field = field,
                          density_fit = density_fit,
                          temperature_fit = temperature_fit)

    # df = immediate_analysis_3D(
    #     params,
    #     solver_output,
    #     field,
    #     density_fit,
    #     temperature_fit,
    #     hamiltonian,
    #     hamiltonian_other,
    #     params.launch_angular_frequency,
    #     params.mode_flag_initial,
    #     params.delta_X,
    #     params.delta_Y,
    #     params.delta_Z,
    #     params.delta_K_X,
    #     params.delta_K_Y,
    #     params.delta_K_Z,
    #     Psi_3D_lab_launch = None,             # Not used yet, so set vacuumLaunch_flag = False
    #     Psi_3D_lab_entry  = None,             # Not used yet, so set vacuumLaunch_flag = False
    #     distance_from_launch_to_entry = None, # Not used yet, so set vacuumLaunch_flag = False
    #     vacuumLaunch_flag = False,
    #     output_path = params.output_path,
    #     output_filename_suffix = params.output_filename_suffix,
    #     dH = dH,
    # )

    if further_analysis_flag:
        analysis = further_analysis_3D(
            params,
            inputs,
            solver_output,
            df,
            Psi_3D_entry_labframe_cartesian = Psi_3D_output_labframe_cartesian[0], # TO DELETE : this is only temporary, because vacuumLaunch_flag = False
            output_path = params.output_path,
            output_filename_suffix = params.output_filename_suffix,
            field = field,
            detailed_analysis_flag = False, # Set to False to disable localisation analysis (not implemented with 3D Scotty anyway)
            dH = dH,
        )

        df.update(analysis)
        dt["analysis"] = datatree.DataTree(df)

    # We need to use h5netcdf and invalid_netcdf in order to easily
    # write complex numbers
    dt.to_netcdf(
        params.output_path / f"scotty_output{params.output_filename_suffix}.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )

    if further_analysis_flag and figure_flag:
        print(f"Plotting and saving data")
        # plot_curvatures - TO REMOVE: need to complete this
        plot_delta_theta_m          (dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"delta_theta_m{params.output_filename_suffix}.png"))
        plot_dispersion_relation    (dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"H_cardano{params.output_filename_suffix}.png"))
        plot_localisations          (dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"localisations{params.output_filename_suffix}"))
        plot_theta_m                (dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"theta_m{params.output_filename_suffix}.png"))
        plot_trajectory             (dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"trajectory{params.output_filename_suffix}.png"))
        plot_trajectories_individual(dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"trajectories{params.output_filename_suffix}.png"))
        plot_wavevector             (dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"wavevector{params.output_filename_suffix}.png"))
        plot_widths                 (dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"widths{params.output_filename_suffix}.png"))

        # TO REMOVE -- experimental, need to clean up
        plot_trajectories_poloidal(field, dt.inputs, dt.solver_output, dt.analysis, filename=(params.output_path / f"poloidal_trajectory{params.output_filename_suffix}.png"))
        print(f"Figures saved to \n{params.output_path}")
    
    # TO REMOVE ??? idk
    print(f"##################################################")
    print(f"#")
    print(f"# FINISHED RUN !")
    print(f"#")
    print(f"##################################################")

    if return_dt_field: return dt, field, hamiltonian, hamiltonian_other



def make_fit(
    method: Optional[Union[str, ProfileFitLike]],
    poloidal_flux_zero_density: float,
    parameters: Optional[Sequence],
    filename: Optional[PathLike]
) -> ProfileFitLike:
    
    log = logging.getLogger(__name__)
    log.debug(f"Making density fit profile")
    
    if callable(method):
        return method
    
    if not isinstance(method, (str, type(None))):
        raise TypeError(
            f"Unexpected method type. Expected callable, str, or None, but got '{type(method)}'"
        )
    
    if parameters is None:
        raise ValueError(
            f"Passing `density_fit_method` ({method}) as string or None requires a list or array of parameters"
        )
    
    return profile_fit(method, poloidal_flux_zero_density, parameters, filename)