import datetime
import datatree
import json
import numpy as np
import pathlib
from scipy.integrate import solve_ivp
from scotty.analysis_3D import immediate_analysis_3D, further_analysis_3D
from scotty.checks_3D import Parameters, check_user_inputs, check_input_before_ray_tracing
from scotty.fun_evolution_3D import pack_beam_parameters_3D, unpack_beam_parameters_3D, beam_evolution_fun_3D
from scotty.fun_general_3D import timer
from scotty.geometry_3D import MagneticField_3D_Cartesian, InterpolatedField_3D_Cartesian
from scotty.hamiltonian_3D import initialise_hamiltonians, assign_hamiltonians
from scotty.launch_3D import find_plasma_entry_position, find_auto_delta_signs, find_plasma_entry_parameters
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
    mode_flag: Union[int, str],

    # Launch parameters (only used if ray propagation begins in the plasma)
    plasmaLaunch_K_cartesian: FloatArray = np.zeros(3),
    plasmaLaunch_Psi_3D_lab_cartesian: FloatArray = np.zeros([3, 3]),
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,

    # Input, output, and plotting arguments
    ne_data_path = pathlib.Path("."),
    magnetic_data_path = pathlib.Path("."),
    Te_data_path = pathlib.Path("."),
    input_filename_suffix: str = "",
    output_path = pathlib.Path("."),
    output_filename_suffix: str = "",
    figure_flag: bool = True,
    detailed_analysis_flag: bool = True,
    further_analysis_flag: bool = False,

    # Solver arguments
    shot=None,
    equil_time=None,
    find_B_method: Union[str, MagneticField_3D_Cartesian] = "torbeam",
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    vacuumLaunch_flag: bool = True,
    relativistic_flag: bool = False, # includes relativistic corrections to electron mass when set to True
    vacuum_propagation_flag: bool = False,
    Psi_BC_flag: Optional[Union[bool, str]] = None,
    poloidal_flux_enter: float = 1.0,
    poloidal_flux_zero_density: float = 1.0, # When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: float = 1.0, # Temperature analogue of poloidal_flux_zero_density

    # Finite-difference and solver parameters
    field: Optional[MagneticField_3D_Cartesian] = None,
    auto_delta_sign: bool = True,  # For flipping signs to maintain forward difference. Applies to delta_X, delta_Y, delta_Z
    delta_X: float = 0.0001, # in the same units as data_X_coord
    delta_Y: float = 0.0001, # in the same units as data_Y_coord
    delta_Z: float = 0.0001, # in the same units as data_Z_coord
    delta_K_X: float = 0.1,  # in the same units as K_X
    delta_K_Y: float = 0.1,  # in the same units as K_Y
    delta_K_Z: float = 0.1,  # in the same units as K_Z
    interp_order: Union[str, int] = 5, # For the 3D interpolation functions
    interp_smoothing = 5,  # For the 3D interpolation functions (specifically, make_fit)
    len_tau: int = 102,
    rtol: float = 1e-3, # for solve_ivp of the beam solver
    atol: float = 1e-6, # for solve_ivp of the beam solver

    # Additional flags
    quick_run: bool = False,       # For quick runs (only ray tracing)
    verbose_run: bool = False,     # For printing more messages
    return_dt_field: bool = False, # For returning the datatree and the field class

    # Only used for circular flux surfaces
    B_T_axis = None,
    B_p_a = None,
    R_axis = None,
    minor_radius_a = None,
) -> datatree.DataTree:
    
    print(f"##################################################")
    print(f"#")
    print(f"# Beam trace me up, Scotty!")
    print(f"# scotty version {__version__}")
    print(f"# Run ID: {uuid.uuid4()}")
    print(f"#")
    # TO REMOVE ??? idk
    print(f"##################################################")
    print(f"#")
    print(f"# STARTING RUN for")
    print(f"# pol. angle = {poloidal_launch_angle_Torbeam}")
    print(f"# tor. angle = {toroidal_launch_angle_Torbeam}")
    print(f"# launch freq = {launch_freq_GHz} GHz")
    print(f"#")
    print(f"##################################################")



    ##################################################
    #
    # STARTING ROUTINE
    #
    ##################################################

    # Loading all the parameters into `params`
    q_launch_cartesian = launch_position_cartesian # TO REMOVE -- left here for backward compatibility
    params = Parameters(poloidal_launch_angle_Torbeam,
                        toroidal_launch_angle_Torbeam,
                        mode_flag,
                        launch_freq_GHz,
                        launch_beam_width,
                        launch_beam_curvature,

                        q_launch_cartesian,
                        
                        # TO REMOVE -- can put back into here when we remove the vacuumLaunch processing below
                        # None, # K_plasmaLaunch_cartesian
                        # None, # Psi_3D_plasmaLaunch_labframe_cartesian,
                        
                        vacuumLaunch_flag,
                        vacuum_propagation_flag,
                        Psi_BC_flag,
                        
                        poloidal_flux_enter,
                        poloidal_flux_zero_density,
                        poloidal_flux_zero_temperature,

                        auto_delta_sign,
                        delta_X,
                        delta_Y,
                        delta_Z,
                        delta_K_X,
                        delta_K_Y,
                        delta_K_Z,
                        len_tau,
                        rtol,
                        atol,
                        
                        interp_order,
                        interp_smoothing,
                        density_fit_parameters,
                        density_fit_method,
                        temperature_fit_parameters,
                        temperature_fit_method,

                        magnetic_data_path,
                        ne_data_path,
                        Te_data_path,
                        input_filename_suffix,
                        output_path,
                        output_filename_suffix,

                        verbose_run,

                        plasmaLaunch_K_cartesian,
                        plasmaLaunch_Psi_3D_lab_cartesian,
                        )
    
    # Checking validity of user-specified inputs
    check_user_inputs(params)

    # Setting experimental profiles
    params.set_experimental_profiles()

    # Using the electron density data, create a spline fit as a
    # function of poloidal flux, i.e.:
    # density_fit(poloidal_flux) = electron_density
    if verbose_run: print(f"Making density fit profile")
    density_fit = make_fit(params.density_fit_method,
                           params.poloidal_flux_zero_density,
                           params.density_fit_parameters,
                           params.ne_filename)

    # TODO THIS SOON
    # Using the temperature data, create a spline fit as a
    # function of poloidal flux, i.e.:
    # temperature_fit(poloidal_flux) = temperature
    if verbose_run: print(f"Making temperature fit profile")
    if verbose_run: print(f"Skipping temperature fit profile -- code not done yet")
    temperature_fit = None



    ##################################################
    #
    # LAUNCH ROUTINE
    #
    ##################################################

    # If the user has already pre-loaded a field, use it
    # Otherwise, create a new one from the specified data pathway
    if field is None:
        if verbose_run: print(f"Reading and creating field profile")
        (field, duration_field_interpolation) = timer(create_magnetic_geometry_3D,
                                                      find_B_method,
                                                      params.magnetic_data_path,
                                                      params.input_filename_suffix,
                                                      params.interp_order_magnetic_data,
                                                      shot,
                                                      equil_time)
        if verbose_run: print(f"Reading and creating field profile took {duration_field_interpolation} s")
    else:
        if verbose_run: print(f"Using existing field profile")
    
    # Calculating the plasma entry position and auto_delta_sign
    if verbose_run: print(f"Calculating ray entry position")
    params.q_initial_cartesian = find_plasma_entry_position(params.poloidal_launch_angle_deg_Torbeam,
                                                            params.toroidal_launch_angle_deg_Torbeam,
                                                            params.q_launch_cartesian,
                                                            field,
                                                            params.poloidal_flux_enter,
                                                            boundary_adjust = 1e-6)

    # Setting the delta_signs
    if verbose_run: print(f"Setting delta signs")
    params.delta_X, params.delta_Y, params.delta_Z = find_auto_delta_signs(params.auto_delta_sign,
                                                                           params.q_initial_cartesian,
                                                                           params.delta_X,
                                                                           params.delta_Y,
                                                                           params.delta_Z,
                                                                           field)

    # Initialises the Hamiltonian H for mode_flags +1 and -1
    if verbose_run: print(f"Initialising the Hamiltonians for `mode_flag` = +1 and -1")
    hamiltonian_pos1, hamiltonian_neg1 = initialise_hamiltonians(params.launch_angular_frequency,
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
    print(f"Beam launched from outside the plasma")
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
                                                       hamiltonian_neg1)
    
    # Assigning the correct Hamiltonian
    hamiltonian, hamiltonian_other, params.mode_flag_initial = assign_hamiltonians(params.mode_flag_launch,
                                                                                   params.mode_flag_initial,
                                                                                   hamiltonian_pos1,
                                                                                   hamiltonian_neg1,
                                                                                   params.q_initial_cartesian,
                                                                                   params.K_initial_cartesian,
                                                                                   tol_H = 1e-5,
                                                                                   verbose_run = params.verbose_run)

    # Checking validity of user-specified arguments
    # TO REMOVE -- should put this in its own file
    # also should update this to make it cleaner? using OOP
    if verbose_run: print(f"Checking validity of user-specified arguments again")
    check_input_before_ray_tracing(params)
    


    ##################################################
    #
    # RAY TRACING ROUTINE
    #
    ##################################################

    print(f"Starting the ray solver")
    ray_solver_output, duration_ray_solver = timer(propagate_ray,
                                                   params.poloidal_flux_enter,
                                                   params.launch_angular_frequency,
                                                   field,
                                                   params.q_initial_cartesian,
                                                   params.K_initial_cartesian,
                                                   hamiltonian,
                                                   params.rtol,
                                                   params.atol,
                                                   quick_run,
                                                   params.len_tau)
    print(f"Ray solver took {duration_ray_solver} s")

    # TO REMOVE -- should change quick_run flag into ray_tracing only or something flag
    if quick_run: return ray_solver_output

    tau_leave, tau_points = cast(tuple, ray_solver_output)



    ##################################################
    #
    # BEAM TRACING ROUTINE
    #
    ##################################################

    # Packing the initial q_X, q_Y, q_Z, K_X, K_Y, K_Z, and the components of Psi_3D
    _beam_parameters_initial = pack_beam_parameters_3D(*params.q_initial_cartesian,
                                                       *params.K_initial_cartesian,
                                                        params.Psi_3D_initial_labframe_cartesian)
    
    print(f"Starting the beam solver")
    solver_beam_output, duration_beam_solver = timer(solve_ivp,
                                                     beam_evolution_fun_3D,
                                                     [0, tau_leave],
                                                     _beam_parameters_initial,
                                                     method = "RK45",
                                                     t_eval = tau_points,
                                                     dense_output = False,
                                                     events = None,
                                                     vectorized = False,
                                                     args = (hamiltonian,),
                                                     rtol = params.rtol,
                                                     atol = params.atol)
    
    print(f"Beam solver took {duration_beam_solver} s")
    print(f"Number of beam evolution evaluations: {solver_beam_output.nfev}")
    print(f"Time per beam evolution evaluation: {duration_beam_solver / solver_beam_output.nfev}")

    solver_status = solver_beam_output.status
    tau_array = solver_beam_output.t
    _beam_parameters_final = solver_beam_output.y

    (q_X_array, q_Y_array, q_Z_array,
     K_X_array, K_Y_array, K_Z_array,
     Psi_3D_output_labframe_cartesian) = unpack_beam_parameters_3D(_beam_parameters_final)

    print(f"Main loop complete")



    ##################################################
    #
    # DATA SAVING ROUTINE
    #
    ##################################################

    inputs = xr.Dataset(
        {
            # Data pathways
            "ne_data_path": str(params.ne_data_path),
            "magnetic_data_path": str(params.magnetic_data_path),
            "Te_data_path": str(params.Te_data_path),
            "output_path": str(params.output_path),
            "input_filename_suffix": str(params.input_filename_suffix),
            "output_filename_suffix": str(params.output_filename_suffix),
            
            # Important ray/beam stuff
            "poloidal_launch_angle_Torbeam": params.poloidal_launch_angle_deg_Torbeam,
            "toroidal_launch_angle_Torbeam": params.toroidal_launch_angle_deg_Torbeam,
            "launch_position_cartesian": (["col"], params.q_launch_cartesian),
            "launch_freq_GHz": params.launch_frequency_GHz,
            "launch_angular_frequency": params.launch_angular_frequency,
            "launch_beam_width": params.launch_beam_width,
            "launch_beam_curvature": params.launch_beam_curvature,
            "K_launch_cartesian": params.K_launch_cartesian,
            "mode_flag": mode_flag, # TO REMOVE?
            "mode_flag_launch": params.mode_flag_launch, # This must be the same as "mode_flag"
            "mode_flag_initial": params.mode_flag_initial,
            "mode_index": params.mode_index,
            "initial_position_cartesian": (["col"], params.q_initial_cartesian),
            "initial_K_cartesian": (["col"], params.K_initial_cartesian),
            "initial_Psi_3D_lab_cartesian": (["row","col"], params.Psi_3D_initial_labframe_cartesian),
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

            # Poloidal flux parameters
            "poloidal_flux_enter": poloidal_flux_enter,
            "poloidal_flux_zero_density": poloidal_flux_zero_density,
            "poloidal_flux_zero_temperature": poloidal_flux_zero_temperature,

            # Flags
            "vacuumLaunch_flag": vacuumLaunch_flag,
            "vacuum_propagation_flag": vacuum_propagation_flag,
            "relativistic_flag": relativistic_flag,
            "Psi_BC_flag": Psi_BC_flag,
            "quick_run": quick_run,
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
            # Solver output
            "solver_status": solver_status,
            "q_X": (["tau"], q_X_array, {"long_name": "X", "units": "m"}),
            "q_Y": (["tau"], q_Y_array, {"long_name": "Y", "units": "m"}),
            "q_Z": (["tau"], q_Z_array, {"long_name": "Z", "units": "m"}),
            "K_X": (["tau"], K_X_array),
            "K_Y": (["tau"], K_Y_array),
            "K_Z": (["tau"], K_Z_array),
            "Psi_3D_launch_labframe_cartesian": (["row","col"], params.Psi_3D_launch_labframe_cartesian),
            "Psi_3D_entry_labframe_cartesian": (["row","col"], params.Psi_3D_entry_labframe_cartesian),
            "initial_Psi_3D_lab_cartesian": (["row","col"], params.Psi_3D_initial_labframe_cartesian), # TO REMOVE -- change key name?
            "Psi_3D_labframe_cartesian": (["tau","row","col"], Psi_3D_output_labframe_cartesian), # TO REMOVE -- change key name?
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
        print(f"Beam solver did not reach completion")
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

    df = immediate_analysis_3D(
        params,
        solver_output,
        field,
        density_fit,
        temperature_fit,
        hamiltonian,
        params.launch_angular_frequency,
        params.mode_flag_initial,
        params.delta_X,
        params.delta_Y,
        params.delta_Z,
        params.delta_K_X,
        params.delta_K_Y,
        params.delta_K_Z,
        Psi_3D_lab_launch = None,             # Not used yet, so set vacuumLaunch_flag = False
        Psi_3D_lab_entry  = None,             # Not used yet, so set vacuumLaunch_flag = False
        distance_from_launch_to_entry = None, # Not used yet, so set vacuumLaunch_flag = False
        vacuumLaunch_flag = False,
        output_path = params.output_path,
        output_filename_suffix = params.output_filename_suffix,
        dH = dH,
    )

    if further_analysis_flag:
        analysis = further_analysis_3D(
            params,
            inputs,
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
        if verbose_run: print(f"Plotting and saving data")
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
        if verbose_run: print(f"Figures saved to \n{params.output_path}")
    
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





def create_magnetic_geometry_3D(
    find_B_method: Union[str, MagneticField_3D_Cartesian],
    magnetic_data_path: Optional[pathlib.Path] = None,
    input_filename_suffix: str = "",
    interp_order: int = 5,
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    **kwargs
) -> MagneticField_3D_Cartesian:
    
    if isinstance(find_B_method, MagneticField_3D_Cartesian): return find_B_method

    def missing_arg(argument: str) -> str:
        return f"Missing '{argument}' for find_B_method='{find_B_method}'"
    
    if find_B_method == "omfit_3D":
        print("Using OMFIT JSON Torbeam file for B and polflux")
        topfile_filename = magnetic_data_path / f"topfile{input_filename_suffix}.json"

        with open(topfile_filename) as f:
            data = json.load(f)
        
        X_grid = np.array(data["X"])
        Y_grid = np.array(data["Y"])
        Z_grid = np.array(data["Z"])

        try: B_X = np.array(data["Bx"])
        except KeyError: B_X = np.array(data["B_X"])
        else: B_X = np.array(data["Bx"])

        try: B_Y = np.array(data["By"])
        except KeyError: B_Y = np.array(data["B_Y"])
        else: B_Y = np.array(data["By"])

        try: B_Z = np.array(data["Bz"])
        except KeyError: B_Z = np.array(data["B_Z"])
        else: B_Z = np.array(data["Bz"])
        
        polflux = np.array(data["pol_flux"])

        return InterpolatedField_3D_Cartesian(
            X_grid,
            Y_grid,
            Z_grid,
            B_X,
            B_Y,
            B_Z,
            polflux,
            interp_order)