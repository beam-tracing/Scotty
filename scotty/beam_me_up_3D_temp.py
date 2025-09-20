import datetime
import datatree
import json
import numpy as np
import pathlib
from scipy.integrate import solve_ivp
from scotty.analysis_3D import immediate_analysis_3D, further_analysis_3D
from scotty.check_input import check_input
from scotty.fun_evolution_3D import pack_beam_parameters_3D, unpack_beam_parameters_3D, beam_evolution_fun_3D
from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.geometry_3D import MagneticField_3D_Cartesian, InterpolatedField_3D_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D
from scotty.launch_3D import LaunchParameters # launch_beam_3D, find_entry_point_3D
from scotty.plotting_3D import (
    plot_delta_theta_m,
    plot_dispersion_relation,
    plot_localisations,
    plot_theta_m,
    plot_trajectory,
    plot_trajectories_individually,
    plot_wavevector,
    plot_widths
)
from scotty.profile_fit import ProfileFitLike, profile_fit
from scotty.ray_solver_3D import propagate_ray
from scotty.typing import ArrayLike, FloatArray, PathLike
from scotty._version import __version__
import time
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
    mode_flag: int,

    # Launch parameters (only used if ray propagation begins in the plasma)
    plasmaLaunch_K_cartesian = np.zeros(3),
    plasmaLaunch_Psi_3D_lab_cartesian = np.zeros([3, 3]),
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,

    # Input, output, and plotting arguments
    ne_data_path = pathlib.Path("."),
    magnetic_data_path = pathlib.Path("."),
    Te_data_path = pathlib.Path("."),
    input_filename_suffix = "",
    field = None,
    output_path = pathlib.Path("."),
    output_filename_suffix = "",
    figure_flag = True,
    detailed_analysis_flag = True,
    further_analysis_flag = False,

    # Solver arguments
    shot=None,
    equil_time=None,
    find_B_method: Union[str, MagneticField_3D_Cartesian] = "torbeam",
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    vacuumLaunch_flag: bool = True,
    relativistic_flag: bool = False, # includes relativistic corrections to electron mass when set to True
    vacuum_propagation_flag: bool = False,
    Psi_BC_flag: Union[bool, str, None] = None,
    poloidal_flux_enter: float = 1.0,
    poloidal_flux_zero_density: float = 1.0, # When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: float = 1.0, # Temperature analogue of poloidal_flux_zero_density

    # Finite-difference and solver parameters
    auto_delta_sign = True,  # For flipping signs to maintain forward difference. Applies to delta_X, delta_Y, delta_Z
    delta_X: float = 0.0001, # in the same units as data_X_coord
    delta_Y: float = 0.0001, # in the same units as data_Y_coord
    delta_Z: float = 0.0001, # in the same units as data_Z_coord
    delta_K_X: float = 0.1,  # in the same units as K_X
    delta_K_Y: float = 0.1,  # in the same units as K_Y
    delta_K_Z: float = 0.1,  # in the same units as K_Z
    interp_order = 5,     # For the 3D interpolation functions
    interp_smoothing = 5, # For the 3D interpolation functions (specifically, density_fit)
    len_tau: int = 102,
    rtol: float = 1e-3,  # for solve_ivp of the beam solver
    atol: float = 1e-6,  # for solve_ivp of the beam solver

    # For quick runs (only ray tracing)
    quick_run: bool = False,

    # Only used for circular flux surfaces
    B_T_axis = None,
    B_p_a = None,
    R_axis = None,
    minor_radius_a = None,
) -> datatree.DataTree:

    print("Beam trace me up, Scotty!")
    print(f"scotty version {__version__}")
    run_id = uuid.uuid4()
    print(f"Run ID: {run_id}")

    # ------------------------------
    # Starting Routine
    # ------------------------------

    launch_angular_frequency = freq_GHz_to_angular_frequency(launch_freq_GHz)

    # Ensure paths are `pathlib.Path`
    ne_data_path = pathlib.Path(ne_data_path)
    magnetic_data_path = pathlib.Path(magnetic_data_path)
    Te_data_path = pathlib.Path(Te_data_path)
    output_path = pathlib.Path(output_path)

    # Setting experimental profiles
    if density_fit_parameters is None and (density_fit_method in [None, "smoothing-spline-file"]):
        ne_filename = ne_data_path / f"ne{input_filename_suffix}.dat"
        density_fit_parameters = [ne_filename, interp_order, interp_smoothing]

        # FIXME: Read data so it can be saved later
        ne_data = np.fromfile(ne_filename, dtype=float, sep="   ")
        # ne_data_density_array = ne_data[2::2]
        # ne_data_radialcoord_array = ne_data[1::2]
    else: ne_filename = None

    # Using the electron density data, create a spline fit as a
    # function of poloidal flux, i.e.:
    # find_density_1D(poloidal_flux) = electron_density
    find_density_1D = make_fit(
        density_fit_method,
        poloidal_flux_zero_density,
        density_fit_parameters,
        ne_filename,
    )

    # TODO THIS SOON
    # INSERT RELATIVISTIC_FLAG STUFF
    find_temperature_1D = None
    
    # If the user has already pre-loaded a field, use it
    # Otherwise, create a new one from the specified data pathway
    if field is None:
        field = create_magnetic_geometry_3D(
            find_B_method,
            magnetic_data_path,
            input_filename_suffix,
            interp_order,
            shot,
            equil_time,
        )
    









    # TO REMOVE -- just trying OOP
    q_launch_cartesian = launch_position_cartesian
    params = LaunchParameters(mode_flag,
                              poloidal_launch_angle_Torbeam,
                              toroidal_launch_angle_Torbeam,
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
                              auto_delta_sign,
                              delta_X,
                              delta_Y,
                              delta_Z,
                              delta_K_X,
                              delta_K_Y,
                              delta_K_Z)
    
    # Calculating the entry position and auto_delta_sign
    params.calculate_entry_position(field)

    # Initialises the Hamiltonian H
    hamiltonian = Hamiltonian_3D(
        field,
        launch_angular_frequency,
        mode_flag,
        find_density_1D,
        delta_X,
        delta_Y,
        delta_Z,
        delta_K_X,
        delta_K_Y,
        delta_K_Z,
        find_temperature_1D
    )

    # TO REMOVE -- should put this in its own file
    # also should update this to make it cleaner? using OOP
    # Checking validity of user-specified arguments
    check_input(
        mode_flag,
        poloidal_flux_enter,
        q_launch_cartesian,
        field,
        poloidal_flux_zero_density,
    )

    # ------------------------------
    # Launch parameters
    # ------------------------------
    if params.vacuumLaunch_flag:
        print("Beam launched from outside the plasma")
        params.calculate_beam_launch(field, hamiltonian, temperature=None)
    else:
        print("Beam launched from inside the plasma")
        params.q_initial_cartesian = q_launch_cartesian # TO REMOVE -- put this directly in calculate_entry_position?
                                                        # by inserting : if vacuumLaunch: self.q_initial_cartesian = q_launch_cartesian?
                                                        # around line 183
        # TO REMOVE -- can put the vacuumLaunch processing from the main beam_me_up_3D routine into launch_3D.py?
        params.K_launch_cartesian = None
        params.K_initial_cartesian = plasmaLaunch_K_cartesian
        params.Psi_3D_launch_labframe_cartesian = None
        params.Psi_3D_entry_labframe_cartesian = None
        params.Psi_3D_initial_labframe_cartesian = plasmaLaunch_Psi_3D_lab_cartesian
        params.distance_from_launch_to_entry = None
    









    # Old code -- TO REMOVE -- when updating overhauling OOP?
    """
    # Flips the sign of any of the delta_X, delta_Y, delta_Z
    # depending on the orientation of the flux surface. This
    # is to ensure a forward difference across the plasma boundary.
    if auto_delta_sign:
        entry_coords = find_entry_point_3D(
            launch_position_cartesian,
            poloidal_launch_angle_Torbeam,
            toroidal_launch_angle_Torbeam,
            poloidal_flux_enter,
            field,
        )

        entry_X, entry_Y, entry_Z = entry_coords

        if field.d_polflux_dX(entry_X, entry_Y, entry_Z, delta_X) > 0: delta_X = -1 * abs(delta_X)
        else: delta_X = abs(delta_X)

        if field.d_polflux_dY(entry_X, entry_Y, entry_Z, delta_Y) > 0: delta_Y = -1 * abs(delta_Y)
        else: delta_Y = abs(delta_Y)

        if field.d_polflux_dZ(entry_X, entry_Y, entry_Z, delta_Z) > 0: delta_Z = -1 * abs(delta_Z)
        else: delta_Z = abs(delta_Z)
    
    # Initialises the Hamiltonian H
    hamiltonian = Hamiltonian_3D(
        field,
        launch_angular_frequency,
        mode_flag,
        find_density_1D,
        delta_X,
        delta_Y,
        delta_Z,
        delta_K_X,
        delta_K_Y,
        delta_K_Z,
        find_temperature_1D
    )

    # Checking validity of user-specified arguments
    check_input(
        mode_flag,
        poloidal_flux_enter,
        launch_position_cartesian,
        field,
        poloidal_flux_zero_density,
    )

    # ------------------------------
    # Launch parameters
    # ------------------------------
    if vacuumLaunch_flag:
        print("Beam launched from outside the plasma")
        (
            q_launch_cartesian,  # q_launch_cartesian,
            q_initial_cartesian, # q_initial_cartesian,
            K_launch_cartesian,  # K_launch_cartesian,
            K_initial_cartesian, # K_initial_cartesian,
            Psi_3D_launch_labframe_cartesian,  # Psi_3D_launch_labframe_cartesian,
            Psi_3D_entry_labframe_cartesian,   # Psi_3D_entry_labframe_cartesian,
            Psi_3D_initial_labframe_cartesian, # Psi_3D_initial_labframe_cartesian,
            distance_from_launch_to_entry, # distance_from_launch_to_entry,
        ) = launch_beam_3D(
            poloidal_launch_angle_Torbeam,
            toroidal_launch_angle_Torbeam,
            launch_angular_frequency,
            launch_beam_width,
            launch_beam_curvature,
            launch_position_cartesian,
            mode_flag,
            field,
            hamiltonian,
            vacuumLaunch_flag,
            vacuum_propagation_flag,
            Psi_BC_flag,
            poloidal_flux_enter,
            delta_X,
            delta_Y,
            delta_Z,
            None,
        )
    else:
        print("Beam launched from inside the plasma")
        q_launch_cartesian = launch_position_cartesian
        q_initial_cartesian = launch_position_cartesian
        K_launch_cartesian = None
        K_initial_cartesian = plasmaLaunch_K_cartesian
        Psi_3D_launch_labframe_cartesian = None
        Psi_3D_entry_labframe_cartesian = None
        Psi_3D_initial_labframe_cartesian = plasmaLaunch_Psi_3D_lab_cartesian
        distance_from_launch_to_entry = None
    
            # ------------------------------
    # Propagating the ray
    # ------------------------------
    print("Starting the solvers")
    ray_solver_output = propagate_ray(
        poloidal_flux_enter,
        launch_angular_frequency,
        field,
        q_initial_cartesian,
        K_initial_cartesian,
        hamiltonian,
        rtol,
        atol,
        quick_run,
        len_tau,
    )

    if quick_run:
        return ray_solver_output

    tau_leave, tau_points = cast(tuple, ray_solver_output)

    # ------------------------------
    # Propagating the beam
    # ------------------------------

    # Initial conditions for the solver
    beam_parameters_initial = pack_beam_parameters_3D(
        q_initial_cartesian[0],
        q_initial_cartesian[1],
        q_initial_cartesian[2],
        K_initial_cartesian[0],
        K_initial_cartesian[1],
        K_initial_cartesian[2],
        Psi_3D_initial_labframe_cartesian,
    )

    solver_start_time = time.time()

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

    inputs = xr.Dataset(
        {
            # Data pathways
            "ne_data_path": str(ne_data_path),
            "magnetic_data_path": str(magnetic_data_path),
            "Te_data_path": str(Te_data_path),
            "output_path": str(output_path),
            "input_filename_suffix": str(input_filename_suffix),
            "output_filename_suffix": str(output_filename_suffix),
            
            # Important ray/beam stuff
            "poloidal_launch_angle_Torbeam": poloidal_launch_angle_Torbeam,
            "toroidal_launch_angle_Torbeam": toroidal_launch_angle_Torbeam,
            "launch_freq_GHz": launch_freq_GHz,
            "launch_angular_frequency": launch_angular_frequency,
            "launch_beam_width": launch_beam_width,
            "launch_beam_curvature": launch_beam_curvature,
            "mode_flag": mode_flag,
            "initial_position_cartesian": (["col"], q_initial_cartesian),
            "initial_K_cartesian": (["col"], K_initial_cartesian),
            "initial_Psi_3D_lab_cartesian": (["row","col"], Psi_3D_initial_labframe_cartesian),
            "delta_X": delta_X,
            "delta_Y": delta_Y,
            "delta_Z": delta_Z,
            "delta_K_X": delta_K_X,
            "delta_K_Y": delta_K_Y,
            "delta_K_Z": delta_K_Z,
            "interp_order": interp_order,
            "interp_smoothing": interp_smoothing,
            "len_tau": len_tau,
            "rtol": rtol,
            "atol": atol,

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
            "Psi_3D_launch_labframe_cartesian": (["row","col"], Psi_3D_launch_labframe_cartesian),
            "Psi_3D_entry_labframe_cartesian": (["row","col"], Psi_3D_entry_labframe_cartesian),
            "initial_Psi_3D_lab_cartesian": (["row","col"], Psi_3D_initial_labframe_cartesian),
            "Psi_3D_labframe_cartesian": (["tau","row","col"], Psi_3D_output),
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
        "id": str(run_id),
    }

    if solver_status == -1:
        # If the solver doesn't finish, end the function here
        print("Solver did not reach completion")
        return
    
    # -------------------
    # Process the data from the main loop to give a bunch of useful stuff
    # -------------------
    print("Analysing data")
    dH = hamiltonian.derivatives(q_X_array, q_Y_array, q_Z_array, K_X_array, K_Y_array, K_Z_array, second_order=True)

    df = immediate_analysis_3D(
        solver_output,
        field,
        find_density_1D,
        find_temperature_1D,
        hamiltonian,
        launch_angular_frequency,
        mode_flag,
        delta_X,
        delta_Y,
        delta_Z,
        delta_K_X,
        delta_K_Y,
        delta_K_Z,
        Psi_3D_lab_launch = None,             # Not used yet, so set vacuumLaunch_flag = False
        Psi_3D_lab_entry  = None,             # Not used yet, so set vacuumLaunch_flag = False
        distance_from_launch_to_entry = None, # Not used yet, so set vacuumLaunch_flag = False
        vacuumLaunch_flag = False,
        output_path = output_path,
        output_filename_suffix = output_filename_suffix,
        dH = dH,
    )

    if further_analysis_flag:
        analysis = further_analysis_3D(
            inputs,
            df,
            Psi_3D_entry_labframe_cartesian = Psi_3D_output[0], # TO DELETE : this is only temporary, because vacuumLaunch_flag = False
            output_path = output_path,
            output_filename_suffix = output_filename_suffix,
            field = field,
            detailed_analysis_flag = False, # Set to False to disable localisation analysis (not implemented with 3D Scotty anyway)
            dH = dH,
        )

        df.update(analysis)
        dt["analysis"] = datatree.DataTree(df)

    # We need to use h5netcdf and invalid_netcdf in order to easily
    # write complex numbers
    dt.to_netcdf(
        output_path / f"scotty_output{output_filename_suffix}.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )

    if further_analysis_flag and figure_flag:
        plot_delta_theta_m(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"delta_theta_m{output_filename_suffix}.png"))
        plot_dispersion_relation(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"H{output_filename_suffix}.png"))
        plot_localisations(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"localisations{output_filename_suffix}.png"))
        plot_theta_m(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"theta_m{output_filename_suffix}.png"))
        plot_trajectory(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"trajectory{output_filename_suffix}.png"))
        plot_trajectories_individually(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"trajectories{output_filename_suffix}.png"))
        plot_wavevector(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"wavevector{output_filename_suffix}.png"))

    # return dt, field
    # TO REMOVE
    return dt, field, hamiltonian, q_launch_cartesian, q_initial_cartesian, K_launch_cartesian, K_initial_cartesian, Psi_3D_initial_labframe_cartesian
    """










    # ------------------------------
    # Propagating the ray
    # ------------------------------
    print("Starting the solvers")
    ray_solver_output = propagate_ray(
        poloidal_flux_enter,
        launch_angular_frequency,
        field,
        params.q_initial_cartesian,
        params.K_initial_cartesian,
        hamiltonian,
        rtol,
        atol,
        quick_run,
        len_tau,
    )

    # TO REMOVE -- should change quick_run flag into ray_tracing only or something flag
    if quick_run:
        return ray_solver_output

    tau_leave, tau_points = cast(tuple, ray_solver_output)

    # ------------------------------
    # Propagating the beam
    # ------------------------------

    # Initial conditions for the solver
    beam_parameters_initial = pack_beam_parameters_3D(
        *params.q_initial_cartesian,
        *params.K_initial_cartesian,
        params.Psi_3D_initial_labframe_cartesian
    )

    solver_start_time = time.time()

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

    q_X_array, q_Y_array, q_Z_array, K_X_array, K_Y_array, K_Z_array, Psi_3D_output_labframe_cartesian = unpack_beam_parameters_3D(beam_parameters_final)

    print("Main loop complete")

    inputs = xr.Dataset(
        {
            # Data pathways
            "ne_data_path": str(ne_data_path),
            "magnetic_data_path": str(magnetic_data_path),
            "Te_data_path": str(Te_data_path),
            "output_path": str(output_path),
            "input_filename_suffix": str(input_filename_suffix),
            "output_filename_suffix": str(output_filename_suffix),
            
            # Important ray/beam stuff
            "poloidal_launch_angle_Torbeam": poloidal_launch_angle_Torbeam,
            "toroidal_launch_angle_Torbeam": toroidal_launch_angle_Torbeam,
            "launch_freq_GHz": launch_freq_GHz,
            "launch_angular_frequency": launch_angular_frequency,
            "launch_beam_width": launch_beam_width,
            "launch_beam_curvature": launch_beam_curvature,
            "mode_flag": mode_flag,
            "initial_position_cartesian": (["col"], params.q_initial_cartesian),
            "initial_K_cartesian": (["col"], params.K_initial_cartesian),
            "initial_Psi_3D_lab_cartesian": (["row","col"], params.Psi_3D_initial_labframe_cartesian),
            "delta_X": delta_X,
            "delta_Y": delta_Y,
            "delta_Z": delta_Z,
            "delta_K_X": delta_K_X,
            "delta_K_Y": delta_K_Y,
            "delta_K_Z": delta_K_Z,
            "interp_order": interp_order,
            "interp_smoothing": interp_smoothing,
            "len_tau": len_tau,
            "rtol": rtol,
            "atol": atol,

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
        "id": str(run_id),
    }

    if solver_status == -1:
        # If the solver doesn't finish, end the function here
        print("Solver did not reach completion")
        return
    
    # -------------------
    # Process the data from the main loop to give a bunch of useful stuff
    # -------------------
    print("Analysing data")
    dH = hamiltonian.derivatives(q_X_array, q_Y_array, q_Z_array, K_X_array, K_Y_array, K_Z_array, second_order=True)

    df = immediate_analysis_3D(
        solver_output,
        field,
        find_density_1D,
        find_temperature_1D,
        hamiltonian,
        launch_angular_frequency,
        mode_flag,
        delta_X,
        delta_Y,
        delta_Z,
        delta_K_X,
        delta_K_Y,
        delta_K_Z,
        Psi_3D_lab_launch = None,             # Not used yet, so set vacuumLaunch_flag = False
        Psi_3D_lab_entry  = None,             # Not used yet, so set vacuumLaunch_flag = False
        distance_from_launch_to_entry = None, # Not used yet, so set vacuumLaunch_flag = False
        vacuumLaunch_flag = False,
        output_path = output_path,
        output_filename_suffix = output_filename_suffix,
        dH = dH,
    )

    if further_analysis_flag:
        analysis = further_analysis_3D(
            inputs,
            df,
            Psi_3D_entry_labframe_cartesian = Psi_3D_output_labframe_cartesian[0], # TO DELETE : this is only temporary, because vacuumLaunch_flag = False
            output_path = output_path,
            output_filename_suffix = output_filename_suffix,
            field = field,
            detailed_analysis_flag = False, # Set to False to disable localisation analysis (not implemented with 3D Scotty anyway)
            dH = dH,
        )

        df.update(analysis)
        dt["analysis"] = datatree.DataTree(df)

    # We need to use h5netcdf and invalid_netcdf in order to easily
    # write complex numbers
    dt.to_netcdf(
        output_path / f"scotty_output{output_filename_suffix}.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )

    if further_analysis_flag and figure_flag:
        plot_delta_theta_m(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"delta_theta_m{output_filename_suffix}.png"))
        plot_dispersion_relation(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"H{output_filename_suffix}.png"))
        plot_localisations(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"localisations{output_filename_suffix}"))
        plot_theta_m(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"theta_m{output_filename_suffix}.png"))
        plot_trajectory(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"trajectory{output_filename_suffix}.png"))
        plot_trajectories_individually(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"trajectories{output_filename_suffix}.png"))
        plot_wavevector(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"wavevector{output_filename_suffix}.png"))
        plot_widths(dt.inputs, dt.solver_output, dt.analysis, filename=(output_path / f"widths{output_filename_suffix}.png"))

    # return dt, field
    # TO REMOVE
    return (dt,
            field,
            hamiltonian,
            params.q_launch_cartesian,
            params.q_initial_cartesian,
            params.K_launch_cartesian,
            params.K_initial_cartesian,
            params.Psi_3D_initial_labframe_cartesian)



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
        B_X = np.array(data["Bx"])
        B_Y = np.array(data["By"])
        B_Z = np.array(data["Bz"])
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